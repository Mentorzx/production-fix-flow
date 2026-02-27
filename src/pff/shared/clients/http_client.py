"""Provide module-level functionality for the PFF codebase.



Notes:

    File: src/pff/shared/clients/http_client.py

"""

from __future__ import annotations

import asyncio
import os
import re
import threading
import time
from collections.abc import Callable, Coroutine, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Protocol, TYPE_CHECKING
from urllib.parse import urlsplit

import httpx
import orjson
from dotenv import load_dotenv
from urllib3 import disable_warnings, exceptions

from pff.shared.core.config import API_HOSTS_CONFIG_PATH, settings

from ..core.cache import CacheManager
from ..core.file_manager import FileManager
from ..core.logging import logger

_ROOT = settings.ROOT_DIR
_HOST_FILE: Final = API_HOSTS_CONFIG_PATH
_ALL_HOSTS: dict[str, dict[str, str]] | None = None
_ENV_LOADED = False
_HOSTS_LOCK = threading.Lock()
_ENV_LOCK = threading.Lock()
_TIMEOUTS_BY_ENDPOINT: dict[str, dict[str, float]] = {
    "BIAS": {"connect": 5.0, "read": 15.0, "write": 10.0},
    "BAE": {"connect": 3.0, "read": 5.0, "write": 5.0},
    "CPM": {"connect": 3.0, "read": 8.0, "write": 8.0},
    "RMVIVO": {"connect": 3.0, "read": 3.0, "write": 3.0},
    "default": {"connect": 3.0, "read": 10.0, "write": 10.0},
}
_ENDPOINT_METRICS: dict[str, dict[str, Any]] = {
    "BIAS": {"success": 0, "failures": 0, "latencies_ms": []},
    "BAE": {"success": 0, "failures": 0, "latencies_ms": []},
    "CPM": {"success": 0, "failures": 0, "latencies_ms": []},
    "RMVIVO": {"success": 0, "failures": 0, "latencies_ms": []},
}
_ENDPOINT_METRICS_LOCK = threading.Lock()


def _ensure_env_loaded() -> None:
    """Load .env once to avoid import-time I/O."""
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    with _ENV_LOCK:
        if _ENV_LOADED:
            return
        load_dotenv(_ROOT / ".env", override=False)
        _ENV_LOADED = True


def _load_all_hosts() -> dict[str, dict[str, str]]:
    """Execute load all hosts.



    Returns:

        Return value produced by the callable.

    """

    global _ALL_HOSTS
    if _ALL_HOSTS is not None:
        return _ALL_HOSTS
    with _HOSTS_LOCK:
        if _ALL_HOSTS is None:
            _ensure_env_loaded()

            _ALL_HOSTS = FileManager().read(_HOST_FILE, return_native=True)

            import polars as pl

            if isinstance(_ALL_HOSTS, pl.DataFrame):
                _ALL_HOSTS = _ALL_HOSTS.to_dict(as_series=False)  # type: ignore[assignment]

            return _ALL_HOSTS  # type: ignore[return-value]
    return _ALL_HOSTS  # type: ignore[return-value]


def _resolve_cluster_order(all_hosts: dict[str, dict[str, str]]) -> list[str]:
    order = [
        c.strip().upper()
        for c in os.getenv("API_CLUSTER_ORDER", "JAG,PAL,BAR,FUN").split(",")
        if c.strip().upper() in all_hosts
    ]
    return order or ["BAR"]


class FailoverStrategy(Protocol):
    """Represent FailoverStrategy.



    Notes:

        Encapsulates behavior while preserving architecture boundaries.

    """

    def cycle(self) -> Iterator[str]:
        """Yield hosts in the failover order."""
        ...

    def report_success(self, host: str, latency: float) -> None:
        """Record a successful request for host metrics."""
        ...

    def report_failure(self, host: str) -> None:
        """Record a failed request for host metrics."""
        ...


class RoundRobin(FailoverStrategy):
    """Estratégia stateless que sempre itera na mesma ordem."""

    __slots__ = ("_hosts", "_idx")

    def __init__(self, service: str, order: list[str]):
        """Execute init.



        Args:

            service: Input value used by this callable.

            order: Input value used by this callable.



        Raises:

            Exception: Propagates domain-specific failures with context.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        all_hosts = _load_all_hosts()
        self._hosts: list[str] = []
        for c in order:
            cluster_data = all_hosts.get(c)
            host: str | None = None

            if cluster_data is None:
                pass
            elif hasattr(cluster_data, "columns") and service in getattr(
                cluster_data, "columns", []
            ):
                host = cluster_data[service][0]  # type: ignore[index]
            elif hasattr(cluster_data, "get"):
                host = cluster_data.get(service)  # type: ignore[assignment]

            if host:
                self._hosts.append(host)

        if not self._hosts:
            raise ValueError(
                f"Serviço '{service}' ausente em todos os clusters: {order}"
            )
        self._idx = 0

    @property
    def current(self) -> str:
        """Execute current.



        Returns:

            Return value produced by the callable.

        """

        return self._hosts[self._idx]

    def advance(self) -> None:
        """Execute advance."""

        self._idx = (self._idx + 1) % len(self._hosts)

    def cycle(self) -> Iterator[str]:
        """Execute cycle.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        for _ in range(len(self._hosts)):
            yield self.current
            self.advance()

    def report_success(self, host: str, latency: float) -> None:
        """O RoundRobin não mantém estado, então esta operação é nula."""
        pass

    def report_failure(self, host: str) -> None:
        """O RoundRobin não mantém estado, então esta operação é nula."""
        pass


class LatencyAwareStrategy(FailoverStrategy):
    """
    Stateful strategy that orders hosts based on health (failures)
    and historical latency, prioritizing the fastest and healthiest ones.
    """

    def __init__(self, service: str, order: list[str]):
        """Execute init.



        Args:

            service: Input value used by this callable.

            order: Input value used by this callable.



        Raises:

            Exception: Propagates domain-specific failures with context.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        self._service_name = service
        all_hosts = _load_all_hosts()
        self._hosts: list[str] = []
        for c in order:
            cluster_data = all_hosts.get(c)
            host: str | None = None

            if cluster_data is None:
                pass
            elif hasattr(cluster_data, "columns") and service in getattr(
                cluster_data, "columns", []
            ):
                host = cluster_data[service][0]  # type: ignore[index]
            elif hasattr(cluster_data, "get"):
                host = cluster_data.get(service)  # type: ignore[assignment]

            if host:
                self._hosts.append(host)

        if not self._hosts:
            raise ValueError(
                f"Serviço '{service}' ausente em todos os clusters: {order}"
            )

        self._latencies = {host: 0.1 for host in self._hosts}
        self._failures = {host: 0 for host in self._hosts}
        self._last_failure_time = {host: 0.0 for host in self._hosts}
        self._lock = threading.Lock()

    def report_success(self, host: str, latency: float) -> None:
        """Execute report success.



        Args:

            host: Input value used by this callable.

            latency: Input value used by this callable.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        with self._lock:
            current_latency = self._latencies.get(host, 0.1)
            self._latencies[host] = (current_latency * 0.8) + (latency * 0.2)
            self._failures[host] = 0

    def report_failure(self, host: str) -> None:
        """Execute report failure.



        Args:

            host: Input value used by this callable.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        with self._lock:
            if host in self._hosts:
                self._failures[host] += 1
                self._last_failure_time[host] = time.time()

    def cycle(self) -> Iterator[str]:
        """Execute cycle.



        Returns:

            Return value produced by the callable.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        with self._lock:

            def sort_key(host):
                """Execute sort key.



                Args:

                    host: Input value used by this callable.



                Returns:

                    Return value produced by the callable.

                """

                is_healthy = self._failures.get(host, 0) < 3
                is_recent_failure = (
                    time.time() - self._last_failure_time.get(host, 0)
                ) < 60
                latency = self._latencies.get(host, 999.0)

                return (not is_healthy, is_recent_failure, latency)

            sorted_hosts = sorted(self._hosts, key=sort_key)

        yield from sorted_hosts


class EndpointFactory:
    """Represent EndpointFactory.



    Notes:

        Encapsulates behavior while preserving architecture boundaries.

    """

    def __init__(self, strategy_name: str = "latency-aware"):
        """Execute init.



        Args:

            strategy_name: Optional input value.



        Raises:

            Exception: Propagates domain-specific failures with context.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        all_hosts = _load_all_hosts()
        self._order = _resolve_cluster_order(all_hosts)

        strategy_map = {
            "round-robin": RoundRobin,
            "latency-aware": LatencyAwareStrategy,
        }
        strategy_cls = strategy_map.get(strategy_name.lower())
        if not strategy_cls:
            raise ValueError(f"Estratégia '{strategy_name}' desconhecida.")

        services: set[str] = set()
        for cluster in self._order:
            cluster_data = all_hosts.get(cluster)
            svc_keys: list[str] = []

            if cluster_data is None:
                pass
            elif hasattr(cluster_data, "columns"):
                svc_keys = list(getattr(cluster_data, "columns", []))
            elif hasattr(cluster_data, "to_dict"):
                svc_keys = list(cluster_data.to_dict().keys())  # type: ignore[union-attr]
            elif hasattr(cluster_data, "keys"):
                svc_keys = list(cluster_data.keys())

            for svc in svc_keys:
                services.add(svc)

        self._strategies: dict[str, FailoverStrategy] = {
            svc: strategy_cls(svc, self._order) for svc in services
        }

    def _get_strategy(self, svc: str) -> FailoverStrategy:
        """Execute get strategy.



        Args:

            svc: Input value used by this callable.



        Returns:

            Return value produced by the callable.



        Raises:

            Exception: Propagates domain-specific failures with context.

        """

        strategy = self._strategies.get(svc.strip().upper())
        if strategy is None:
            raise RuntimeError(f"Serviço '{svc}' não configurado.")
        return strategy

    def cycle(self, svc: str) -> Iterator[str]:
        """Execute cycle.



        Args:

            svc: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        return self._get_strategy(svc).cycle()

    def report_success(self, host: str, svc: str, latency: float):
        """Reporta um sucesso para a estratégia do serviço, para que ela aprenda."""
        self._get_strategy(svc).report_success(host, latency)

    def report_failure(self, host: str, svc: str):
        """Reporta uma falha para a estratégia do serviço, para que ela aprenda."""
        self._get_strategy(svc).report_failure(host)

    def build(self, *, path_only: bool = False) -> APIsEndpoints:
        """Execute build.



        Args:

            path_only: Optional input value.



        Returns:

            Return value produced by the callable.

        """

        host_fn: Callable[[str], str] = (lambda _s: "") if path_only else self.host
        return APIsEndpoints(host_fn)

    def host(self, svc: str) -> str:
        """Retorna o melhor host disponível para o serviço."""
        strategy = self._get_strategy(svc)
        return next(strategy.cycle())


@dataclass(slots=True)
class APIsEndpoints:
    """Represent APIsEndpoints.



    Notes:

        Encapsulates behavior while preserving architecture boundaries.

    """

    _host: Callable[[str], str]

    def customer_enquiry(self, msisdn: str) -> tuple[str, str]:
        """Execute customer enquiry.



        Args:

            msisdn: Input value used by this callable.



        Returns:

            Return value produced by the callable.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        service_type = "BIAS"
        url = join(
            self._host(service_type),
            "bias/bssfsdCustomerEnquiry/v1/customer"
            f"?includeTerminatedContracts=true&includeContracts=true"
            f"&communicationIdType=E.164&communicationId=55{msisdn}",
        )
        return url, service_type

    def customer_enquiry_by_customer(self, cid: str) -> tuple[str, str]:
        """Execute customer enquiry by customer.



        Args:

            cid: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        service_type = "BIAS"
        url = join(
            self._host(service_type),
            f"bias/bssfsdCustomerEnquiry/v1/customer?customerId={cid}",
        )
        return url, service_type

    def individual_party_enquiry(self, ext: str) -> tuple[str, str]:
        """Execute individual party enquiry.



        Args:

            ext: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        service_type = "BAE"
        url = join(
            self._host(service_type),
            f"bae/bssfIndividualPartyEnquiry/v1/individualParty/?externalId={ext}",
        )
        return url, service_type

    def read_contract(self, ctt: str, cust: str) -> tuple[str, str]:
        """Execute read contract.



        Args:

            ctt: Input value used by this callable.

            cust: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        service_type = "CPM"
        url = join(
            self._host(service_type),
            f"cpm/business/v1/readContract/customer/{cust}/contract/{ctt}?validAt=*",
        )
        return url, service_type

    @property
    def update_contract_status(self) -> tuple[str, str]:
        """Execute update contract status.



        Returns:

            Return value produced by the callable.

        """

        service_type = "BIAS"
        url = join(
            self._host(service_type), "bias/vivoUpdateContractStatus/v1/updateStatus"
        )
        return url, service_type

    def deactivate_contract(self, msisdn: str) -> tuple[str, str]:
        """Execute deactivate contract.



        Args:

            msisdn: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        service_type = "BIAS"
        url = join(
            self._host(service_type),
            f"bias/vivoDeactivateContract/v1/communicationId/{msisdn}/communicationIdType/E.164",
        )
        return url, service_type

    def activate_product(self, msisdn: str) -> tuple[str, str]:
        """Execute activate product.



        Args:

            msisdn: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        service_type = "BIAS"
        url = join(
            self._host(service_type),
            f"bias/vivoActivateProduct/v1/communicationId/{msisdn}/communicationIdType/E.164",
        )
        return url, service_type

    @property
    def manage_consumer_list(self) -> tuple[str, str]:
        """Execute manage consumer list.



        Returns:

            Return value produced by the callable.

        """

        service_type = "BIAS"
        url = join(
            self._host(service_type),
            "bias/bssfsdSubscriptionManagement/v1/rmil-manage-consumer-list/",
        )
        return url, service_type

    @property
    def create_client(self) -> tuple[str, str]:
        """Execute create client.



        Returns:

            Return value produced by the callable.

        """

        service_type = "BIAS"
        url = join(self._host(service_type), "bias/vivoCreateClient/v1/customer")
        return url, service_type

    def subscription(self, cust: str, ctt: str) -> tuple[str, str]:
        """Execute subscription.



        Args:

            cust: Input value used by this callable.

            ctt: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        service_type = "BAE"
        url = join(
            self._host(service_type),
            f"bae/bssfSubscriptionManagement/v1/customer/{cust}/contract/{ctt}",
        )
        return url, service_type

    def delete_contract(self, cid: str, ctid: str) -> tuple[str, str]:
        """Execute delete contract.



        Args:

            cid: Input value used by this callable.

            ctid: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        service_type = "BAE"
        url = join(
            self._host(service_type),
            f"bae/bssfSubscriptionManagement/v1/customer/{cid}/contract/{ctid}",
        )
        return url, service_type

    def party_cascade(self, pid: str) -> tuple[str, str]:
        """Execute party cascade.



        Args:

            pid: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        service_type = "CPM"
        url = join(self._host(service_type), f"cpm/business/v1/updateParty/party/{pid}")
        return url, service_type


def join(base: str, path: str) -> str:
    """Execute join.



    Args:

        base: Input value used by this callable.

        path: Input value used by this callable.



    Returns:

        Return value produced by the callable.

    """

    return f"{base.rstrip('/')}/{path.lstrip('/')}"


_api_factory_instance: EndpointFactory | None = None
_API_instance: APIsEndpoints | None = None


def _get_api_factory() -> EndpointFactory:
    """Execute get api factory.



    Returns:

        Return value produced by the callable.

    """

    global _api_factory_instance
    if _api_factory_instance is None:
        _api_factory_instance = EndpointFactory()
    return _api_factory_instance


def _fallback_hosts_from_config(service: str) -> list[str]:
    """Extract host candidates from generic service-style host config."""
    all_hosts = _load_all_hosts()
    candidates: list[str] = []

    services = all_hosts.get("services")
    if isinstance(services, dict):
        for value in services.values():
            if not isinstance(value, dict):
                continue
            host = value.get("host")
            if isinstance(host, str) and host.strip():
                candidates.append(host.strip())

    if not candidates:
        logger.warning(
            f"No explicit hosts found for service '{service}'. Falling back to localhost."
        )
        return ["localhost"]

    return list(dict.fromkeys(candidates))


if TYPE_CHECKING:
    API: APIsEndpoints

_DEFAULT_TIMEOUT = 10.0
_DEFAULT_RETRIES = 3
_DEFAULT_BACKOFF = 0.5
_BENIGN_ERRORS = {
    (403, "BIAS.UpdateContractStatusTransitionNotAllowed"),
    (409, "BIAS.DuplicateResource"),
}
_FRIENDLY_PT = {
    400: "Requisição inválida.",
    401: "Não autorizado.",
    403: "Operação não permitida.",
    404: "Recurso não encontrado.",
    409: "O recurso já existe.",
    500: "Erro interno do servidor. (Operação redundante ?).",
    503: "Serviço temporariamente indisponível.",
    599: "Falha de comunicação com o servidor. Verifique sua conexão local ou sua VPN.",
}


class HttpClient:
    """A reusable and robust asynchronous HTTP client for external APIs with HTTP/2 support."""

    _vpn_logged: bool = False

    def __init__(
        self,
        *,
        observation_callback: Callable[..., Coroutine[Any, Any, None]] | None = None,
        **kwargs,
    ) -> None:
        """Execute init.



        Args:

            observation_callback: Optional input value.

            **kwargs: Additional keyword arguments.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        self._timeout = kwargs.get("timeout", _DEFAULT_TIMEOUT)
        self._retries = kwargs.get("retries", _DEFAULT_RETRIES)
        self._backoff = kwargs.get("backoff", _DEFAULT_BACKOFF)
        self._observation_callback = observation_callback
        self.cache = CacheManager()

        self._ca_bundle = os.getenv("PFF_CA_BUNDLE")
        verify_ssl: bool | str = True
        if self._ca_bundle and Path(self._ca_bundle).exists():
            logger.debug("HTTPS verification using CA bundle: {}", self._ca_bundle)
            verify_ssl = self._ca_bundle
        else:
            disable_warnings(exceptions.InsecureRequestWarning)
            verify_ssl = False

        http2_enabled = False
        try:
            import importlib.util

            http2_enabled = importlib.util.find_spec("h2") is not None
            if http2_enabled:
                logger.debug("HTTP/2 support enabled (h2 package available)")
        except ImportError:
            pass

        self._client = httpx.AsyncClient(
            http2=http2_enabled,
            timeout=httpx.Timeout(
                connect=3.0,
                read=self._timeout,
                write=self._timeout,
                pool=self._timeout * 2,
            ),
            limits=httpx.Limits(
                max_connections=100,
                max_keepalive_connections=50,
            ),
            verify=verify_ssl,
            follow_redirects=True,
        )

        self._last_response: httpx.Response | None = None

    async def __aenter__(self) -> HttpClient:
        return self

    async def __aexit__(self, *_exc) -> None:
        await self.close()

    async def close(self) -> None:
        """Close the underlying httpx client."""
        if self._client:
            await self._client.aclose()

    def _build_host_candidates(
        self, url: str, method: str, **request_kwargs
    ) -> list[tuple[tuple, dict]]:
        """
        Build a list of parameter combinations to be tried by async failover.
        It supports:
          * Absolute URLs (passed untouched)
          * Relative paths resolved through EndpointFactory fail-over
        """
        parsed = urlsplit(url)
        if not parsed.netloc and re.match(r"^[\w.\-]+:\d+", parsed.path):
            host, _, rest = parsed.path.partition("/")
            parsed = parsed._replace(netloc=host, path="/" + rest)

        base_path = parsed.path + (f"?{parsed.query}" if parsed.query else "")
        combinations: list[tuple[tuple, dict]] = []
        if parsed.netloc:
            full_url = f"{parsed.scheme + '://' if parsed.scheme else ''}{parsed.netloc}{base_path}"
            combinations.append(
                ((), {**request_kwargs, "method": method, "url": full_url})
            )
            return combinations

        service = (
            "BIAS"
            if "/bias/" in url.lower()
            else (
                "CPM"
                if "/cpm/" in url.lower()
                else "RMVIVO" if "rmvivo" in url.lower() else "BAE"
            )
        )
        try:
            hosts = list(_get_api_factory().cycle(service))
        except RuntimeError:
            hosts = _fallback_hosts_from_config(service)

        for host in hosts:
            for scheme in (parsed.scheme,) if parsed.scheme else ("http", "https"):
                combinations.append(
                    (
                        (),
                        {
                            **request_kwargs,
                            "method": method,
                            "url": f"{scheme}://{host}{base_path}",
                        },
                    )
                )
        return combinations

    async def _extract_response_content(
        self, response: httpx.Response, tag: str | None
    ) -> Any:
        """Execute extract response content.



        Args:

            response: Input value used by this callable.

            tag: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        content = response.content
        if not content:
            return {}
        try:
            return orjson.loads(content)
        except orjson.JSONDecodeError:
            text = response.text
            logger.debug(
                "A resposta para a tag '{}' não era um JSON válido. Retornando como texto.",
                tag or "desconhecida",
            )
            return text

    async def _handle_response_error(
        self,
        response: httpx.Response,
        warning_message: str | None,
        tag: str | None,
    ) -> bool:
        """Centralized HTTP error handling (benign vs blocking)."""
        status_code = response.status_code
        payload: dict = {}
        try:
            if response.content:
                payload = orjson.loads(response.content)
        except orjson.JSONDecodeError:
            pass
        error_code = payload.get("code")
        is_benign = (status_code, error_code) in _BENIGN_ERRORS
        details = payload.get("details", "")
        code = f" ({payload.get('code')})" if payload.get("code") else ""
        http_status = f" (HTTP {status_code})"
        final_message = (
            f"{details}{code}{http_status}"
            if details
            else f"{warning_message}{code}{http_status}"
        )
        if self._observation_callback:
            msisdn = self._extract_msisdn_from_response(response, payload)
            request_type = self._extract_request_type_from_url(str(response.url))
            if msisdn:
                await self._observation_callback(
                    msisdn=msisdn,
                    request=request_type,
                    obs=final_message,
                    payload=payload,
                )

        if is_benign:
            logger.warning("[{}] Benign error ignored: {}", tag or "N/A", final_message)
            return False

        logger.error("[{}] API error: {}", tag or "N/A", final_message)

        if 501 <= status_code < 600:
            raise RuntimeError(f"Non-recoverable server error: {final_message}")

        return False

    async def _attempt_single_request(self, **kwargs) -> httpx.Response:
        """Attempt a single HTTP request with retry logic."""
        method = kwargs.pop("method")
        url = kwargs.pop("url")
        view_response = False

        self._strip_internal_request_kwargs(kwargs)
        for attempt in range(self._retries + 1):
            self._log_request_debug(
                attempt=attempt,
                view_response=view_response,
                method=method,
                url=url,
                kwargs=kwargs,
            )
            try:
                response = await self._client.request(method, url, **kwargs)
                self._log_response_debug(response=response, view_response=view_response)
                return response
            except (httpx.RequestError, httpx.TimeoutException):
                await self._handle_retry_backoff(attempt)

        raise RuntimeError("Máximo de retentativas excedido")

    @staticmethod
    def _strip_internal_request_kwargs(kwargs: dict[str, Any]) -> None:
        """Execute strip internal request kwargs.



        Args:

            kwargs: Input value used by this callable.

        """

        for key in ["ok_msg", "warn_msg", "tag"]:
            kwargs.pop(key, None)

    @staticmethod
    def _log_request_debug(
        *,
        attempt: int,
        view_response: bool,
        method: str,
        url: str,
        kwargs: dict[str, Any],
    ) -> None:
        """Execute log request debug.



        Args:

            attempt: Input value used by this callable.

            view_response: Input value used by this callable.

            method: Input value used by this callable.

            url: Input value used by this callable.

            kwargs: Input value used by this callable.

        """

        if attempt != 0 or not view_response:
            return
        logger.debug("--- HTTP Request Details ---")
        logger.debug(f"Method: {method.upper()}")
        logger.debug(f"URL: {url}")
        headers = kwargs.get("headers")
        if headers:
            logger.debug(
                f"Headers: {orjson.dumps(headers, option=orjson.OPT_INDENT_2).decode()}"
            )
        else:
            logger.debug("Headers: None")

        body = kwargs.get("json")
        if body:
            logger.debug(
                f"Body: {orjson.dumps(body, option=orjson.OPT_INDENT_2).decode()}"
            )
        else:
            logger.debug("Body: None")
        logger.debug("----------------------------")

    @staticmethod
    def _log_response_debug(*, response: httpx.Response, view_response: bool) -> None:
        """Execute log response debug.



        Args:

            response: Input value used by this callable.

            view_response: Input value used by this callable.

        """

        if not view_response or response.status_code in (200, 204):
            return
        logger.debug("--- HTTP Response Details ---")
        logger.debug(f"Status Code: {response.status_code} {response.reason_phrase}")
        response_headers = dict(response.headers)
        logger.debug(
            f"Response Headers: {orjson.dumps(response_headers, option=orjson.OPT_INDENT_2).decode() if response_headers else 'None'}"
        )
        response_text = response.text
        if response_text:
            try:
                response_json = orjson.loads(response_text)
                logger.debug(
                    f"Response Body: {orjson.dumps(response_json, option=orjson.OPT_INDENT_2).decode()}"
                )
            except orjson.JSONDecodeError:
                logger.debug(f"Response Body (non-JSON): {response_text}")
        else:
            logger.debug("Response Body: None")
        logger.debug("-----------------------------")

    async def _handle_retry_backoff(self, attempt: int) -> None:
        """Execute handle retry backoff.



        Args:

            attempt: Input value used by this callable.



        Raises:

            Exception: Propagates domain-specific failures with context.

        """

        if attempt == self._retries:
            raise RuntimeError("No retry attempts remaining")
        await asyncio.sleep(self._backoff * (2**attempt))

    async def _execute_async_failover(
        self, combinations: list[tuple[tuple, dict]], service_type: str
    ) -> httpx.Response:
        """
        Implements asynchronous failover. It now correctly stops on the first
        valid HTTP response (2xx, 4xx, 5xx), not just on 2xx success.
        """
        failures: list[BaseException] = []
        tasks, task_to_host = self._create_failover_tasks(combinations, failures)

        try:
            response = await self._consume_failover_tasks(
                tasks=tasks,
                task_to_host=task_to_host,
                failures=failures,
                service_type=service_type,
            )
            if response is not None:
                return response
        finally:
            await self._cancel_remaining_tasks(tasks)

        self._report_failover_connect_errors(failures)
        raise RuntimeError(
            f"Nenhum host respondeu ao serviço '{service_type}'. Verifique VPN ou rede."
        )

    def _create_failover_tasks(
        self,
        combinations: list[tuple[tuple, dict]],
        failures: list[BaseException],
    ) -> tuple[set[asyncio.Task], dict[asyncio.Task, str]]:
        """Execute create failover tasks.



        Args:

            combinations: Input value used by this callable.

            failures: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        tasks: set[asyncio.Task] = set()
        task_to_host: dict[asyncio.Task, str] = {}

        def _swallow_exc(task: asyncio.Task) -> None:
            """Execute swallow exc.



            Args:

                task: Input value used by this callable.

            """

            if task.cancelled():
                return
            try:
                exc = task.exception()
                if exc:
                    failures.append(exc)
            except asyncio.CancelledError:
                return
            except Exception as exc:
                failures.append(exc)

        for _, kwargs in combinations:
            task = asyncio.create_task(self._attempt_single_request(**kwargs))
            task.add_done_callback(_swallow_exc)
            tasks.add(task)
            task_to_host[task] = urlsplit(kwargs["url"]).netloc
        return tasks, task_to_host

    async def _consume_failover_tasks(
        self,
        *,
        tasks: set[asyncio.Task],
        task_to_host: dict[asyncio.Task, str],
        failures: list[BaseException],
        service_type: str,
    ) -> httpx.Response | None:
        """Execute consume failover tasks.



        Args:

            tasks: Input value used by this callable.

            task_to_host: Input value used by this callable.

            failures: Input value used by this callable.

            service_type: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        while tasks:
            done, pending = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_COMPLETED
            )
            tasks.clear()
            tasks.update(pending)
            for task in done:
                host = task_to_host[task]
                response = await self._handle_failover_task_result(
                    task=task,
                    host=host,
                    service_type=service_type,
                    pending_tasks=tasks,
                    failures=failures,
                )
                if response is not None:
                    return response
        return None

    async def _handle_failover_task_result(
        self,
        *,
        task: asyncio.Task,
        host: str,
        service_type: str,
        pending_tasks: set[asyncio.Task],
        failures: list[BaseException],
    ) -> httpx.Response | None:
        """Execute handle failover task result.



        Args:

            task: Input value used by this callable.

            host: Input value used by this callable.

            service_type: Input value used by this callable.

            pending_tasks: Input value used by this callable.

            failures: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        try:
            resp = task.result()
            if resp:
                _get_api_factory().report_success(
                    host, service_type, resp.elapsed.total_seconds()
                )
                await self._cancel_remaining_tasks(pending_tasks)
                return resp
        except (httpx.ConnectTimeout, httpx.ReadTimeout, httpx.ConnectError):
            _get_api_factory().report_failure(host, service_type)
        except Exception as exc:
            failures.append(exc)
            _get_api_factory().report_failure(host, service_type)
        return None

    @staticmethod
    async def _cancel_remaining_tasks(tasks: set[asyncio.Task]) -> None:
        """Execute cancel remaining tasks.



        Args:

            tasks: Input value used by this callable.

        """

        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    @staticmethod
    def _report_failover_connect_errors(failures: list[BaseException]) -> None:
        """Execute report failover connect errors.



        Args:

            failures: Input value used by this callable.

        """

        unique_errors = {str(e) for e in failures if isinstance(e, httpx.ConnectError)}
        for error_msg in unique_errors:
            logger.critical(
                "Erro de conexão irrecuperável: {}. Verifique a rede/VPN.",
                error_msg,
            )
        if unique_errors:
            HttpClient._vpn_logged = True

    async def _execute_json_request(
        self,
        url: str,
        *,
        method: str = "GET",
        service_type: str,
        json_data: dict | None = None,
        headers: dict | None = None,
        timeout: float | None = None,
        success_message: str | None = None,
        warning_message: str | None = None,
        tag: str | None = None,
    ) -> str | bool | dict | list:
        """
        Smart async HTTP requester with host fail-over and HTTP/2 multiplexing.
        Raises *RuntimeError* with friendly Portuguese message when network
        errors exhaust all candidates.
        """
        request_kwargs: dict[str, Any] = {
            "json": json_data,
            "headers": headers,
            "ok_msg": success_message,
            "warn_msg": warning_message,
            "tag": tag,
        }

        if timeout:
            request_kwargs["timeout"] = httpx.Timeout(
                connect=timeout,
                read=timeout,
                write=timeout,
                pool=timeout * 2,
            )

        combinations = self._build_host_candidates(url, method, **request_kwargs)

        try:
            response = await self._execute_async_failover(
                combinations, service_type=service_type
            )
            self._last_response = response
        except httpx.ConnectTimeout as exc:
            if not HttpClient._vpn_logged:
                logger.error(f"Communication failure (VPN/Network): {exc}")
                HttpClient._vpn_logged = True
            raise
        except Exception as exc:
            logger.error(f"Fail-over error: {type(exc).__name__}: {exc}")
            raise

        if response.is_success:
            if success_message:
                logger.success(success_message)
            return await self._extract_response_content(response, tag)  # type: ignore[no-any-return]

        return await self._handle_response_error(response, warning_message, tag)

    async def make_request(
        self, endpoint_config: dict[str, Any], subscriber_data: dict[str, Any]
    ) -> dict[str, Any] | None:
        """
        High-level async HTTP request method with intelligent caching and HTTP/2.

        Args:
            endpoint_config: Configuration dict with url, method, headers, json, etc.
            subscriber_data: Data used for URL templating and cache keys

        Returns:
            Response data as dict, or None if unsuccessful
        """
        url = endpoint_config.get("url", "")
        endpoint_type = endpoint_config.get("type", "unknown")
        method = endpoint_config.get("method", "GET")
        headers_config = endpoint_config.get("headers")

        cached = self.cache.templates.get(url, endpoint_type, method)
        if cached:
            logger.info(f"Cache HIT para {endpoint_type} (host em cache)")
            final_url = self.cache.templates.apply_template(
                cached.template,
                {k: str(v) for k, v in subscriber_data.items() if v is not None},
            )
            headers = {**cached.headers, **(headers_config or {})}

            response = await self._execute_json_request(
                url=final_url,
                method=method,
                service_type=endpoint_type,
                json_data=endpoint_config.get("json"),
                headers=headers,
                timeout=endpoint_config.get("timeout"),
                success_message=endpoint_config.get("ok_msg"),
                warning_message=endpoint_config.get("warn_msg"),
                tag=endpoint_type,
            )

            if isinstance(response, dict) or response is True:
                return response if isinstance(response, dict) else None

            logger.warning(
                f"Host em cache falhou para {endpoint_type}, entrando em fallback"
            )
            self.cache.templates.remove(
                self.cache.templates._generate_cache_key(url, endpoint_type, method)
            )
        else:
            logger.info(f"Cache MISS para {endpoint_type}")

        final_url = url
        headers = headers_config or {}

        response = await self._execute_json_request(
            url=final_url,
            method=method,
            service_type=endpoint_type,
            json_data=endpoint_config.get("json"),
            headers=headers,
            timeout=endpoint_config.get("timeout"),
            success_message=endpoint_config.get("ok_msg"),
            warning_message=endpoint_config.get("warn_msg"),
            tag=endpoint_type,
        )
        if (
            (response is not None and response is not False)
            and self._last_response
            and self._last_response.url
        ):
            real_url = str(self._last_response.url)
            self.cache.templates.set(
                url=real_url,
                endpoint_type=endpoint_type,
                method=method,
                headers=headers or {},
                ttl_days=7,
                subscriber_data=subscriber_data,
            )
            logger.info(f"Template salvo em cache para {endpoint_type}")

        return response if isinstance(response, dict) else None

    @staticmethod
    def _extract_msisdn(message: str | None) -> str:
        """Extract MSISDN from a warning string like '[55119999…]'."""
        match = re.search(r"\[(\d+)]", message or "")
        return match.group(1) if match else "N/A"

    def _generate_unique_path(self, folder: Path, stem: str, suffix: str) -> Path:
        """Return non-conflicting path like `foo.xlsx`, `foo(1).xlsx`, …."""
        path = folder / f"{stem}{suffix}"
        counter = 1
        while path.exists():
            path = folder / f"{stem}({counter}){suffix}"
            counter += 1
        return path

    def _extract_msisdn_from_response(
        self, response: httpx.Response, payload: dict
    ) -> str | None:
        """Tries to find an MSISDN from the request URL or response payload."""
        if "communicationId" in payload:
            return payload["communicationId"]  # type: ignore[no-any-return]
        url_str = str(response.request.url)
        match = re.search(r"55(\d{10,13})", url_str)
        if match:
            return match.group(0)

        return None

    def _extract_request_type_from_url(self, url: str) -> str:
        """Extracts a request type identifier from the URL path."""
        path_segments = url.split("/")
        last_segment = path_segments[-1].split("?")[0]
        if last_segment:
            return last_segment

        return path_segments[-2] if len(path_segments) > 1 else "unknown_request"


async def check_http_status(url: str, *, timeout_s: float = 1.0) -> bool:
    """Check whether an HTTP endpoint responds with status 200."""
    timeout = max(float(timeout_s), 0.1)
    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=timeout,
                read=timeout,
                write=timeout,
                pool=timeout * 2,
            ),
            follow_redirects=True,
        ) as client:
            response = await client.get(url)
        return response.status_code == 200
    except (httpx.RequestError, httpx.TimeoutException):
        return False


def __getattr__(name: str) -> Any:
    if name == "API":
        global _API_instance
        if _API_instance is None:
            _API_instance = _get_api_factory().build(path_only=True)
        return _API_instance
    if name == "api_factory":
        return _get_api_factory()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
