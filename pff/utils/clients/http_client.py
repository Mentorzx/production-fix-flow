from __future__ import annotations

import asyncio

import os
import re
from pathlib import Path
from typing import Any, Callable, Coroutine
from urllib.parse import urlsplit

import httpx
import orjson
from urllib3 import disable_warnings, exceptions

from pff.utils import (
    CacheManager,
    EndpointFactory,
    logger,
)

# ───────────────────────── constants & helpers ────────────────────────── #

api_factory = EndpointFactory()
API = api_factory.build(path_only=True)

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


# ---------------------main class definition---------------------#
class HttpClient:
    """A reusable and robust asynchronous HTTP client for external APIs with HTTP/2 support."""

    _vpn_logged: bool = False

    def __init__(
        self,
        *,
        observation_callback: Callable[..., Coroutine[Any, Any, None]] | None = None,
        **kwargs,
    ) -> None:
        self._timeout = kwargs.get("timeout", _DEFAULT_TIMEOUT)
        self._retries = kwargs.get("retries", _DEFAULT_RETRIES)
        self._backoff = kwargs.get("backoff", _DEFAULT_BACKOFF)
        self._observation_callback = observation_callback
        self.cache = CacheManager()

        # CA-bundle / self-signed handling
        self._ca_bundle = os.getenv("PFF_CA_BUNDLE")
        verify_ssl = True
        if self._ca_bundle and Path(self._ca_bundle).exists():
            logger.debug("Verificação HTTPS usando CA bundle: {}", self._ca_bundle)
            verify_ssl = self._ca_bundle
        else:
            disable_warnings(exceptions.InsecureRequestWarning)
            # logger.debug("Verificação HTTPS desabilitada")
            verify_ssl = False

        # Initialize httpx client with HTTP/2 support
        self._client = httpx.AsyncClient(
            http2=False,
            timeout=httpx.Timeout(
                connect=self._timeout,
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

        # Store last response for caching
        self._last_response: httpx.Response | None = None

    async def __aenter__(self) -> "HttpClient":
        return self

    async def __aexit__(self, *_exc) -> None:
        await self.close()

    async def close(self) -> None:
        """Close the underlying httpx client."""
        if self._client:
            await self._client.aclose()

    # ──────────────────────── private internals ─────────────────────── #

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
            else "CPM"
            if "/cpm/" in url.lower()
            else "RMVIVO"
            if "rmvivo" in url.lower()
            else "BAE"
        )
        for host in api_factory.cycle(service):
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

    # ------- response helpers -------------------------------------------- #

    async def _extract_response_content(
        self, response: httpx.Response, tag: str | None
    ) -> Any:
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
            logger.warning(
                "[{}] Erro benigno ignorado: {}", tag or "N/A", final_message
            )
            return False

        # Para erros não benignos, o log é mais severo
        logger.error("[{}] Erro na API: {}", tag or "N/A", final_message)

        if 501 <= status_code < 600:
            raise RuntimeError(f"Erro de servidor não recuperável: {final_message}")

        return False

    # ------- async failover implementation ----------------------------- #

    async def _attempt_single_request(self, **kwargs) -> httpx.Response:
        """Attempt a single HTTP request with retry logic."""
        method = kwargs.pop("method")
        url = kwargs.pop("url")
        view_response = False  # Enable detailed logging of the response for debugging

        # Remove custom keys not expected by httpx
        for key in ["ok_msg", "warn_msg", "tag"]:
            kwargs.pop(key, None)
        for attempt in range(self._retries + 1):
            if attempt == 0 and view_response:
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
            try:
                response = await self._client.request(method, url, **kwargs)
                if view_response and response.status_code not in (200, 204):
                    logger.debug("--- HTTP Response Details ---")
                    logger.debug(
                        f"Status Code: {response.status_code} {response.reason_phrase}"
                    )
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
                return response
            except (httpx.RequestError, httpx.TimeoutException):
                if attempt == self._retries:
                    raise
                await asyncio.sleep(self._backoff * (2**attempt))

        raise RuntimeError("Máximo de retentativas excedido")

    async def _execute_async_failover(
        self, combinations: list[tuple[tuple, dict]], service_type: str
    ) -> httpx.Response:
        """
        Implements asynchronous failover. It now correctly stops on the first
        valid HTTP response (2xx, 4xx, 5xx), not just on 2xx success.
        """
        tasks: set[asyncio.Task] = set()
        task_to_host: dict[asyncio.Task, str] = {}
        failures: list[BaseException] = []  # To collect network errors

        def _swallow_exc(t: asyncio.Task) -> None:
            if not t.cancelled():
                try:
                    exc = t.exception()
                    if exc:
                        failures.append(exc)
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    failures.append(e)

        for _, kwargs in combinations:
            task = asyncio.create_task(self._attempt_single_request(**kwargs))
            task.add_done_callback(_swallow_exc)
            tasks.add(task)
            task_to_host[task] = urlsplit(kwargs["url"]).netloc

        try:
            while tasks:
                done, tasks = await asyncio.wait(
                    tasks, return_when=asyncio.FIRST_COMPLETED
                )
                for task in done:
                    host = task_to_host[task]
                    try:
                        resp = task.result()
                        if resp:
                            api_factory.report_success(
                                host, service_type, resp.elapsed.total_seconds()
                            )
                            for t in tasks:
                                t.cancel()
                            await asyncio.gather(*tasks, return_exceptions=True)
                            return resp
                    except (
                        httpx.ConnectTimeout,
                        httpx.ReadTimeout,
                        httpx.ConnectError,
                    ):
                        api_factory.report_failure(host, service_type)
                    except Exception:
                        api_factory.report_failure(host, service_type)
        finally:
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

        unique_errors = {str(e) for e in failures if isinstance(e, httpx.ConnectError)}
        for error_msg in unique_errors:
            logger.critical(
                "Erro de conexão irrecuperável: {}. Verifique a rede/VPN.",
                error_msg,
            )
        if unique_errors:
            HttpClient._vpn_logged = True

        raise RuntimeError(
            f"Nenhum host respondeu ao serviço '{service_type}'. Verifique VPN ou rede."
        )

    # ------- high-level request ------------------------------------------ #

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
        request_kwargs = {
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
            # Pass the service_type to the failover
            response = await self._execute_async_failover(
                combinations, service_type=service_type
            )
            self._last_response = response
        except httpx.ConnectTimeout as exc:
            if not HttpClient._vpn_logged:
                logger.error(f"Falha de comunicação (VPN/Rede): {exc}")
                HttpClient._vpn_logged = True
            raise
        except Exception as exc:
            logger.error(f"Falha no fail-over: {type(exc).__name__}: {exc}")
            raise

        if response.is_success:
            if success_message:
                logger.success(success_message)
            return await self._extract_response_content(response, tag)

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

        # Try to get cached template
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

        # No cache or cache failed - do full failover
        final_url = url
        headers = headers_config

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

    # ------- misc small helpers ----------------------------------------- #

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
            return payload["communicationId"]
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
