from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from typing import Callable, Final, Iterator, Protocol

import yaml
from dotenv import load_dotenv

from pff.config import settings

_ROOT = settings.ROOT_DIR
load_dotenv(_ROOT / ".env", override=False)
_HOST_FILE: Final = _ROOT / "config" / "api_hosts.yaml"

with _HOST_FILE.open("r", encoding="utf-8") as fp:
    _ALL_HOSTS: dict[str, dict[str, str]] = yaml.safe_load(fp)

_ORDER: Final[list[str]] = [
    c.strip().upper()
    for c in os.getenv("API_CLUSTER_ORDER", "JAG,PAL,BAR,FUN").split(",")
    if c.strip().upper() in _ALL_HOSTS
] or ["BAR"]


class FailoverStrategy(Protocol):
    def cycle(self) -> Iterator[str]: ...
    def report_success(self, host: str, latency: float) -> None: ...
    def report_failure(self, host: str) -> None: ...


class RoundRobin(FailoverStrategy):
    """Estratégia stateless que sempre itera na mesma ordem."""

    __slots__ = ("_hosts", "_idx")

    def __init__(self, service: str, order: list[str]):
        self._hosts = [
            _ALL_HOSTS[c][service] for c in order if service in _ALL_HOSTS[c]
        ]
        if not self._hosts:
            raise ValueError(
                f"Serviço '{service}' ausente em todos os clusters: {order}"
            )
        self._idx = 0

    @property
    def current(self) -> str:
        return self._hosts[self._idx]

    def advance(self) -> None:
        self._idx = (self._idx + 1) % len(self._hosts)

    def cycle(self) -> Iterator[str]:
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
        self._service_name = service
        self._hosts = [
            _ALL_HOSTS[c][service] for c in order if service in _ALL_HOSTS[c]
        ]
        if not self._hosts:
            raise ValueError(
                f"Serviço '{service}' ausente em todos os clusters: {order}"
            )

        self._latencies = {host: 0.1 for host in self._hosts}
        self._failures = {host: 0 for host in self._hosts}
        self._last_failure_time = {host: 0.0 for host in self._hosts}
        self._lock = threading.Lock()

    def report_success(self, host: str, latency: float) -> None:
        with self._lock:
            current_latency = self._latencies.get(host, 0.1)
            self._latencies[host] = (current_latency * 0.8) + (latency * 0.2)
            self._failures[host] = 0

    def report_failure(self, host: str) -> None:
        with self._lock:
            if host in self._hosts:
                self._failures[host] += 1
                self._last_failure_time[host] = time.time()

    def cycle(self) -> Iterator[str]:
        with self._lock:

            def sort_key(host):
                is_healthy = self._failures.get(host, 0) < 3
                is_recent_failure = (
                    time.time() - self._last_failure_time.get(host, 0)
                ) < 60
                latency = self._latencies.get(host, 999.0)

                return (not is_healthy, is_recent_failure, latency)

            sorted_hosts = sorted(self._hosts, key=sort_key)

        yield from sorted_hosts


class EndpointFactory:
    def __init__(self, strategy_name: str = "latency-aware"):
        self._order = _ORDER

        strategy_map = {
            "round-robin": RoundRobin,
            "latency-aware": LatencyAwareStrategy,
        }
        strategy_cls = strategy_map.get(strategy_name.lower())
        if not strategy_cls:
            raise ValueError(f"Estratégia '{strategy_name}' desconhecida.")

        services = {
            svc for cluster in self._order for svc in _ALL_HOSTS[cluster].keys()
        }
        self._strategies: dict[str, FailoverStrategy] = {
            svc: strategy_cls(svc, self._order) for svc in services
        }

    def _get_strategy(self, svc: str) -> FailoverStrategy:
        strategy = self._strategies.get(svc.strip().upper())
        if strategy is None:
            raise RuntimeError(f"Serviço '{svc}' não configurado.")
        return strategy

    def cycle(self, svc: str) -> Iterator[str]:
        return self._get_strategy(svc).cycle()

    def report_success(self, host: str, svc: str, latency: float):
        """Reporta um sucesso para a estratégia do serviço, para que ela aprenda."""
        self._get_strategy(svc).report_success(host, latency)

    def report_failure(self, host: str, svc: str):
        """Reporta uma falha para a estratégia do serviço, para que ela aprenda."""
        self._get_strategy(svc).report_failure(host)

    def build(self, *, path_only: bool = False) -> "APIsEndpoints":
        host_fn: Callable[[str], str] = (lambda _s: "") if path_only else self.host
        return APIsEndpoints(host_fn)

    def host(self, svc: str) -> str:
        """Retorna o melhor host disponível para o serviço."""
        strategy = self._get_strategy(svc)
        return next(strategy.cycle())


@dataclass(slots=True)
class APIsEndpoints:
    _host: Callable[[str], str]

    # GET --------------------------------------------------------------
    def customer_enquiry(self, msisdn: str) -> tuple[str, str]:
        service_type = "BIAS"
        url = join(
            self._host(service_type),
            "bias/bssfsdCustomerEnquiry/v1/customer"
            f"?includeTerminatedContracts=true&includeContracts=true"
            f"&communicationIdType=E.164&communicationId=55{msisdn}",
        )
        return url, service_type

    def customer_enquiry_by_customer(self, cid: str) -> tuple[str, str]:
        service_type = "BIAS"
        url = join(
            self._host(service_type),
            f"bias/bssfsdCustomerEnquiry/v1/customer?customerId={cid}",
        )
        return url, service_type

    def individual_party_enquiry(self, ext: str) -> tuple[str, str]:
        service_type = "BAE"
        url = join(
            self._host(service_type),
            f"bae/bssfIndividualPartyEnquiry/v1/individualParty/?externalId={ext}",
        )
        return url, service_type

    def read_contract(self, ctt: str, cust: str) -> tuple[str, str]:
        service_type = "CPM"
        url = join(
            self._host(service_type),
            f"cpm/business/v1/readContract/customer/{cust}/contract/{ctt}?validAt=*",
        )
        return url, service_type

    # POST -------------------------------------------------------------
    @property
    def update_contract_status(self) -> tuple[str, str]:
        service_type = "BIAS"
        url = join(
            self._host(service_type), "bias/vivoUpdateContractStatus/v1/updateStatus"
        )
        return url, service_type

    def deactivate_contract(self, msisdn: str) -> tuple[str, str]:
        service_type = "BIAS"
        url = join(
            self._host(service_type),
            f"bias/vivoDeactivateContract/v1/communicationId/{msisdn}/communicationIdType/E.164",
        )
        return url, service_type

    def activate_product(self, msisdn: str) -> tuple[str, str]:
        service_type = "BIAS"
        url = join(
            self._host(service_type),
            f"bias/vivoActivateProduct/v1/communicationId/{msisdn}/communicationIdType/E.164",
        )
        return url, service_type

    @property
    def manage_consumer_list(self) -> tuple[str, str]:
        service_type = "BIAS"
        url = join(
            self._host(service_type),
            "bias/bssfsdSubscriptionManagement/v1/rmil-manage-consumer-list/",
        )
        return url, service_type

    @property
    def create_client(self) -> tuple[str, str]:
        service_type = "BIAS"
        url = join(self._host(service_type), "bias/vivoCreateClient/v1/customer")
        return url, service_type

    # PATCH / DELETE idem (usar self._host(…))
    def subscription(self, cust: str, ctt: str) -> tuple[str, str]:
        service_type = "BAE"
        url = join(
            self._host(service_type),
            f"bae/bssfSubscriptionManagement/v1/customer/{cust}/contract/{ctt}",
        )
        return url, service_type

    def delete_contract(self, cid: str, ctid: str) -> tuple[str, str]:
        service_type = "BAE"
        url = join(
            self._host(service_type),
            f"bae/bssfSubscriptionManagement/v1/customer/{cid}/contract/{ctid}",
        )
        return url, service_type

    def party_cascade(self, pid: str) -> tuple[str, str]:
        service_type = "CPM"
        url = join(
            self._host(service_type), f"cpm/business/v1/updateParty/party/{pid}"
        )
        return url, service_type


join = lambda base, path: f"{base.rstrip('/')}/{path.lstrip('/')}"  # noqa: E731
