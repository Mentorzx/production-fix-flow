"""
LineService Base - Infrastructure and Core Utilities

This module contains the base class with:
- Circuit breakers setup
- Request coalescing infrastructure
- Resilient request execution
- Utility methods (save_object, set_observation)

Part of Sprint 4 refactoring (line_service.py split into 4 files).
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import Callable, Coroutine
from datetime import timedelta
from functools import wraps
from pathlib import Path
from typing import Any, Concatenate, ParamSpec, TypeVar

import polars as pl
from aiobreaker import CircuitBreaker, CircuitBreakerError
from pydantic import BaseModel

from pff.application.ports.file_manager import FileManagerPort
from pff.application.ports.http_client import HttpClientPort
from pff.application.ports.line_api import LineApiPort
from pff.application.ports.settings import SettingsPort
from pff.shared.clients import HttpClient
from pff.shared.core.config import settings as default_settings
from pff.shared.core.file_manager import FileManager
from pff.shared.core.logging import logger
from pff.shared.research import Research

from .config import load_line_service_config

_P = ParamSpec("_P")
_R = TypeVar("_R")
_Self = TypeVar("_Self")


def _resolve_default_line_api() -> LineApiPort:
    """Resolve default line API lazily to avoid static coupling at import time."""
    from pff.shared.clients.http_client import API

    return API


def capture_collector(
    fn: Callable[Concatenate[_Self, _P], Coroutine[Any, Any, _R]],
) -> Callable[Concatenate[_Self, _P], Coroutine[Any, Any, _R]]:
    """Decorator that stores *collector* (if provided) in `self._collector`."""

    @wraps(fn)
    async def wrapper(self: _Self, *args: _P.args, **kwargs: _P.kwargs) -> _R:
        """Execute wrapper.



        Args:

            *args: Additional positional arguments.

            **kwargs: Additional keyword arguments.



        Returns:

            Return value produced by the callable.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        if (collector := kwargs.pop("collector", None)) is not None:
            self._collector = collector  # type: ignore[attr-defined]
        return await fn(self, *args, **kwargs)

    return wrapper


class LineServiceBase:
    """
    Base class for LineService with infrastructure utilities.

    Provides:
    - Circuit breakers for resilience
    - Request coalescing for deduplication
    - HTTP client wrapper
    - File manager integration
    """

    _enquiry_breaker: CircuitBreaker
    _individual_party_breaker: CircuitBreaker
    _contract_breaker: CircuitBreaker
    _contract_status_breaker: CircuitBreaker
    _product_status_breaker: CircuitBreaker
    _delete_contract_breaker: CircuitBreaker
    _party_termination_breaker: CircuitBreaker
    _consumer_list_breaker: CircuitBreaker
    _create_client_breaker: CircuitBreaker

    def __init__(
        self,
        *,
        http_client: HttpClientPort | None = None,
        file_manager: FileManagerPort | None = None,
        api_client: LineApiPort | None = None,
        settings_obj: SettingsPort | None = None,
        **kwargs,
    ) -> None:
        """Initialize base infrastructure."""
        self._config = load_line_service_config()
        self._settings = settings_obj or default_settings
        self._http_client = http_client or HttpClient(
            observation_callback=self.set_observation, **kwargs
        )
        self._file_manager = file_manager or FileManager()
        self._api = api_client or _resolve_default_line_api()
        self._research = Research()
        self._unique_path = self._http_client._generate_unique_path
        self.make_request = self._http_client.make_request

        self._init_circuit_breakers()

        self._request_locks: defaultdict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self._request_cache: dict[str, Any] = {}

    def _init_circuit_breakers(self) -> None:
        """Initialize all circuit breakers with appropriate configurations."""

        read_breakers = [
            "enquiry",
            "individual_party",
            "contract",
        ]
        for name in read_breakers:
            setattr(
                self,
                f"_{name}_breaker",
                CircuitBreaker(
                    fail_max=self._config.read_breaker.fail_max,
                    timeout_duration=timedelta(
                        seconds=self._config.read_breaker.timeout_duration_s
                    ),
                ),
            )

        write_breakers = [
            "contract_status",
            "product_status",
            "delete_contract",
            "party_termination",
            "consumer_list",
            "create_client",
        ]
        for name in write_breakers:
            setattr(
                self,
                f"_{name}_breaker",
                CircuitBreaker(
                    fail_max=self._config.write_breaker.fail_max,
                    timeout_duration=timedelta(
                        seconds=self._config.write_breaker.timeout_duration_s
                    ),
                ),
            )

    async def close(self):
        """Closes the underlying http client session."""
        await self._http_client.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        return False

    async def _clear_coalescing_cache(self, key: str, delay: int | None = None):
        """Removes a key from the short-term coalescing cache."""
        actual_delay = delay if delay is not None else self._config.coalescing_delay_s
        await asyncio.sleep(actual_delay)
        self._request_cache.pop(key, None)
        self._request_locks.pop(key, None)
        logger.debug(f"Coalescing cache cleared for key: {key}")

    async def _execute_resilient_request(
        self,
        breaker: CircuitBreaker,
        cache_key: str,
        request_coro: Callable[[], Coroutine[Any, Any, Any]],
        identifier: str,
        operation_name: str,
    ) -> dict[str, Any]:
        """
        Executes a resilient request with coalescing and circuit breaking.

        Args:
            breaker: Circuit breaker instance to use.
            cache_key: Key for request coalescing cache.
            request_coro: Coroutine function to execute the actual request.
            identifier: Identifier for logging (e.g., MSISDN, customer_id).
            operation_name: Name of the operation for logging.

        Returns:
            The result dictionary or an error dictionary.
        """
        async with self._request_locks[cache_key]:
            if cache_key in self._request_cache:
                logger.debug(f"Returning coalesced result for {cache_key}")
                return self._request_cache[cache_key]  # type: ignore[no-any-return]

            try:
                logger.debug(
                    f"Executing network call for {cache_key} under circuit breaker"
                )

                result = await breaker.call_async(request_coro)
                result = result if isinstance(result, dict) else {}

                self._request_cache[cache_key] = result
                asyncio.create_task(self._clear_coalescing_cache(cache_key))

                return result

            except CircuitBreakerError as e:
                logger.error(
                    f"component=line_service event=circuit_breaker_open operation={operation_name} "
                    f"id={identifier} error={e}"
                )
                return {"error": "Service temporarily unavailable", "details": str(e)}
            except Exception as e:
                logger.error(
                    f"component=line_service event=unexpected_error operation={operation_name} "
                    f"id={identifier} error={e}"
                )
                return {
                    "error": f"Failed to execute {operation_name}",
                    "details": str(e),
                }

    async def _execute_state_changing_request(
        self,
        breaker: CircuitBreaker,
        request_coro: Callable[[], Coroutine[Any, Any, Any]],
        identifier: str,
        operation_name: str,
    ) -> bool:
        """
        Executes a state-changing request with circuit breaking protection.

        Args:
            breaker: Circuit breaker instance to use.
            request_coro: Coroutine function to execute the actual request.
            identifier: Identifier for logging (e.g., MSISDN, customer_id).
            operation_name: Name of the operation for logging.

        Returns:
            True if the operation succeeded, False otherwise.
        """
        try:
            logger.debug(
                f"Executing state-changing operation [{operation_name}] for "
                f"{identifier} under circuit breaker"
            )

            result = await breaker.call_async(request_coro)
            return bool(result)

        except CircuitBreakerError as e:
            logger.error(
                f"component=line_service evento=circuit_breaker_aberto operacao={operation_name} "
                f"id={identifier} erro={e}"
            )
            return False
        except Exception as e:
            logger.exception(
                f"component=line_service evento=erro_inesperado operacao={operation_name} "
                f"id={identifier} erro={e}"
            )
            return False

    @capture_collector
    async def save_object(self, obj: Any, var_name: str) -> None:
        """
        Persist *obj* under `self._settings.OUTPUTS_DIR/objects/`.

        Supported:
            • pydantic.BaseModel → JSON
            • dict / list[dict]  → JSON
            • polars.DataFrame   → XLSX
            • str path to CSV|XLS|TXT → loads & re-exports XLSX
        """
        out_dir = self._settings.OUTPUTS_DIR / "objects"
        self._file_manager.ensure_dir(out_dir)

        if isinstance(obj, BaseModel):
            obj = obj.model_dump()

        if isinstance(obj, (dict, list)):
            self._file_manager.save(obj, self._unique_path(out_dir, var_name, ".json"))
            return

        if isinstance(obj, pl.DataFrame):
            self._file_manager.save(obj, self._unique_path(out_dir, var_name, ".xlsx"))
            return

        if isinstance(obj, str) and (p := Path(obj)) and self._file_manager.exists(p):
            try:
                self._file_manager.assert_supported_path(
                    p, allowed_exts={".csv", ".xls", ".xlsx", ".txt"}
                )
            except ValueError:
                pass
            else:
                bundle = self._file_manager.read(p)
                self._file_manager.save(
                    bundle, self._unique_path(out_dir, var_name, ".xlsx")
                )
                return

        raise RuntimeError(
            f"Não foi possível salvar '{var_name}': tipo '{type(obj).__name__}' não suportado."
        )

    @capture_collector
    async def set_observation(
        self,
        msisdn: str | None = None,
        request: str | None = None,
        obs: dict[str, Any] | None = None,
        payload: dict[str, Any] | None = None,
        endpoint: str | None = None,
        response: dict[str, Any] | None = None,
    ) -> None:
        """
        Store HTTP observation data for analysis.

        Args:
            msisdn: MSISDN identifier (for sequence YAML compatibility)
            request: Request type (for sequence YAML compatibility)
            obs: Observation data (for sequence YAML compatibility)
            payload: The observation data to store (legacy parameter)
            endpoint: Optional endpoint identifier
            response: Optional response data
        """

        actual_payload = obs if obs is not None else payload
        actual_endpoint = request if request is not None else endpoint

        if actual_payload is None:
            logger.warning("set_observation called without observation data")
            return

        obs_dir = self._settings.OUTPUTS_DIR / "observations"
        self._file_manager.ensure_dir(obs_dir)

        observation = {
            "msisdn": msisdn,
            "endpoint": actual_endpoint,
            "request": request,
            "payload": actual_payload,
            "response": response,
        }

        import time

        filename = f"{actual_endpoint or 'obs'}_{int(time.time())}.json"
        self._file_manager.save(observation, obs_dir / filename)
