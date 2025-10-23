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
from datetime import timedelta
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Concatenate, Coroutine, ParamSpec, TypeVar

import polars as pl
from aiobreaker import CircuitBreaker, CircuitBreakerError
from pydantic import BaseModel

from pff.config import settings
from pff.utils import FileManager, Research, logger
from pff.utils.clients import HttpClient

_P = ParamSpec("_P")
_R = TypeVar("_R")
_Self = TypeVar("_Self")


def capture_collector(
    fn: Callable[
        Concatenate[_Self, _P], Coroutine[Any, Any, _R]
    ],
) -> Callable[Concatenate[_Self, _P], Coroutine[Any, Any, _R]]:
    """Decorator that stores *collector* (if provided) in `self._collector`."""

    @wraps(fn)
    async def wrapper(
        self: _Self, *args: _P.args, **kwargs: _P.kwargs
    ) -> _R:
        if (collector := kwargs.pop("collector", None)) is not None:
            self._collector = collector  # type: ignore[attr-defined]
        return await fn(self, *args, **kwargs)

    return wrapper  # type: ignore


class LineServiceBase:
    """
    Base class for LineService with infrastructure utilities.

    Provides:
    - Circuit breakers for resilience
    - Request coalescing for deduplication
    - HTTP client wrapper
    - File manager integration
    """

    # Type hints for circuit breakers (created dynamically in __init__)
    _enquiry_breaker: CircuitBreaker
    _individual_party_breaker: CircuitBreaker
    _contract_breaker: CircuitBreaker
    _contract_status_breaker: CircuitBreaker
    _product_status_breaker: CircuitBreaker
    _delete_contract_breaker: CircuitBreaker
    _party_termination_breaker: CircuitBreaker
    _consumer_list_breaker: CircuitBreaker
    _create_client_breaker: CircuitBreaker

    def __init__(self, **kwargs) -> None:
        """Initialize base infrastructure."""
        self._http_client = HttpClient(
            observation_callback=self.set_observation, **kwargs
        )
        self._file_manager = FileManager()
        self._research = Research()
        self._unique_path = self._http_client._generate_unique_path
        self.make_request = self._http_client.make_request

        # Circuit Breaker configurations
        self._init_circuit_breakers()

        # Request Coalescing structures
        self._request_locks = defaultdict(asyncio.Lock)
        self._request_cache: dict[str, Any] = {}  # Short-term cache for coalescing

    def _init_circuit_breakers(self) -> None:
        """Initialize all circuit breakers with appropriate configurations."""
        # Read operations - more lenient (5 failures, 60s timeout)
        read_breakers = [
            "enquiry",
            "individual_party",
            "contract",
        ]
        for name in read_breakers:
            setattr(
                self,
                f"_{name}_breaker",
                CircuitBreaker(fail_max=5, timeout_duration=timedelta(seconds=60))
            )

        # Write operations - stricter (3 failures, 30s timeout)
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
                CircuitBreaker(fail_max=3, timeout_duration=timedelta(seconds=30))
            )

    async def close(self):
        """Closes the underlying http client session."""
        await self._http_client.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        # Note: close() is async, so this won't call it
        # Users should call await service.close() manually
        return False

    async def _clear_coalescing_cache(self, key: str, delay: int = 10):
        """Removes a key from the short-term coalescing cache."""
        await asyncio.sleep(delay)
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
                return self._request_cache[cache_key]

            try:
                logger.debug(
                    f"Executing network call for {cache_key} under circuit breaker"
                )

                # CRITICAL FIX: Use call_async for async functions, not call
                result = await breaker.call_async(request_coro)
                result = result if isinstance(result, dict) else {}

                # Store in short-term coalescing cache
                self._request_cache[cache_key] = result
                asyncio.create_task(self._clear_coalescing_cache(cache_key))

                return result

            except CircuitBreakerError as e:
                logger.error(
                    f"Circuit breaker open for {operation_name} [{identifier}]: {e}"
                )
                return {"error": "Service temporarily unavailable", "details": str(e)}
            except Exception as e:
                logger.error(
                    f"Unexpected error in {operation_name} [{identifier}]: {e}"
                )
                return {"error": f"Failed to {operation_name}", "details": str(e)}

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
                f"Executing state-changing operation [{operation_name}] for {identifier} under circuit breaker"
            )

            # CRITICAL FIX: Use call_async for async functions, not call
            result = await breaker.call_async(request_coro)
            return bool(result)

        except CircuitBreakerError as e:
            logger.error(
                f"Circuit breaker open for {operation_name} [{identifier}]: {e}"
            )
            return False
        except Exception as e:
            logger.exception(
                f"Unexpected error in {operation_name} [{identifier}]: {e}"
            )
            return False

    # ──────────────────────── public helpers ────────────────────────── #

    @capture_collector
    async def save_object(self, obj: Any, var_name: str) -> None:
        """
        Persist *obj* under `settings.OUTPUTS_DIR/objects/`.

        Supported:
            • pydantic.BaseModel → JSON
            • dict / list[dict]  → JSON
            • polars.DataFrame   → XLSX
            • str path to CSV|XLS|TXT → loads & re-exports XLSX
        """
        out_dir = settings.OUTPUTS_DIR / "objects"
        out_dir.mkdir(parents=True, exist_ok=True)

        if isinstance(obj, BaseModel):
            obj = obj.model_dump()

        if isinstance(obj, (dict, list)):
            self._file_manager.save(obj, self._unique_path(out_dir, var_name, ".json"))
            return

        if isinstance(obj, pl.DataFrame):
            self._file_manager.save(obj, self._unique_path(out_dir, var_name, ".xlsx"))
            return

        if (
            isinstance(obj, str)
            and (p := Path(obj)).exists()
            and p.suffix.lower() in {".csv", ".xls", ".xlsx", ".txt"}
        ):
            df = self._file_manager.read(p)
            self._file_manager.save(df, self._unique_path(out_dir, var_name, ".xlsx"))
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
        # Handle both legacy and new parameter styles
        # Priority: obs > payload
        actual_payload = obs if obs is not None else payload
        actual_endpoint = request if request is not None else endpoint

        if actual_payload is None:
            logger.warning("set_observation called without observation data")
            return

        # Store observation for later analysis
        obs_dir = settings.OUTPUTS_DIR / "observations"
        obs_dir.mkdir(parents=True, exist_ok=True)

        observation = {
            "msisdn": msisdn,
            "endpoint": actual_endpoint,
            "request": request,
            "payload": actual_payload,
            "response": response,
        }

        # Save to file for analysis
        timestamp = asyncio.get_event_loop().time()
        filename = f"observation_{int(timestamp * 1000)}.json"

        self._file_manager.save(
            observation,
            obs_dir / filename
        )
