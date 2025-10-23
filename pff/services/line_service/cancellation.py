"""
LineService Cancellation - Cancellation Operations

This module contains cancellation-specific operations:
- set_soft_cancel_control
- set_soft_cancel_postpaid

Part of Sprint 4 refactoring (line_service.py split into 4 files).
These methods have high complexity and are isolated for easier maintenance.
"""

from __future__ import annotations

from pff.utils.clients import API

from .base import LineServiceBase, capture_collector


class LineServiceCancellation(LineServiceBase):
    """
    Cancellation operations for LineService.

    Handles soft-cancel operations for control and postpaid contracts.
    """

    # ──────────────────────── CANCELLATION OPERATIONS ──────────────────────

    @capture_collector
    async def set_soft_cancel_control(self, msisdn: str) -> bool:
        """
        Soft-cancels a control contract for the specified MSISDN.
        Enhanced with circuit breaking protection.

        Args:
            msisdn: MSISDN identifier.

        Returns:
            True if the operation succeeded.
        """
        data = {
            "communicationId": msisdn,
            "communicationIdType": "E.164",
            "status": [{"status": "CtrlCancelado"}],
        }
        url, service_type = API.update_contract_status

        async def request_coro():
            return await self.make_request(
                endpoint_config={
                    "url": url,
                    "type": service_type,
                    "method": "POST",
                    "json": data,
                    "ok_msg": f"[{msisdn}] Soft‑cancel controle OK",
                    "warn_msg": f"[{msisdn}] Soft‑cancel controle falhou",
                },
                subscriber_data={"msisdn": msisdn},
            )

        return await self._execute_state_changing_request(
            breaker=self._contract_status_breaker,
            request_coro=request_coro,
            identifier=msisdn,
            operation_name="set_soft_cancel_control",
        )

    @capture_collector
    async def set_soft_cancel_postpaid(self, msisdn: str, reason: str) -> bool:
        """
        Soft-cancels a postpaid contract for the specified MSISDN and reason.
        Enhanced with circuit breaking protection.

        Args:
            msisdn: MSISDN identifier.
            reason: Reason for soft-cancel.

        Returns:
            True if the operation succeeded.
        """
        data = {"reason": reason}
        url, service_type = API.deactivate_contract(msisdn)

        async def request_coro():
            return await self.make_request(
                endpoint_config={
                    "url": url,
                    "type": service_type,
                    "method": "POST",
                    "json": data,
                    "ok_msg": f"[{msisdn}] Soft‑cancel pós‑pago OK",
                    "warn_msg": f"[{msisdn}] Soft‑cancel pós-pago falhou",
                },
                subscriber_data={"msisdn": msisdn},
            )

        return await self._execute_state_changing_request(
            breaker=self._contract_status_breaker,
            request_coro=request_coro,
            identifier=msisdn,
            operation_name="set_soft_cancel_postpaid",
        )
