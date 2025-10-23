"""
LineService Mutations - Write Operations

This module contains all POST/PATCH/DELETE operations:
- set_contract_status
- set_product_status
- delete_contract
- set_party_terminated
- set_consumer_list
- set_create_client

Part of Sprint 4 refactoring (line_service.py split into 4 files).
"""

from __future__ import annotations

from typing import Any

from pff.utils.clients import API

from .base import LineServiceBase, capture_collector


class LineServiceMutations(LineServiceBase):
    """
    Mutation operations for LineService (POST/PATCH/DELETE).

    All methods modify state and use circuit breakers for resilience.
    """

    # ──────────────────────── STATE‑CHANGERS ────────────────────────────

    @capture_collector
    async def set_contract_status(
        self,
        msisdn: str | None = None,
        status: str = "",
        *,
        customer_id: str | None = None,
        contract_id: str | None = None,
    ) -> bool:
        """
        Updates the status of a contract, identified by either (customer_id + contract_id) or by msisdn.
        Enhanced with circuit breaking protection.

        Args:
            msisdn: MSISDN identifier.
            status: New contract status value.
            customer_id: Customer ID (required with contract_id).
            contract_id: Contract ID (required with customer_id).

        Raises:
            ValueError: If neither msisdn nor (customer_id and contract_id) are provided.

        Returns:
            True if the operation succeeded.
        """
        if customer_id and contract_id:
            identifier = f"{customer_id}/{contract_id}"
            url, service_type = API.subscription(customer_id, contract_id)

            async def request_coro():
                return await self.make_request(
                    endpoint_config={
                        "url": url,
                        "type": service_type,
                        "method": "PATCH",
                        "json": {"status": [{"status": status}]},
                        "ok_msg": f"[{customer_id}] Contrato {contract_id} atualizado com sucesso",
                        "warn_msg": f"[{customer_id}] Erro ao atualizar contrato {contract_id}",
                    },
                    subscriber_data={
                        "customer_id": customer_id,
                        "contract_id": contract_id,
                    },
                )

            return await self._execute_state_changing_request(
                breaker=self._contract_status_breaker,
                request_coro=request_coro,
                identifier=identifier,
                operation_name="set_contract_status",
            )

        elif msisdn:
            data = {
                "communicationId": msisdn,
                "communicationIdType": "E.164",
                "status": [{"status": status}],
            }
            url, service_type = API.update_contract_status

            async def request_coro():
                return await self.make_request(
                    endpoint_config={
                        "url": url,
                        "type": service_type,
                        "method": "POST",
                        "json": data,
                        "ok_msg": f"[{msisdn}] Status do contrato atualizado para {status}",
                        "warn_msg": f"[{msisdn}] Erro ao atualizar status do contrato",
                    },
                    subscriber_data={"msisdn": msisdn},
                )

            return await self._execute_state_changing_request(
                breaker=self._contract_status_breaker,
                request_coro=request_coro,
                identifier=msisdn,
                operation_name="set_contract_status",
            )

        else:
            raise ValueError(
                "set_contract_status: passe 'msisdn' ou ('customer_ext_id' e 'contract_ext_id')."
            )

    @capture_collector
    async def set_product_status(
        self,
        msisdn: str,
        activate: bool,
        *,
        product_offering_external_id: str | None = None,
        product_external_id: str | None = None,
        product_id: str | None = None,
        contract_id: str | None = None,
        customer_id: str | None = None,
    ) -> bool:
        """
        Activates or terminates a product for the given MSISDN.
        Enhanced with circuit breaking protection.

        Args:
            msisdn: MSISDN identifier.
            activate: Whether to activate (True) or terminate (False) the product.
            product_offering_external_id: Product offering external ID (for activation).
            product_external_id: Product external ID (for activation).
            product_id: Product ID (for termination).
            contract_id: Contract ID (for termination).
            customer_id: Customer ID (for termination).

        Raises:
            ValueError: If required parameters for activation or termination are missing.

        Returns:
            True if the operation succeeded.
        """
        if activate:
            data = {
                "product": [
                    {
                        "productOfferingExternalId": product_offering_external_id,
                        "externalId": product_external_id,
                        "status": [{"status": "ProductActive"}],
                    }
                ]
            }
            url, service_type = API.activate_product(msisdn)

            async def request_coro():
                return await self.make_request(
                    endpoint_config={
                        "url": url,
                        "type": service_type,
                        "method": "POST",
                        "json": data,
                        "ok_msg": f"[{msisdn}] Produto ativado com sucesso",
                        "warn_msg": f"[{msisdn}] Erro ao ativar produto",
                    },
                    subscriber_data={"msisdn": msisdn},
                )

            return await self._execute_state_changing_request(
                breaker=self._product_status_breaker,
                request_coro=request_coro,
                identifier=msisdn,
                operation_name="set_product_status (activate)",
            )
        else:
            if not customer_id or not contract_id:
                raise ValueError(
                    "Both 'customer_id' and 'contract_id' must be provided."
                )
            data = {
                "product": [
                    {"id": product_id, "status": [{"status": "ProductTerminated"}]}
                ]
            }
            url, service_type = API.subscription(customer_id, contract_id)

            async def request_coro():
                return await self.make_request(
                    endpoint_config={
                        "url": url,
                        "type": service_type,
                        "method": "PATCH",
                        "json": data,
                        "ok_msg": f"[{msisdn}] Produto terminado com sucesso",
                        "warn_msg": f"[{msisdn}] Erro ao terminar produto",
                    },
                    subscriber_data={
                        "customer_id": customer_id,
                        "contract_id": contract_id,
                        "product_id": product_id,
                    },
                )

            return await self._execute_state_changing_request(
                breaker=self._product_status_breaker,
                request_coro=request_coro,
                identifier=f"{msisdn}/{customer_id}/{contract_id}",
                operation_name="set_product_status (terminate)",
            )

    @capture_collector
    async def delete_contract(self, customer_id: str, contract_id: str) -> bool:
        """
        Deletes a contract for the given customer.
        Enhanced with circuit breaking protection.

        Args:
            customer_id: Customer ID.
            contract_id: Contract ID.

        Returns:
            True if the contract was deleted successfully.
        """
        identifier = f"{customer_id}/{contract_id}"
        url, service_type = API.delete_contract(customer_id, contract_id)

        async def request_coro():
            return await self.make_request(
                endpoint_config={
                    "url": url,
                    "type": service_type,
                    "method": "DELETE",
                    "ok_msg": f"[{customer_id}] Contrato deletado com sucesso",
                    "warn_msg": f"[{customer_id}] Erro ao deletar contrato",
                },
                subscriber_data={
                    "customer_id": customer_id,
                    "contract_id": contract_id,
                },
            )

        return await self._execute_state_changing_request(
            breaker=self._delete_contract_breaker,
            request_coro=request_coro,
            identifier=identifier,
            operation_name="delete_contract",
        )

    @capture_collector
    async def set_party_terminated(self, party_id: str) -> bool:
        """
        Terminates a party by marking its status as 'PartyTerminated'.
        Enhanced with circuit breaking protection.

        Args:
            party_id: The party's identifier.

        Returns:
            True if the party was terminated successfully.
        """
        url, service_type = API.party_cascade(party_id)

        async def request_coro():
            return await self.make_request(
                endpoint_config={
                    "url": url,
                    "type": service_type,
                    "method": "POST",
                    "json": {"resource": {"statuses": [{"status": "PartyTerminated"}]}},
                    "headers": {"ERICSSON.Cascade-Termination": "true"},
                    "ok_msg": f"[{party_id}] Party terminated com sucesso",
                    "warn_msg": f"[{party_id}] Erro ao terminar party",
                },
                subscriber_data={"party_id": party_id},
            )

        return await self._execute_state_changing_request(
            breaker=self._party_termination_breaker,
            request_coro=request_coro,
            identifier=party_id,
            operation_name="set_party_terminated",
        )

    @capture_collector
    async def set_consumer_list(
        self,
        provider_customer_ext_id: str,
        provider_contract_ext_id: str,
        provider_product_ext_id: str,
        consumer_entries: list[dict[str, str]],
    ) -> bool:
        """
        Updates the consumer list for a provider's product.
        Enhanced with circuit breaking protection.

        Args:
            provider_customer_ext_id: Provider's customer external ID.
            provider_contract_ext_id: Provider's contract external ID.
            provider_product_ext_id: Provider's product external ID.
            consumer_entries: List of consumer entries (already formatted for the API).

        Returns:
            True if the operation succeeded.
        """
        data = {
            "providerCustomerExternalId": provider_customer_ext_id,
            "providerContractExternalId": provider_contract_ext_id,
            "providerProductExternalId": provider_product_ext_id,
            "consumer": consumer_entries,
        }

        identifier = f"{provider_customer_ext_id}/{provider_contract_ext_id}/{provider_product_ext_id}"
        url, service_type = API.manage_consumer_list

        async def request_coro():
            return await self.make_request(
                endpoint_config={
                    "url": url,
                    "type": service_type,
                    "method": "POST",
                    "json": data,
                    "headers": {"ERICSSON.External-User-Id": "BKO"},
                    "ok_msg": f"[{provider_customer_ext_id}] Consumer list atualizado",
                    "warn_msg": f"[{provider_customer_ext_id}] Consumer list falhou",
                },
                subscriber_data={
                    "provider_customer_ext_id": provider_customer_ext_id,
                    "provider_contract_ext_id": provider_contract_ext_id,
                    "provider_product_ext_id": provider_product_ext_id,
                },
            )

        return await self._execute_state_changing_request(
            breaker=self._consumer_list_breaker,
            request_coro=request_coro,
            identifier=identifier,
            operation_name="set_consumer_list",
        )

    @capture_collector
    async def set_create_client(self, data: dict[str, Any]) -> bool:
        """
        Sends a POST request to create a new client using the provided data.
        Enhanced with circuit breaking protection.

        Args:
            data: A dictionary containing the client information to be created.

        Returns:
            True if the client was created successfully, False otherwise.
        """
        subscriber_data = {}
        if "msisdn" in data:
            subscriber_data["msisdn"] = data["msisdn"]
        if "externalId" in data:
            subscriber_data["external_id"] = data["externalId"]

        identifier = data.get("msisdn") or data.get("externalId") or "unknown"
        url, service_type = API.create_client

        async def request_coro():
            return await self.make_request(
                endpoint_config={
                    "url": url,
                    "type": service_type,
                    "method": "POST",
                    "json": data,
                    "headers": {"ERICSSON.External-User-Id": "BKO"},
                    "ok_msg": "Cliente criado com sucesso",
                    "warn_msg": "Erro ao criar cliente",
                },
                subscriber_data=subscriber_data,
            )

        return await self._execute_state_changing_request(
            breaker=self._create_client_breaker,
            request_coro=request_coro,
            identifier=str(identifier),
            operation_name="set_create_client",
        )
