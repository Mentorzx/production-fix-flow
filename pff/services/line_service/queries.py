"""
LineService Queries - Read Operations

This module contains all GET/read-only operations:
- get_customer_enquiry
- get_individual_party
- get_contract (+ _resolve_enquiry, _fetch_single_contract)
- get_product (+ helper methods)

Part of Sprint 4 refactoring (line_service.py split into 4 files).
"""

from __future__ import annotations

from typing import Any

from aiobreaker import CircuitBreakerError
from aiocache import cached
from aiocache.serializers import MsgPackSerializer

from pff.utils import logger, progress_bar
from pff.utils.clients import API

from .base import LineServiceBase, capture_collector


class LineServiceQueries(LineServiceBase):
    """
    Query operations for LineService (GET/read-only).

    All methods are read-only and use circuit breakers + caching.
    """

    # ──────────────────────── FETCHERS ─────────────────────────────────

    @capture_collector
    @cached(ttl=60, serializer=MsgPackSerializer(), namespace="customer_enquiry")
    async def get_customer_enquiry(
        self, msisdn: str | None = None, customer_id: str | None = None
    ) -> dict[str, Any]:
        """
        Fetches customer enquiry data using either an MSISDN or a customer ID.
        Enhanced with caching, circuit breaking, and request coalescing.

        Args:
            msisdn: MSISDN identifier.
            customer_id: Customer ID.

        Raises:
            ValueError: If neither msisdn nor customer_id is provided.

        Returns:
            The customer enquiry data.
        """
        if not msisdn and not customer_id:
            raise ValueError("Either 'msisdn' or 'customer_id' must be provided.")

        identifier = msisdn or customer_id
        cache_key = f"enquiry_{identifier}"

        if msisdn:
            url, service_type = API.customer_enquiry(msisdn)
            subscriber_data = {"msisdn": msisdn}
        else:
            url, service_type = API.customer_enquiry_by_customer(customer_id or "")
            subscriber_data = {"customer_id": customer_id}

        async def request_coro():
            return await self.make_request(
                endpoint_config={
                    "url": url,
                    "type": service_type,
                    "method": "GET",
                    "ok_msg": f"[{identifier}] Dados obtidos com sucesso da customer enquiry",
                    "warn_msg": f"[{identifier}] Erro ao obter dados da customer enquiry",
                },
                subscriber_data=subscriber_data,
            )

        return await self._execute_resilient_request(
            breaker=self._enquiry_breaker,
            cache_key=cache_key,
            request_coro=request_coro,
            identifier=str(identifier),
            operation_name="get_customer_enquiry",
        )

    @capture_collector
    @cached(ttl=30, serializer=MsgPackSerializer(), namespace="individual_party")
    async def get_individual_party(self, external_id: str) -> dict[str, Any]:
        """
        Fetches individual party details by external ID.
        Enhanced with caching and circuit breaking.

        Args:
            external_id: External identifier for the party.

        Returns:
            The individual party data.
        """
        cache_key = f"party_{external_id}"
        url, service_type = API.individual_party_enquiry(external_id)

        async def request_coro():
            return await self.make_request(
                endpoint_config={
                    "url": url,
                    "type": service_type,
                    "method": "GET",
                    "ok_msg": f"[{external_id}] Dados obtidos com sucesso da individual party",
                    "warn_msg": f"[{external_id}] Erro ao obter dados da individual party",
                },
                subscriber_data={"external_id": external_id},
            )

        return await self._execute_resilient_request(
            breaker=self._individual_party_breaker,
            cache_key=cache_key,
            request_coro=request_coro,
            identifier=external_id,
            operation_name="get_individual_party",
        )

    # ──────────────────────── LOCATORS ──────────────────────────────────

    @capture_collector
    async def get_contract(
        self,
        *,
        enquiry: dict[str, Any] | None = None,
        search: dict[str, Any],
        product_status: str = "ProductActive",
    ) -> dict[str, Any]:
        """
        Searches for a contract matching the specified filters.
        Enhanced with circuit breaking for direct contract fetching.

        Args:
            enquiry: Pre-fetched enquiry data.
            search: Search filters for contract and product.
            product_status: Status of the product to filter by (default "ProductActive").

        Raises:
            ValueError: If neither 'enquiry', 'msisdn', nor 'customer_id' is provided in search.
            RuntimeError: If the search yields none or more than one contract.

        Returns:
            Dictionary containing customer and contract information.
        """
        enquiry = enquiry or await self._resolve_enquiry(search)
        if not enquiry or "externalId" not in enquiry:
            raise RuntimeError(
                f"Customer enquiry não retornou 'externalId' "
                f"para {search.get('msisdn') or search.get('customer_id')}"
            )
        if {"customer_id", "id"} <= search.keys():
            contract = await self._fetch_single_contract(search)
        else:
            all_contracts = enquiry.get("contract", [])
            contract_criteria = {
                k: v
                for k, v in search.items()
                if k not in ("customer_id", "productOfferingExternalId")
            }
            offering_id = search.get("productOfferingExternalId")
            matched = (
                await self._research.search_in(all_contracts, contract_criteria)
                if contract_criteria
                else all_contracts
            )

            if offering_id:
                prod_crit = {
                    "productOfferingExternalId": offering_id,
                    "status": [{"status": product_status}],
                }

                async def has_active_offering(ct):
                    return bool(
                        await self._research.search_in(ct.get("product", []), prod_crit)
                    )

                matched = [ct for ct in matched if await has_active_offering(ct)]
            if len(matched) != 1:
                raise RuntimeError(
                    f"Esperava 1 contrato, encontrou {len(matched)} (search={search!r})"
                )
            contract = matched[0]
            logger.success(
                f"Cliente - [{enquiry['externalId']}] Contrato localizado: "
                f"{contract.get('id')} ({contract.get('externalId')})"
            )

        return {
            "customer_id": enquiry["id"],
            "customer_ext_id": enquiry["externalId"],
            "contract": contract,
        }

    async def _resolve_enquiry(self, search: dict[str, Any]) -> dict[str, Any]:
        """
        Resolve or fetch the customer enquiry based on provided search parameters.

        Args:
            search:
                A dictionary of search criteria which may include:
                - 'externalId': list of external identifiers (e.g. MSISDN)
                - 'customer_id': the customer's ID
                - 'id': the contract ID

        Returns:
            The full enquiry data as a dictionary, obtained either directly
            or by calling `get_customer_enquiry`.

        Raises:
            ValueError:
                If neither an externalId nor a valid customer_id (without contract id)
                is present in `search`.
        """
        ext_ids = (
            await self._research.search_in(search, {"externalId": self._research.ANY})
            or []
        )
        msisdn = next(iter(ext_ids[0].values())) if ext_ids else None
        cust_id = search.get("customer_id")

        if msisdn:
            return await self.get_customer_enquiry(msisdn=str(msisdn))
        elif cust_id:
            return await self.get_customer_enquiry(customer_id=cust_id)

        raise ValueError(
            "Passe 'enquiry' ou inclua identificadores suficientes em search."
        )

    async def _fetch_single_contract(self, search: dict[str, Any]) -> dict[str, Any]:
        """
        Fetch a single contract directly from the API using customer_id and contract id.
        Enhanced with circuit breaking protection.

        Args:
            search:
                Must contain:
                - 'customer_id': the customer's ID
                - 'id': the contract ID

        Returns:
            A dictionary contract: the contract dict

        Raises:
            ValueError:
                If either 'customer_id' or 'id' is missing or falsy.
            RuntimeError:
                If the API call returns no contract or a non-dict response.
        """
        cust_id = search.get("customer_id")
        ctt_id = search.get("id")
        if not cust_id or not ctt_id:
            raise ValueError("Both 'customer_id' and 'id' must be provided.")

        try:
            url, service_type = API.read_contract(str(ctt_id), str(cust_id))
            contract = await self._contract_breaker.call(
                self.make_request,
                endpoint_config={
                    "url": url,
                    "type": service_type,
                    "method": "GET",
                    "ok_msg": f"Cliente - [{cust_id}] Contrato {ctt_id} obtido",
                    "warn_msg": f"[{cust_id}] Erro ao ler contrato {ctt_id}",
                },
                subscriber_data={
                    "customer_id": str(cust_id),
                    "contract_id": str(ctt_id),
                },
            )

            if not isinstance(contract, dict):
                raise RuntimeError(
                    f"Contrato id={ctt_id} não encontrado para customer_id={cust_id}"
                )

            return contract

        except CircuitBreakerError as e:
            logger.error(
                f"Circuit breaker open for contract fetch [{cust_id}/{ctt_id}]: {e}"
            )
            raise RuntimeError(f"Service temporarily unavailable for contract {ctt_id}")

    @capture_collector
    async def get_product(
        self,
        *,
        contract: dict[str, Any] | None = None,
        enquiry: dict[str, Any] | None = None,
        search: dict[str, Any] | None = None,
        status: str | None = "ProductActive",
    ) -> dict[str, Any] | list[dict[str, Any]] | None:
        """
        Retrieve product(s) based on the provided contract, enquiry, search criteria, and status.

        Args:
            contract: A dictionary representing a contract. If provided,
                products will be retrieved from this contract. Defaults to None.
            enquiry: A dictionary representing an enquiry. If provided,
                products will be retrieved from the contracts within the enquiry. Defaults to None.
            search: Search criteria to filter products. If None, products
                are filtered only by status. Defaults to None.
            status: The status of the products to filter. If "ProductActive", only
                active products are retrieved. If "all", all products are retrieved regardless of
                status. Defaults to "ProductActive".

        Returns:
            - A single product dictionary if exactly one product matches the criteria.
            - A list of product dictionaries if multiple products match the criteria.
            - None if no products match the criteria.

        Raises:
            ValueError: If neither `contract` nor `enquiry` is provided.

        Notes:
            - Logs a success message for each product found, including its ID and search criteria.
            - Logs a warning if no products are found.
        """
        matched_products: list[dict[str, Any]] = []
        if contract:
            all_contracts = [contract]
        elif enquiry:
            all_contracts = enquiry.get("contract", [])
        else:
            raise ValueError("Missing 'contract' or 'enquiry'.")

        for ctt in progress_bar(all_contracts, desc="Buscando nos contratos..."):
            products = self._extract_products(ctt) if ctt else []
            if not search:
                matches = [
                    product
                    for product in products
                    if self._status_ok(product, status or "")
                ]
            else:
                criteria = [search]
                if status and str(status).lower() != "all":
                    criteria.append({"status": [{"status": status}]})
                matches = await self._research.search_in(products, criteria)

            for prod in progress_bar(matches, desc="Procurando produto..."):
                prefix = (
                    f"[{enquiry['externalId']}] "
                    if enquiry and "externalId" in enquiry
                    else ""
                )
                logger.success(
                    f"{prefix}Produto localizado: {prod.get('id')} usando {(search if search else 'nenhum critério')!r}"
                )
                ext_id = (
                    prod.get("productOfferingExternalId")
                    or prod.get("productOffering", {}).get("productOfferingExternalId")
                    or prod.get("productOffering", {}).get("productOfferingId")
                )
                if ext_id:
                    prod["product_offering_external_id"] = ext_id
                    prod["product_external_id"] = ext_id
            matched_products.extend(matches)

        if not matched_products:
            logger.warning(
                f"Nenhum produto encontrado usando {search!r} em {len(all_contracts)} contrato(s)"
            )
            return None
        return matched_products[0] if len(matched_products) == 1 else matched_products

    # ──────────────────────── HELPER METHODS ────────────────────────────

    def _status_matches(self, prod: dict[str, Any], wanted: str | None) -> bool:
        """
        Determines if the status of a given product matches the desired status.

        Args:
            prod: A dictionary representing the product. It may contain
                a "status" key with a string value, a "statuses" key with a list of dictionaries,
                or a "status" key within a nested dictionary.
            wanted: The desired status to match.

        Returns:
            True if the product's status matches the desired status, False otherwise.
        """
        raw = prod.get("status") or prod.get("statuses")
        if not raw:
            return False
        if isinstance(raw, str):
            return raw == wanted
        if isinstance(raw, dict):
            return raw.get("status") == wanted
        if isinstance(raw, list):
            return any(item.get("status") == wanted for item in raw)
        return False

    def _extract_products(self, ctt: dict[str, Any]) -> list[dict[str, Any]]:
        """
        Extracts a list of products from the given dictionary.

        This method retrieves products from the "product" key and
        extends the list with products found under the "resources" key.

        Args:
            ctt: A dictionary containing product and resource information.

        Returns:
            A list of dictionaries representing the extracted products.
        """
        prods = list(ctt.get("product", []))
        for res in ctt.get("resources", []):
            prods.extend(res.get("products", []))
        return prods

    def _status_ok(self, prod: dict[str, Any], wanted: str) -> bool:
        """
        Determines if the status of a product matches the desired status.

        Args:
            prod: A dictionary representing the product, which may contain
                a "status" key (single status) or a "statuses" key (list of statuses).
            wanted: The desired status to check against. If `None` or "all" (case-insensitive),
                the function will always return True.

        Returns:
            True if the desired status matches the product's status or statuses,
            or if `wanted` is `None` or "all". False otherwise.
        """
        if wanted is None or str(wanted).lower() == "all":
            return True
        rows = prod.get("status") or prod.get("statuses") or []
        if isinstance(rows, dict):
            rows = [rows]
        if isinstance(rows, str):
            return rows == wanted
        return any(r.get("status") == wanted for r in rows)
