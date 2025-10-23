from __future__ import annotations

import asyncio
import os
import re
from typing import Any

import polars as pl
from simpleeval import simple_eval

from pff.config import settings
from pff.utils import FileManager, Research, logger
from pff.utils.polars_extensions import PolarsResearch, ResponseToDataFrameConverter

PLACEHOLDER_PATTERN = re.compile(r"{{\s*([^{}]+?)\s*}}")
SINGLE_PLACEHOLDER_PATTERN = re.compile(r"^{{\s*([^{}]+?)\s*}}$")


class SequenceService:
    """
    SequenceService provides a framework for executing predefined sequences of
    steps, where each step can invoke methods from registered services or
    internal handlers. It supports conditional execution, argument resolution
    with context-based placeholders, and result storage for use in subsequent
    steps. This enables flexible, configurable workflows for processing data or
    orchestrating service calls asynchronously.
    """

    def __init__(self, services: dict[str, Any]) -> None:
        """
        Initialize SequenceService with a dictionary of available services.
        """
        self._services = services
        self._file_manager = FileManager()
        self._dict_research = Research()
        self._polars_research = PolarsResearch()
        self._df_converter = ResponseToDataFrameConverter()

        config_path = settings.ROOT_DIR / "config" / "sequences.yaml"
        self._sequences = self._file_manager.read(config_path)

    async def run(
        self, msisdn: str, sequence_name: str, *, collector=None, **row_vars: Any
    ) -> None:
        """
        Execute a predefined sequence of steps for a given MSISDN asynchronously.
        """
        with logger.contextualize(
            msisdn=os.path.basename(msisdn) if "/" in msisdn else msisdn,
            sequence=sequence_name,
        ):
            logger.debug(f"Iniciando sequência '{sequence_name}'")
            line_service = self._services.get("line")
            if line_service and collector:
                line_service._collector = collector

            context = self._initialize_context(msisdn, collector, **row_vars)

            if sequence_name not in self._sequences:
                raise KeyError(f"Sequence '{sequence_name}' not defined.")

            await self._execute_steps(self._sequences[sequence_name], context)

    def _initialize_context(
        self, msisdn_or_data: Any, collector: Any, **row_vars: Any
    ) -> dict[str, Any]:
        """
        Create the initial context for sequence execution.
        """
        context = {"msisdn": msisdn_or_data, "collector": collector, **row_vars}
        return context

    async def _execute_steps(
        self, steps: list[dict[str, Any]], context: dict[str, Any]
    ) -> None:
        for step_index, step in enumerate(steps):
            try:
                if "when" in step:
                    condition_result = await self._evaluate_expression(
                        step["when"], context
                    )
                    if not bool(condition_result):
                        logger.debug(
                            f"Pulei o passo {step_index + 1}: condição 'when' é falsa."
                        )
                        continue
                if "loop_over" in step:
                    await self._execute_with_loop(step, context)
                elif "next_sequence" in step:
                    await self._execute_next_sequence(step, context)
                elif "method" in step:
                    await self._execute_method(step, context)
                elif "set" in step:
                    await self._execute_set(step, context)

            except Exception as e:
                logger.error(f"Erro no passo {step_index + 1}: {e}", exc_info=True)
                raise

    async def _execute_with_loop(
        self, step: dict[str, Any], context: dict[str, Any]
    ) -> None:
        """Execute step with loop_over."""
        loop_var = step["loop_over"]
        items = await self._evaluate_expression(loop_var, context)

        if not items:
            logger.debug(f"Loop sobre '{loop_var}' está vazio, pulando")
            return

        for item in items:
            loop_context = context.copy()
            loop_context["item"] = item

            if "next_sequence" in step:
                await self._execute_next_sequence(step, loop_context)
            elif "method" in step:
                await self._execute_method(step, loop_context)

    async def _execute_next_sequence(
        self, step: dict[str, Any], context: dict[str, Any]
    ) -> None:
        """Execute another sequence."""
        sequence_name = step["next_sequence"]

        if sequence_name not in self._sequences:
            raise KeyError(f"Subsequence '{sequence_name}' not found.")

        logger.debug(f"Executando subsequência: {sequence_name}")
        await self._execute_steps(self._sequences[sequence_name], context)

    async def _execute_set(self, step: dict[str, Any], context: dict[str, Any]) -> None:
        """Execute set operation."""
        var_name = step["set"]
        value = await self._resolve_arguments(step["value"], context)
        context[var_name] = value
        logger.debug(f"Variável '{var_name}' = {value}")

    async def _execute_method(
        self, step: dict[str, Any], context: dict[str, Any]
    ) -> None:
        """
        Execute a method specified in the step asynchronously.
        """
        method_path = step["method"]
        resolved_args = await self._resolve_arguments(step.get("args", {}), context)
        result = None

        if "." in method_path:
            service_name, method_name = method_path.split(".", 1)
            if service_name not in self._services:
                raise AttributeError(
                    f"Serviço '{service_name}' não foi registrado no SequenceService."
                )

            service_instance = self._services[service_name]
            if not hasattr(service_instance, method_name):
                raise AttributeError(
                    f"Método '{method_name}' não encontrado no serviço '{service_name}'."
                )

            method = getattr(service_instance, method_name)

            # Check if method is a coroutine and await it if necessary
            if asyncio.iscoroutinefunction(method):
                result = await method(**resolved_args)
            else:
                result = method(**resolved_args)
                if asyncio.iscoroutine(result):
                    result = await result

        elif hasattr(self, f"_handle_{method_path}"):
            method = getattr(self, f"_handle_{method_path}")

            # Check if internal handler is a coroutine
            if asyncio.iscoroutinefunction(method):
                result = await method(resolved_args)
            else:
                result = method(resolved_args)
        else:
            raise ValueError(
                f"Formato de método inválido: '{method_path}'. Use 'servico.metodo'."
            )

        if "save_as" in step:
            var_name = step["save_as"]
            context[var_name] = result

            df_result = self._df_converter.json_to_dataframe(result)
            if df_result is not None:
                df_var_name = f"{var_name}_df"
                context[df_var_name] = df_result
                logger.debug(
                    f"DataFrame salvo como '{df_var_name}' com shape {df_result.shape}"
                )

    async def _handle_search_in_dict(self, args: dict) -> list:
        """
        Search in a dictionary using given criteria.
        """
        return await self._dict_research.search_in(
            data=args["data"], criterias=args["criteria"]
        )

    def _handle_search_in_df(self, args: dict) -> pl.DataFrame:
        """
        Search in a DataFrame using given criteria.
        """
        return self._polars_research.search_dataframe(
            df=args["dataframe"], criteria=args["criteria"]
        )

    async def _evaluate_expression(
        self, expression: str, context: dict[str, Any]
    ) -> Any:
        try:
            rendered_expr = await self._render(expression, context)
            logger.debug(f"Avaliando: '{expression}' → '{rendered_expr}'")

            if "info" in context:
                status = context["info"]["contract"]["status"][-1]["status"]
                logger.debug(f"Status atual do contrato: {status}")

            # Use simple_eval instead of eval() for security
            result = simple_eval(rendered_expr, names=context)
            return result
        except Exception as e:
            logger.error(f"Erro ao avaliar '{expression}': {e}")
            return False

    async def _render(self, value: Any, context: dict[str, Any]) -> Any:
        """
        Render placeholders in a string using the context asynchronously.
        """
        if not isinstance(value, str):
            return value

        match = SINGLE_PLACEHOLDER_PATTERN.match(value)
        if match:
            expression = match.group(1).strip()
            return await self._evaluate_expression(expression, context)

        async def repl(m: re.Match[str]) -> str:
            val = await self._evaluate_expression(m.group(1).strip(), context)
            return str(val)

        # Handle multiple placeholders
        result = value
        for match in PLACEHOLDER_PATTERN.finditer(value):
            replacement = await repl(match)
            result = result.replace(match.group(0), replacement)

        return result

    async def _resolve_arguments(self, value: Any, context: dict[str, Any]) -> Any:
        """
        Recursively resolve arguments, rendering placeholders as needed asynchronously.
        """
        if isinstance(value, str):
            return await self._render(value, context)
        if isinstance(value, dict):
            resolved_dict = {}
            for k, v in value.items():
                resolved_dict[k] = await self._resolve_arguments(v, context)
            return resolved_dict
        if isinstance(value, list):
            resolved_list = []
            for v in value:
                resolved_list.append(await self._resolve_arguments(v, context))
            return resolved_list
        return value
