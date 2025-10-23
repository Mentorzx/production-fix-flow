from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Body, HTTPException, status, Depends
from pydantic import BaseModel, Field, model_validator

from pff.config import settings
from pff.utils import CacheManager, FileManager, logger

from ..models import SequenceInfo
from ..deps import verify_api_key

# from ..auth import get_current_user

"""
Sequences router for managing YAML sequence definitions.

This module provides CRUD operations for sequences that define
the steps to process lines/MSISDNs in the PFF system.
"""

# Get sequences file path from settings
SEQS_FILE = Path(settings.CONFIG_DIR) / "sequences.yaml"
file_manager = FileManager()
cache_manager = CacheManager()
_YAML_LOCK = threading.Lock()
router = APIRouter()


# ── Pydantic models ────────────────────────────────────────────────────
class Step(BaseModel):
    """Represents a single step in a sequence"""

    method: str | None = Field(
        default=None,
        min_length=1,
        description="Name of the method to call (omitted if using next_sequence)",
    )
    args: dict[str, Any] = Field(default_factory=dict)
    when: str | None = None
    next_sequence: str | None = None
    loop_over: str | None = None
    save_as: str | None = None
    set: str | None = None
    value: str | None = None
    description: str | None = None


class SequencePayload(BaseModel):
    """Payload for creating a new sequence"""

    name: str = Field(min_length=1, pattern=r"^[\w\.\-]+$")
    steps: list[Step] = Field(min_length=1)

    @model_validator(mode="after")
    def _rules(self):
        """Validate step rules"""
        for i, step in enumerate(self.steps, 1):
            if step.set and step.value is None:
                raise ValueError(f"step {i}: 'set' requires 'value'")
            if step.next_sequence and step.method:
                raise ValueError(
                    f"step {i}: use either 'next_sequence' OR 'method', not both"
                )
        return self


class SequenceUpdate(BaseModel):
    """Payload for updating an existing sequence"""

    steps: list[Step] = Field(min_length=1)

    @model_validator(mode="after")
    def _rules(self):
        """Validate step rules"""
        for i, step in enumerate(self.steps, 1):
            if step.set and step.value is None:
                raise ValueError(f"step {i}: 'set' requires 'value'")
            if step.next_sequence and step.method:
                raise ValueError(
                    f"step {i}: use either 'next_sequence' OR 'method', not both"
                )
        return self


# ════════════════════════════════════════════════════════════════════════════
#                               GET
# ════════════════════════════════════════════════════════════════════════════
@router.get("/", response_model=list[SequenceInfo])
def list_sequences(
    api_key: str = Depends(verify_api_key)
):
    """
    Retrieves a list of all available sequences.

    Returns summary information for each sequence including name,
    number of steps, and description from first step if available.

    Returns:
        List of SequenceInfo objects
    """
    # Try cache first
    cache_key = "sequences:list"
    cached = cache_manager.get(cache_key)
    if cached:
        return cached

    data = file_manager.read(SEQS_FILE)
    if not isinstance(data, dict):
        data = {}

    sequences = [
        SequenceInfo(
            name=k,
            steps=len(v),
            description=(
                v[0].get("description") if v and isinstance(v[0], dict) else None
            ),
        )
        for k, v in data.items()
        if isinstance(v, list)
    ]

    # Cache for 5 minutes
    cache_manager.set(cache_key, sequences, ttl=300)
    logger.info(f"Listadas {len(sequences)} sequências disponíveis")

    return sequences


@router.get("/{name}", response_model=list[dict[str, Any]])
def get_sequence(
    name: str,
    # user: Annotated[dict, Depends(get_current_user)]  # Temporariamente comentado
):
    """
    Retrieve a specific sequence by name.

    Returns the complete sequence definition including all steps
    and their configurations.

    Args:
        name: Sequence name

    Returns:
        List of step dictionaries

    Raises:
        HTTPException: If sequence not found
    """
    # Try cache first
    cache_key = f"sequence:{name}"
    cached = cache_manager.get(cache_key)
    if cached:
        return cached

    data = file_manager.read(SEQS_FILE)
    if not isinstance(data, dict):
        data = {}

    if name not in data:
        logger.warning(f"Sequência não encontrada: {name}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Sequence '{name}' not found"
        )

    sequence = data[name]

    # Cache individual sequence for 10 minutes
    cache_manager.set(cache_key, sequence, ttl=600)

    return sequence


# ════════════════════════════════════════════════════════════════════════════
#                               POST
# ════════════════════════════════════════════════════════════════════════════
@router.post("/", status_code=status.HTTP_201_CREATED)
def create_sequence(
    payload: SequencePayload,
    # user: Annotated[dict, Depends(get_current_user)]  # Temporariamente comentado
):
    """
    Create a new sequence definition.

    The sequence name must be unique and follow the pattern [a-zA-Z0-9.-_]+

    Args:
        payload: SequencePayload with name and steps

    Returns:
        Success message with sequence details

    Raises:
        HTTPException: If sequence already exists
    """
    with _YAML_LOCK:
        data = file_manager.read(SEQS_FILE)
        if not isinstance(data, dict):
            data = {}

        if payload.name in data:
            logger.warning(f"Tentativa de criar sequência duplicada: {payload.name}")
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Sequence '{payload.name}' already exists",
            )

        # Convert steps to dict format
        steps_data = []
        for step in payload.steps:
            step_dict = step.model_dump(exclude_none=True)
            steps_data.append(step_dict)

        data[payload.name] = steps_data

        # Save to file
        file_manager.save(data, SEQS_FILE)

        # Clear cache
        if "sequences:list" in cache_manager:
            del cache_manager["sequences:list"]

        logger.success(
            f"Sequência '{payload.name}' criada com {len(payload.steps)} passos"
        )

    return {
        "message": f"Sequence '{payload.name}' created successfully",
        "name": payload.name,
        "steps": len(payload.steps),
    }


# ════════════════════════════════════════════════════════════════════════════
#                               PUT
# ════════════════════════════════════════════════════════════════════════════
@router.put("/{name}")
def update_sequence(
    name: str,
    payload: SequenceUpdate,
    # user: Annotated[dict, Depends(get_current_user)]  # Temporariamente comentado
):
    """
    Update an existing sequence definition.

    Replaces all steps in the sequence with the provided steps.

    Args:
        name: Sequence name to update
        payload: SequenceUpdate with new steps

    Returns:
        Success message with sequence details

    Raises:
        HTTPException: If sequence not found
    """
    with _YAML_LOCK:
        data = file_manager.read(SEQS_FILE)
        if not isinstance(data, dict):
            data = {}

        if name not in data:
            logger.warning(f"Tentativa de atualizar sequência inexistente: {name}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Sequence '{name}' not found",
            )

        # Convert steps to dict format
        steps_data = []
        for step in payload.steps:
            step_dict = step.model_dump(exclude_none=True)
            steps_data.append(step_dict)

        data[name] = steps_data

        # Save to file
        file_manager.save(data, SEQS_FILE)

        # Clear cache
        if "sequences:list" in cache_manager:
            del cache_manager["sequences:list"]
        if f"sequence:{name}" in cache_manager:
            del cache_manager[f"sequence:{name}"]

        logger.info(f"Sequência '{name}' atualizada com {len(payload.steps)} passos")

    return {
        "message": f"Sequence '{name}' updated successfully",
        "name": name,
        "steps": len(payload.steps),
    }


# ════════════════════════════════════════════════════════════════════════════
#                               DELETE
# ════════════════════════════════════════════════════════════════════════════
@router.delete("/{name}")
def delete_sequence(
    name: str,
    # user: Annotated[dict, Depends(get_current_user)]  # Temporariamente comentado
):
    """
    Delete a sequence from the configuration.

    Cannot delete sequences that are referenced by other sequences.

    Args:
        name: Sequence name to delete

    Returns:
        Success message

    Raises:
        HTTPException: If sequence not found or is referenced by others
    """
    with _YAML_LOCK:
        data = file_manager.read(SEQS_FILE)
        if not isinstance(data, dict):
            data = {}

        if name not in data:
            logger.warning(f"Tentativa de deletar sequência inexistente: {name}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Sequence '{name}' not found",
            )

        # Check if sequence is referenced by others
        referenced_by = []
        for seq_name, steps in data.items():
            if seq_name != name and isinstance(steps, list):
                for step in steps:
                    if isinstance(step, dict) and step.get("next_sequence") == name:
                        referenced_by.append(seq_name)

        if referenced_by:
            logger.error(f"Sequência '{name}' é referenciada por: {referenced_by}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Cannot delete sequence '{name}' because it's referenced by: {', '.join(referenced_by)}",
            )

        del data[name]

        # Save to file
        file_manager.save(data, SEQS_FILE)

        # Clear cache
        if "sequences:list" in cache_manager:
            del cache_manager["sequences:list"]
        if f"sequence:{name}" in cache_manager:
            del cache_manager[f"sequence:{name}"]

        logger.success(f"Sequência '{name}' deletada com sucesso")

    return {"message": f"Sequence '{name}' deleted successfully"}


# ════════════════════════════════════════════════════════════════════════════
#                               PATCH
# ════════════════════════════════════════════════════════════════════════════
@router.patch("/{name}/rename")
def rename_sequence(
    name: str,
    new_name: str = Body(..., embed=True),
    # user: Annotated[dict, Depends(get_current_user)]  # Temporariamente comentado
):
    """
    Rename an existing sequence.

    Updates all references to the sequence in other sequences.

    Args:
        name: Current sequence name
        new_name: New sequence name

    Returns:
        Success message with old and new names

    Raises:
        HTTPException: If sequence not found or new name already exists
    """
    if not new_name or not new_name.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="New name cannot be empty"
        )

    with _YAML_LOCK:
        data = file_manager.read(SEQS_FILE)
        if not isinstance(data, dict):
            data = {}

        if name not in data:
            logger.warning(f"Tentativa de renomear sequência inexistente: {name}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Sequence '{name}' not found",
            )

        if new_name in data:
            logger.warning(f"Novo nome já existe: {new_name}")
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Sequence '{new_name}' already exists",
            )

        # Copy sequence with new name
        data[new_name] = data[name]
        del data[name]

        # Update references in other sequences
        updated_refs = 0
        for seq_name, steps in data.items():
            if isinstance(steps, list):
                for step in steps:
                    if isinstance(step, dict) and step.get("next_sequence") == name:
                        step["next_sequence"] = new_name
                        updated_refs += 1

        # Save to file
        file_manager.save(data, SEQS_FILE)

        # Clear cache
        if "sequences:list" in cache_manager:
            del cache_manager["sequences:list"]
        if f"sequence:{name}" in cache_manager:
            del cache_manager[f"sequence:{name}"]

        logger.info(
            f"Sequência renomeada: '{name}' -> '{new_name}' ({updated_refs} referências atualizadas)"
        )

    return {
        "message": f"Sequence renamed from '{name}' to '{new_name}' successfully",
        "old_name": name,
        "new_name": new_name,
        "updated_references": updated_refs,
    }


# ════════════════════════════════════════════════════════════════════════════
#                               VALIDATE
# ════════════════════════════════════════════════════════════════════════════
@router.post("/validate")
def validate_sequence(
    payload: SequencePayload,
    # user: Annotated[dict, Depends(get_current_user)]  # Temporariamente comentado
):
    """
    Validate a sequence definition without saving.

    Useful for testing sequence syntax and checking references
    before creation.

    Args:
        payload: SequencePayload to validate

    Returns:
        Validation result with warnings if any
    """
    data = file_manager.read(SEQS_FILE)
    if not isinstance(data, dict):
        data = {}

    # Check if next_sequences exist
    missing_sequences = []
    for step in payload.steps:
        if step.next_sequence and step.next_sequence not in data:
            missing_sequences.append(step.next_sequence)

    warnings = []
    if missing_sequences:
        warnings.append(
            f"Referenced sequences not found: {', '.join(set(missing_sequences))}"
        )

    # Check for method existence (basic validation)
    known_methods = [
        "get_contract",
        "get_customer_enquiry",
        "set_contract_status",
        "validate_contract",
        "search_in",
        "set_observation",
        "get_validation",
    ]

    unknown_methods = []
    for step in payload.steps:
        if step.method and step.method not in known_methods:
            unknown_methods.append(step.method)

    if unknown_methods:
        warnings.append(f"Unknown methods: {', '.join(set(unknown_methods))}")

    logger.info(f"Validação de sequência '{payload.name}': {len(warnings)} avisos")

    return {
        "valid": True,
        "name": payload.name,
        "steps": len(payload.steps),
        "warnings": warnings,
    }
