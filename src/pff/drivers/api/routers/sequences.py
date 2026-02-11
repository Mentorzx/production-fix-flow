from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, HTTPException, status, Depends
from pydantic import BaseModel, Field, model_validator

from pff.shared.core.config import SEQUENCES_CONFIG_PATH
from pff.shared import CacheManager, FileManager, logger
from pff.shared.acceleration.concurrency import get_lock

from ..models import SequenceInfo
from ..deps import verify_api_key

"""
Sequences router for managing YAML sequence definitions.

This module provides CRUD operations for sequences that define
the steps to process lines/MSISDNs in the PFF system.
"""

SEQS_FILE = SEQUENCES_CONFIG_PATH
file_manager = FileManager()
cache_manager = CacheManager()
_YAML_LOCK = get_lock()
router = APIRouter()


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
                raise ValueError(f"step {i}: use either 'next_sequence' OR 'method', not both")
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
                raise ValueError(f"step {i}: use either 'next_sequence' OR 'method', not both")
        return self


@router.get("/", response_model=list[SequenceInfo])
def list_sequences(api_key: str = Depends(verify_api_key)):
    """
    Retrieves a list of all available sequences.
    """
    cache_key = "sequences:list"
    cached = cache_manager.get(cache_key)
    if cached:
        return cached

    data = file_manager.read(SEQS_FILE, return_native=True)
    if not isinstance(data, dict):
        data = {}

    sequences = [
        SequenceInfo(
            name=k,
            steps=len(v),
            description=(v[0].get("description") if v and isinstance(v[0], dict) else None),
        )
        for k, v in data.items()
        if isinstance(v, list)
    ]

    cache_manager.set(cache_key, sequences, ttl=300)
    logger.info(f"Listadas {len(sequences)} sequências disponíveis")

    return sequences


@router.get("/{name}", response_model=list[dict[str, Any]])
def get_sequence(
    name: str,
):
    """
    Retrieve a specific sequence by name.
    """
    cache_key = f"sequence:{name}"
    cached = cache_manager.get(cache_key)
    if cached:
        return cached

    data = file_manager.read(SEQS_FILE, return_native=True)
    if not isinstance(data, dict):
        data = {}

    if name not in data:
        logger.warning(f"Sequence not found: {name}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Sequence '{name}' not found"
        )

    sequence = data[name]

    cache_manager.set(cache_key, sequence, ttl=600)

    return sequence


@router.post("/", status_code=status.HTTP_201_CREATED)
def create_sequence(
    payload: SequencePayload,
):
    """
    Create a new sequence definition.
    """
    with _YAML_LOCK:
        data = file_manager.read(SEQS_FILE, return_native=True)
        if not isinstance(data, dict):
            data = {}

        if payload.name in data:
            logger.warning(f"Attempt to create duplicate sequence: {payload.name}")
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Sequence '{payload.name}' already exists",
            )

        steps_data = []
        for step in payload.steps:
            step_dict = step.model_dump(exclude_none=True)
            steps_data.append(step_dict)

        data[payload.name] = steps_data

        file_manager.save(data, SEQS_FILE)

        if "sequences:list" in cache_manager:
            del cache_manager["sequences:list"]

        logger.success(f"Sequência '{payload.name}' criada com {len(payload.steps)} passos")

    return {
        "message": f"Sequence '{payload.name}' created successfully",
        "name": payload.name,
        "steps": len(payload.steps),
    }


@router.put("/{name}")
def update_sequence(
    name: str,
    payload: SequenceUpdate,
):
    """
    Update an existing sequence definition.
    """
    with _YAML_LOCK:
        data = file_manager.read(SEQS_FILE, return_native=True)
        if not isinstance(data, dict):
            data = {}

        if name not in data:
            logger.warning(f"Attempt to update non-existent sequence: {name}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Sequence '{name}' not found",
            )

        steps_data = []
        for step in payload.steps:
            step_dict = step.model_dump(exclude_none=True)
            steps_data.append(step_dict)

        data[name] = steps_data

        file_manager.save(data, SEQS_FILE)

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


@router.delete("/{name}")
def delete_sequence(
    name: str,
):
    """
    Delete a sequence from the configuration.
    """
    with _YAML_LOCK:
        data = file_manager.read(SEQS_FILE, return_native=True)
        if not isinstance(data, dict):
            data = {}

        if name not in data:
            logger.warning(f"Attempt to delete non-existent sequence: {name}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Sequence '{name}' not found",
            )

        referenced_by = []
        for seq_name, steps in data.items():
            if seq_name != name and isinstance(steps, list):
                for step in steps:
                    if isinstance(step, dict) and step.get("next_sequence") == name:
                        referenced_by.append(seq_name)

        if referenced_by:
            logger.error(f"Sequence '{name}' is referenced by: {referenced_by}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Cannot delete sequence '{name}' because it's referenced by: {', '.join(referenced_by)}",
            )

        del data[name]

        file_manager.save(data, SEQS_FILE)

        if "sequences:list" in cache_manager:
            del cache_manager["sequences:list"]
        if f"sequence:{name}" in cache_manager:
            del cache_manager[f"sequence:{name}"]

        logger.success(f"Sequência '{name}' deletada com sucesso")

    return {"message": f"Sequence '{name}' deleted successfully"}


@router.patch("/{name}/rename")
def rename_sequence(
    name: str,
    new_name: str = Body(..., embed=True),
):
    """
    Rename an existing sequence.
    """
    if not new_name or not new_name.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="New name cannot be empty"
        )

    with _YAML_LOCK:
        data = file_manager.read(SEQS_FILE, return_native=True)
        if not isinstance(data, dict):
            data = {}

        if name not in data:
            logger.warning(f"Attempt to rename non-existent sequence: {name}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Sequence '{name}' not found",
            )

        if new_name in data:
            logger.warning(f"New name already exists: {new_name}")
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Sequence '{new_name}' already exists",
            )

        data[new_name] = data[name]
        del data[name]

        updated_refs = 0
        for seq_name, steps in data.items():
            if isinstance(steps, list):
                for step in steps:
                    if isinstance(step, dict) and step.get("next_sequence") == name:
                        step["next_sequence"] = new_name
                        updated_refs += 1

        file_manager.save(data, SEQS_FILE)

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


@router.post("/validate")
def validate_sequence(
    payload: SequencePayload,
):
    """
    Validate a sequence definition without saving.
    """
    data = file_manager.read(SEQS_FILE, return_native=True)
    if not isinstance(data, dict):
        data = {}

    missing_sequences = []
    for step in payload.steps:
        if step.next_sequence and step.next_sequence not in data:
            missing_sequences.append(step.next_sequence)

    warnings = []
    if missing_sequences:
        warnings.append(f"Referenced sequences not found: {', '.join(set(missing_sequences))}")

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
