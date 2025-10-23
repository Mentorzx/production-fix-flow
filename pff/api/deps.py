from functools import lru_cache
from pathlib import Path
from typing import Generator

from fastapi import Depends, HTTPException, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from pff.services.sequence_service import SequenceService
from pff.services.line_service import LineService
from pff.services.business_service import BusinessService
from pff.utils.file_manager import FileManager
from pff.api.security import API_KEY

SEQS_FILE = Path(__file__).resolve().parents[2] / "config" / "sequences.yaml"


def get_line_service() -> Generator[LineService, None, None]:
    """
    Yields an instance of LineService within a context manager.

    This dependency function is intended for use with FastAPI's dependency injection system.
    It ensures that the LineService instance is properly managed and cleaned up after use.

    Yields:
        LineService: An instance of the LineService class.
    """
    with LineService() as svc:
        yield svc


def get_validator_service() -> Generator[BusinessService, None, None]:
    """
    Yields an instance of ValidatorService using a context manager.

    This generator function is intended for dependency injection in frameworks like FastAPI.
    It ensures that the ValidatorService is properly initialized and cleaned up after use.

    Yields:
        ValidatorService: An instance of the ValidatorService class.
    """
    with BusinessService() as validator:
        validator._ensure_models_loaded()
        yield validator


def get_engine(
    # trunk-ignore(ruff/B008)
    service: LineService = Depends(get_line_service),  # 👈 OK here
    # trunk-ignore(ruff/B008)
    validator: BusinessService = Depends(get_validator_service),  # 👈 OK here
) -> SequenceService:
    """
    Creates and returns a SequenceEngine instance using the provided LineService dependency.

    Args:
        service (LineService): The line service dependency, injected via FastAPI's Depends.

    Returns:
        SequenceEngine: An instance of SequenceEngine initialized with the given LineService.
    """
    return SequenceService(service, validator)


async def verify_api_key(x_api_key: str = Header(None)):
    """
    Verifies the API key from the X-API-Key header.

    Args:
        x_api_key: The API key from the request header

    Raises:
        HTTPException: If API key is missing or invalid
    """
    if x_api_key is None:
        raise HTTPException(
            status_code=403,
            detail="API key is required. Please provide X-API-Key header."
        )

    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=403,
            detail="Invalid API key"
        )

    return x_api_key


@lru_cache(maxsize=1)
def get_sequences_yaml_cached():
    """
    Loads and returns the sequences data from a YAML file, utilizing caching if available.

    Returns:
        dict: The parsed contents of the sequences YAML file.
    """
    return FileManager.load_yaml(SEQS_FILE)
