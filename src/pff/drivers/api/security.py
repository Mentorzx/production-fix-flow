"""Provide module-level functionality for the PFF codebase.



Notes:

    File: src/pff/drivers/api/security.py

"""

import os

from dotenv import load_dotenv
from fastapi import Header, HTTPException

load_dotenv()

API_KEY = os.getenv("API_KEY")


def verify_token(x_api_key: str = Header(...)):
    """
    Verifies the provided API key from the request header.

    Args:
        x_api_key (str): The API key provided in the 'X-API-KEY' header.

    Raises:
        HTTPException: If the provided API key does not match the expected API_KEY, raises a 401 Unauthorized error.
    """
    if not API_KEY:
        raise HTTPException(500, "API_KEY is not configured")
    if x_api_key != API_KEY:
        raise HTTPException(401, "Invalid token")
