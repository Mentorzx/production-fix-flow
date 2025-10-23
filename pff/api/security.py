import os
from fastapi import Header, HTTPException
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("API_KEY", "fallback-insecure-key-DO-NOT-USE")

def verify_token(x_api_key: str = Header(...)):
    """
    Verifies the provided API key from the request header.

    Args:
        x_api_key (str): The API key provided in the 'X-API-KEY' header.

    Raises:
        HTTPException: If the provided API key does not match the expected API_KEY, raises a 401 Unauthorized error.
    """
    if x_api_key != API_KEY:
        raise HTTPException(401, "Invalid token")