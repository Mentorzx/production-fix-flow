"""
Tests for Security Module (pff/api/security.py)

Tests cover:
- API key loading from environment
- Token verification
- Security error handling
"""

import os
import pytest
from unittest.mock import Mock
from fastapi import HTTPException
from pff.api.security import verify_token, API_KEY


@pytest.mark.unit
class TestSecurityAPIKey:
    """Test API key loading and validation."""

    def test_api_key_loaded_from_env(self, monkeypatch):
        """Test that API_KEY is loaded from environment variable."""
        # Note: API_KEY is loaded at module import time, so we test the mechanism
        # by verifying it's not the fallback value when .env is properly configured
        assert API_KEY is not None
        assert len(API_KEY) > 10  # Should be a proper token

    def test_api_key_not_fallback_in_production(self):
        """Test that API_KEY is not using the insecure fallback in production."""
        # In production, API_KEY should NEVER be the fallback
        assert API_KEY != "fallback-insecure-key-DO-NOT-USE"


@pytest.mark.unit
class TestVerifyToken:
    """Test token verification functionality."""

    def test_verify_token_success(self):
        """Test successful token verification."""
        # Mock a valid token that matches API_KEY
        valid_token = API_KEY

        # Should not raise exception
        try:
            verify_token(x_api_key=valid_token)
        except HTTPException:
            pytest.fail("verify_token() raised HTTPException with valid token")

    def test_verify_token_invalid_raises_401(self):
        """Test that invalid token raises 401 HTTPException."""
        invalid_token = "invalid-token-12345"

        with pytest.raises(HTTPException) as exc_info:
            verify_token(x_api_key=invalid_token)

        assert exc_info.value.status_code == 401
        assert "Invalid token" in str(exc_info.value.detail)

    def test_verify_token_empty_raises_401(self):
        """Test that empty token raises 401 HTTPException."""
        with pytest.raises(HTTPException) as exc_info:
            verify_token(x_api_key="")

        assert exc_info.value.status_code == 401

    def test_verify_token_none_raises_error(self):
        """Test that None token raises error (should be caught by FastAPI)."""
        # FastAPI's Header(...) makes x_api_key required, but test the function behavior
        with pytest.raises((HTTPException, TypeError)):
            verify_token(x_api_key=None)


@pytest.mark.integration
class TestSecurityIntegration:
    """Integration tests for security module."""

    def test_api_key_strong_enough(self):
        """Test that API_KEY meets security requirements."""
        # API key should be at least 32 characters for security
        assert len(API_KEY) >= 32, "API_KEY should be at least 32 characters"

        # Should contain mix of characters (not just numbers or letters)
        has_letters = any(c.isalpha() for c in API_KEY)
        has_numbers_or_special = any(c.isdigit() or not c.isalnum() for c in API_KEY)

        assert has_letters, "API_KEY should contain letters"
        assert has_numbers_or_special, "API_KEY should contain numbers or special chars"

    def test_verify_token_case_sensitive(self):
        """Test that token verification is case-sensitive."""
        if API_KEY.isupper() or API_KEY.islower():
            pytest.skip("API_KEY is all one case, cannot test case sensitivity")

        # Try with wrong case
        wrong_case_token = API_KEY.swapcase()

        with pytest.raises(HTTPException) as exc_info:
            verify_token(x_api_key=wrong_case_token)

        assert exc_info.value.status_code == 401


@pytest.mark.unit
class TestSecurityBestPractices:
    """Test security best practices are followed."""

    def test_no_hardcoded_secrets_in_code(self):
        """Test that security.py doesn't contain hardcoded production secrets."""
        from pathlib import Path

        security_file = Path(__file__).parents[1] / "pff" / "api" / "security.py"
        content = security_file.read_text()

        # Should not contain obvious hardcoded secrets
        dangerous_patterns = [
            "super-secret-token",  # Old hardcoded value
            "password123",
            "admin",
            "secret123",
        ]

        for pattern in dangerous_patterns:
            # Allow pattern in comments/strings for documentation, but not as assignment
            if f'API_KEY = "{pattern}"' in content or f"API_KEY = '{pattern}'" in content:
                pytest.fail(f"Found hardcoded secret pattern: {pattern}")

    def test_dotenv_loaded(self):
        """Test that dotenv is properly loaded in security.py."""
        from pathlib import Path

        security_file = Path(__file__).parents[1] / "pff" / "api" / "security.py"
        content = security_file.read_text()

        # Should import and load dotenv
        assert "from dotenv import load_dotenv" in content or "import dotenv" in content
        assert "load_dotenv()" in content

    def test_api_key_uses_env_variable(self):
        """Test that API_KEY is loaded from environment variable."""
        from pathlib import Path

        security_file = Path(__file__).parents[1] / "pff" / "api" / "security.py"
        content = security_file.read_text()

        # Should use os.getenv() or similar
        assert 'os.getenv("API_KEY"' in content or 'os.environ.get("API_KEY"' in content
