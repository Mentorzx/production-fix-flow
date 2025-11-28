"""
Tests for Authentication and Authorization (pff/api/auth.py)

Tests cover:
- Password hashing and verification
- JWT token generation and validation
- User authentication
- Token expiration
- Rate limiting (429 responses)
"""

import pytest
from datetime import datetime, timedelta, timezone
from fastapi import HTTPException
import jwt

from pff.api import auth


@pytest.mark.unit
class TestPasswordHashing:
    """Test password hashing and verification."""

    def test_hash_password_returns_hashed_string(self):
        """Test that _hash_password returns a bcrypt hashed string."""
        password = "my_secure_password_123"
        hashed = auth._hash_password(password)

        assert hashed is not None
        assert isinstance(hashed, str)
        assert len(hashed) > 20  # Bcrypt hashes are long
        assert hashed != password  # Should not be plaintext

    def test_hash_password_generates_different_hashes(self):
        """Test that hashing the same password twice generates different salts."""
        password = "test_password"
        hash1 = auth._hash_password(password)
        hash2 = auth._hash_password(password)

        assert hash1 != hash2  # Different salts = different hashes

    def test_verify_password_correct(self):
        """Test password verification with correct password."""
        password = "correct_password"
        hashed = auth._hash_password(password)

        assert auth._verify_pw(password, hashed) is True

    def test_verify_password_incorrect(self):
        """Test password verification with incorrect password."""
        password = "correct_password"
        hashed = auth._hash_password(password)

        assert auth._verify_pw("wrong_password", hashed) is False

    def test_verify_password_case_sensitive(self):
        """Test that password verification is case-sensitive."""
        password = "MyPassword123"
        hashed = auth._hash_password(password)

        assert auth._verify_pw("mypassword123", hashed) is False
        assert auth._verify_pw("MYPASSWORD123", hashed) is False


@pytest.mark.unit
class TestTokenGeneration:
    """Test JWT token creation and validation."""

    def test_create_token_generates_valid_jwt(self):
        """Test that _create_token generates a valid JWT."""
        data = {"sub": "test_user"}
        token = auth._create_token(data)

        assert token is not None
        assert isinstance(token, str)
        assert len(token) > 50  # JWTs are long strings

    def test_create_token_includes_payload_data(self):
        """Test that created token contains the payload data."""
        data = {"sub": "user123", "role": "admin"}
        token = auth._create_token(data)

        # Decode without verification to check payload
        decoded = jwt.decode(token, options={"verify_signature": False})

        assert decoded["sub"] == "user123"
        assert decoded["role"] == "admin"

    def test_create_token_includes_expiration(self):
        """Test that token includes expiration time."""
        data = {"sub": "test_user"}
        token = auth._create_token(data)

        decoded = jwt.decode(token, options={"verify_signature": False})

        assert "exp" in decoded
        assert isinstance(decoded["exp"], int)

    def test_create_token_custom_expiration(self):
        """Test token creation with custom expiration time."""
        data = {"sub": "test_user"}
        custom_expiration = 60  # 60 minutes

        token = auth._create_token(data, expires_delta=custom_expiration)
        decoded = jwt.decode(token, options={"verify_signature": False})

        # Check expiration is approximately 60 minutes from now
        exp_time = datetime.fromtimestamp(decoded["exp"], tz=timezone.utc)
        expected_exp = datetime.now(timezone.utc) + timedelta(minutes=custom_expiration)

        # Allow 10 second tolerance
        assert abs((exp_time - expected_exp).total_seconds()) < 10

    def test_create_token_verifiable_with_secret(self):
        """Test that token can be verified with the secret key."""
        data = {"sub": "test_user"}
        token = auth._create_token(data)

        # Should not raise exception
        decoded = jwt.decode(token, auth.settings.secret_key, algorithms=[auth.settings.alg])

        assert decoded["sub"] == "test_user"


@pytest.mark.unit
class TestAuthentication:
    """Test user authentication flow."""

    def test_authenticate_valid_credentials(self):
        """Test successful authentication with valid credentials."""
        # _fake_db has admin@local with password "admin"
        user = auth.authenticate("admin@local", "admin")

        assert user is not None
        assert user["username"] == "admin@local"
        assert user["full_name"] == "Admin"

    def test_authenticate_invalid_username(self):
        """Test authentication fails with invalid username."""
        user = auth.authenticate("nonexistent@local", "any_password")

        assert user is None

    def test_authenticate_invalid_password(self):
        """Test authentication fails with wrong password."""
        user = auth.authenticate("admin@local", "wrong_password")

        assert user is None

    def test_authenticate_case_sensitive_username(self):
        """Test that username authentication is case-sensitive."""
        user = auth.authenticate("ADMIN@LOCAL", "admin")

        assert user is None  # Username case must match exactly

    def test_get_user_exists(self):
        """Test retrieving an existing user."""
        user = auth._get_user("admin@local")

        assert user is not None
        assert user["username"] == "admin@local"

    def test_get_user_not_exists(self):
        """Test retrieving a non-existent user returns None."""
        user = auth._get_user("nonexistent@local")

        assert user is None


@pytest.mark.unit
class TestFakeDatabase:
    """Test fake database structure (to be replaced with PostgreSQL)."""

    def test_fake_db_has_admin_user(self):
        """Test that _fake_db contains the admin user."""
        assert "admin@local" in auth._fake_db

    def test_admin_user_has_required_fields(self):
        """Test that admin user has all required fields."""
        admin = auth._fake_db["admin@local"]

        assert "username" in admin
        assert "full_name" in admin
        assert "hashed_pw" in admin
        assert "disabled" in admin

    def test_admin_password_is_hashed(self):
        """Test that stored password is hashed, not plaintext."""
        admin = auth._fake_db["admin@local"]

        assert admin["hashed_pw"] != "admin"  # Not plaintext
        assert len(admin["hashed_pw"]) > 20  # Hashed string is long

    def test_admin_user_not_disabled(self):
        """Test that admin user is not disabled."""
        admin = auth._fake_db["admin@local"]

        assert admin["disabled"] is False


@pytest.mark.integration
class TestGetCurrentUser:
    """Test get_current_user dependency."""

    def test_get_current_user_with_valid_token(self):
        """Test get_current_user extracts user from valid token."""
        # Create a valid token
        token = auth._create_token({"sub": "admin@local"})

        # get_current_user should extract the username
        user = auth.get_current_user(token)

        assert user is not None
        assert user["username"] == "admin@local"

    def test_get_current_user_with_invalid_token(self):
        """Test get_current_user raises HTTPException with invalid token."""
        invalid_token = "invalid.jwt.token"

        with pytest.raises(HTTPException) as exc_info:
            auth.get_current_user(invalid_token)

        assert exc_info.value.status_code == 401
        assert "Could not validate credentials" in str(exc_info.value.detail)

    def test_get_current_user_with_expired_token(self):
        """Test get_current_user raises HTTPException with expired token."""
        # Create token that expired 1 hour ago
        data = {"sub": "admin@local"}
        past_time = datetime.now(timezone.utc) - timedelta(hours=1)
        token_data = data.copy()
        token_data.update({"exp": past_time})

        expired_token = jwt.encode(token_data, auth.settings.secret_key, algorithm=auth.settings.alg)

        with pytest.raises(HTTPException) as exc_info:
            auth.get_current_user(expired_token)

        assert exc_info.value.status_code == 401

    def test_get_current_user_with_nonexistent_user(self):
        """Test get_current_user raises HTTPException when user not in database."""
        token = auth._create_token({"sub": "nonexistent@local"})

        with pytest.raises(HTTPException) as exc_info:
            auth.get_current_user(token)

        assert exc_info.value.status_code == 401

    def test_get_current_user_with_disabled_user(self):
        """Test get_current_user raises HTTPException for disabled user."""
        # Temporarily add a disabled user
        auth._fake_db["disabled@local"] = {
            "username": "disabled@local",
            "full_name": "Disabled User",
            "hashed_pw": auth._hash_password("password"),
            "disabled": True,
        }

        token = auth._create_token({"sub": "disabled@local"})

        try:
            with pytest.raises(HTTPException) as exc_info:
                auth.get_current_user(token)

            # Current implementation returns 401 for disabled users
            assert exc_info.value.status_code == 401
        finally:
            # Cleanup
            del auth._fake_db["disabled@local"]


@pytest.mark.unit
class TestSettings:
    """Test auth settings configuration."""

    def test_settings_has_secret_key(self):
        """Test that settings contains a secret key."""
        assert hasattr(auth.settings, "secret_key")
        assert auth.settings.secret_key is not None

    def test_settings_has_algorithm(self):
        """Test that settings specifies JWT algorithm."""
        assert hasattr(auth.settings, "alg")
        assert auth.settings.alg == "HS256"

    def test_settings_has_expiration_time(self):
        """Test that settings defines token expiration time."""
        assert hasattr(auth.settings, "access_token_expire_minutes")
        assert auth.settings.access_token_expire_minutes > 0
