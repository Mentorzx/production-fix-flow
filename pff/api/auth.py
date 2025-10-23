from datetime import datetime, timedelta, timezone
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
import jwt
from jwt import InvalidTokenError as JWTError
import bcrypt
from pydantic_settings import BaseSettings


# === configuration ===========================================================
class Settings(BaseSettings):
    secret_key: str = "insecure"
    alg: str = "HS256"
    access_token_expire_minutes: int = 30


settings = Settings()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")

# === helpers =================================================================


def _verify_pw(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))


def _hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def _get_user(username: str):
    return _fake_db.get(username)


def _create_token(data: dict, expires_delta: int | None = None):
    """
    Generates a JSON Web Token (JWT) with the provided data and expiration time.

    Args:
        data (dict): The payload data to include in the token.
        expires_delta (int | None, optional): The expiration time in minutes for the token.
            If None, uses the default expiration time from settings.

    Returns:
        str: The encoded JWT as a string.
    """
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (
        timedelta(minutes=expires_delta)
        if expires_delta
        else timedelta(minutes=settings.access_token_expire_minutes)
    )
    to_encode.update({"exp": expire})
    token = jwt.encode(to_encode, settings.secret_key, algorithm=settings.alg)
    return token


def authenticate(username: str, password: str):
    """
    Authenticate a user by verifying the provided username and password.

    Args:
        username (str): The username of the user attempting to authenticate.
        password (str): The plaintext password provided by the user.

    Returns:
        dict or None: The user object (as a dictionary) if authentication is successful;
        otherwise, None.
    """
    user = _get_user(username)
    if not user or not _verify_pw(password, user["hashed_pw"]):
        return None
    return user


# === fake bank users (need develop) ==========================================


_fake_db = {
    "admin@local": {
        "username": "admin@local",
        "full_name": "Admin",
        "hashed_pw": _hash_password("admin"),
        "disabled": False,
    },
    "admin": {
        "username": "admin",
        "full_name": "Admin",
        "hashed_pw": _hash_password("admin"),
        "disabled": False,
    }
}


# === dependency for protected routes =========================================


def get_current_user(token: Annotated[str, Depends(oauth2_scheme)]):
    """
    Retrieves the current authenticated user based on the provided JWT token.

    Args:
        token (str): The JWT token extracted from the request's Authorization header.

    Raises:
        HTTPException: If the token is invalid, expired, or the user cannot be validated.

    Returns:
        dict: The user object if authentication is successful.

    Notes:
        - The function expects the JWT token to contain a "sub" claim with the username.
        - Raises a 401 Unauthorized error if authentication fails or the user is disabled.
    """
    cred_exc = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=[settings.alg])
        username: str | None = payload.get("sub")
        if username is None:
            raise cred_exc
    except JWTError as err:
        raise cred_exc from err
    user = _get_user(username)
    if user is None or user.get("disabled"):
        raise cred_exc
    return user


# === routes ==================================================================

router = APIRouter(tags=["auth"])


@router.post("/token")
def login(form: Annotated[OAuth2PasswordRequestForm, Depends()]):
    """
    Authenticate a user and generate an access token.

    Args:
        form (OAuth2PasswordRequestForm): The form containing the username and password.

    Returns:
        dict: A dictionary containing the access token and its type.

    Raises:
        HTTPException: If authentication fails due to invalid credentials.
    """
    user = authenticate(form.username, form.password)
    if not user:
        raise HTTPException(401, "invalid credentials")
    token = _create_token({"sub": user["username"]})
    return {"access_token": token, "token_type": "bearer"}


@router.post("/login")
def login_json(credentials: dict):
    """
    Authenticate a user with JSON body and generate an access token.

    This endpoint accepts JSON with username/password fields (alternative to OAuth2 form).

    Args:
        credentials (dict): Dictionary with 'username' and 'password' keys.

    Returns:
        dict: A dictionary containing the access token and its type.

    Raises:
        HTTPException: If authentication fails due to invalid credentials.
    """
    username = credentials.get("username")
    password = credentials.get("password")

    if not username or not password:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Username and password are required"
        )

    user = authenticate(username, password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password"
        )
    token = _create_token({"sub": user["username"]})
    return {"access_token": token, "token_type": "bearer"}


@router.post("/register", status_code=201)
def signup(username: str, password: str, full_name: str | None = None):
    """
    Registers a new user with the provided username, password, and optional full name.

    Args:
        username (str): The unique username for the new user.
        password (str): The password for the new user, which will be hashed before storage.
        full_name (str | None, optional): The full name of the user. Defaults to None, in which case the username is used as the full name.

    Raises:
        HTTPException: If the username already exists in the database.

    Returns:
        dict: A dictionary with a message indicating successful creation.
    """
    if username in _fake_db:
        raise HTTPException(400, "User already exists")
    _fake_db[username] = {
        "username": username,
        "full_name": full_name or username,
        "hashed_pw": _hash_password(password),
        "disabled": False,
    }
    return {"msg": "created"}
