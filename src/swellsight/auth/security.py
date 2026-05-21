"""JWT and password hashing."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

import bcrypt
from jose import jwt

from swellsight.platform.settings import get_settings

ALGORITHM = "HS256"


def _secret_key() -> str:
    return get_settings().jwt_secret


def _expire_minutes() -> int:
    return get_settings().jwt_expire_minutes


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))


def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=_expire_minutes())
    )
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, _secret_key(), algorithm=ALGORITHM)


def decode_access_token(token: str) -> Dict[str, Any]:
    return jwt.decode(token, _secret_key(), algorithms=[ALGORITHM])
