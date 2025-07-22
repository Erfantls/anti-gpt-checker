import os
import hashlib
import hmac
import json
import base64

from fastapi import HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from api.server_config import API_SHARED_SECRET_KEY

security = HTTPBearer(auto_error=False)

def _decode_token(raw_token: str) -> dict:
    """
    Bearer value is expected to be a URL‑safe base64 string
    that decodes to a JSON object holding
    {
        "User":  "<user_id>",
        "Nonce": "<cryptographic_nonce>",
        "Auth":  "<hex_digest>"
    }
    """
    try:
        padded = raw_token + "=" * (-len(raw_token) % 4)      # base64 padding
        data = base64.urlsafe_b64decode(padded).decode()
        return json.loads(data)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Malformed bearer token",
        )

def _expected_signature(user_id: str, nonce: str) -> str:
    payload = f"{user_id}:{nonce}".encode()
    key = API_SHARED_SECRET_KEY.encode()
    sig = hmac.new(key, payload, hashlib.sha256).hexdigest()
    return sig

async def verify_token(
    credentials: HTTPAuthorizationCredentials | None = Security(security),
) -> str:
    """
    Dependency to be added via Depends(verify_token).

    On success it returns the user_id that was authenticated.
    On failure it raises an HTTPException so the request never
    reaches your endpoint logic.
    """
    # 1. Missing or non‑bearer Authorization header
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid Authorization header",
        )

    # 2. Decode, validate structure, compute expected HMAC
    token_fields = _decode_token(credentials.credentials)

    try:
        user_id = token_fields["User"]
        nonce = token_fields["Nonce"]
        received_sig = token_fields["Auth"]
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token is missing required fields",
        )

    expected_sig = _expected_signature(user_id, nonce)

    # 3. Constant‑time comparison to avoid timing attacks
    if not hmac.compare_digest(received_sig, expected_sig):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token is invalid or expired",
        )

    # 4. All good
    return str(user_id)

def generate_salt(length: int = 16) -> str:
    return base64.b64encode(os.urandom(length)).decode('utf-8')

def hash_password_with_salt(password: str, salt: str) -> str:
    salted = (password + salt).encode('utf-8')
    hash_digest = hashlib.sha256(salted).digest()
    return base64.b64encode(hash_digest).decode('utf-8')