from functools import lru_cache

import httpx
from fastapi import HTTPException, Security, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from api.config import API_WEB_APP_IP, API_WEB_APP_PORT

security = HTTPBearer(auto_error=False)

@lru_cache
def _verify_url() -> str:
    """Build the auth-service URL once and cache it."""
    return f"http://{API_WEB_APP_IP}:{API_WEB_APP_PORT}/api/token/verify/"


# ── Dependency ────────────────────────────────────────────────────────────────
async def verify_token(
    credentials: HTTPAuthorizationCredentials | None = Security(security),
) -> bool:
    """
    Dependency to be added via `Depends(verify_token)`.

    On success it simply returns `True`. On failure it raises the appropriate
    HTTPException, so the request never reaches your endpoint logic.
    """
    # 1. No / malformed `Authorization` header
    if credentials is None or not credentials.scheme.lower() == "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid Authorization header",
        )

    # 2. Ask the auth service to verify the token
    async with httpx.AsyncClient(timeout=5.0) as client:
        try:
            resp = await client.post(_verify_url(), json={"token": credentials.credentials})
        except httpx.RequestError:
            # Upstream auth service is down or unreachable
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Authorization service unavailable, try again later",
            )

    # 3. Token rejected by auth service
    if resp.status_code != status.HTTP_200_OK:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token is invalid or expired",
        )

    # 4. All good – you could also return resp.json() if you need user info
    return True