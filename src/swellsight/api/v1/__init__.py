from fastapi import APIRouter

from .analyses import router as analyses_router
from .auth import router as auth_router
from .health import router as health_router

api_v1_router = APIRouter(prefix="/v1")
api_v1_router.include_router(health_router)
api_v1_router.include_router(auth_router)
api_v1_router.include_router(analyses_router)
