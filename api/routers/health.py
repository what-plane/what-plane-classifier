from fastapi import APIRouter, Path, Query
from pydantic import BaseModel

from ..dependancies import model

router = APIRouter()


class HealthSet(BaseModel):
    model_version: str
    ping: str


@router.get(
    "/health", response_model=HealthSet, status_code=200, tags=["health"],
)
async def health_check() -> HealthSet:
    """API to check status and model version

    Returns:
        HealthSet: Model version and a ping
    """
    model_version = getattr(model.whatplane_model, "version", "Not Set")
    return HealthSet(model_version=model_version, ping="Pong!")
