from typing import List
from fastapi import APIRouter, Path, Query
from pydantic import BaseModel

from ..dependancies import model

router = APIRouter()


class WPClass(BaseModel):
    class_name: List[str] = model.get_wp_classes()


@router.get(
    "/classes/whatplane", response_model=WPClass, status_code=200, tags=["classes"],
)
async def get_classes() -> WPClass:
    return WPClass()
