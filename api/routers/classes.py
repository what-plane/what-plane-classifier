from typing import List
from fastapi import APIRouter, Path, Query
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from ..dependancies import model

router = APIRouter()


class WPClass(BaseModel):
    class_name: List[str] = []


@router.get(
    "/classes/whatplane", response_model=WPClass, status_code=200, tags=["classes"],
)
async def get_classes():
    json_compatible_item_data = jsonable_encoder(model.get_wp_classes())
    return JSONResponse(content=json_compatible_item_data)
