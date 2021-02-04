import uvicorn

from fastapi import FastAPI, Path, Query, HTTPException
from pydantic import BaseModel

from .routers import predict, health


app = FastAPI(
    title="WhatPlane", description="Recognising Aircraft with Deep Learning", version="0.3.1"
)

app.include_router(predict.router)
app.include_router(health.router)
