import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routers import predict, health

ORIGINS = os.getenv("CORS_ORIGINS").split(",")
ORIGINS_REGEX = os.getenv("CORS_ORIGINS_REGEX")

app = FastAPI(
    title="WhatPlane", description="Recognising Aircraft with Deep Learning", version="0.3.1"
)

app.include_router(predict.router)
app.include_router(health.router)


app.add_middleware(
    CORSMiddleware,
    allow_origins=ORIGINS,
    allow_origin_regex=ORIGINS_REGEX,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
