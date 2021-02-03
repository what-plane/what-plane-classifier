import uvicorn

from fastapi import FastAPI, Path, Query, HTTPException
from pydantic import BaseModel

print('__file__={0:<35} | __name__={1:<20} | __package__={2:<20}'.format(__file__,__name__,str(__package__)))

from .routers import predict, health


app = FastAPI(
    title="WhatPlane", description="Recognising Aircraft with Deep Learning", version="0.3.1"
)

app.include_router(predict.router)
app.include_router(health.router)