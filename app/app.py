import os
import io
import sys
from pathlib import Path as PyPath
import json
from typing import List

import uvicorn
from fastapi import FastAPI, Path, Query, HTTPException
from pydantic import BaseModel
from PIL import Image
from azure.storage.blob import BlobServiceClient
import torch

sys.path.insert(0, "..")

import whatplane.models.model_helpers as mh
from whatplane.models.data_helpers import PREDICT_TRANSFORM
from whatplane.models.predict_model import predict_image_data

CWD = PyPath(".")
IMAGE_UPLOAD_CONTAINTER = "uploaded-images"
CLASSIFIED_IMAGE_CONTAINER = "uploaded-images-airliners"
CONNECT_STR = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
blob_service_client = BlobServiceClient.from_connection_string(CONNECT_STR)

TEMPLATE_RESPONSE = {"data": {"predictions": [], "topk": 0, "predictor": ""}}

TEMPLATE_PRED = {"class_name": "", "class_pred": 0.00}

with open(CWD / "imagenet_class_index.json") as f:
    imagenet_class_index = json.load(f)

imagenet_model = mh.initialize_model(
    "densenet161",
    [item[1] for item in list(imagenet_class_index.values())],
    replace_classifier=False,
)
whatplane_model = mh.load_model(CWD / "../models/model.pth")

app = FastAPI(
    title="WhatPlane", description="Recognising Aircraft with Deep Learning", version="0.3.0"
)


class Prediction(BaseModel):
    class_name: str = "Boeing 757"
    class_prob: float = 0.92


class PredictionSet(BaseModel):
    predictions: List[Prediction]
    topk: int = 1
    predictor: str = "whatplane"


def prepare_response(probs, class_names, predictor):
    predictions = [
        Prediction(class_name=class_name, class_prob=round(probs[i], 3))
        for i, class_name in enumerate(class_names)
    ]
    return PredictionSet(predictions=predictions, topk=len(predictions), predictor=predictor)


@app.get(
    "/predict/{filename}",
    response_model=PredictionSet,
    status_code=200,
    tags=["predict"],
    responses={404: {"description": "File not found or is the wrong type"}},
)
async def image_prediction_api(
    filename: str = Path(
        ..., title="The filename of the image uploaded by the frontend application", min_length=36
    ),
    topk: int = Query(1, title="The number of classes returned ordered by probability", ge=1, le=5),
):

    uploaded_blob = blob_service_client.get_blob_client(
        container=IMAGE_UPLOAD_CONTAINTER, blob=filename
    )

    if not uploaded_blob.exists():
        raise HTTPException(status_code=404, detail="Resource not found")

    content_type = uploaded_blob.get_blob_properties()["content_settings"]["content_type"]

    if content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=404, detail=f"File type {content_type} not supported")

    image = Image.open(io.BytesIO(uploaded_blob.download_blob().readall()))

    imagenet_probs, imagenet_classes = predict_image_data(image, imagenet_model, topk=5)

    # Take only classes with prob > 0.5
    imagenet_likely_class = [
        imagenet_classes[i] for i, prob in enumerate(imagenet_probs) if prob > 0.5
    ]

    # If image is an airliner, load inference model
    if "airliner" not in imagenet_likely_class:
        response = prepare_response(imagenet_probs[:topk], imagenet_classes[:topk], "imagenet")
    else:
        whatplane_probs, whatplane_classes = predict_image_data(image, whatplane_model, topk=topk)
        response = prepare_response(whatplane_probs, whatplane_classes, "whatplane")

        # Transfer blob to classified image container
        airliner_blob = blob_service_client.get_blob_client(
            container=CLASSIFIED_IMAGE_CONTAINER, blob="/".join([whatplane_classes[0], filename])
        )
        airliner_blob.start_copy_from_url(uploaded_blob.url)

    return response


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
