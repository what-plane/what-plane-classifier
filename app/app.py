import os
import io
import sys
from pathlib import Path
import json

from flask import Flask, jsonify, abort
from PIL import Image
from azure.storage.blob import BlobServiceClient
import torch

sys.path.insert(0, "..")

import whatplane.models.model_helpers as mh
from whatplane.models.data_helpers import PREDICT_TRANSFORM
from whatplane.models.predict_model import predict_image_data

app = Flask(__name__)

CWD = Path(".")
IMAGE_UPLOAD_CONTAINTER = "uploaded-images"
CLASSIFIED_IMAGE_CONTAINER = "uploaded-images-airliners"
CONNECT_STR = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
blob_service_client = BlobServiceClient.from_connection_string(CONNECT_STR)


with open(CWD / "imagenet_class_index.json") as f:
    imagenet_class_index = json.load(f)

imagenet_model = mh.initialize_model(
    "densenet161",
    [item[1] for item in list(imagenet_class_index.values())],
    replace_classifier=False,
)
airliner_model = mh.load_model(CWD / "../models/model.pth")


@app.errorhandler(404)
def resource_not_found(e):
    return jsonify(error=str(e)), 404

@app.route("/predict/<string:filename>", methods=["GET"])
def image_predict_api(filename):

    if len(filename) == 0:
        abort(404, description="Resource not found")

    uploaded_blob = blob_service_client.get_blob_client(
        container=IMAGE_UPLOAD_CONTAINTER, blob=filename
    )

    if not uploaded_blob.exists():
        abort(404, description="Resource not found")

    content_type = uploaded_blob.get_blob_properties()["content_settings"]["content_type"]

    if content_type not in ["image/jpeg", "image/png"]:
        abort(404, description=f"File type {content_type} not supported")

    image = Image.open(io.BytesIO(uploaded_blob.download_blob().readall()))

    class_ids, class_names = predict_image_data(image, imagenet_model, topk=5)
    # If image is an airliner, load inference model
    if "airliner" not in class_names:
        result = jsonify({"class_name": class_names[0], "class_id": str(class_ids[0])})
    else:
        top_probs, top_classes = predict_image_data(image, airliner_model, topk=1)
        result = jsonify({"class_name": top_classes[0], "class_pred": round(top_probs[0] * 100, 1)})

        # Transfer blob to classified image container
        airliner_blob = blob_service_client.get_blob_client(
            container=CLASSIFIED_IMAGE_CONTAINER, blob="/".join([top_classes[0], filename])
        )
        airliner_blob.start_copy_from_url(uploaded_blob.url)

    return result


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
