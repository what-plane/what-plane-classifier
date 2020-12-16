from flask import Flask, request, jsonify
from PIL import Image
from werkzeug.utils import secure_filename
import torchvision.transforms as transforms
from torchvision import models
import torch
import torch.optim as optim
from pathlib import Path
import sys
import os
import json
import io
from io import BytesIO
sys.path.insert(0, '..')  # noqa
import src.models.model_helpers as mh  # noqa
from src.models.data_helpers import load_data, PREDICT_TRANSFORM  # noqa
from src.models.train_model import train_model  # noqa
from src.models.predict_model import test, predict  # noqa
import src.models.visualise_helpers as vh  # noqa


app = Flask(__name__)

with open("./imagenet_class_index.json") as f:
    imagenet_class_index = json.load(f)

imagenet_model = mh.initialize_model(
    "densenet161", [item[1] for item in list(imagenet_class_index.values())], replace_classifier=False)
imagenet_model.eval()


def load_inference_model(model_path):
    model = mh.load_model(model_path)
    return model


def transform_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    return PREDICT_TRANSFORM(image).unsqueeze(0)


@app.route('/predict', methods=['GET', 'POST'])
def image_predict():
    if request.method == 'POST':
        file = request.files['image']
        class_ids, class_names = predict(file, imagenet_model, topk=5)
        # If image is an airliner, load inference model
        if 'airliner' not in class_names:
            return jsonify({'class_name': class_names[0], 'class_id': str(class_ids[0])})
        else:
            model = load_inference_model(
                '../models/model_ash_densenet161_SGD.pth')
            top_probs, top_classes = predict(file, model, topk=1)
            return jsonify({'class_name': top_classes[0], 'class_pred': round(top_probs[0]*100, 1)})


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=80)
