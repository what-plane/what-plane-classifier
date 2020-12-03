from flask import Flask, request, render_template, jsonify
from PIL import Image
import torchvision.transforms as transforms
from torchvision import models
import torch
from pathlib import Path
import sys
import os
import json
import io
sys.path.insert(0, '..')  # noqa
import src.models.model_helpers as mh  # noqa
from src.models.data_helpers import load_data, imshow, plot_image  # noqa
from src.models.train_model import train_model  # noqa
from src.models.predict_model import test, predict, caption_image  # noqa

app = Flask(__name__)

with open("./data/imagenet_class_index.json") as f:
    imagenet_class_index = json.load(f)

model = models.densenet121(pretrained=True)
model.eval()

# def load_model(model_path):
#     model


def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)


def imagenet_pred(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    outputs = model.forward(tensor)
    _, y_hat = outputs.max(1)
    predicted_idx = str(y_hat.item())
    return imagenet_class_index[predicted_idx]


@app.route('/predict', methods=['POST'])
def imagenet_predict():
    if request.method == 'POST':
        file = request.files['file']
        img_bytes = file.read()
        class_id, class_name = imagenet_pred(image_bytes=img_bytes)
        return jsonify({'class_id': class_id, 'class_name': class_name})


if __name__ == '__main__':
    app.run()
