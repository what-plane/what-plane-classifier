from flask import Flask, request, render_template, jsonify, redirect
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


UPLOAD_FOLDER = './static/uploads'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

with open("./data/imagenet_class_index.json") as f:
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


@app.route('/', methods=['GET', 'POST'])
def image_predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        img = Image.open(os.path.join(
            app.config['UPLOAD_FOLDER'], filename))
        buf = BytesIO()
        img.save(buf, 'jpeg')
        buf.seek(0)
        buf.close()
        # Load ImageNet model to check image type
        class_ids, class_names = predict(file, imagenet_model, topk=5)

        print("imagenet:", class_ids, class_names)
        # If image is an airliner, load inference model
        if 'airliner' not in class_names:
            return render_template('index.html', class_name=class_names[0], class_id=str(class_ids[0]), filename=filename)
        else:
            model = load_inference_model(
                '../models/model_airliners_net_balanced_densenet161_SGD.pth')
            top_probs, top_classes = predict(file, model, topk=1)
            return render_template('index.html', class_name=top_classes[0], class_id=round(top_probs[0]*100, 1), filename=filename)

    return render_template('index.html')


if __name__ == '__main__':
    app.run()
