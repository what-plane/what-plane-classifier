from pathlib import Path
import sys
import json

from flask import Flask, request, jsonify
from PIL import Image
import torch

sys.path.insert(0, '..')

import src.models.model_helpers as mh
from src.models.data_helpers import PREDICT_TRANSFORM
from src.models.predict_model import predict

app = Flask(__name__)

with open("./imagenet_class_index.json") as f:
    imagenet_class_index = json.load(f)

imagenet_model = mh.initialize_model(
    "densenet161", [item[1] for item in list(imagenet_class_index.values())], replace_classifier=False)
imagenet_model.eval()

airliner_model = mh.load_model('../models/model.pth')

@app.route('/predict', methods=['GET'])
def image_predict():
    if request.method == 'GET':
        file = request.files['image']
        class_ids, class_names = predict(file, imagenet_model, topk=5)
        # If image is an airliner, load inference model
        if 'airliner' not in class_names:
            return jsonify({'class_name': class_names[0], 'class_id': str(class_ids[0])})
        else:
            top_probs, top_classes = predict(file, airliner_model, topk=1)
            return jsonify({'class_name': top_classes[0], 'class_pred': round(top_probs[0]*100, 1)})


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
