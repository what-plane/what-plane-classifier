import io
import json

import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import models
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, jsonify, request
from data_loaders import load_data
from model_helpers import init_model
from train_model import train, test
from predict import predict as predict_func


use_cuda = torch.cuda.is_available()

app = Flask(__name__)
with open("./models/airlinersnet_mapping.json", "r") as f:
    cat_to_name = json.load(f)

with open("./models/torch_mapping.json", "r") as f:
    torch_mapping = json.load(f)

class_names = [cat_to_name[x] for x in torch_mapping.keys()]


def load_chkpoint():
    model_transfer = torch.load('./models/densenet_full_-30-model.pth',
                                map_location=torch.device('cpu'))

    criterion_transfer = nn.CrossEntropyLoss()
    optimizer_transfer = optim.SGD(
        model_transfer.parameters(), lr=0.001, momentum=0.9)
    model_transfer.load_state_dict(torch.load(
        './models/model_densenet_sgd_30_airlinersnet.pt', map_location=torch.device('cpu')))
    return model_transfer


def predict_aircraft(img_path):
    idx = predict_func(img_path, load_chkpoint())
    return class_names[idx]


@ app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        class_name = predict_aircraft(file)
        return jsonify({'class_name': class_name})


if __name__ == '__main__':
    app.run()
