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
with open("./models/new_mapping.json", "r") as f:
    cat_to_name = json.load(f)

model_transfer = init_model()


criterion_transfer = nn.CrossEntropyLoss()
optimizer_transfer = optim.SGD(
    model_transfer.parameters(), lr=0.001, momentum=0.9)

transform = transforms.Compose([
    transforms.Resize(240),
    transforms.RandomRotation(10),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])])

model = torch.load('./models/densenet_full_-30-model.pth',
                   map_location=torch.device('cpu'))
# model_transfer.load_state_dict(torch.load(
#     './models/model_densenet_sgd_30_airlinersnet.pt', map_location=torch.device('cpu')))


def predict_aircraft(img_path):
    idx = predict_func(img_path, model_transfer)
    return idx


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        class_id = predict_aircraft(file)
        return jsonify({'class_name': cat_to_name[f'{class_id}'], 'class_id': (str(class_id))})


if __name__ == '__main__':
    app.run()
