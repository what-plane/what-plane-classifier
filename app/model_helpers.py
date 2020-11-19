import os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from collections import OrderedDict

import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.optim import lr_scheduler
from torchvision import datasets
from torch.utils.tensorboard import SummaryWriter

use_cuda = torch.cuda.is_available()


def init_model():
    model_transfer = models.densenet161(pretrained=True)

    num_classes = 24
    num_features = model_transfer.classifier.in_features
    input_size = model_transfer.classifier.state_dict()[next(
        iter(model_transfer.classifier.state_dict()))].size(1)
    hidden_units = int(input_size/2)

    model_transfer.classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(input_size, hidden_units)),
        ('relu', nn.ReLU()),
        ('drop', nn.Dropout(0.4)),
        ('fc2', nn.Linear(hidden_units, num_classes))
    ]))

    if use_cuda:
        model_transfer = model_transfer.cuda()
    return model_transfer
