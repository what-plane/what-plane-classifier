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

transform = transforms.Compose([transforms.Resize(224),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406],
                                                     [0.229, 0.224, 0.225])])


def predict(img_path, model):

    img = Image.open(img_path)

    inputs = transform(img).unsqueeze(dim=0)

    model.eval()

    with torch.no_grad():
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
    # model.train()

    return preds.cpu().numpy()[0]
