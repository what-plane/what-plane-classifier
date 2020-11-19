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


def load_data(data_dir, batch_size):
    # Data loading and normalization for training
    norm = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(240),
            transforms.RandomRotation(10),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(*norm)
        ]),
        'valid': transforms.Compose([
            transforms.Resize(255),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(*norm)
        ]),
        'test': transforms.Compose([
            transforms.Resize(255),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(*norm)
        ]),
    }

    # Load the datasets with ImageFolder
    image_datasets = {x: datasets.ImageFolder(os.path.join(
        data_dir, x), data_transforms[x]) for x in ['train', 'valid', 'test']}

    # Dataloaders
    dataloaders = {x: torch.utils.data.DataLoader(
        image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'valid', 'test']}
    dataset_sizes = {x: len(image_datasets[x])
                     for x in ['train', 'valid', 'test']}
    return image_datasets, dataloaders, dataset_sizes
