from pathlib import Path

import numpy as np
import math
import matplotlib.pyplot as plt
from PIL import Image

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

IMAGE_SIZE = 224
BANNERHEIGHT = 12
ROTATION_ANGLE = 10
NORM = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

l = IMAGE_SIZE / 2
rad = math.radians(ROTATION_ANGLE)
c = math.cos(rad)
s = math.sin(rad)

x = l * c - l * s
y = l * s + l * c
rotpad = math.ceil(max(x, y) - l)

TRAIN_TRANSFORM = transforms.Compose(
    [
        transforms.Pad((0, 0, 0, -BANNERHEIGHT)),  # Crop banner from bottom edge of image
        transforms.Resize(IMAGE_SIZE),
        transforms.RandomResizedCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.Pad(
            rotpad, padding_mode="reflect"
        ),  # Mirror boundary to avoid empty corners of rotated image
        transforms.RandomRotation(ROTATION_ANGLE),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(*NORM),
    ]
)

PREDICT_TRANSFORM = transforms.Compose(
    [
        transforms.Pad((0, 0, 0, -BANNERHEIGHT)),
        transforms.Resize(IMAGE_SIZE),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(*NORM),
    ]
)


def load_data(data_dir, batch_size, num_workers=4):
    # TODO: Write docstring
    """[summary]

    Args:
        data_dir ([type]): [description]

    Returns:
        [type]: [description]
    """
    data_dir = Path(data_dir)

    data_transforms = {
        "train": TRAIN_TRANSFORM,
        "valid": PREDICT_TRANSFORM,
        "test": PREDICT_TRANSFORM,
    }

    image_datasets = {
        x: datasets.ImageFolder(data_dir / x, data_transforms[x])
        for x in ["train", "valid", "test"]
    }

    dataloaders = {
        x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=num_workers)
        for x in ["train", "valid", "test"]
    }

    class_names = image_datasets["train"].classes

    return class_names, image_datasets, dataloaders


def process_image(image_path):
    """Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array

    Args:
        image_path ([type]): [description]

    Returns:
        [type]: [description]
    """
    img_pil = Image.open(image_path)

    adjustments = PREDICT_TRANSFORM

    img_tensor = adjustments(img_pil)

    return img_tensor


def unnormalize_img_tensor(img_tensor):
    """Imshow for Tensor

    Args:
        img_tensor ([type]): [description]

    Returns:
        [type]: [description]
    """
    img_tensor = img_tensor.numpy().transpose((1, 2, 0))
    mean = np.array(NORM[0])
    std = np.array(NORM[1])
    img_tensor = std * img_tensor + mean
    img_tensor = np.clip(img_tensor, 0, 1)
    return img_tensor

def visualize_sample(dataloader, model):
    """[summary]

    Args:
        dataloader ([type]): [description]
        model ([type]): [description]
    """
    # Get a batch of data
    images, labels = next(iter(dataloader))

    images = images[:10]
    labels = labels[:10]

    fig = plt.figure(figsize=(25, 2))
    for i in np.arange(10):
        ax = fig.add_subplot(1, 10, i+1, xticks=[], yticks=[])
        plt.imshow(unnormalize_img_tensor(images[i]))
        ax.set_title(model.classes[labels[i]])

    return

def plot_image(img_path):
    img_pil = Image.open(img_path)
    plt.imshow(img_pil)
    plt.axis("off")
    plt.show()

    return
