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


def load_data(data_dir, batch_size):
    # TODO: Write docstring
    """[summary]

    Args:
        data_dir ([type]): [description]

    Returns:
        [type]: [description]
    """
    data_dir = Path(data_dir)
    train_dir = data_dir / "train"
    test_dir = data_dir / "test"
    valid_dir = data_dir / "valid"

    l = IMAGE_SIZE/2

    rad = math.radians(ROTATION_ANGLE)
    c = math.cos(rad)
    s = math.sin(rad)

    x = l*c-l*s
    y = l*s+l*c
    rotpad = math.ceil(max(x,y)-l)

    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.Pad((0,0,0,-BANNERHEIGHT)), # Crop banner from bottom edge of image
                transforms.Resize(240),
                transforms.RandomResizedCrop(IMAGE_SIZE),
                transforms.RandomHorizontalFlip(),
                transforms.Pad(rotpad, padding_mode='reflect'), # Mirror boundary to avoid empty corners of rotated image
                transforms.RandomRotation(ROTATION_ANGLE),
                transforms.CenterCrop(IMAGE_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(*NORM)
            ]
        ),
        "valid": transforms.Compose(
            [
                transforms.Pad((0,0,0,-BANNERHEIGHT)),
                transforms.Resize(256),
                transforms.CenterCrop(IMAGE_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(*NORM)
            ]
        ),
        "test": transforms.Compose(
            [
                transforms.Pad((0,0,0,-BANNERHEIGHT)),
                transforms.Resize(256),
                transforms.CenterCrop(IMAGE_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(*NORM)
            ]
        ),
    }

    image_datasets = {
        x: datasets.ImageFolder(data_dir / x, data_transforms[x])
        for x in ["train", "valid", "test"]
    }

    dataloaders = {
        x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4)
        for x in ["train", "valid", "test"]
    }

    dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "test", "valid"]}

    class_names = image_datasets["train"].classes

    return dataloaders, dataset_sizes, image_datasets


def process_image(image_path):
    """Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array

    Args:
        image_path ([type]): [description]

    Returns:
        [type]: [description]
    """
    img_pil = Image.open(image_path)

    adjustments = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(*NORM)
        ]
    )

    img_tensor = adjustments(img_pil)

    return img_tensor


def imshow(img_tensor, title=None):
    """Imshow for Tensor."""
    img_tensor = img_tensor.numpy().transpose((1, 2, 0))
    mean = np.array(NORM[0])
    std = np.array(NORM[1])
    img_tensor = std * img_tensor + mean
    img_tensor = np.clip(img_tensor, 0, 1)
    plt.imshow(img_tensor)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

    return
