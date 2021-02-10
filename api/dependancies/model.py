from pathlib import Path
import json
from typing import List, Tuple

from PIL.Image import Image
from torch import hub

import whatplane.models.model_helpers as mh
from whatplane.models.predict_model import predict_image_data

BASE_DIR = Path(".")
MODELS_DIR = BASE_DIR / "models"

with open(BASE_DIR / "api/imagenet_class_index.json") as f:
    imagenet_class_index = json.load(f)

# Set PyTorch cache directory to where the model is stored
hub.set_dir(str(MODELS_DIR.resolve()))
imagenet_model = mh.initialize_model(
    "densenet161", [item[1] for item in list(imagenet_class_index.values())], replace_classifier=False,
)
whatplane_model = mh.load_model(MODELS_DIR / "model.pth")


def should_predict_whatplane(imagenet_probs: List[float], imagenet_classes: List[str]) -> bool:
    """Function to check which model to use for prediction.

    Args:
        imagenet_probs (List[float]): List of probabilities returned from ImageNet
        imagenet_classes (List[str]): List of class names returned from ImageNetwork

    Returns:
        bool: Returns True if WhatPlane is to be used, otherwise returns False
    """
    VALID_CLASSES = ["Airliner", "Wing"]

    imagenet_likely_class = set([imagenet_classes[i] for i, prob in enumerate(imagenet_probs) if prob > 0.5])

    if imagenet_classes[0] in VALID_CLASSES:
        return True
    elif imagenet_likely_class.intersection(set(VALID_CLASSES)):
        return True

    return False


def get_wp_classes() -> List:
    """Function to extract the the list of classes from whatplane

    Returns:
        List: List of all classes from WhatPlane
    """
    return whatplane_model.class_names
