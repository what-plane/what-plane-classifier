from pathlib import Path
import json

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
    "densenet161",
    [item[1] for item in list(imagenet_class_index.values())],
    replace_classifier=False,
)
whatplane_model = mh.load_model(MODELS_DIR / "model.pth")


def predict_imagenet(image, topk):

    imagenet_probs, imagenet_classes = predict_image_data(image, imagenet_model, topk)

    # Take only classes with prob > 0.5
    imagenet_likely_class = [
        imagenet_classes[i] for i, prob in enumerate(imagenet_probs) if prob > 0.5
    ]
    return imagenet_probs, imagenet_classes, imagenet_likely_class


def predict_whatplane(image, topk):
    return predict_image_data(image, whatplane_model, topk)
