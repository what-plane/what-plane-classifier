from collections import OrderedDict
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models


def initialize_model(
    arch,
    class_names,
    hidden_units=1024,
    dropout=0.4,
    device=torch.device("cpu"),
    pre_trained=True,
    freeze_model=True,
    replace_classifier=True,
):
    # TODO Docstring
    """Create PyTorch model based on Pretrained Network with custom classifier

    Downloads pretrained model and freezes parameters. Sets up custom classifier with
    1 hidden layer of the specified size.

    Args:
        arch (str): Name of pretrained model as per method, i.e. vgg13 for models.vgg13()
                    (see https://pytorch.org/docs/stable/torchvision/models.html)
        hidden_units (int): Number of hidden units in classifier
        class_names (obj): class_names attribute of dataloader object.
        dropout (float): Dropout on fc1

    Returns:
        model (obj): PyTorch model

    Example:
        >>> model = create_model('densenet121', 512, class_to_idx)
    """

    model = getattr(models, arch)(pretrained=pre_trained)
    model.arch = arch

    if "classifier" not in str(list(model.named_modules())[-1]):
        raise RuntimeError("Requested Model Architecture not supported")

    # Replace classifier
    input_units = model.classifier.state_dict()[next(iter(model.classifier.state_dict()))].size(1)

    if freeze_model:
        for param in model.parameters():
            param.requires_grad = False

    if replace_classifier:
        classifier = nn.Sequential(
            OrderedDict(
                [
                    ("fc1", nn.Linear(input_units, hidden_units)),
                    ("relu1", nn.ReLU()),
                    ("drop1", nn.Dropout(dropout)),
                    ("fc2", nn.Linear(hidden_units, len(class_names))),
                ]
            )
        )

        model.classifier = classifier

    model.class_names = class_names

    model.accuracies = {"train": [], "valid": []}
    model.losses = {"train": [], "valid": []}

    model.device = device
    model.to(model.device)

    return model


def save_checkpoint(model, optimizer, tag, models_folder=Path("../models")):

    save_path = models_folder / gen_model_path(model, optimizer, tag)

    print("Saving checkpoint: ", save_path)

    model.cpu()
    checkpoint = {
        "arch": model.arch,
        "class_names": model.class_names,
        "model_state": model.state_dict(),
        "optimizer": optimizer.__class__.__name__,
        "optimizer_state": optimizer.state_dict(),
        "losses": model.losses,
        "accuracies": model.accuracies,
    }

    torch.save(checkpoint, save_path)

    model.to(model.device)


def save_model(model, optimizer, tag, models_folder=Path("../models")):

    tag = "_".join(["model", tag])

    save_path = models_folder / gen_model_path(model, optimizer, tag)

    print("Saving model: ", save_path)

    model.cpu()
    torch.save({"model": model}, save_path)
    model.to(model.device)


def load_checkpoint(model, optimizer, load_path):

    model.cpu()
    # load checkpoint
    print("Loading checkpoint: ", load_path)
    checkpoint = torch.load(load_path)

    # model
    assert model.arch == checkpoint["arch"], "Trying to load {} into {}".format(
        checkpoint["arch"], model.arch
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(model.device)

    model.class_names = checkpoint["class_names"]
    model.losses = checkpoint["losses"]
    model.accuracies = checkpoint["accuracies"]

    # optimizer
    assert (
        optimizer.__class__.__name__ == checkpoint["optimizer"]
    ), "Trying to load {} into {}".format(checkpoint["optimizer"], optimizer.__class__.__name__)
    optimizer.load_state_dict(checkpoint["optimizer_state"])

    return model, optimizer


def load_model(load_path, device=torch.device("cpu")):

    print("Loading model: ", load_path)
    checkpoint = torch.load(load_path)

    model = checkpoint["model"]
    model.device = device
    model.to(model.device)

    return model


def gen_model_path(model, optimizer, tag):
    return "_".join([tag, model.arch, optimizer.__class__.__name__]) + ".pth"
