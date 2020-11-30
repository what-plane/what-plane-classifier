from collections import OrderedDict
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models


def initialize_model(arch, hidden_units, class_to_idx, dropout, freeze_model=True):
    # TODO Docstring
    """Create PyTorch model based on Pretrained Network with custom classifier

    Downloads pretrained model and freezes parameters. Sets up custom classifier with
    1 hidden layer of the specified size.

    Args:
        arch (str): Name of pretrained model as per method, i.e. vgg13 for models.vgg13()
                    (see https://pytorch.org/docs/stable/torchvision/models.html)
        hidden_units (int): Number of hidden units in classifier
        class_to_idx (obj): class_to_idx attribute of dataloader object.
        dropout (float): Dropout on fc1

    Returns:
        model (obj): PyTorch model

    Example:
        >>> model = create_model('densenet121', 512, class_to_idx)
    """

    if arch == "densenet121":
        model = models.densenet121(pretrained=True)
        input_units = model.classifier.in_features
    elif arch == "densenet161":
        model = models.densenet161(pretrained=True)
        input_units = model.classifier.in_features
    elif arch == "vgg13":
        model = models.vgg13(pretrained=True)
        input_units = model.classifier[0].in_features
    elif arch == "vgg16":
        model = models.vgg16(pretrained=True)
        input_units = model.classifier[0].in_features
    else:
        raise ValueError("Requested arch not available")

    if freeze_model:
        for param in model.parameters():
            param.requires_grad = False

    classifier = nn.Sequential(
        OrderedDict(
            [
                ("fc1", nn.Linear(input_units, hidden_units)),
                ("relu1", nn.ReLU()),
                ("drop1", nn.Dropout(dropout)),
                ("fc2", nn.Linear(hidden_units, len(class_to_idx))),
                # ("output", nn.Softmax(dim=1)),
            ]
        )
    )

    model.classifier = classifier

    model.class_to_idx = class_to_idx

    return model


def save_checkpoint(checkpoint_folder, checkpoint_name, model):
    # TODO Docstring
    save_path = checkpoint_folder / f"{checkpoint_name}_checkpoint.pth"

    model.cpu()
    torch.save(
        {"model_state_dict": model.state_dict()}, save_path,
    )

    return


def load_checkpoint(checkpoint_path, model):
    # TODO Docstring
    model.load_state_dict(torch.load(checkpoint_path))
    return model


def save_full_model(checkpoint_folder, checkpoint_name, model, optimiser, criterion, losses, acc):
    # TODO Docstring
    save_path = checkpoint_folder / f"{checkpoint_name}_full_model.pth"

    model.cpu()
    torch.save(
        {
            "model": model,
            "losses": losses,
            "acc": acc,
            "criterion": criterion,
            "optimiser": optimiser,
        },
        save_path,
    )

    return

def load_full_model(checkpoint_path):
    # TODO Docstring
    # Load the saved file
    checkpoint = torch.load(checkpoint_path)

    # Load the model
    model = checkpoint["model"]

    # Load stuff from checkpoint
    losses = checkpoint["losses"]
    acc = checkpoint["acc"]
    criterion = checkpoint["criterion"]

    optimiser = checkpoint["optimiser"]
    return model, losses, acc, criterion, optimiser