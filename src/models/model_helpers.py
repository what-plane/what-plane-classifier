from collections import OrderedDict
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

def initialize_model(arch, hidden_units, class_to_idx, dropout):
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

    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(
        OrderedDict(
            [
                ("fc1", nn.Linear(input_units, hidden_units)),
                ("relu1", nn.ReLU()),
                ("drop1", nn.Dropout(dropout)),
                ("fc2", nn.Linear(hidden_units, len(class_to_idx))),
                ('output', nn.LogSoftmax(dim=1))
            ]
        )
    )
    print(classifier)
    model.classifier = classifier

    model.class_to_idx = class_to_idx

    return model


# def initialize_model(model_name, dropout, num_classes, hidden_layer, lr, device):

#     classifier = nn.Sequential(
#         OrderedDict(
#             [
#                 ("fc1", nn.Linear(input_size, hidden_layer)),
#                 ("relu", nn.ReLU()),
#                 ("dropout", nn.Dropout(dropout)),
#                 ("fc2", nn.Linear(hidden_layer, num_classes)),
#                 ("output", nn.LogSoftmax(dim=1)),
#             ]
#         )
#     )

#     model.classifier = classifier
#     criterion = nn.NLLLoss()
#     optimizer = optim.Adam(model.classifier.parameters(), lr)
#     exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
#     model.to(device)

#     return model


def save_checkpoint(
    checkpoint_path, arch, image_datasets, model, optimizer, num_epochs, lr, dropout, hidden_layer
):
    # TODO: Rework this
    save_path = f"./checkpoint-{arch}.pth"

    model.class_to_idx = image_datasets["train"].class_to_idx
    model.cpu()
    torch.save(
        {
            "model": model,
            "hidden_layer": hidden_layer,
            "lr": lr,
            "num_epochs": num_epochs,
            "dropout": dropout,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "class_to_idx": model.class_to_idx,
        },
        save_path,
    )


def load_checkpoint(checkpoint_path):
    # Load the saved file
    checkpoint = torch.load(checkpoint_path)
    model = checkpoint["model"]
    print(model)
    # Load stuff from checkpoint
    model.class_to_idx = checkpoint["class_to_idx"]
    model.load_state_dict(checkpoint["model_state_dict"])

    return model


def save_model(model, optimizer, losses, acc, model_name, arch, dropout, path=""):
    """Save PyTorch model with optimiser

    Saves a PyTorch model state_dict and optimiser state_dict, along with log
    of previous training (no. epochs, losses, accuracy)

    Args:
        model (object): A PyTorch model
        optimizer (object): A PyTorch optimizer
        losses (dict): Dictionary of the logged training and validation loss
        acc (dict): Dictionary of the logged training and validation accuracy
        model_name (str): Name to save model as
        arch (str): Name of pretrained model as per method, i.e. vgg13 for models.vgg13()
                    (see https://pytorch.org/docs/stable/torchvision/models.html)
        path (str): Directory to save model as

    Returns:
        None

    Example:
        >>> save_model(model, optimizer, losses, accuracy, 'model_512hu_classifier', densenet121, path='home/checkpoints')
    """

    if path != "":
        model_path = os.path.join(path, model_name + ".pt")

        if not os.path.exists(path):
            os.makedirs(path)
    else:
        model_path = model_name + ".pt"

    torch.save(
        {
            "epoch": len(losses["train"]),
            "arch": arch,
            "dropout": dropout,
            "model_state_dict": model.state_dict(),
            "class_to_idx": model.class_to_idx,
            "optimizer_state_dict": optimizer.state_dict(),
            "losses": losses,
            "acc": acc,
        },
        model_path,
    )

    print(f"Model saved as: {model_path}")

    return


def load_model(model_path, optimizer):
    """Load PyTorch model with optimiser in order to continue training

    Load a PyTorch model state_dict and optimiser state_dict along with log
    of previous training (no. epochs, losses, accuracy) from checkpoint

    Args:
        model_path (str): Location of the saved model checkpoint
        optimizer (object): A PyTorch optimizer (Must be the same as the one in
                            the model checkpoint)

    Returns:
        epoch (int): Number of epochs the model has been trained for
        model (object): A PyTorch model now with the checkpoint state_dict loaded
        optimizer (object): A PyTorch optimizer now with the checkpoint state_dict loaded
        losses (dict): Dictionary of the logged training and validation loss
        acc (dict): Dictionary of the logged training and validation accuracy

    Example:
        >>> epoch, model, optimizer, losses, accuracy = load_model_train('checkpoints/model_512hu_classifier.pt', optimizer)
    """

    checkpoint = torch.load(model_path)

    model = initialize_model(
        checkpoint["arch"],
        checkpoint["model_state_dict"]["classifier.fc2.weight"].shape[1],
        checkpoint["class_to_idx"],
        checkpoint["dropout"]
    )

    epoch = checkpoint["epoch"]
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    losses = checkpoint["losses"]
    acc = checkpoint["acc"]

    print(f"Model loaded from: {model_path}")

    return epoch, model, optimizer, losses, acc
