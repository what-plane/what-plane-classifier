from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models


def initialize_model(model_name, dropout, num_classes, hidden_layer, lr, device):

    if model_name == "densenet":
        model = models.densenet121(pretrained=True)
        input_size = 1024

    elif model_name == "vgg":
        model = models.vgg11_bn(pretrained=True)
        input_size = 25088

    elif model_name == "alexnet":
        model = models.alexnet(pretrained=True)
        input_size = 9216

    else:
        print("Invalid model name")

    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(
        OrderedDict(
            [
                ("fc1", nn.Linear(input_size, hidden_layer)),
                ("relu", nn.ReLU()),
                ("dropout", nn.Dropout(dropout)),
                ("fc2", nn.Linear(hidden_layer, num_classes)),
                ("output", nn.LogSoftmax(dim=1)),
            ]
        )
    )

    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr)
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    model.to(device)

    return model

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

