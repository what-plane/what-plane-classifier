import time
import copy
from collections import OrderedDict

from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.tensorboard import SummaryWriter


def train_model(model, optimizer, losses, acc, dataloaders, criterion, n_epochs=3, device=torch.device("cpu")):
    """Train PyTorch model with specified optimizer & dataloaders

    With specified model, optimiser, dataloaders, train the model using training data for
    the specified number of epochs. Use validation data to assess model after epoch.
    Shows progress of training and validation including loss, and prints training and
    validation losses and accuracy after each epoch

    Args:
        model (object): A PyTorch model
        optimizer (object): A PyTorch optimizer
        losses (dict): Dictionary to save training and validation loss
        acc (dict): Dictionary to save training and validation accuracy
        dataloaders (dict of obj): Dictionary of PyTorch dataloaders for 'train' and
                                   'validation' data
        criterion (obj): PyTorch criterion to use, e.g. nn.NLLLoss()
        n_epochs (int): number of epochs to train for
        device (obj): PyTorch device to use for training

    Returns:
        model (obj): Trained PyTorch model
        optimizer (obj): PyTorch optimiser used for training
        losses (dict): Dictionary of the logged training and validation loss
        acc (dict): Dictionary of the logged training and validation accuracy

    Example:
        >>> output_model, output_optimizer, output_losses, output_acc = train_model(model,
                                        optimizer, {'train':[],'valid':[]}, {'train':[],'valid':[]},
                                        dataloaders, torch.nn.NLLLoss(), 5, torch.device('cpu'))
    """

    writer = SummaryWriter()
    writer.flush()

    start_epoch = len(acc["valid"])

    model.to(device)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(start_epoch, start_epoch + n_epochs):

        for phase in ["train", "valid"]:

            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0
            running_acc = 0

            steps = 0
            step_bar = tqdm(total=len(dataloaders[phase]), desc="Steps", position=0)

            postfix_dict = {"Phase": phase, "Loss": "None"}

            for images, labels in dataloaders[phase]:
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):

                    outputs = model(images)

                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                probs = torch.exp(outputs)
                top_p, top_class = probs.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)

                running_acc += torch.mean(equals.type(torch.FloatTensor)).item()
                running_loss += loss.item()

                steps += 1
                postfix_dict["Loss"] = str(round(running_loss / steps, 3))
                step_bar.set_postfix(postfix_dict)
                step_bar.update(1)

            step_bar.close()

            epoch_loss = running_loss / len(dataloaders[phase])
            epoch_acc = running_acc / len(dataloaders[phase])

            losses[phase].append(epoch_loss)
            acc[phase].append(epoch_acc)
            writer.add_scalar(f"Loss/{phase}", epoch_loss, epoch)

            if phase == "valid":
                print("-" * 8)
                print(
                    "Epoch: {}/{}.. ".format(epoch + 1, start_epoch + n_epochs),
                    "Training Loss: {:.3f}.. ".format(losses["train"][-1]),
                    "Training Accuracy: {:.3f}.. ".format(acc["train"][-1]),
                    "Validation Loss: {:.3f}.. ".format(losses["valid"][-1]),
                    "Validation Accuracy: {:.3f}".format(acc["valid"][-1]),
                )

                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_wts)
    return model, optimizer, losses, acc
