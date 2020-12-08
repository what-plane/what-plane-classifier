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

from .model_helpers import save_checkpoint, load_checkpoint, gen_model_path


def train_model(model, optimizer, dataloaders, criterion, n_epochs, tag, models_folder):
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

    PHASES = ["train", "valid"]

    model.to(model.device)

    # Tensorboard
    writer = SummaryWriter()
    comment = "Training {} for {} epochs on {}\n".format(tag, n_epochs, model.device)
    print(comment)
    writer.add_text("TRAIN", comment)
    comment = "Model: {}\nCriterion: {}\nOptimizer: {}\n".format(
        model.arch, optimizer.__class__.__name__, criterion.__class__.__name__
    )
    print(comment)
    writer.add_text("TRAIN", comment)
    writer.flush()

    start_epoch = len(model.losses["valid"])

    best_loss = np.Inf

    if start_epoch == 0:  # Fresh start
        best_loss = np.Inf
    else:  # Resume training
        comment = "Loading previous {} epochs\n".format(start_epoch)
        print(comment)
        writer.add_text("TRAIN", comment)
        for i in range(start_epoch):
            writer.add_scalars("Loss", {x: model.losses[x][i] for x in PHASES}, i)
            writer.add_scalars("Accuracy", {x: model.accuracies[x][i] for x in PHASES}, i)
        writer.flush()
        best_loss = model.losses["valid"][-1]

    for epoch in range(start_epoch, start_epoch + n_epochs):

        for phase in PHASES:

            running_loss = 0
            running_acc = 0

            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            # Init progress bar
            steps = 0
            step_bar = tqdm(total=len(dataloaders[phase]), desc="Steps", position=0)
            postfix_dict = {"Phase": phase, "Loss": "None"}

            for images, labels in dataloaders[phase]:
                images, labels = images.to(model.device), labels.to(model.device)

                with torch.set_grad_enabled(phase == "train"):

                    outputs = model(images)

                    loss = criterion(outputs, labels)

                    if phase == "train":
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item()

                _, top_class = torch.max(outputs, 1)
                equals = top_class == labels.view(*top_class.shape)
                running_acc += torch.mean(equals.type(torch.FloatTensor)).item()

                # Update Progress Bar
                steps += 1
                postfix_dict["Loss"] = str(round(running_loss / steps, 3))
                step_bar.set_postfix(postfix_dict)
                step_bar.update(1)

            step_bar.close()

            epoch_loss = running_loss / len(dataloaders[phase])
            model.losses[phase].append(epoch_loss)

            epoch_acc = running_acc / len(dataloaders[phase])
            model.accuracies[phase].append(epoch_acc)

            if phase == "valid":
                print("-" * 8)
                print(
                    "Epoch: {}/{}.. ".format(epoch + 1, start_epoch + n_epochs),
                    "Training Loss: {:.3f}.. ".format(model.losses["train"][-1]),
                    "Training Accuracy: {:.3f}.. ".format(model.accuracies["train"][-1]),
                    "Validation Loss: {:.3f}.. ".format(model.losses["valid"][-1]),
                    "Validation Accuracy: {:.3f}".format(model.accuracies["valid"][-1]),
                )
                writer.add_scalars("Loss", {x: model.losses[x][-1] for x in PHASES}, epoch)
                writer.add_scalars("Accuracy", {x: model.accuracies[x][-1] for x in PHASES}, epoch)
                writer.flush()

                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    save_checkpoint(model, optimizer, tag, models_folder)

    load_path = models_folder / gen_model_path(model, optimizer, tag)
    model, optimizer = load_checkpoint(model, optimizer, load_path)
    return model, optimizer
