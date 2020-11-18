import time
import copy
from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.tensorboard import SummaryWriter


def train(n_epochs, loaders, model, optimizer, criterion, use_cuda, save_path):
    # Tensorboard writer
    writer = SummaryWriter()
    """returns trained model"""
    writer.flush()

    print("Training for {} epochs on {}\n".format(n_epochs, "GPU" if use_cuda else "CPU"))
    # initialize tracker for minimum validation loss
    valid_loss_min = np.Inf

    losses = {"train": [], "validation": []}
    for epoch in range(1, n_epochs + 1):
        # initialize variables to monitor training and validation loss
        train_loss = 0.0
        valid_loss = 0.0

        ###################
        # train the model #
        ###################
        model.train()
        for batch_idx, (data, target) in enumerate(loaders["train"]):
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            ## find the loss and update the model parameters accordingly
            ## record the average training loss, using something like
            ## train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += (1 / (batch_idx + 1)) * (loss.data - train_loss)

        ######################
        # validate the model #
        ######################
        model.eval()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(loaders["valid"]):
                # move to GPU
                if use_cuda:
                    data, target = data.cuda(), target.cuda()
                output = model(data)
                loss = criterion(output, target)
                valid_loss += (1 / (batch_idx + 1)) * (loss.data - valid_loss)
                ## update the average validation loss

        # print training/validation statistics
        print(
            "Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}".format(
                epoch, train_loss, valid_loss
            )
        )
        losses["train"].append(train_loss)
        losses["validation"].append(valid_loss)
        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Valid", valid_loss, epoch)

        ## TODO: save the model if validation loss has decreased
        if valid_loss < valid_loss_min:
            print("Saving model", save_path)
            valid_loss_min = valid_loss
            torch.save(model.state_dict(), save_path)

    # plotting losses
    plt.plot(losses["train"], label="Training Loss")
    plt.plot(losses["validation"], label="Validation Loss")
    plt.legend()
    _ = plt.ylim()

    # return trained model
    return model


