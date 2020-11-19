from PIL import ImageFile
import os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from collections import OrderedDict

import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.optim import lr_scheduler
from torchvision import datasets
from torch.utils.tensorboard import SummaryWriter


use_cuda = torch.cuda.is_available()

ImageFile.LOAD_TRUNCATED_IMAGES = True


def train(n_epochs, loaders, model, optimizer, criterion, use_cuda, save_path):
    # Tensorboard writer
    writer = SummaryWriter()
    """returns trained model"""
    writer.flush()

    print('Training for {} epochs on {}\n'.format(
        n_epochs, 'GPU' if use_cuda else 'CPU'))
    # initialize tracker for minimum validation loss
    valid_loss_min = np.Inf

    losses = {'train': [], 'validation': []}
    for epoch in range(1, n_epochs+1):
        # initialize variables to monitor training and validation loss
        train_loss = 0.0
        valid_loss = 0.0

        ###################
        # train the model #
        ###################
        model.train()
        for batch_idx, (data, target) in enumerate(loaders['train']):
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            # find the loss and update the model parameters accordingly
            # record the average training loss, using something like
            ## train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += ((1/(batch_idx+1))*(loss.data-train_loss))

        ######################
        # validate the model #
        ######################
        model.eval()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(loaders['valid']):
                # move to GPU
                if use_cuda:
                    data, target = data.cuda(), target.cuda()
                output = model(data)
                loss = criterion(output, target)
                valid_loss += ((1/(batch_idx+1))*(loss.data-valid_loss))
                # update the average validation loss

        # print training/validation statistics
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch,
            train_loss,
            valid_loss
        ))
        losses['train'].append(train_loss)
        losses['validation'].append(valid_loss)
        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Valid", valid_loss, epoch)

        # TODO: save the model if validation loss has decreased
        if valid_loss < valid_loss_min:
            print('Saving model', save_path)
            valid_loss_min = valid_loss
            torch.save(model.state_dict(), save_path)

    # plotting losses
    plt.plot(losses['train'], label='Training Loss')
    plt.plot(losses['validation'], label='Validation Loss')
    plt.legend()
    _ = plt.ylim()

    # return trained model
    return model


def test(loaders, model, criterion, use_cuda):

    # monitor test loss and accuracy
    test_loss = 0.
    correct = 0.
    total = 0.

    model.eval()
    for batch_idx, (data, target) in enumerate(loaders['test']):
        # move to GPU
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the loss
        loss = criterion(output, target)
        # update average test loss
        test_loss = test_loss + ((1 / (batch_idx + 1))
                                 * (loss.data - test_loss))
        # convert output probabilities to predicted class
        pred = output.data.max(1, keepdim=True)[1]
        # compare predictions to true label
        correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
        total += data.size(0)

    print('Test Loss: {:.6f}\n'.format(test_loss))

    print('\nTest Accuracy: %2d%% (%2d/%2d)' % (
        100. * correct / total, correct, total))
