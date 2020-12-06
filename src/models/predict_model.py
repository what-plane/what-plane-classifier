import torch
import numpy as np
import matplotlib.pyplot as plt

from .data_helpers import process_image, plot_image, unnormalize_img_tensor

def test(loaders, model, criterion, use_cuda):
    # TODO Refactor this
    # monitor test loss and accuracy
    test_loss = 0.0
    correct = 0.0
    total = 0.0

    model.eval()
    for batch_idx, (data, target) in enumerate(loaders["test"]):
        # move to GPU
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the loss
        loss = criterion(output, target)
        # update average test loss
        test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss.data - test_loss))
        # convert output probabilities to predicted class
        pred = output.data.max(1, keepdim=True)[1]
        # compare predictions to true label
        correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
        total += data.size(0)

    print("Test Loss: {:.6f}\n".format(test_loss))
    print("\nTest Accuracy: %2d%% (%2d/%2d)" % (100.0 * correct / total, correct, total))


def predict(image_path, model, topk=1, device=torch.device('cpu')):
    ''' Predict the class (or classes) of an image using a trained deep learning model.

    Args:
        image_path (str): Location of the image file
        model (object): A trained PyTorch model
        cat_to_name (dict): Dict which maps category numbers to category names
        top_k (int): Number of top classes to return
        device (obj): Device to perform inference on

    Returns:
        prediction_dict (dict): Dictionary of top classes predicted for that image

    Example:
        >>> result = predict('images/flower.jpg', model, cat_to_name, 5, torch.device('cpu'))
    '''

    image = process_image(image_path).float().unsqueeze(0)

    image = image.to(device)
    model.to(device)

    model.eval()

    with torch.set_grad_enabled(False):
        output = model(image)

    if "LogSoftmax" in str(model.classifier[-1]):
        probs = torch.exp(output)
    else:
        probs = torch.nn.functional.softmax(output, dim=1)

    top_probs, top_classes = probs.topk(topk)
    top_probs = top_probs.cpu().numpy().tolist()[0]

    top_classes = [model.classes[i] for i in top_classes.cpu().numpy().tolist()[0]]

    return top_probs, top_classes

def predict_aircraft(image_path, model):
    _, classes = predict(image_path, model)

    return classes[0]


def visualize_results(dataloader, model):
    # Visualize some sample data

    device = model.device

    # Get a batch of data
    images, labels = next(iter(dataloader))

    images = images[:10]
    labels = labels[:10]

    model.eval()

    with torch.no_grad():
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        predicted_classes = preds.cpu().numpy()
        correct_classes = labels.cpu().numpy

    model.train()

    fig = plt.figure(figsize=(25, 3))
    for i in np.arange(10):
        ax = fig.add_subplot(1, 10, i+1, xticks=[], yticks=[])
        imshow(images[i])
        ax.set_title('{}\n({})'.format(model.classes[predicted_classes[i]], model.classes[correct_classes[i]]),
                 color=('green' if predicted_classes[i]==correct_classes[i] else 'red'))