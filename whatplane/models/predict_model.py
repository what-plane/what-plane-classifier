import torch
import numpy as np
import matplotlib.pyplot as plt

from .data_helpers import process_image_data, process_image_file


def test(dataloaders, model, criterion):
    # TODO Refactor this
    # monitor test loss and accuracy

    test_dataloader = dataloaders["test"]
    test_loss = 0.0
    test_accuracy = 0.0

    predicted_classes = []
    correct_classes = []

    model.eval()

    with torch.no_grad():
        for images, labels in test_dataloader:
            images, labels = images.to(model.device), labels.to(model.device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct = preds == labels.view(*preds.shape)
            test_accuracy += torch.mean(correct.type(torch.FloatTensor)).item()
            predicted_classes.extend(preds.cpu().numpy().tolist())
            correct_classes.extend(labels.cpu().numpy().tolist())

    test_loss /= len(test_dataloader)
    test_accuracy /= len(test_dataloader)

    print("Test Loss: {:.6f}".format(test_loss))
    print("Test Accuracy: {:.2f}".format(100 * test_accuracy))

    return test_loss, test_accuracy, predicted_classes, correct_classes


def predict_image_data(image_data, model, topk=1):
    image = process_image_data(image_data).float().unsqueeze(0)
    return predict_normalized(image, model, topk)

def predict(image_path, model, topk=1):
    image = process_image_file(image_path).float().unsqueeze(0)
    return predict_normalized(image, model, topk)

def predict_normalized(processed_image, model, topk):
    """ Predict the class (or classes) of an image using a trained deep learning model.

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
    """

    processed_image = processed_image.to(model.device)

    model.eval()

    with torch.set_grad_enabled(False):
        output = model(processed_image)

    probs = torch.nn.functional.softmax(output, dim=1)

    top_probs, top_classes = probs.topk(topk)
    top_probs = top_probs.cpu().numpy().tolist()[0]

    top_classes = [model.class_names[i] for i in top_classes.cpu().numpy().tolist()[0]]

    return top_probs, top_classes


def predict_aircraft(image_path, model):
    _, classes = predict(image_path, model)

    return classes[0]
