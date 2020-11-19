import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

from .data_helpers import process_image

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


def predict(image_path, model, cat_to_name, topk=1, device=torch.device('cpu')):
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

    image = torch.from_numpy(process_image(image_path)).float().unsqueeze(0)

    image = image.to(device)
    model.to(device)

    model.eval()

    with torch.set_grad_enabled(False):
        output = model(image)

    probs = torch.exp(output)
    top_probs, top_classes = probs.topk(topk)

    top_probs = top_probs.cpu().numpy().tolist()[0]
    top_classes = top_classes.cpu().numpy().tolist()[0]

    idx_to_class = {val: key for key, val in model.class_to_idx.items()}

    if cat_to_name:
        top_classes = [cat_to_name[idx_to_class[class_no]] for class_no in top_classes]
    else:
        top_classes = [idx_to_class[class_no] for class_no in top_classes]

    prediction_dict = dict(zip(top_classes,top_probs))

    return prediction_dict

def plot_image(img_path):
    img = cv2.imread(img_path)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(rgb)
    plt.axis('off')
    plt.show()

def caption_images(img_path, top_k, model, cat_to_name):
    probs, classes = predict(img_path, model, top_k)
    class_names = [cat_to_name[x] for x in classes]
    print(f"Top {top_k} predictions: {list(zip(class_names, probs))}")
    plot_image(img_path)
