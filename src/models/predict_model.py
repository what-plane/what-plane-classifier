import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

from data_helpers import process_image


def test(loaders, model, criterion, use_cuda):
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

def predict(image_path, model, topk=1):
    image = process_image(image_path)

    image = image.unsqueeze_(0)
    image = image.float()

    with torch.no_grad():
        output = model.forward(image)

    output_prob = torch.exp(output)

    probs, indeces = output_prob.topk(topk)
    probs = probs.numpy()
    indeces = indeces.numpy()

    probs = probs.tolist()[0]
    indeces = indeces.tolist()[0]

    mapping = {val: key for key, val in model.class_to_idx.items()}
    classes = [mapping[item] for item in indeces]
    classes = np.array(classes)
    return probs, classes

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
