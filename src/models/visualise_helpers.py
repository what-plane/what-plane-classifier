from itertools import product
import random

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import confusion_matrix

from .data_helpers import unnormalize_img_tensor
from .predict_model import predict

def visualize_sample(dataloader, model):
    """[summary]

    Args:
        dataloader ([type]): [description]
        model ([type]): [description]
    """
    # Get a batch of data
    images, labels = next(iter(dataloader))

    images = images[:10]
    labels = labels[:10]

    fig = plt.figure(figsize=(25, 2))
    for i in np.arange(10):
        ax = fig.add_subplot(1, 10, i + 1, xticks=[], yticks=[])
        plt.imshow(unnormalize_img_tensor(images[i]))
        ax.set_title(model.class_names[labels[i]])

    return


def plot_image(img_path):
    img_pil = Image.open(img_path)
    plt.imshow(img_pil)
    plt.axis("off")
    plt.show()

    return


def visualize_results(dataloader, model):

    # Get a batch of data
    images, labels = next(iter(dataloader))

    images = images[:10]
    labels = labels[:10]

    model.eval()

    with torch.no_grad():
        images, labels = images.to(model.device), labels.to(model.device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        predicted_classes = preds.cpu().numpy()
        correct_classes = labels.cpu().numpy()

    model.train()

    fig = plt.figure(figsize=(25, 3))
    for i in np.arange(10):
        ax = fig.add_subplot(1, 10, i + 1, xticks=[], yticks=[])
        plt.imshow(unnormalize_img_tensor(images[i]))
        predicted_class = model.class_names[predicted_classes[i]]
        correct_class = model.class_names[correct_classes[i]]
        plt_title = f"{predicted_class}\n({correct_class})"
        ax.set_title(
            plt_title, color=("green" if predicted_classes[i] == correct_classes[i] else "red"),
        )

def predict_test_image_folder(test_image_dir, model):
    test_images = test_image_dir.glob("*/*.jpg")
    fig = plt.figure(figsize=(25, 12))
    for i, img_path in enumerate(random.sample(list(test_images),3)):
        probs, classes = predict(img_path, model, 5)
        ax1 = fig.add_subplot(2, 3, i+1, xticks=[], yticks=[])
        plt.imshow(Image.open(img_path))
        ax1.set_title(classes[0])
        ax2 = fig.add_subplot(2, 3, 3+i+1)
        ax2.barh(classes, probs)
        ax2.invert_yaxis()


def plot_confusion_matrix(
    correct_classes,
    predicted_classes,
    classes,
    normalize=False,
    title="Confusion matrix",
    cmap=plt.cm.Blues,
):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting normalize=True.
    """

    cnf_matrix = confusion_matrix(correct_classes, predicted_classes)

    if normalize:
        cnf_matrix = cnf_matrix.astype("float") / cnf_matrix.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(25, 10))
    plt.imshow(cnf_matrix, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)  # 45
    plt.yticks(tick_marks, classes)

    thresh = cnf_matrix.max() / 2.0
    for i, j in product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):
        plt.text(
            j,
            i,
            str(round(cnf_matrix[i, j], 2)) if normalize else cnf_matrix[i, j],
            horizontalalignment="center",
            color="white" if cnf_matrix[i, j] > thresh else "black",
        )

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.show()

    return plt
