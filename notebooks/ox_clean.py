import os
import copy
import shutil

data_dir = "../data/raw/data"
source = "../data/raw/data/images/"
train = "../data/raw/data/images/train/"
test = "../data/raw/data/images/test/"
valid = "../data/raw/data/images/valid/"

test_images = []
train_images = []
val_images = []


def get_images_names():

    with open(data_dir+'images_family_test.txt', 'r') as f:
        for line in f:
            test_images.append(line.split()[0]+".jpg")

    with open(data_dir+'images_family_train.txt', 'r') as f:
        for line in f:
            train_images.append(line.split()[0]+".jpg")

    with open(data_dir+'images_family_trainval.txt', 'r') as f:
        for line in f:
            val_images.append(line.split()[0]+".jpg")


def move_files():

    files = os.listdir(source)

    for f in files:
        if f in test_images:
            shutil.move(source+f, test)
        elif f in train_images:
            shutil.move(source+f, train)
        elif f in val_images:
            shutil.move(source+f, valid)
