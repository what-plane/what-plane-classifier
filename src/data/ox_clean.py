import os
import copy
import json
import shutil

data_dir = "../../data/raw/data/"
source = "../../data/raw/data/images/"

test_images = []
train_images = []
val_images = []
families = []


with open('ox_class_mapping.json') as f:
    class_dict = json.load(f)

with open('ox_folder_mapping.json') as f:
    family_dict = json.load(f)


def get_family_names():
    with open(data_dir+'families.txt', 'r') as f:
        for line in f:
            families.append(line)


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


def create_dataset_folders(folder):
    folder = os.mkdir(os.path.join(source, folder))
    return str(folder)


train = "../../data/raw/data/images/train/"
test = "../../data/raw/data/images/test/"
valid = "../../data/raw/data/images/valid/"

folders = [train, test, valid]


def move_files():

    files = os.listdir(source)

    for f in files:
        if f in test_images:
            shutil.move(source+f, test)
        elif f in train_images:
            shutil.move(source+f, train)
        elif f in val_images:
            shutil.move(source+f, valid)


def create_family_folders():
    for folder in folders:
        for family in family_dict.keys():
            os.mkdir(os.path.join(os.path.abspath(folder), str(family)))


def move_images_to_folders():
    for folder in folders:
        files = os.listdir(folder)
        for file in files:
            if file in class_dict.keys():
                family = class_dict[file]
                for k, v in family_dict.items():
                    if family == v:
                        shutil.move(folder+file, folder+str(k))


def remove_unused_images():
    for folder in folders:
        files = os.listdir(folder)
        for file in files:
            if file.endswith(".jpg"):
                os.remove(folder+file)


if __name__ == "__main__":
    get_family_names()
    get_images_names()
    create_dataset_folders("train")
    create_dataset_folders("test")
    create_dataset_folders("valid")
    move_files()
    create_family_folders()
    move_images_to_folders()
    remove_unused_images()
