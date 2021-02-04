import argparse

from torchvision import models

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_arch", type=str, help="Architecture of the torchvision model")
    args = parser.parse_args()

    model = getattr(models, args.model_arch)(pretrained=True)
