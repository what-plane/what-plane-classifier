import argparse
from pathlib import Path

from torch import hub
from torchvision import models

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_arch", type=str, help="Architecture of the torchvision model")
    parser.add_argument("torch_hub", type=str, help="Path to set as torch hub")
    args = parser.parse_args()

    hub_path = Path(args.torch_hub)
    hub.set_dir(str(hub_path.resolve()))
    model = getattr(models, args.model_arch)(pretrained=True)
