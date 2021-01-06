from pathlib import Path

import pandas as pd


def check_img_exists(row, raw_image_path=Path(".")):
    img_path = raw_image_path / "/".join(
        [row["Set"], str(row["Class"]), str(row["PhotoId"]) + ".jpg"]
    )
    return img_path.exists()


if __name__ == "__main__":

    DATASET_NAME = "airlinersnet"

    BASE_PATH = Path(".")
    RAW_IMG_PATH = BASE_PATH / "/".join(["data", "raw", DATASET_NAME])

    selected_samples = pd.read_pickle(RAW_IMG_PATH / "airlinersnet_catalog.pkl")

    # Check deleted photos
    selected_samples["StillExists"] = selected_samples.apply(
        check_img_exists, raw_image_path=RAW_IMG_PATH, axis=1
    )

    selected_samples.to_pickle(RAW_IMG_PATH / "airlinersnet_catalog.pkl")
