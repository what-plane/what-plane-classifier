# -*- coding: utf-8 -*-
import json
from pathlib import Path
import math
from functools import partial

from tqdm.contrib.concurrent import process_map
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from airlinersnet import airlinersConnector
from actype_classes import ACTYPE_DICT

SEED = 42
np.random.seed(seed=SEED)


def sample_class(class_df, n_samples=1000, train_test_val_split=(0.6, 0.2, 0.2)):

    THRESHOLD_DATE = pd.Timestamp("2000-01-01 00:00:00")

    if len(class_df) < n_samples:
        selected_index = class_df.index.tolist()

    elif len(class_df[(class_df["Date"] < THRESHOLD_DATE)]) > 0.25 * n_samples:
        pre_threshold = math.floor(0.25 * n_samples)
        post_threshold = math.ceil(0.75 * n_samples)

        selected_index = (
            class_df[class_df["Date"] >= THRESHOLD_DATE]
            .sample(post_threshold, random_state=SEED)
            .index.tolist()
        )
        selected_index.extend(
            class_df[class_df["Date"] < THRESHOLD_DATE]
            .sample(pre_threshold, random_state=SEED)
            .index.tolist()
        )

    else:
        selected_index = class_df.sample(n_samples, random_state=SEED).index.tolist()

    class_df = class_df.loc[selected_index, :]

    train_size, test_size, val_size = train_test_val_split

    test_val_size = test_size + val_size

    train_idx, test_val_idx = train_test_split(
        class_df.index.values, train_size=train_size, test_size=test_val_size, random_state=SEED
    )

    test_idx, val_idx = train_test_split(
        test_val_idx,
        train_size=test_size / test_val_size,
        test_size=val_size / test_val_size,
        random_state=SEED,
    )

    class_df.loc[train_idx, "Set"] = "train"
    class_df.loc[test_idx, "Set"] = "test"
    class_df.loc[val_idx, "Set"] = "valid"

    return class_df


def create_dir(path):
    if not path.exists():
        path.mkdir()
    return


def fetch_photo(row, raw_path, base_url, headers, proxies):
    img_path = raw_path / "/".join([row["Set"], row["Class"], str(row["PhotoId"]) + ".jpg"])
    if not img_path.exists():
        airlinersConnector.get_image_from_url(base_url, row["URL"], img_path, proxies, headers)

    return


if __name__ == "__main__":
    __spec__ = None

    BASE_PATH = Path(".")
    DATA_PATH = Path(".") / "data"
    with open(BASE_PATH / "src/data/airlinersnet.json") as json_data_file:
        info_dict = json.load(json_data_file)

    ac = airlinersConnector(info_dict)

    output_df_list = []

    for ac_manu in info_dict["manufacturers"].keys():
        out_filepath = DATA_PATH / "/".join(["raw", f"{ac_manu}.pickle"])
        if out_filepath.exists():
            output_df = pd.read_pickle(out_filepath)
        else:
            print(f"Fetching data for {ac_manu}...")
            output_df = ac.get_all_results_by_manufacturer(ac_manu)
            print(f"Saving data for {ac_manu}...")
            output_df.to_pickle(out_filepath)

        output_df_list.append(output_df)

    all_df = pd.concat(output_df_list, axis=0, ignore_index=True)

    all_df["PhotoId"] = pd.to_numeric(all_df["PhotoId"]).astype("Int32")

    all_df["Class"] = "Unknown"

    for ac_class, class_info in ACTYPE_DICT.items():
        this_class_loc = all_df["ACType"].str.contains(class_info["match_regex"], regex=True)
        all_df.loc[this_class_loc, "Class"] = ac_class

    all_df["Date"] = pd.to_datetime(all_df["Date"], errors="coerce")

    all_df = all_df[all_df["Class"] != "Unknown"].dropna(axis=0, how="any")

    selected_samples = (
        all_df.groupby("Class").apply(sample_class, n_samples=1000).reset_index(drop=True)
    )

    RAW_IMG_PATH = DATA_PATH / "/".join(["raw", "airlinersnet"])

    create_dir(RAW_IMG_PATH)

    for dataset in selected_samples["Set"].unique():
        folder_path = RAW_IMG_PATH / dataset
        create_dir(folder_path)

    for dataset in selected_samples["Set"].unique():
        for ac_class in ACTYPE_DICT.keys():
            folder_path = RAW_IMG_PATH / "/".join([dataset, ac_class])
            create_dir(folder_path)

    img_to_fetch = selected_samples.to_dict(orient="records")

    fetch_photo_func = partial(
        fetch_photo,
        raw_path=RAW_IMG_PATH,
        base_url=ac._base_url,
        headers=ac._headers,
        proxies=ac._proxies,
    )

    process_map(fetch_photo_func, img_to_fetch)

    def check_img_exists(row):
        img_path = RAW_IMG_PATH / "/".join(
            [row["Set"], str(row["Class"]), str(row["PhotoId"]) + ".jpg"]
        )
        return img_path.exists()

    # Check deleted photos
    selected_samples["StillExists"] = selected_samples.apply(check_img_exists, axis=1)