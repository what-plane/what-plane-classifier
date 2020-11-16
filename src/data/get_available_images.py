# -*- coding: utf-8 -*-
import json
from pathlib import Path

import pandas as pd

from airlinersnet import airlinersConnector
from actype_classes import ACTYPE_DICT

if __name__ == '__main__':
    BASE_PATH = Path(".")
    with open(BASE_PATH / "src/data/airlinersnet.json") as json_data_file:
        info_dict = json.load(json_data_file)

    ac = airlinersConnector(info_dict)

    output_df_list = []

    for ac_manu in info_dict["manufacturers"].keys():
        out_filepath = BASE_PATH / "/".join(["data", "raw", f"{ac_manu}.pickle"])
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
        all_df.loc[all_df["ACType"].str.contains(class_info["match_regex"], regex=True), "Class"] = ac_class
