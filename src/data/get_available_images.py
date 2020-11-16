# -*- coding: utf-8 -*-
import json
from pathlib import Path

from airlinersnet import airlinersConnector

if __name__ == '__main__':
    BASE_PATH = Path(".")
    with open(BASE_PATH / "src/data/airlinersnet.json") as json_data_file:
        info_dict = json.load(json_data_file)

    ac = airlinersConnector(info_dict)

    for ac_manu in info_dict["manufacturers"].keys():
        print(f"Fetching data for {ac_manu}...")
        output_df = ac.get_all_results_by_manufacturer(ac_manu)
        print(f"Saving data for {ac_manu}...")
        output_df.to_pickle(f"{ac_manu}.pickle")

