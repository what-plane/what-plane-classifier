# -*- coding: utf-8 -*-

import math
import re
from pathlib import Path

from tqdm import tqdm
import pandas as pd
import requests
from bs4 import BeautifulSoup


class AirlinersConnector:
    """
    # TODO write docstring for class
    """

    def __init__(self, info_dict={}):
        self._base_url = info_dict["base_url"]
        self._search_ext = info_dict["search_ext"]

        self._proxies = {
            "http": None,
            "https": None,
        }

        self._headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.198 Safari/537.36"
        }

        self._permitted_users = info_dict["permitted_users"]
        self._manufacturer_dict = info_dict["manufacturers"]

    def get_all_results_by_manufacturer(self, ac_manufacturer):

        output_df_list = []
        for user in self._permitted_users["userids"]:
            print(f"Fetching results for {user}...")
            output_df_list.append(self.get_search_results(ac_manufacturer, user))

        out_df = pd.concat(output_df_list, axis=0, ignore_index=True)

        return out_df

    def get_search_results(self, ac_manufacturer, username):

        ac_manufacturer_key = self._manufacturer_dict[ac_manufacturer]["id"]

        soup = self._fetch_search_page(ac_manufacturer_key, username, 1)

        search_info = soup.find("div", class_="ps-v2-header-message").text
        num_matches = pd.to_numeric(
            re.search(r"total of ([\d,]+) matches", search_info).group(1).replace(",", "")
        )

        num_pages = math.ceil(num_matches / 84)

        output_list = []

        output_list.extend(self._extract_results_from_soup(soup))

        for page in tqdm(range(2, num_pages + 1)):
            soup = self._fetch_search_page(ac_manufacturer_key, username, page)
            output_list.extend(self._extract_results_from_soup(soup))

        out_df = pd.DataFrame.from_records(output_list)

        out_df["Username"] = username
        out_df["Manufacturer"] = ac_manufacturer

        return out_df

    def _fetch_search_page(self, ac_manufacturer, user, page):
        requests_url = self._base_url + self._search_ext.format(
            manu=ac_manufacturer, user=user, page=page
        )
        init_page = requests.get(requests_url, proxies=self._proxies, headers=self._headers)
        soup = BeautifulSoup(init_page.text, "html.parser")
        return soup

    def _extract_results_from_soup(self, soup):
        def parse_messy_text(string):
            # re.sub(r"[\s#]", "", string)
            return string.replace(r"\n", "").strip()

        TEMPLATE_DICT = {
            "PhotoId": "",
            "URL": "",
            "Airline": "",
            "ACType": "",
            "Reg": "",
            "MSN": "",
            "Location": "",
            "Country": "",
            "Date": "",
            "Photographer": "",
        }
        output_result = []

        these_results = soup.find_all("div", class_="ps-v2-results-row")

        for result in these_results:
            this_output = TEMPLATE_DICT.copy()

            # Parse Photo Info
            photo_div = result.find(
                "div", class_="ps-v2-results-col-title-half-width ps-v2-results-col-title-photo-id"
            )
            this_output["PhotoId"] = parse_messy_text(photo_div.text).replace("#", "")
            this_output["URL"] = photo_div.a["href"]

            # Parse A/C info
            ac_div = result.find("div", class_="ps-v2-results-col ps-v2-results-col-aircraft")
            ac_info = ac_div.find_all("div", class_="ps-v2-results-display-detail-no-wrapping")
            if len(ac_info) > 1:
                this_output["Airline"] = parse_messy_text(ac_info[0].a.text)
            this_output["ACType"] = parse_messy_text(ac_info[-1].a.text)

            # Parse Reg and ID info
            id_div = result.find("div", class_="ps-v2-results-col ps-v2-results-col-id-numbers")
            id_info = id_div.find_all("div", class_="ps-v2-results-display-detail-no-wrapping")
            for id_entry in id_info:
                if "REG:" in id_entry.text:
                    this_output["Reg"] = parse_messy_text(id_entry.a.text)
                if "MSN:" in id_entry.text:
                    this_output["MSN"] = parse_messy_text(id_entry.a.text)

            # Parse Location and Date info
            loc_date = result.find(
                "div", class_="ps-v2-results-col ps-v2-results-col-location-date"
            )
            loc_date_info = loc_date.find_all(
                "div", class_="ps-v2-results-display-detail-no-wrapping"
            )
            this_output["Location"] = parse_messy_text(loc_date_info[0].text)

            loc_date_a = loc_date_info[1].find_all("a")
            this_output["Country"] = ", ".join([info.text for info in loc_date_a[:-1]])
            this_output["Date"] = parse_messy_text(loc_date_a[-1].text)

            # ua-name
            grapher_div = result.find("div", class_="ua-name")
            this_output["Photographer"] = parse_messy_text(grapher_div.a.text)

            output_result.append(this_output)

        return output_result

    @staticmethod
    def get_image_from_url(base_url, url, outfile_path, proxies, headers):
        img_page_url = base_url[:-1] + url
        photo_page = requests.get(img_page_url, proxies=proxies, headers=headers)
        soup = BeautifulSoup(photo_page.text, "html.parser")
        img_src = soup.find("div", class_="pdp-image-wrapper").find("img")["src"]
        img_src = img_src.split("?")[0]

        photo = requests.get(img_src, proxies=proxies, headers=headers)
        outfile_path.write_bytes(photo.content)

        return
