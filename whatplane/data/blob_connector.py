# -*- coding: utf-8 -*-

import math
import re
from pathlib import Path

from tqdm import tqdm
import pandas as pd
import requests
from bs4 import BeautifulSoup


class BlobConnector:
    """[summary]
    """
    def __init__(self, info_dict={}):

