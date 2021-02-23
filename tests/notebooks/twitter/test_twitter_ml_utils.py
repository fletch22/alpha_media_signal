from pathlib import Path
from statistics import mean
from typing import List

import pandas as pd

from ams.config import constants, logger_factory
from ams.twitter.twitter_ml_utils import add_nasdaq_roi_new

logger = logger_factory.create(__name__)

def calc_return(rois: List[str]):
    ini_inv = 1000
    total = ini_inv
    for r in rois:
        total = (total * r) + total

    return total


def test_add_nasdaq_roi():
    # Arrange
    rows = [
        {"f22_ticker": "IBM", "date": "2020-08-10", "roi": .01},
        {"f22_ticker": "IBM", "date": "2020-08-11", "roi": .02},
        {"f22_ticker": "IBM", "date": "2020-08-12", "roi": .03},
        {"f22_ticker": "IBM", "date": "2020-08-13", "roi": .04},
        {"f22_ticker": "IBM", "date": "2020-08-14", "roi": .05},
        {"f22_ticker": "IBM", "date": "2020-08-17", "roi": .06},
        {"f22_ticker": "IBM", "date": "2020-08-18", "roi": .07},
        {"f22_ticker": "IBM", "date": "2020-08-19", "roi": .08},
        {"f22_ticker": "IBM", "date": "2020-08-20", "roi": .09},
        {"f22_ticker": "IBM", "date": "2020-08-21", "roi": .1},
    ]
    df = pd.DataFrame(rows)
    df_nr = add_nasdaq_roi_new(df=df, num_hold_days=3)

    # Act
    # Assert


def test_read_funky():
    file_path_str = str(Path(constants.TWITTER_MODEL_PREDICTION_DIR_PATH, "performance_2.txt"))
    all_roi = []
    with open(file_path_str, "r+") as rf:
        all_lines = rf.readlines()
        for al in all_lines:
            al = al.replace("\n", "")
            start_token = "Roi 1: "
            if al.startswith(start_token):
                rois_raw_list = al.split(start_token)
                rois_raw = rois_raw_list[1]
                if not rois_raw.startswith("None"):
                    logger.info(rois_raw[:6])
                    all_roi.append(float(rois_raw[:6]))

    logger.info(f"Average roi: {mean(all_roi)}")

