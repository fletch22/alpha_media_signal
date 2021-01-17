from random import shuffle
from typing import List
import pandas as pd

from ams.config import constants
from ams.notebooks.twitter.twitter_ml_utils import add_nasdaq_roi_new


def test_():
    # Arrange
    rois = [.05, .05, .05]
    # Act

    # for i in range(10):
    #     shuffle(rois)
    #     print(f"return: {calc_return(rois)}")

    print((calc_return(rois) - 1000) / 1000)

    # Assert


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
    print(df_nr.head())

    # Assert



