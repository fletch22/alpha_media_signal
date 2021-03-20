from pathlib import Path
from statistics import mean
from typing import List

import pandas as pd

from ams.config import constants, logger_factory
from ams.twitter.twitter_ml_utils import add_nasdaq_roi, remove_ticker_cols
from ams.utils import random_utils

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
    df_nr = add_nasdaq_roi(df=df, num_hold_days=3)

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


def rename_cols(df: pd.DataFrame):
    cols = set(df.columns)

    good_cols = cols.difference(["buy_sell", "purchase_date"])

    ren_dict = dict()
    for ndx, c in enumerate(good_cols):
        ren_dict[c] = f"{random_utils.get_random_string()}"

    df.rename(columns=ren_dict, inplace=True)

    return df


def test_prep_for_auto_ml():
    df = pd.read_parquet(constants.SAMPLE_TWEET_STOCK_TRAIN_DF_PATH)

    cols = remove_ticker_cols(list(df.columns))
    df = df[cols].copy()

    df = rename_cols(df=df)

    df.dropna(subset=["purchase_date"], inplace=True)
    df = df.fillna(value=0)

    date_strs = list(df["purchase_date"].unique())

    num_dates = len(date_strs)

    num_train = int(.75 * num_dates)
    num_test = int(.15 * num_dates)

    train_max_dt_str = sorted(date_strs)[num_train]
    test_max_dt_str = sorted(date_strs)[num_train + num_test]

    df.sort_values(by=["purchase_date"], inplace=True)

    df.loc[df["purchase_date"] <= train_max_dt_str, "ds"] = "TRAIN"
    df.loc[(df["purchase_date"] > train_max_dt_str) & (df["purchase_date"] <= test_max_dt_str), "ds"] = "TEST"
    df.loc[(df["purchase_date"] >= train_max_dt_str), "ds"] = "VALIDATE"

    df.drop(columns=["purchase_date"], inplace=True)
    df.rename(columns={"buy_sell": "bs"}, inplace=True)

    file_path = constants.AUTO_ML_PATH
    # if file_path.exists():
    #     file_path.unlink()
    logger.info(file_path)
    df.to_csv(str(file_path), header=True)