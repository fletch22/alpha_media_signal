import time
from datetime import datetime
from enum import Enum

import pandas as pd

from ams.config import constants, logger_factory
from ams.utils import date_utils

logger = logger_factory.create(__name__)


class RatingMomentum(Enum):
    Reiterated = "Reiterated"
    Upgraded = "Upgraded"
    Initiated = "Initiated"
    Downgraded = "Downgraded"
    Assigned = "Assigned"
    Unrated = "Unrated"


# NOTE: 2021-04-04: chris.flesche: Rating: 1 == buy, 0==hold, -1==sell; This attempts to combine
# a cat field with the numeric field; the intent is to show the direction of movement
def assign_rating_momentum(rating: str, rating_initiating):
    rm = rating
    if rating_initiating == RatingMomentum.Reiterated.name:
        rm = 0
    elif rating_initiating == RatingMomentum.Upgraded.name:
        rm = 1
    elif rating_initiating == RatingMomentum.Downgraded.name:
        if rating == 0:
            rm = -1
        elif rating == -1:
            rm = -1

    return rm


def get_timestamp(date_str: str):
    dt = date_utils.parse_std_datestring(date_str)
    return time.mktime(dt.timetuple())


def get_age_in_days(timestamp_1: float, dt: datetime):
    dt_1 = datetime.fromtimestamp(timestamp_1)

    return (dt - dt_1).days


def add_rolling_values(df: pd.DataFrame):
    df.loc[:, "rating_momentum"] = df.apply(lambda row: assign_rating_momentum(row["rating"], row["rating_initiating"]), axis=1)

    all_dfs = []
    with pd.option_context('mode.chained_assignment', None):
        df_grouped = df.groupby(by=["ticker"])

        for _, df_g in df_grouped:
            df_g.sort_values(by=["rating_date"], inplace=True)
            earliest_date_str = df_g["rating_date"].min()
            oldest_date_str = df_g["rating_date"].max()
            num_days = date_utils.get_days_between(date_str_1=earliest_date_str, date_str_2=oldest_date_str)
            num_days = 2 if num_days < 2 else num_days

            df_g.loc[:, "rating_momentum"] = df_g["rating_momentum"].rolling(window=num_days, min_periods=1).mean().astype("float64")
            df_g.loc[:, "target_price"] = df_g["target_price"].rolling(window=num_days, min_periods=1).mean().astype("float64")
            df_g.loc[:, "tr_rating_roi"] = df_g["tr_rating_roi"].rolling(window=num_days, min_periods=1).mean().astype("float64")
            df_g.loc[:, "rank"] = df_g["rank"].rolling(window=num_days, min_periods=1).mean().astype("float64")
            df_g.loc[:, "rating"] = df_g["rating"].rolling(window=num_days, min_periods=1).mean().astype("float64")
            all_dfs.append(df_g)

    df = pd.concat(all_dfs, axis=0)

    return df[["ticker", "target_price", "rating_date", "rating_momentum", "tr_rating_roi", "rank", "rating"]]


def agg_tipranks(df_stocks: pd.DataFrame):
    df = pd.read_parquet(constants.TIP_RANKS_STOCK_DATA_PATH)
    df = add_rolling_values(df=df)
    df.rename(columns={"rating_date": "date"}, inplace=True)
    df.sort_values(by=["ticker", "date"], inplace=True)

    df_merged = pd.merge(df_stocks, df, how="left", on=["ticker", "date"])
    df_merged.sort_values(by=["ticker", "date"], inplace=True)

    df_grouped = df_merged.groupby(by=["ticker"])

    all_dfs = []
    for _, df_g in df_grouped:
        df_g.sort_values(by=["date"], inplace=True)
        df_g.loc[:, "rating_momentum"] = df_g["rating_momentum"].ffill()
        df_g.loc[:, "tr_rating_roi"] = df_g["tr_rating_roi"].ffill()
        df_g.loc[:, "rank"] = df_g["rank"].ffill()
        df_g.loc[:, "rating"] = df_g["rating"].ffill()
        df_g.loc[:, "target_price"] = df_g["target_price"].ffill()
        all_dfs.append(df_g)

    df = pd.concat(all_dfs, axis=0)

    return df