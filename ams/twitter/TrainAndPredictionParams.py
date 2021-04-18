from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path

import pandas as pd

import ams.utils.date_utils
from ams.config import logger_factory
from ams.utils import date_utils

logger = logger_factory.create(__name__)


class PredictionMode(Enum):
    RealMoneyStockRecommender = "RealMoneyStockRecommender"
    DevelopmentAndTraining = "DevelopmentAndTraining"


class PredictionMaxRounds(Enum):
    ONE = "one"
    ALL = "all"


class TrainAndPredictionParams:
    df: pd.DataFrame = None
    tweet_date_str: str = None
    num_days_until_purchase = None
    num_hold_days: int = None
    clean_pure_run: bool = None  # NOTE: When True: re-fetches core data (excluding tweets); will run more slowly
    min_train_size: int = None
    min_date_str: str = None  # NOTE: The earliest date (with tweets) we want to predict.
    max_date_str: str = None  # NOTE: The most recent date (with tweets) we want to predict.
    oldest_tweet_date = None  # NOTE: The oldest date (with tweets) we want to train with.
    prediction_mode: PredictionMode = None
    require_balance = None  # NOTE: 2021-03-06: chris.flesche: If True, will remove rows in order to balance the dataset.
    min_price = None

    def __init__(self, df: pd.DataFrame,
                 tweet_date_str: str,
                 num_hold_days: int,
                 min_date_str: str,
                 max_date_str: str,
                 clean_pure_run: bool = False,
                 min_train_size: int = 500,
                 num_days_until_purchase: int = 1,
                 oldest_tweet_date: str = "2020-08-10",
                 prediction_mode: PredictionMode = PredictionMode.DevelopmentAndTraining,
                 require_balance: bool = True,
                 min_price: float = 0.,
                 ):

        self.df = df
        self.tweet_date_str = tweet_date_str
        self.num_days_until_purchase = num_days_until_purchase
        self.num_hold_days = num_hold_days
        self.clean_pure_run = clean_pure_run
        self.min_train_size = min_train_size
        self.min_date_str = min_date_str
        self.max_date_str = max_date_str
        self.oldest_tweet_date = oldest_tweet_date
        self.prediction_mode = prediction_mode
        self.require_balance = require_balance
        self.min_price = min_price

        self.validate_init()

    def validate_init(self):
        assert (self.df is not None
                and self.tweet_date_str is not None
                and self.num_days_until_purchase is not None
                and self.num_hold_days is not None
                and self.clean_pure_run is not None
                and self.min_train_size is not None
                and self.max_date_str is not None
                and self.oldest_tweet_date is not None
                and self.prediction_mode is not None)

    @property
    def dt_start(self):
        return date_utils.parse_std_datestring(self.tweet_date_str)

    @property
    def predict_num_rows(self):
        return self.df[self.df["date"] == self.tweet_date_str].shape[0]

    @property
    def purchase_date_str(self):
        return ams.utils.date_utils.get_next_market_day_no_count_closed_days(date_str=self.tweet_date_str, num_days=self.num_days_until_purchase)

    def is_valid_and_in_range(self, dt_tweet: datetime):
        tweet_dt_tmp = date_utils.get_standard_ymd_format(dt_tweet)
        df_train = self.df[self.df["date"] < tweet_dt_tmp]
        df_predict = self.df[self.df["date"] == tweet_dt_tmp]

        in_range = self.min_date_str <= tweet_dt_tmp <= self.max_date_str

        has_enough_training_rows = df_train.shape[0] > self.min_train_size
        has_prediction_rows = df_predict.shape[0] > 0

        is_valid = False
        if not in_range:
            logger.info(f"Purchase date {tweet_dt_tmp} not in range.")
        elif not has_enough_training_rows:
            logger.info(f"Not enough training rows on {tweet_dt_tmp}")
        elif not has_prediction_rows:
            logger.info(f"Not enough prediction rows on {tweet_dt_tmp}")
        else:
            is_valid = is_good_market_date(dt_tweet) and has_enough_training_rows and has_prediction_rows

        return is_valid, in_range

    def validate_tweet_date_str(self):
        dt = date_utils.parse_std_datestring(self.tweet_date_str)
        return self.is_valid_and_in_range(dt_tweet=dt)

    def subtract_day(self):
        dt_start_tmp = self.dt_start

        is_at_end = False
        while True:
            dt_start_tmp = dt_start_tmp + timedelta(days=-1)
            is_valid, in_range = self.is_valid_and_in_range(dt_tweet=dt_start_tmp)
            if not in_range:
                is_at_end = True
                break
            if is_valid:
                break

        if not is_at_end:
            self.tweet_date_str = date_utils.get_standard_ymd_format(dt_start_tmp)
            logger.info(f"New tweet_date_str: {self.tweet_date_str}")

        return is_at_end


def is_good_market_date(dt, verbose: bool = True):
    result = True
    is_closed, reached_end_of_data = date_utils.is_stock_market_closed(dt)
    if is_closed or reached_end_of_data:
        date_str = date_utils.get_standard_ymd_format(dt)
        if verbose and reached_end_of_data:
            logger.info(f"No can do. Reached end of data on {date_str}")
        if verbose and is_closed:
            logger.info(f"No can do. Market closed on {date_str}")
        result = False
    return result