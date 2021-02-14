from datetime import datetime, timedelta

import pandas as pd

from ams.config import logger_factory
from ams.utils import date_utils

logger = logger_factory.create(__name__)


class TrainParams:
    min_price: float = None
    max_price: float = None
    balance_labels: bool = True


class PredictionMode:
    RealMoneyStockRecommender = "RealMoneyStockRecommender"
    DevelopmentAndTraining = "DevelopmentAndTraining"


class PredictionParams:
    df: pd.DataFrame = None
    predict_date_str: str = None
    num_hold_days: int = 1
    clean_pure_run: bool = True  # NOTE: When True: re-fetches core data (excluding tweets); will run more slowly
    train_params: TrainParams = None
    persist_predictions: bool = True
    min_train_size: int = 500
    min_date_str: str = None  # NOTE: The earliest date (with tweets) we want to predict.
    max_date_str: str = None  # NOTE: The most recent date (with tweets) we want to predict.
    oldest_tweet_date = "2020-08-10"  # NOTE: The oldest date (with tweets) we want to train with.
    prediction_mode: PredictionMode = PredictionMode.DevelopmentAndTraining

    @property
    def dt_start(self):
        return date_utils.parse_std_datestring(self.predict_date_str)

    @property
    def predict_num_rows(self):
        return self.df[self.df["date"] == self.predict_date_str].shape[0]

    def is_valid_dt(self, dt_tmp: datetime):
        pred_date_tmp_str = date_utils.get_standard_ymd_format(dt_tmp)
        df_train = self.df[self.df["date"] < pred_date_tmp_str]
        df_predict = self.df[self.df["date"] == pred_date_tmp_str]

        in_range = self.min_date_str <= pred_date_tmp_str <= self.max_date_str

        has_enough_training_rows = df_train.shape[0] > self.min_train_size
        has_prediction_rows = df_predict.shape[0] > 0

        if not in_range:
            logger.info(f"Prediction date {pred_date_tmp_str} not in range.")
        elif not has_enough_training_rows:
            logger.info(f"Not enough training rows on {pred_date_tmp_str}")
        elif not has_prediction_rows:
            logger.info(f"Not enough prediction rows on {pred_date_tmp_str}")

        return in_range and is_good_market_date(dt_tmp) and has_enough_training_rows and has_prediction_rows

    def validate_prediction_date_str(self):
        dt = date_utils.parse_std_datestring(self.predict_date_str)
        return self.is_valid_dt(dt_tmp=dt)

    def subtract_day(self):
        dt_start_tmp = self.dt_start + timedelta(days=-1)

        min_dt = date_utils.parse_std_datestring(self.min_date_str)

        is_valid = True
        while not self.is_valid_dt(dt_tmp=dt_start_tmp):
            dt_start_tmp = dt_start_tmp + timedelta(days=-1)
            if dt_start_tmp < min_dt:
                is_valid = False
                break

        self.predict_date_str = date_utils.get_standard_ymd_format(dt_start_tmp)

        return is_valid


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