from datetime import datetime

import pandas as pd

from ams.services.twitter_service import EARLIEST_TWEET_DATE_STR
from ams.twitter.TrainAndPredictionParams import TrainAndPredictionParams, PredictionMode
from ams.utils import date_utils


class TrainAndPredictionParamFactory:

    @classmethod
    def create_generic_trainer(self, df: pd.DataFrame, num_hold_days: int,
                               max_date_str: str = None, min_date_str: str = None,
                               require_balance: bool = True):
        if min_date_str is None:
            min_date_str = EARLIEST_TWEET_DATE_STR

        if max_date_str is None:
            max_date_str = TrainAndPredictionParamFactory.get_today_str()

        pred_params = TrainAndPredictionParams(df=df,
                                               prediction_mode=PredictionMode.DevelopmentAndTraining,
                                               tweet_date_str=max_date_str,
                                               num_days_until_purchase=1,
                                               min_date_str=min_date_str,
                                               max_date_str=max_date_str,
                                               num_hold_days=num_hold_days,
                                               require_balance=require_balance)

        return pred_params

    @classmethod
    def get_today_str(cls):
        return date_utils.get_standard_ymd_format(datetime.now())

    @classmethod
    def create_mid_january_trainer(cls, df: pd.DataFrame, num_hold_days):
        date_str = "2021-01-13"
        return TrainAndPredictionParamFactory.create_generic_trainer(df=df, num_hold_days=num_hold_days, max_date_str=date_str)

    @classmethod
    def create_october_trainer(cls, df: pd.DataFrame, num_hold_days):
        date_str = "2020-10-31"
        return TrainAndPredictionParamFactory.create_generic_trainer(df=df, num_hold_days=num_hold_days, max_date_str=date_str)
