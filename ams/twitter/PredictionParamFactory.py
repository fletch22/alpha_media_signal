from datetime import datetime

import pandas as pd

from ams.services.twitter_service import EARLIEST_TWEET_DATE_STR
from ams.twitter.PredictionParams import PredictionParams, PredictionMode, TrainParams
from ams.utils import date_utils


class PredictionParamFactory:

    @classmethod
    def create_generic_trainer(self, df: pd.DataFrame, num_hold_days, max_date_str: str = None, min_date_str: str = None):
        if min_date_str is None:
            min_date_str = EARLIEST_TWEET_DATE_STR

        if max_date_str is None:
            max_date_str = date_utils.get_standard_ymd_format(datetime.now())

        pred_params = PredictionParams(df=df,
                                   prediction_mode=PredictionMode.DevelopmentAndTraining,
                                   tweet_date_str=max_date_str,
                                   num_days_until_purchase=1,
                                   min_date_str=min_date_str,
                                   max_date_str=max_date_str,
                                   num_hold_days=num_hold_days)

        return pred_params

    @classmethod
    def create_mid_january_trainer(cls, df: pd.DataFrame, num_hold_days):
        date_str = "2021-01-13"
        return PredictionParamFactory.create_generic_trainer(df=df, num_hold_days=num_hold_days, max_date_str=date_str)