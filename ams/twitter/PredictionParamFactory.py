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

        pred_params = PredictionParams()
        pred_params.prediction_mode = PredictionMode.DevelopmentAndTraining
        pred_params.min_date_str = min_date_str
        pred_params.predict_date_str = max_date_str
        pred_params.max_date_str = max_date_str
        pred_params.num_hold_days = num_hold_days
        pred_params.df = df

        pred_params.clean_pure_run = False
        pred_params.train_params = TrainParams()

        return pred_params