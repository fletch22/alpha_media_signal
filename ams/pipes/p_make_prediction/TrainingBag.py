import collections
from pathlib import Path
from typing import List, Dict

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

from ams.config import logger_factory

logger = logger_factory.create(__name__)

TrainingFistfull = collections.namedtuple("TrainingFistfull", field_names="model, df_train, df_test")

TRAINING_BAG_REL_PATH = Path("prediction_bucket", "training_bag.dat")


class TrainingBag:
    bag: Dict[str, TrainingFistfull] = {}

    def __init__(self):
        self.bag = {}

    def add_fistfull(self, purchase_date_str: str, model: object, df_train: pd.DataFrame, df_test: pd.DataFrame):
        self.bag[purchase_date_str] = TrainingFistfull(model=model, df_train=df_train, df_test=df_test)

    def get_purchase_dates(self):
        return sorted(self.bag.keys(), reverse=True)

    def get_recent_models(self, purchase_date_str: str, num_models: int) -> List[object]:
        pd_list = self.bag.keys()
        pd_list = sorted(pd_list)
        pd_list = [pd for pd in pd_list if purchase_date_str >= pd]

        avail_models = len(pd_list)
        logger.info(f"Found {avail_models}")
        num_models = num_models if avail_models >= num_models else avail_models

        best_models = []
        for p_date in pd_list[:num_models]:
            logger.info(f"p_date: {p_date}")
            tff = self.bag[p_date]
            best_models.append(tff.model)

        return best_models

    def get_recent_data(self, purchase_date_str: str, num_models: int) -> List[object]:
        pd_list = self.bag.keys()
        pd_list = sorted(pd_list, reverse=True)
        pd_list = [pd for pd in pd_list if purchase_date_str > pd]

        avail_models = len(pd_list)
        logger.info(f"Found {avail_models} dates.")
        num_models = num_models if avail_models >= num_models else avail_models

        best_models = []
        for p_date in pd_list[:num_models]:
            logger.info(f"p_date: {p_date}")
            best_models.append(self.get_data(purchase_date_str=purchase_date_str))

        return best_models

    def get_data(self, purchase_date_str: str) -> (pd.DataFrame, pd.DataFrame, StandardScaler):
        tff = self.bag[purchase_date_str]
        return (tff.df_train, tff.df_test)

    @classmethod
    def persist(cls, training_bag: object, twitter_root: Path):
        p_path = Path(twitter_root, TRAINING_BAG_REL_PATH)
        joblib.dump(training_bag, p_path)

    @classmethod
    def load(cls, twitter_root: Path):
        p_parent_path = Path(twitter_root, TRAINING_BAG_REL_PATH)
        return joblib.load(p_parent_path)