import collections
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler

from ams.pipes.p_make_prediction.TrainingBag import TrainingBag
from ams.twitter.TrainAndPredictionParams import TrainAndPredictionParams

ModelInfo = collections.namedtuple("ModelInfo",field_names="model, standard_scaler")

class DayPredictionInfo:
    df = None
    tapp = None
    training_bag = None
    output_path = None
    models_info = None
    max_models = None

    def __init__(self, tapp: TrainAndPredictionParams,
                 training_bag: TrainingBag,
                 output_path: Path,
                 max_models: int):
        self.tapp = tapp
        self.training_bag = training_bag
        self.output_path = output_path
        self.max_models = max_models
        self.models_info = collections.deque()

    def set_df(self, df: pd.DataFrame):
        self.df = df

    def append_model_info(self, model: object, standard_scaler: StandardScaler):
        mi = ModelInfo(model=model, standard_scaler=standard_scaler)
        self.models_info.appendleft(mi)
        if len(self.models_info) > self.max_models:
            self.models_info.pop()

    def get_models_info(self):
        return list(self.models_info)