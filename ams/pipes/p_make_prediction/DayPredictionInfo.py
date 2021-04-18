import collections
from pathlib import Path
from typing import List, Set

import pandas as pd
import xgboost
from sklearn.preprocessing import StandardScaler

from ams.pipes.p_make_prediction.TrainingBag import TrainingBag
from ams.twitter.TrainAndPredictionParams import TrainAndPredictionParams

ModelInfo = collections.namedtuple("ModelInfo", field_names="model, standard_scaler")
ImportantFeatures = collections.namedtuple("ImportantFeatures", field_names="important_features, feature_columns")


class DayPredictionInfo:
    df = None
    tapp = None
    training_bag = None
    output_path = None
    models_info = None
    max_models = None
    important_feats = None

    def __init__(self, tapp: TrainAndPredictionParams,
                 training_bag: TrainingBag,
                 output_path: Path,
                 max_models: int):
        self.tapp = tapp
        self.training_bag = training_bag
        self.output_path = output_path
        self.max_models = max_models
        self.models_info = collections.deque()
        self.important_feats: List[ImportantFeatures] = []
        self.narrow_cols: Set[str] = set()

    def set_df(self, df: pd.DataFrame):
        self.df = df

    def append_model_info(self,
                          model: xgboost,
                          standard_scaler: StandardScaler,
                          feature_cols: List[str],
                          narrow_cols: List[str]):
        self.narrow_cols = set(narrow_cols)

        important_features = set()
        fc = set()
        for ndx, feat_imp in enumerate(model.feature_importances_):
            if feat_imp > 0.:
                important_features.add(feat_imp)
                fc.add(feature_cols[ndx])

        im_feat = ImportantFeatures(important_features=important_features, feature_columns=fc)
        self.important_feats.append(im_feat)

        mi = ModelInfo(model=model, standard_scaler=standard_scaler)
        self.models_info.appendleft(mi)
        if len(self.models_info) > self.max_models:
            self.models_info.pop()

    def get_models_info(self):
        return list(self.models_info)