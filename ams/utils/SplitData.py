import numpy as np
import pandas as pd
from typing import List

class SplitData():
    X_train = None
    y_train = None

    X_test = None
    y_test = None

    df_test_raw = None
    df_test_std = None

    train_cols = None

    has_enough_data = None

    def __init__(self, X_train: np.array,
                 y_train: np.array,
                 X_test: np.array,
                 y_test: np.array,
                 df_test_raw: pd.DataFrame,
                 df_test_std: pd.DataFrame,
                 train_cols: List[str],
                 has_enough_data: bool
                 ):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.df_test_raw = df_test_raw
        self.df_test_std = df_test_std
        self.train_cols = train_cols
        self.has_enough_data = has_enough_data

SplitData(X_train=None,
          y_train=None,
          X_test=None,
          y_test=None)