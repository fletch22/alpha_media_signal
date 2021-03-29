import collections
import warnings
from typing import List

import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from ams.config import logger_factory
from ams.pipes.p_make_prediction.TrainingBag import TrainingBag
from ams.twitter.twitter_ml_utils import transform_to_numpy, get_data_for_predictions

logger = logger_factory.create(__name__)


def get_weights(df):
    import numpy as np
    weights = df["days_since_earliest_date"].to_numpy()
    weights = np.array([pow(w, 3) for w in weights])
    scaler = MinMaxScaler()
    results = scaler.fit_transform(weights.reshape(-1, 1))

    return results


def predict_with_model(df_test: pd.DataFrame,
                       narrow_cols: List[str],
                       standard_scaler: StandardScaler,
                       model: object):
    X_predict = get_data_for_predictions(df=df_test, narrow_cols=narrow_cols, standard_scaler=standard_scaler)

    logger.info("Invoking model prediction ...")

    return model.predict(X_predict)


def train_predict(df_train: pd.DataFrame,
                  df_test: pd.DataFrame,
                  narrow_cols: List[str],
                  training_bag: TrainingBag,
                  purchase_date_str: str,
                  label_col: str = "buy_sell",
                  require_balance: bool = True,
                  buy_thresh: float = 0.):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="invalid value encountered in true_divide")
        X_train, y_train, standard_scaler = transform_to_numpy(df=df_train,
                                                               narrow_cols=narrow_cols,
                                                               label_col=label_col,
                                                               require_balance=require_balance)

    if X_train is None or X_train.shape[0] == 0 or y_train is None:
        logger.info("Not enough training data.")
        return None, None

    # TODO: 2021-03-24: chris.flesche: Experimental
    # xgb_args = dict(seed=42, reg_lambda=2,
    #                 tree_method='gpu_hist', gpu_id=0, learning_rate=1.0,
    #                 max_depth=7, n_estimators=110)

    # NOTE: 2021-03-24: chris.flesche: 1.1% roi
    xgb_args = dict(seed=42, max_depth=8, tree_method='gpu_hist', gpu_id=0)

    if not require_balance:
        num_buy = df_train[df_train["stock_val_change"] > buy_thresh].shape[0]
        num_sell = df_train[df_train["stock_val_change"] <= buy_thresh].shape[0]

        balance_ratio = num_sell / num_buy

        logger.info(f"Train Sell: {num_sell} / Buy: {num_buy}; ratio: {balance_ratio}")

        xgb_args["scale_pos_weight"] = balance_ratio

    # grid_search(X_train=X_train, y_train=y_train)

    xgb_reg = xgb.XGBRegressor(**xgb_args)

    model = xgb_reg.fit(X_train, y_train, sample_weight=get_weights(df=df_train))

    training_bag.add_fistfull(purchase_date_str=purchase_date_str, model=model, df_train=df_train, df_test=df_test)

    return predict_with_model(narrow_cols=narrow_cols,
                              df_test=df_test,
                              standard_scaler=standard_scaler,
                              model=model)