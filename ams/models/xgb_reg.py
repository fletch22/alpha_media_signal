import warnings
from typing import List

import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler

from ams.config import logger_factory
from ams.pipes.p_make_prediction.DayPredictionInfo import DayPredictionInfo
from ams.services.ticker_service import get_ticker_info, make_one_hotted, get_ticker_eod_data
from ams.twitter.twitter_ml_utils import transform_to_numpy, get_data_for_predictions, one_hot

logger = logger_factory.create(__name__)


# def get_weights(df):
#     import numpy as np
#     weights = df["days_since_earliest_date"].to_numpy()
#     power = 3
#     weights = np.array([pow(w, power) for w in weights])
#     scaler = MinMaxScaler()
#     results = scaler.fit_transform(weights.reshape(-1, 1))
#
#     return results

# FIXME: 2021-04-18: chris.flesche: Testing
def get_weights(df):
    import numpy as np
    weights = list(df["close"].to_numpy())
    power = 3
    weights = np.array([pow(w, power) for w in weights])
    scaler = MinMaxScaler()
    results = scaler.fit_transform(weights.reshape(-1, 1))

    return results


def predict_with_model(df_test: pd.DataFrame,
                       narrow_cols: List[str],
                       dpi: DayPredictionInfo):
    logger.info("Invoking model prediction ...")

    all_models = dpi.get_models_info()
    all_preds = []
    for mi in all_models:
        X_predict = get_data_for_predictions(df=df_test, narrow_cols=narrow_cols, standard_scaler=mi.standard_scaler)
        pred = mi.model.predict(X_predict)
        all_preds.append(pred)

    return all_preds


def split_train_test(df: pd.DataFrame, tweet_date_str: str) -> (pd.DataFrame, pd.DataFrame):
    logger.info(f"Splitting on tweet date {tweet_date_str}")

    df_train = df[df["date"] < tweet_date_str].copy()
    df_test = df[df["date"] == tweet_date_str].copy()

    max_test = df_test['future_date'].max()
    df_train = df_train[df_train["future_date"] < max_test].copy()

    # FIXME: 2021-04-09: chris.flesche: For testing only. This avoids predicting when
    # we have less than n rows.
    if df_train.shape[0] < 10000:
        return None, None

    logger.info(f"Test data size: {df_test.shape[0]}")
    logger.info(f"Train data size: {df_train.shape[0]}")
    logger.info(f"Oldest date of train data (future_date): {df_train['future_date'].max()}")
    logger.info(f"Oldest date of test data (future_date): {df_test['future_date'].max()}")

    return df_train, df_test

def train_predict(dpi: DayPredictionInfo,
                  label_col: str = "buy_sell",
                  require_balance: bool = True,
                  buy_thresh: float = 0.):
    narrow_cols = list(dpi.df.columns)

    # one hot
    df_all_tickers = get_ticker_info()
    columns = [c for c in dpi.df.columns if str(dpi.df[c].dtype) == "object"]
    omit_cols = {'f22_ticker', 'future_date', 'date', 'purchase_date'}
    columns = list(set(columns) - omit_cols)

    df = make_one_hotted(df=dpi.df, df_all_tickers=df_all_tickers, cols=columns)
    df, _ = one_hot(df=df)

    # df.drop(columns=["f22_ticker"], inplace=True)

    df_train, df_test = split_train_test(df=df, tweet_date_str=dpi.tapp.tweet_date_str)

    if df_test is None or df_test.shape[0] == 0:
        return None, None

    df_train = df_train.fillna(value=0)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="invalid value encountered in true_divide")
        X_train, y_train, standard_scaler, feature_cols = transform_to_numpy(df=df_train,
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

    xgb_args = dict(seed=42, max_depth=4, tree_method='gpu_hist', gpu_id=0)

    if not require_balance:
        num_buy = df_train[df_train["stock_val_change"] > buy_thresh].shape[0]
        num_sell = df_train[df_train["stock_val_change"] <= buy_thresh].shape[0]

        balance_ratio = num_sell / num_buy

        logger.info(f"Train Sell: {num_sell} / Buy: {num_buy}; ratio: {balance_ratio}")

        xgb_args["scale_pos_weight"] = balance_ratio

    # grid_search(X_train=X_train, y_train=y_train)

    xgb_reg = xgb.XGBRegressor(**xgb_args)
    model = xgb_reg.fit(X_train, y_train, sample_weight=get_weights(df=df_train))
    dpi.append_model_info(model=model, standard_scaler=standard_scaler, feature_cols=feature_cols, narrow_cols=narrow_cols)

    return predict_with_model(narrow_cols=narrow_cols,
                              df_test=df_test,
                              dpi=dpi), df_test