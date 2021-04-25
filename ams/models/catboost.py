import warnings
from typing import List, Tuple, Optional

import numpy
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from ams.config import logger_factory, constants
from ams.pipes.p_make_prediction.DayPredictionInfo import DayPredictionInfo
from ams.services import twitter_service
from ams.twitter.twitter_ml_utils import get_data_for_predictions

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
    power = 2
    return np.array([pow(w, power) for w in weights])


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


def transform_to_numpy(df: pd.DataFrame, narrow_cols: List[str], label_col: str = "buy_sell", require_balance: bool = True) -> \
    Tuple[Optional[numpy.array], Optional[numpy.array], Optional[List[str]]]:
    feature_cols = twitter_service.get_feature_columns(narrow_cols)

    X_train_raw, y_train = twitter_service.split_df_for_learning(df=df, label_col=label_col, train_cols=feature_cols, require_balance=require_balance)

    if X_train_raw is None or X_train_raw.shape[0] == 0:
        return None, None, None

    # standard_scaler = StandardScaler()
    # standard_scaler = standard_scaler.fit(X_train_raw)
    # X_train = standard_scaler.transform(X_train_raw)

    X_train = X_train_raw

    return X_train, y_train, feature_cols


def train_predict(dpi: DayPredictionInfo,
                  label_col: str = "buy_sell",
                  require_balance: bool = True,
                  buy_thresh: float = 0.):
    narrow_cols = list(dpi.df.columns)

    df = dpi.df
    # one hot
    # df_all_tickers = get_ticker_info()
    columns = [c for c in dpi.df.columns if str(dpi.df[c].dtype) == "object"]
    omit_cols = {'f22_ticker', 'future_date', 'date', 'purchase_date'}
    cat_columns = list(set(columns) - omit_cols)
    # for c in columns:
    #     logger.info(c)
    #
    # df = make_one_hotted(df=dpi.df, df_all_tickers=df_all_tickers, cols=columns)
    # df, _ = one_hot(df=df)

    # df.drop(columns=["f22_ticker"], inplace=True)

    df_train, df_test = split_train_test(df=df, tweet_date_str=dpi.tapp.tweet_date_str)

    if df_test is None or df_test.shape[0] == 0:
        return None, None

    df_train = df_train.fillna(value=0)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="invalid value encountered in true_divide")
        X_train, y_train, feature_cols = transform_to_numpy(df=df_train,
                                              narrow_cols=narrow_cols,
                                              label_col=label_col,
                                              require_balance=require_balance)

    if X_train is None or X_train.shape[0] == 0 or y_train is None:
        logger.info("Not enough training data.")
        return None, None

    # cat_args = dict(loss_function='RMSE', iterations=50)
    # if not require_balance:
    #     num_buy = df_train[df_train["stock_val_change"] > buy_thresh].shape[0]
    #     num_sell = df_train[df_train["stock_val_change"] <= buy_thresh].shape[0]
    #
    #     balance_ratio = num_sell / num_buy
    #
    #     logger.info(f"Train Sell: {num_sell} / Buy: {num_buy}; ratio: {balance_ratio}")
    #
    #     cat_args["scale_pos_weight"] = balance_ratio

    # grid_search(X_train=X_train, y_train=y_train)

    import catboost as cb
    model = cb.CatBoostRegressor(loss_function='RMSE', iterations=5, silent=True, train_dir=constants.CATBOOST_TRAIN_DIR)

    import numpy as np
    df_skinny = df[feature_cols].copy()
    categorical_features_indices = np.where(df_skinny.dtypes != np.float)[0]

    # grid = {'learning_rate': [0.03, 0.1],
    #         'depth': [4, 6, 10],
    #         'l2_leaf_reg': [1, 3, 5, 7, 9]}
    # results: {'depth': 14, 'l2_leaf_reg': 7, 'learning_rate': 0.1}
    #
    # grid = {'learning_rate': [0.01, 0.03, 0.1],
    #         'depth': [6, 10, 12, 14],
    #         'l2_leaf_reg': [1, 3, 5, 7, 9]}
    # results = {'depth': 6, 'l2_leaf_reg': 1, 'learning_rate': 0.1}
    # grid_search_result = model.grid_search(grid,
    #                                        X=X_train,
    #                                        y=y_train,
    #                                        plot=True)
    # logger.info(grid_search_result)
    # raise Exception("foo")

    model.fit(X_train, y_train, cat_features=categorical_features_indices, sample_weight=get_weights(df=df_train))

    dpi.append_model_info(model=model, standard_scaler=None, feature_cols=feature_cols, narrow_cols=narrow_cols)

    df_test.fillna("", inplace=True)
    return predict_with_model(narrow_cols=narrow_cols,
                              df_test=df_test,
                              dpi=dpi), df_test