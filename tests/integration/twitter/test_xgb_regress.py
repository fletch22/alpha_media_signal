from datetime import datetime
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, KFold
from sklearn.model_selection import train_test_split

from ams.config import constants, logger_factory
from ams.config.constants import xgb_defaults
from ams.twitter.pred_perf_testing import get_days_roi_from_prediction_table
from ams.twitter.twitter_ml_utils import transform_to_numpy, get_data_for_predictions

logger = logger_factory.create(__name__)


def test_regress_boston_data():
    # read data in

    boston = load_boston()
    x, y = boston.data, boston.target
    X_train, xtest, y_train, ytest = train_test_split(x, y, test_size=0.15)

    xgbr = xgb_defaults.XGBRegressor(verbosity=0)
    xgbr.fit(X_train, y_train)

    score = xgbr.score(X_train, y_train)
    print("Training score: ", score)

    scores = cross_val_score(xgbr, X_train, y_train, cv=10)
    print("Mean cross-validation score: %.2f" % scores.mean())

    kfold = KFold(n_splits=10, shuffle=True)
    kf_cv_scores = cross_val_score(xgbr, X_train, y_train, cv=kfold)
    print("K-fold CV average score: %.2f" % kf_cv_scores.mean())

    ypred = xgbr.predict(xtest)
    mse = mean_squared_error(ytest, ypred)
    print("MSE: %.2f" % mse)
    print("RMSE: %.2f" % (mse ** (1 / 2.0)))

    x_ax = range(len(ytest))
    plt.plot(x_ax, ytest, label="original")
    plt.plot(x_ax, ypred, label="predicted")
    plt.title("Boston test and predicted data")
    plt.legend()
    plt.show()


def train_predict(df_train: pd.DataFrame, df_predict: pd.DataFrame, narrow_cols: List[str], label_col: str = "buy_sell", require_balance: bool = True, buy_thresh: float = 0.):
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="invalid value encountered in true_divide")
        X_train, y_train, standard_scaler = transform_to_numpy(df=df_train,
                                                               narrow_cols=narrow_cols,
                                                               label_col=label_col,
                                                               require_balance=require_balance)

        if X_train is None or X_train.shape[0] == 0 or y_train is None:
            logger.info("Not enough training data.")
            return None

        xgb_args = dict(seed=42, max_depth=8)

        if not require_balance:
            num_buy = df_train[df_train["stock_val_change"] > buy_thresh].shape[0]
            num_sell = df_train[df_train["stock_val_change"] <= buy_thresh].shape[0]

            balance_ratio = num_sell / num_buy

            logger.info(f"Sell: {num_sell} / Buy: {num_buy}; ratio: {balance_ratio}")

            xgb_args["scale_pos_weight"] = balance_ratio

        # grid_search(X_train=X_train, y_train=y_train)

        logger.info(f"Using XGB Args: {xgb_args}")
        model = xgb.XGBRegressor(**xgb_args)

        model.fit(X_train,
                  y_train)

    X_predict = get_data_for_predictions(df=df_predict, narrow_cols=narrow_cols, standard_scaler=standard_scaler)

    logger.info("Invoking model prediction ...")
    prediction = model.predict(X_predict)

    df_predict.loc[:, "prediction"] = prediction

    return df_predict, model


def test_regress():
    df_train = pd.read_parquet(constants.SAMPLE_TWEET_STOCK_TRAIN_DF_PATH)
    df_test = pd.read_parquet(constants.SAMPLE_TWEET_STOCK_TEST_DF_PATH)
    df_train = df_train.fillna(value=0)

    print(f"Num rows: {df_train.shape[0]}")

    # buy_thresh = df_train["stock_val_change"].mean()
    df_predict, _ = train_predict(df_train=df_train,
                               df_predict=df_test,
                               narrow_cols=list(df_train.columns),
                               label_col="stock_val_change",
                               require_balance=False,
                               buy_thresh=0)

    top_divisor = 0
    if top_divisor != 0 and df_predict.shape[0] >= top_divisor:
        df_predict.sort_values(by=["prediction"], ascending=False, inplace=True)

        num_rows = df_predict.shape[0]
        quint = int(num_rows / top_divisor)

        min_predict = df_predict["prediction"].to_list()[quint]

        logger.info(f"Finding predictions above min: {min_predict}")
        df_buy = df_predict[df_predict["prediction"] > min_predict]
    else:
        df_buy = df_predict

    num_buys = df_buy.shape[0]
    logger.info(f"df_predict num rows: {df_predict.shape[0]}: buy predictions: {num_buys}")

    df_buy = df_buy[["f22_ticker", "purchase_date", "future_date", "marketcap"]].copy()

    num_hold_days = 1
    if df_buy is None or df_buy.shape[0] == 0:
        logger.info("No buy predictions.")
    else:
        df_buy.loc[:, "num_hold_days"] = num_hold_days
        df_buy.loc[:, "run_timestamp"] = datetime.timestamp(datetime.now())

        purchase_date_str = df_predict["purchase_date"].to_list()[0]
        get_days_roi_from_prediction_table(df_preds=df_buy,
                                           purchase_date_str=purchase_date_str,
                                           num_hold_days=num_hold_days,
                                           min_price=0.,
                                           addtl_hold_days=1)


def test_min_max():
    from sklearn.preprocessing import MinMaxScaler
    # define data
    data = np.array([[0.001],
                     [0.025],
                     [0.005],
                     [0.07],
                     [0.1]])
    print(data)
    # define min max scaler
    scaler = MinMaxScaler()
    # transform data
    scaled = scaler.fit_transform(data)
    print(scaled)
    print(np.mean(data))