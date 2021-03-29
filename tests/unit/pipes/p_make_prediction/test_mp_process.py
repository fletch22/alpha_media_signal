import warnings
from datetime import datetime
from statistics import mean

import pandas as pd
import xgboost
from sklearn.preprocessing import StandardScaler

from ams.config import logger_factory, constants
from ams.models.xgb_reg import get_weights
from ams.pipes.p_make_prediction.TrainingBag import TrainingBag
from ams.twitter.pred_perf_testing import get_days_roi_from_prediction_table
from ams.twitter.twitter_ml_utils import get_data_for_predictions, transform_to_numpy

logger = logger_factory.create(__name__)


def test_foo():
    # Arrange
    rows = [
        {"f22_ticker": "abc", "purchase_date": "2020-01-01"},
        {"f22_ticker": "abc", "purchase_date": "2020-01-01"},
        {"f22_ticker": "cde", "purchase_date": "2020-01-01"}
    ]

    df_buy = pd.DataFrame(rows, columns=["f22_ticker", "purchase_date"])

    df_again = df_buy.groupby(["f22_ticker", "purchase_date"]).size().reset_index(name='counts')

    # Act
    logger.info(df_again.head())

    # Assert


def test_bagger():
    # Arrange
    tb: TrainingBag = TrainingBag.load(twitter_root=constants.TWITTER_OUTPUT_RAW_PATH)
    # Act

    purchase_date_list = tb.get_purchase_dates()

    all_rois = []
    for p_date in purchase_date_list:
        logger.info(f"Primary purchase date: {pd}")
        df_train, df_test = tb.get_data(purchase_date_str=p_date)
        model_1, standard_scaler = get_model(df_train)

        X_predict = get_data_for_predictions(df=df_test, narrow_cols=list(df_train.columns), standard_scaler=standard_scaler)

        rec_data = tb.get_recent_data(purchase_date_str=p_date, num_models=2)
        all_models = [model_1]
        for df_train, df_test in rec_data:
            model, _ = get_model(df_train=df_train)
            all_models.append(model)

        predictions = []
        for mod in all_models:
            pred = mod.predict(X_predict)
            predictions.append(pred)

        buys = []
        for ndx, pred in enumerate(predictions):
            col = f"prediction"
            df_tmp = df_test.copy()
            df_tmp.loc[:, col] = pred
            df_buy = df_tmp[df_tmp[col] > 0].copy()
            buys.append(df_buy)
            num_buys = df_buy.shape[0]
            logger.info(f"df_predict num rows: {df_test.shape[0]}: buy predictions: {num_buys}")

        df_buy = pd.concat(buys, axis=0)

        roi = handle_buy_predictions(df_buy)
        all_rois.append(roi)

        logger.info(f"Ongoing roi: {mean(all_rois)}")

    logger.info(f"Overall roi: {mean(all_rois)}")
    # Assert


def get_model(df_train):
    label_col = "stock_val_change"
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="invalid value encountered in true_divide")
        X_train, y_train, standard_scaler = transform_to_numpy(df=df_train,
                                                               narrow_cols=df_train.columns,
                                                               label_col=label_col,
                                                               require_balance=False)
    xgb_reg = xgboost.XGBRegressor(seed=42, max_depth=8, tree_method='gpu_hist', gpu_id=0)
    model = xgb_reg.fit(X_train, y_train, sample_weight=get_weights(df=df_train))
    return model, standard_scaler


def handle_buy_predictions(df_buy):
    df_buy = df_buy[["f22_ticker", "purchase_date", "future_date", "marketcap"]].copy()

    df_buy = df_buy.groupby(["f22_ticker", "purchase_date"]).size().reset_index(name='counts')
    df_buy_3 = df_buy[df_buy["counts"] == 3]
    df_buy_2 = df_buy[df_buy["counts"] == 2]
    df_buy_1 = df_buy[df_buy["counts"] == 1]

    logger.info(f"Found {df_buy_3.shape[0]} with 3 counts.")
    logger.info(f"Found {df_buy_2.shape[0]} with 2 counts.")
    logger.info(f"Found {df_buy_1.shape[0]} with 1 counts.")

    df_buy = df_buy_3
    if df_buy.shape[0] == 0:
        df_buy = df_buy_2

    if df_buy.shape[0] == 0:
        df_buy = df_buy_1

    roi = None
    num_hold_days = 1
    if df_buy is None or df_buy.shape[0] == 0:
        logger.info("No buy predictions.")
    else:
        df_buy.loc[:, "num_hold_days"] = num_hold_days
        df_buy.loc[:, "run_timestamp"] = datetime.timestamp(datetime.now())

        logger.info(f"Purchase dts: {df_buy['purchase_date'].unique()}")

        purchase_date_str = df_buy["purchase_date"].to_list()[0]
        roi = get_days_roi_from_prediction_table(df_preds=df_buy,
                                                 purchase_date_str=purchase_date_str,
                                                 num_hold_days=num_hold_days,
                                                 min_price=0.,
                                                 addtl_hold_days=1)
    return roi