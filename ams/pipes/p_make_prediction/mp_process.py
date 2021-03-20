from datetime import datetime
from pathlib import Path
from random import shuffle
from statistics import mean
from typing import Union

import pandas as pd

from ams.config import constants, logger_factory
from ams.config.constants import ensure_dir
from ams.models import xgb_reg
from ams.pipes.p_stock_merge.sm_process import STOCKS_MERGED_FILENAME, PRED_PARAMS_FILENAME
from ams.services import pickle_service, slack_service
from ams.twitter import skip_day_predictor
from ams.twitter.TrainAndPredictionParams import TrainAndPredictionParams, PredictionMode
from ams.twitter.TwitterStackingModel import TwitterStackingModel
from ams.twitter.pred_perf_testing import get_days_roi_from_prediction_table
from ams.utils import date_utils
from ams.utils.date_utils import get_next_market_day_no_count_closed_days

logger = logger_factory.create(__name__)

PREDICTIONS_CSV = "predictions.csv"
MONEY_PREDICTIONS_CSV = "real_money_predictions.csv"


def split_train_test(tapp: TrainAndPredictionParams) -> (pd.DataFrame, pd.DataFrame):
    df = tapp.df

    logger.info(f"Splitting on tweet date {tapp.tweet_date_str}")

    df_train = df[df["date"] < tapp.tweet_date_str].copy()
    df_test = df[df["date"] == tapp.tweet_date_str].copy()
    df_test.dropna(subset=["future_date"], inplace=True)

    logger.info(f"Test data size: {df_test.shape[0]}")
    logger.info(f"Train data size: {df_train.shape[0]}")
    logger.info(f"Oldest date of train data (future_date): {df_train['future_date'].max()}")
    logger.info(f"Oldest date of test data (future_date): {df_test['future_date'].max()}")

    return df_train, df_test


def get_real_money_preds_path(output_path: Path):
    return Path(output_path, MONEY_PREDICTIONS_CSV)


def persist_predictions(df_buy: pd.DataFrame, tapp: TrainAndPredictionParams, output_path: Path, rev_ndx: int):
    df_preds = None

    pred_path = Path(output_path, PREDICTIONS_CSV)
    if tapp.prediction_mode == PredictionMode.RealMoneyStockRecommender:
        pred_path = get_real_money_preds_path(output_path)

    is_new = not pred_path.exists()

    if not is_new:
        df_preds = pd.read_csv(pred_path)
        df_preds = df_preds[~((df_preds["purchase_date"] == tapp.purchase_date_str)
                              & (df_preds["num_hold_days"] == tapp.num_hold_days)
                              & (df_preds["revolution_ndx"] == rev_ndx))]

    if df_preds is not None:
        df_combined = pd.concat([df_preds, df_buy], axis=0)
    else:
        df_combined = df_buy

    logger.info(f"Wrote {df_buy.shape[0]} predictions to {tapp.prediction_mode.name} output file.")
    df_combined.to_csv(pred_path, index=False)


def predict_day(tapp: TrainAndPredictionParams, output_path: Path, rev_ndx: int) -> (Union[None, float], Union[None, object]):
    df_train, df_test = split_train_test(tapp=tapp)

    if df_test.shape[0] == 0:
        return None, None

    df_train = df_train.fillna(value=0)

    df_predict, model = xgb_reg.train_predict(df_train=df_train,
                                              df_test=df_test,
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
        df_buy = df_predict[df_predict["prediction"] > 0]

    num_buys = df_buy.shape[0]
    logger.info(f"df_predict num rows: {df_predict.shape[0]}: buy predictions: {num_buys}")

    df_buy = df_buy[["f22_ticker", "purchase_date", "future_date", "marketcap"]].copy()

    roi = None
    num_hold_days = 1
    if df_buy is None or df_buy.shape[0] == 0:
        logger.info("No buy predictions.")
    else:
        df_buy.loc[:, "num_hold_days"] = num_hold_days
        df_buy.loc[:, "run_timestamp"] = datetime.timestamp(datetime.now())
        df_buy.loc[:, "revolution_ndx"] = rev_ndx

        logger.info(f"Purchase dts: {df_buy['purchase_date'].unique()}")

        persist_predictions(df_buy=df_buy, tapp=tapp, output_path=output_path, rev_ndx=rev_ndx)

        if tapp.prediction_mode == PredictionMode.DevelopmentAndTraining:
            purchase_date_str = df_predict["purchase_date"].to_list()[0]
            roi = get_days_roi_from_prediction_table(df_preds=df_buy,
                                                     purchase_date_str=purchase_date_str,
                                                     num_hold_days=num_hold_days,
                                                     min_price=0.,
                                                     addtl_hold_days=1)

    return roi, model


def start(src_path: Path, dest_path: Path, prediction_mode: PredictionMode):
    ensure_dir(dest_path)
    stocks_merged_path = Path(src_path, STOCKS_MERGED_FILENAME)
    df = pd.read_parquet(stocks_merged_path)

    tran_and_pred_path = Path(src_path, PRED_PARAMS_FILENAME)
    tapp: TrainAndPredictionParams = pickle_service.load(tran_and_pred_path)
    tapp.prediction_mode = prediction_mode

    logger.info(f"Stock-merged size: {df.shape[0]}")

    roi_all = []
    nth_sell_day = 1 + tapp.num_days_until_purchase + tapp.num_hold_days

    today_date_str = date_utils.get_standard_ymd_format(date=datetime.now())

    mode_offset = 0 if tapp.prediction_mode == PredictionMode.DevelopmentAndTraining else 2

    tweet_date_str = get_next_market_day_no_count_closed_days(date_str=today_date_str, num_days=-(nth_sell_day - mode_offset))

    for i in range(nth_sell_day):
        tapp.tweet_date_str = tweet_date_str
        tapp.max_date_str = tapp.tweet_date_str

        dates = skip_day_predictor.get_every_nth_tweet_date(nth_sell_day=nth_sell_day, skip_start_days=i)
        dates = list(set(dates) | {tapp.tweet_date_str})
        tapp.df = df[df["date"].isin(dates)].copy()

        roi = train_skipping_data(output_path=dest_path, tapp=tapp, rev_ndx=nth_sell_day)
        if roi is not None:
            roi_all.append(roi)

    overall_roi = None
    if len(roi_all) > 0:
        overall_roi = mean(roi_all)

    if overall_roi is not None:
        logger.info(f"Overall roi: {overall_roi}")


def train_skipping_data(output_path: Path, tapp: TrainAndPredictionParams, rev_ndx: int):
    has_more_days = True
    roi_all = []
    tsm = TwitterStackingModel()
    is_first_loop = True
    while has_more_days:

        roi, model = predict_day(tapp=tapp, output_path=output_path, rev_ndx=rev_ndx)

        if tapp.prediction_mode == PredictionMode.RealMoneyStockRecommender:
            break

        if roi is not None:
            roi_all.append(roi)
            logger.info(f"Ongoing roi: {mean(roi_all)}")

        if model is not None and is_first_loop:
            tsm.add_trained_model(model)
            is_first_loop = False

        is_at_end = tapp.subtract_day()
        has_more_days = not is_at_end

    TwitterStackingModel.persist(twitter_stacking_model=tsm)

    if tapp.prediction_mode == PredictionMode.RealMoneyStockRecommender:
        tickers = inspect_real_pred_results(output_path, tapp.purchase_date_str)
        slack_service.send_direct_message_to_chris(f"{tapp.purchase_date_str}: {str(tickers)}")

    overall_roi = None
    if len(roi_all) > 0:
        overall_roi = mean(roi_all)

    return overall_roi


def inspect_real_pred_results(output_path: Path, purchase_date_str: str):
    real_mon_path = get_real_money_preds_path(output_path)
    df = pd.read_csv(str(real_mon_path))

    logger.info(f"Purchase date str: {purchase_date_str}")

    # TODO: 2021-03-16: chris.flesche:  Not yet known if this is a good strategy
    # df_g = df.groupby(by=["f22_ticker", "purchase_date"]) \
    #             .agg(F.mean(F.col("prediction").alias("mean_prediction")))

    df = df[df["purchase_date"] == purchase_date_str]
    tickers = df["f22_ticker"].to_list()
    shuffle(tickers)

    logger.info(f"Num: {len(tickers)}: {tickers}")

    return set(tickers)


if __name__ == '__main__':
    dest_path = Path(constants.TWITTER_OUTPUT_RAW_PATH, "prediction_bucket")
    # src_path = Path(constants.TWITTER_OUTPUT_RAW_PATH, "stock_merge_drop", "main")
    #
    # prediction_mode = PredictionMode.DevelopmentAndTraining # PredictionMode.DevelopmentAndTraining PredictionMode.RealMoneyStockRecommender
    #
    # start(src_path=src_path,
    #         dest_path=dest_path,
    #         prediction_mode=prediction_mode)

    purchase_date_str = "2021-03-17"
    tick_1 = inspect_real_pred_results(output_path=dest_path, purchase_date_str="2021-03-17")
    tick_2 = inspect_real_pred_results(output_path=dest_path, purchase_date_str="2021-03-18")

    logger.info(f"Overlap: {tick_1.intersection(tick_2)}")
