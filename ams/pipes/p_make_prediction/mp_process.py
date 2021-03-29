import collections
from datetime import datetime
from pathlib import Path
from random import shuffle
from statistics import mean
from typing import Union

import pandas as pd

from ams.config import constants, logger_factory
from ams.config.constants import ensure_dir
from ams.models import xgb_reg
from ams.pipes.p_make_prediction.TrainingBag import TrainingBag
from ams.pipes.p_stock_merge.sm_process import STOCKS_MERGED_FILENAME, PRED_PARAMS_FILENAME
from ams.services import pickle_service, slack_service
from ams.twitter import skip_day_predictor
from ams.twitter.TrainAndPredictionParams import TrainAndPredictionParams, PredictionMode
from ams.twitter.pred_perf_testing import get_days_roi_from_prediction_table
from ams.utils.date_utils import get_next_market_day_no_count_closed_days

logger = logger_factory.create(__name__)

PREDICTIONS_CSV = "predictions.csv"
MONEY_PREDICTIONS_CSV = "real_money_predictions.csv"


def split_train_test(tapp: TrainAndPredictionParams) -> (pd.DataFrame, pd.DataFrame):
    df = tapp.df

    logger.info(f"Splitting on tweet date {tapp.tweet_date_str}")

    df_train = df[df["date"] < tapp.tweet_date_str].copy()
    df_test = df[df["date"] == tapp.tweet_date_str].copy()
    # df_test.dropna(subset=["future_date"], inplace=True)

    logger.info(f"Test data size: {df_test.shape[0]}")
    logger.info(f"Train data size: {df_train.shape[0]}")
    logger.info(f"Oldest date of train data (future_date): {df_train['future_date'].max()}")
    logger.info(f"Oldest date of test data (future_date): {df_test['future_date'].max()}")

    return df_train, df_test


def get_real_money_preds_path(output_path: Path):
    return Path(output_path, MONEY_PREDICTIONS_CSV)


def persist_predictions(df_buy: pd.DataFrame, tapp: TrainAndPredictionParams, output_path: Path):
    df_preds = None

    pred_path = Path(output_path, PREDICTIONS_CSV)
    if tapp.prediction_mode == PredictionMode.RealMoneyStockRecommender:
        pred_path = get_real_money_preds_path(output_path)

    is_new = not pred_path.exists()

    if not is_new:
        df_preds = pd.read_csv(pred_path)
        df_preds = df_preds[~((df_preds["purchase_date"] == tapp.purchase_date_str)
                              & (df_preds["num_hold_days"] == tapp.num_hold_days))]

    if df_preds is not None:
        df_combined = pd.concat([df_preds, df_buy], axis=0)
    else:
        df_combined = df_buy

    logger.info(f"Wrote {df_buy.shape[0]} predictions to {tapp.prediction_mode.name} output file.")
    df_combined.to_csv(pred_path, index=False)


def predict_day(tapp: TrainAndPredictionParams,
                output_path: Path,
                training_bag: TrainingBag) -> (Union[None, float]):
    df_train, df_test = split_train_test(tapp=tapp)

    if df_test is None or df_test.shape[0] == 0:
        return None

    df_train = df_train.fillna(value=0)

    predictions = xgb_reg.train_predict(df_train=df_train,
                                        df_test=df_test.copy(),
                                        narrow_cols=list(df_train.columns),
                                        training_bag=training_bag,
                                        purchase_date_str=tapp.purchase_date_str,
                                        label_col="stock_val_change",
                                        require_balance=False,
                                        buy_thresh=0)

    # top_divisor = 0
    # if top_divisor != 0 and df_predict.shape[0] >= top_divisor:
    #     df_predict.sort_values(by=["prediction"], ascending=False, inplace=True)
    #
    #     num_rows = df_predict.shape[0]
    #     quint = int(num_rows / top_divisor)
    #
    #     min_predict = df_predict["prediction"].to_list()[quint]
    #
    #     logger.info(f"Finding predictions above min: {min_predict}")
    #     df_buy = df_predict[df_predict["prediction"] > min_predict]
    # else:
    #     df_buy = df_predict[df_predict["prediction"] > 0]

    col_preds = "predictions"
    df_test.loc[:, col_preds] = predictions
    df_buy = df_test[df_test[col_preds] > 0].copy()
    logger.info(f"df_predict num rows: {df_test.shape[0]}: buy predictions: {df_buy.shape[0]}")

    roi = handle_buy_predictions(df_buy, output_path, tapp)

    return roi


def handle_buy_predictions(df_buy, output_path: Path, tapp: TrainAndPredictionParams):
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

        persist_predictions(df_buy=df_buy, tapp=tapp, output_path=output_path)

        if tapp.prediction_mode == PredictionMode.DevelopmentAndTraining:
            purchase_date_str = df_buy["purchase_date"].to_list()[0]
            roi = get_days_roi_from_prediction_table(df_preds=df_buy,
                                                     purchase_date_str=purchase_date_str,
                                                     num_hold_days=num_hold_days,
                                                     min_price=0.,
                                                     addtl_hold_days=1)
    return roi


def filter_columns(df: pd.DataFrame):
    cols = list(df.columns)
    cols = [c for c in cols if not c.startswith("location_")]
    cols = [c for c in cols if not c.startswith("currency_")]
    cols = [c for c in cols if not c.startswith("industry_")]
    cols = [c for c in cols if not c.startswith("famaindustry_")]
    cols = [c for c in cols if not c.startswith("category_")]
    cols = [c for c in cols if not c.startswith("sector_")]
    cols = [c for c in cols if not c.startswith("scalerevenue_")]
    cols = [c for c in cols if not c.startswith("table_")]
    cols = [c for c in cols if not c.startswith("sicsector_")]
    cols = [c for c in cols if not c.startswith("scalemarketcap_")]

    return df[cols]


def start(src_path: Path, dest_path: Path, prediction_mode: PredictionMode, purchase_date_str: str, send_msgs: bool = True):
    ensure_dir(dest_path)

    logger.info(f"Getting files from {src_path}")

    stocks_merged_path = Path(src_path, STOCKS_MERGED_FILENAME)
    df = pd.read_parquet(stocks_merged_path)

    tran_and_pred_path = Path(src_path, PRED_PARAMS_FILENAME)
    tapp: TrainAndPredictionParams = pickle_service.load(tran_and_pred_path)
    tapp.prediction_mode = prediction_mode

    logger.info(f"Stock-merged size: {df.shape[0]}")

    training_bag = TrainingBag()
    roi_all = []

    nth_sell_day = 1 + tapp.num_days_until_purchase + tapp.num_hold_days

    mode_offset = 0 if tapp.prediction_mode == PredictionMode.DevelopmentAndTraining else 2
    tweet_date_str = get_next_market_day_no_count_closed_days(date_str=purchase_date_str, num_days=-(nth_sell_day - mode_offset))

    for i in range(nth_sell_day):
        tapp.tweet_date_str = tweet_date_str
        tapp.max_date_str = tapp.tweet_date_str

        dates = skip_day_predictor.get_every_nth_tweet_date(nth_sell_day=nth_sell_day, skip_start_days=i)
        dates = list(set(dates) | {tapp.tweet_date_str})
        tapp.df = df[df["date"].isin(dates)].copy()

        if tapp.prediction_mode == PredictionMode.RealMoneyStockRecommender:
            _ = predict_day(tapp=tapp, output_path=dest_path)

            tickers = inspect_real_pred_results(dest_path, tapp.purchase_date_str)

            if send_msgs:
                slack_service.send_direct_message_to_chris(f"{tapp.purchase_date_str}: {str(tickers)}")
        else:
            roi = train_skipping_data(output_path=dest_path, tapp=tapp, training_bag=training_bag)
            if roi is not None:
                roi_all.append(roi)

        break

    if tapp.prediction_mode == PredictionMode.DevelopmentAndTraining:
        twitter_root_path = src_path.parent.parent
        TrainingBag.persist(training_bag=training_bag, twitter_root=twitter_root_path)

    overall_roi = None
    if len(roi_all) > 0:
        overall_roi = mean(roi_all)

    if overall_roi is not None:
        logger.info(f"Overall roi: {overall_roi}")


def train_skipping_data(output_path: Path, tapp: TrainAndPredictionParams, training_bag: TrainingBag) -> float:
    has_more_days = True
    roi_all = []
    overall_roi = None

    tapp.df.sort_values(by=["date"], inplace=True)

    # FIXME: 2021-03-27: chris.flesche: Temp
    # tapp.df = tapp.df.sample(frac=.1)

    dates = tapp.df["date"].unique()

    prev_model_deque = collections.deque()

    count = 0
    while has_more_days:
        roi, prev_model_deque = predict_day(tapp=tapp, output_path=output_path, prev_model_deque=prev_model_deque, training_bag=training_bag)

        if roi is not None:
            roi_all.append(roi)
            logger.info(f"Ongoing roi: {mean(roi_all)}")

        dates = dates[1:]
        tapp.tweet_date_str = dates[0]
        has_more_days = len(dates) > 1

        count += 1
        if count > 13:
            break

    if len(roi_all) > 0:
        overall_roi = mean(roi_all)

    return overall_roi


def inspect_real_pred_results(output_path: Path, purchase_date_str: str):
    real_mon_path = get_real_money_preds_path(output_path)
    df = pd.read_csv(str(real_mon_path))

    logger.info(f"Purchase date str: {purchase_date_str}")

    # logger.info(list(df.columns))

    # TODO: 2021-03-16: chris.flesche:  Not yet known if this is a good strategy
    # df_g = df.groupby(by=["f22_ticker", "purchase_date"]) \
    #             .agg(F.mean(F.col("prediction").alias("mean_prediction")))

    df = df[df["purchase_date"] == purchase_date_str]
    tickers = df["f22_ticker"].to_list()
    shuffle(tickers)

    logger.info(f"Num: {len(tickers)}: {tickers}")

    return set(tickers)


if __name__ == '__main__':
    twit_root_path = constants.TWITTER_OUTPUT_RAW_PATH  # Path(constants.TEMP_PATH, "twitter")
    # twit_root_path = Path(constants.TEMP_PATH, "twitter")
    src_path = Path(twit_root_path, "stock_merge_drop", "main")
    dest_path = Path(twit_root_path, "prediction_bucket")

    pred_path = Path(dest_path, PREDICTIONS_CSV)
    if pred_path.exists():
        pred_path.unlink()

    prediction_mode = PredictionMode.DevelopmentAndTraining  # PredictionMode.RealMoneyStockRecommender

    # purchase_date_str = date_utils.get_standard_ymd_format(date=datetime.now())
    purchase_date_str = "2021-08-10"
    start(src_path=src_path,
          dest_path=dest_path,
          prediction_mode=prediction_mode,
          send_msgs=False,
          purchase_date_str=purchase_date_str)
