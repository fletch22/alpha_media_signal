import gc
import sys
from datetime import datetime, timedelta
from statistics import mean

from sklearn.preprocessing import StandardScaler

gc.collect()

paths_to_add = ['/home/jovyan/work', '/home/jupyter/alpha_media_signal']

for p in paths_to_add:
    if p not in sys.path:
        sys.path.append(p)

import pandas as pd

from pathlib import Path

from ams.config import constants, logger_factory
from ams.services import twitter_service
from ams.services import ticker_service

import numpy as np
from typing import List
from ams.notebooks.twitter.twitter_ml_utils import WorkflowMode, get_data_for_predictions
from ams.notebooks.twitter import twitter_ml_utils
from ams.utils import date_utils

pd.set_option('display.max_rows', 5000)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

logger = logger_factory.create(__name__)

overall_roi = []

df_rec_quart_drop = None


def get_quarterly_data():
    global df_rec_quart_drop
    if df_rec_quart_drop is None:
        df_rec_quart_drop = twitter_service.get_all_quarterly_data_for_twitter()
    return df_rec_quart_drop.copy()


def process(df_twitter_raw: pd.DataFrame, predict_date_str: str, workflow_mode: WorkflowMode, num_hold_days: int):
    global overall_roi
    cat_uniques = None
    model_xgb = None

    dt = date_utils.parse_std_datestring(predict_date_str)
    if date_utils.is_stock_market_closed(dt):
        logger.info("No can do. Market closed.")
        return False

    if workflow_mode is WorkflowMode.Training:
        logger.info(f"Filtering twitter data to data before '{predict_date_str}'.")
        df_twitter_raw = df_twitter_raw[df_twitter_raw["date"] < predict_date_str]
    else:
        logger.info(f"Filtering twitter data to only '{predict_date_str}'.")
        df_twitter_raw = df_twitter_raw[df_twitter_raw["date"] == predict_date_str]
        model_xgb = twitter_ml_utils.load_model_for_prediction()
        cat_uniques = model_xgb.cat_uniques

    if df_twitter_raw.shape[0] == 0:
        logger.info(f"No twitter data on {predict_date_str}")
        return False

    logger.info(f"Max date: {df_twitter_raw['date'].max()}")
    logger.info(f"Num tweet records: {df_twitter_raw.shape[0]:,}")

    # twitter_ml_utils.show_distribution(df=df_twitter_raw)

    logger.info("Converting twitter data - phase I ...")
    df_booled = twitter_service.convert_to_bool(df=df_twitter_raw)
    df_twitter = twitter_ml_utils.convert_twitter_to_numeric(df=df_booled)

    logger.info("Getting Twitter stock data ...")
    df_stock_data = twitter_ml_utils.get_twitter_stock_data(df_tweets=df_twitter,
                                                            num_hold_days=num_hold_days,
                                                            workflow_mode=workflow_mode)

    logger.info(f"Num Twitter stock data records: {df_stock_data.shape[0]}")

    logger.info("Getting Twitter stock quarterly data ...")

    df_rec_quart_drop = get_quarterly_data()

    columns_fundy = list(df_rec_quart_drop.columns)

    df_result = twitter_ml_utils.merge_fundies_with_stock(df_stock_data=df_stock_data)
    df_drop_init = df_result.dropna(subset=["date"]).drop(columns="lastupdated_eq_fun")
    df_drop_future = df_drop_init[df_drop_init["date"] > df_drop_init["calendardate"]]
    df_drop_future = df_drop_future.sort_values(by=["ticker", "date", "calendardate"], ascending=False)
    df_stock_and_quarter = df_drop_future.drop_duplicates(subset=["ticker", "date"], keep="first")
    logger.info("Finished merging in quarterly stock data.")

    logger.info("Getting Nasdaq categorized ticker columns ...")
    df_nas_tickers_info, cat_uniques = ticker_service.get_nasdaq_tickers(cat_uniques=cat_uniques)

    logger.info(f"Num rows from NASDAQ categorized tickers: {df_nas_tickers_info.shape[0]}")
    col_ticker = "ticker_drop"

    df_stock_quart_info = pd.merge(df_stock_and_quarter, df_nas_tickers_info, how='inner', left_on=["ticker"], right_on=[col_ticker])
    df_sqi = df_stock_quart_info.drop(columns=[col_ticker])

    df_stock_renamed = df_sqi.rename(columns={"ticker": "f22_ticker"})

    if 'None' in df_stock_renamed.columns:
        df_stock_renamed = df_stock_renamed.drop(columns=['None'])

    logger.info("Merging Tweets with stock data ...")
    df_merged = pd.merge(df_twitter, df_stock_renamed, how='inner', left_on=["f22_ticker", "date"], right_on=["f22_ticker", "date"])

    logger.info(f"Num rows from merged {df_merged.shape[0]}")

    if df_merged.shape[0] == 0:
        logger.info("Not enough data after merge.")
        return False

    df_days = twitter_ml_utils.add_days_since_quarter_results(df=df_merged)

    logger.info("Adding meta information about dates (day of week, day of month, etc).")
    df_days_of = twitter_ml_utils.add_calendar_days(df=df_days)

    logger.info("Adding nasdaq roi rates.")
    df_dd = twitter_ml_utils.add_nasdaq_roi_new(df=df_days_of, num_hold_days=num_hold_days)

    # FIXME: 2021-01-15: chris.flesche: "close" should be approximated for when predicting
    df_dd["original_close_price"] = df_dd["close"]
    df_dd["date"].max()
    logger.info(f'Num df_thin_rabbit: {df_dd.shape[0]}')

    # NOTE: 2021-01-03: chris.flesche: For NLP
    # save_twitter_stock_join(df=df_thin_rabbit)

    cols_fundy_numeric = list(set(columns_fundy) - {"ticker", 'calendardate', 'datekey', 'reportperiod'})

    df_no_z = twitter_service.fill_null_numeric(df=df_dd, cols_fundy_numeric=cols_fundy_numeric)

    logger.info("Adding simple moving average data ...")
    df_since_sma = twitter_ml_utils.add_sma_stuff(df=df_no_z)

    df_since_sma["purchase_date"] = df_since_sma["date"]

    logger.info("Adding days until sale ...")
    df_days_until = ticker_service.add_days_until_sale(df=df_since_sma)

    # FIXME: 2021-01-14: chris.flesche: Use previous day's close for refine pool. Or remove.
    df = twitter_service.refine_pool(df=df_days_until, min_volume=None, min_price=None, max_price=None)
    df = twitter_service.omit_columns(df=df)
    df_tweet_counted = twitter_service.add_tweet_count(df=df).drop(columns=["calendardate", "reportperiod", "dimension", "datekey"])

    # NOTE: 2021-01-03: chris.flesche:
    # df_winnowed = twitter_ml_utils.truncate_avail_columns(df=df_tweet_counted)

    df_ranked = twitter_ml_utils.add_tip_ranks(df=df_tweet_counted, tr_file_path=constants.TIP_RANKED_DATA_PATH)

    df_ticker_hotted, unique_tickers = ticker_service.make_f22_ticker_one_hotted(df_ranked=df_ranked, cat_uniques=cat_uniques)
    cat_uniques["f22_ticker"] = unique_tickers

    narrow_cols = list(df_ticker_hotted.columns)

    print(f"Number of train_hotted {df_ticker_hotted.shape[0]}.")

    if workflow_mode == WorkflowMode.Training:
        logger.info("Adding buy/sell label for training ...")
        df_boughted = twitter_service.add_buy_sell(df=df_ticker_hotted)
    else:
        df_boughted = df_dd

    df_train = df_boughted

    logger.info(f"Num rows of prepared data: {df_train.shape[0]}")
    logger.info(f"Oldest date of prepared data (future_date): {df_train['future_date'].max()}")
    logger.info(f"Num unique tickers: {len(cat_uniques['f22_ticker'])}")

    if workflow_mode is WorkflowMode.Training:
        logger.info("Starting XGB training ...")
        sac_roi_list, did_train = twitter_ml_utils.xgb_learning(df=df_train, narrow_cols=narrow_cols, cat_uniques=cat_uniques)

        if not did_train:
            return False

        investment = 1000
        for s in sac_roi_list:
            investment = (investment * s) + investment

        logger.info(f"roi amount: {investment}")
        logger.info(sac_roi_list)
        if len(sac_roi_list) > 0:
            overall_roi.append(mean(sac_roi_list))

        if len(overall_roi) > 0:
            print(f"Overall roi: {mean(overall_roi)}")

    elif workflow_mode is WorkflowMode.Prediction:
        overwrite_file = False
        logger.info("Converting Pandas dataframe to numpy array for prediction step ...")

        X_predict = get_data_for_predictions(df=df_ticker_hotted, narrow_cols=narrow_cols, standard_scaler=model_xgb.scaler)

        logger.info("Invoking model prediction ...")
        prediction = model_xgb.model.predict(X_predict)

        df_ticker_hotted["prediction"] = prediction

        df_buy = df_ticker_hotted[df_ticker_hotted["prediction"] == 1][["f22_ticker", "purchase_date", "future_date"]]
        df_buy["num_hold_days"] = num_hold_days
        df_buy["run_timestamp"] = datetime.timestamp(datetime.now())

        df_preds = pd.read_csv(constants.TWITTER_PREDICTIONS_PATH)
        df_preds = df_preds[~((df_preds["purchase_date"] == predict_date_str) & (df_preds["num_hold_days"] == num_hold_days))]

        logger.info(f"Old rows found: {df_preds.shape[0]}")

        if overwrite_file:
            df_combined = df_buy
        else:
            df_combined = pd.concat([df_preds, df_buy], axis=0)

        logger.info("Writing predictions to output ...")
        df_combined.to_csv(constants.TWITTER_PREDICTIONS_PATH, index=False)
    else:
        raise Exception(f"workflow_mode: {workflow_mode.value} not yet implemented.")

    return True


def pred_and_train(predict_date_str: str, num_hold_days: int, df_tweets: pd.DataFrame):
    had_enough_data = process(df_twitter_raw=df_tweets,
                              predict_date_str=predict_date_str,
                              workflow_mode=WorkflowMode.Training,
                              num_hold_days=num_hold_days)

    if had_enough_data:
        process(df_twitter_raw=df_tweets,
                predict_date_str=predict_date_str,
                workflow_mode=WorkflowMode.Prediction,
                num_hold_days=num_hold_days)


def start():
    global overall_roi
    overall_roi = []
    # today_dt_str = date_utils.get_standard_ymd_format(datetime.now())
    learning_prep_dir = Path(constants.TWITTER_GREAT_REDUCTION_DIR, "main")
    df_tweets = twitter_ml_utils.load_twitter_raw(learning_prep_dir=learning_prep_dir)

    # df_tweets = df_tweets.sample(frac=.25)

    start_date_str = "2020-08-10"
    start_dt = date_utils.parse_std_datestring(start_date_str)
    num_days_train = 160
    num_hold_days = 1

    for day_ndx in range(num_days_train - 1, -1, -1):
        dt = start_dt + timedelta(days=day_ndx)
        predict_date_str = date_utils.get_standard_ymd_format(dt)
        logger.info(f"Prediction date: {predict_date_str}")

        if df_tweets[df_tweets["date"] > predict_date_str].shape[0] == 0:
            continue

        pred_and_train(df_tweets=df_tweets, predict_date_str=predict_date_str, num_hold_days=num_hold_days)

def start_spec():
    global overall_roi
    overall_roi = []
    # today_dt_str = date_utils.get_standard_ymd_format(datetime.now())
    learning_prep_dir = Path(constants.TWITTER_GREAT_REDUCTION_DIR, "main")
    df_tweets = twitter_ml_utils.load_twitter_raw(learning_prep_dir=learning_prep_dir)

    # df_tweets = df_tweets.sample(frac=.25)

    num_hold_days = 1

    predict_date_str = "2020-12-16"
    logger.info(f"Prediction date: {predict_date_str}")

    if df_tweets[df_tweets["date"] > predict_date_str].shape[0] == 0:
        return

    pred_and_train(df_tweets=df_tweets, predict_date_str=predict_date_str, num_hold_days=num_hold_days)


if __name__ == '__main__':
    # TODO: 2021-01-14: chris.flesche:
    #   Combine data with future close and without. Modify code to handle it. Process all at once. Then train and predict in 'xgb_learning_predicting'. Run function to test roi.
    # start_spec()
    start()