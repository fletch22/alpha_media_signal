from pathlib import Path
from statistics import mean

import pandas as pd

from ams.config import constants, logger_factory
from ams.services import slack_service
from ams.twitter import pred_persistence
from ams.twitter import twitter_ml_utils
from ams.twitter.PredictionParamFactory import PredictionParamFactory
from ams.twitter.PredictionParams import PredictionParams, PredictionMode
from ams.twitter.skip_day_predictor import get_every_nth_tweet_date

pd.set_option('display.max_rows', 5000)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

logger = logger_factory.create(__name__)

df_rec_quart_drop = None


def predict_in_range(pp: PredictionParams):
    logger.info("Cleaning prediction file...")
    pred_persistence.clean_prediction_file(pp=pp)

    rois = []
    is_at_end = False
    prediction_ndx_count = 0
    while not is_at_end:
        logger.info(f"\nTraining and predicting for tweet date {pp.tweet_date_str} for purchase on {pp.purchase_date_str} with {pp.predict_num_rows} rows ...\n")

        is_valid, in_range = pp.validate_tweet_date_str()
        if is_valid:
            is_at_end, rois, prediction_ndx_count = twitter_ml_utils.predict_day(pp=pp, prediction_ndx_count=prediction_ndx_count)
            if is_at_end:
                break

        is_at_end = pp.subtract_day()
        if is_at_end:
            break

    if len(rois) > 0:
        logger.info(f"Overall mean roi: {mean(rois):.4f}")


def get_tweet_data():
    df_tweets = twitter_ml_utils.load_twitter_raw(proc_path=constants.TWITTER_END_DROP)
    return twitter_ml_utils.get_stock_matchable(df=df_tweets)


def make_a_real_prediction(tweet_date_str: str, num_hold_days: int) -> str:
    df = get_tweet_data()
    logger.info(f"Num tweets: {df.shape[0]}")

    pred_params = PredictionParams(df=df,
                                   prediction_mode=PredictionMode.RealMoneyStockRecommender,
                                   tweet_date_str=tweet_date_str,
                                   min_date_str=tweet_date_str,
                                   max_date_str=tweet_date_str,
                                   num_hold_days=num_hold_days)

    purchase_date_str = pred_params.purchase_date_str
    predict_in_range(pp=pred_params)

    return purchase_date_str


def get_tweet_data_alt():
    df_tweets = twitter_ml_utils.load_twitter_raw(proc_path=Path(constants.TEMP_PATH, "twitter", "end_drop"))
    return twitter_ml_utils.get_stock_matchable(df=df_tweets)


def make_historical_prediction(num_hold_days: int, max_date_str: str = None, min_date_str: str = None):
    df = get_tweet_data_alt()
    # df = df.sample(frac=.1)

    tweet_days_1 = get_every_nth_tweet_date(nth_sell_day=3, skip_days=0)
    df = df[df["date"].isin(tweet_days_1)].copy()

    logger.info(f"Num tweets: {df.shape[0]}")

    # df = df.sample(frac=.01)

    pred_params = PredictionParamFactory.create_generic_trainer(df=df, num_hold_days=num_hold_days,
                                                                max_date_str=max_date_str, min_date_str=min_date_str,
                                                                require_balance=False)
    # pred_params = PredictionParamFactory.create_mid_january_trainer(df=df, num_hold_days=num_hold_days)
    # pred_params = PredictionParamFactory.create_october_trainer(df=df, num_hold_days=num_hold_days)

    predict_in_range(pp=pred_params)


if __name__ == '__main__':
    # constants.xgb.defaults.max_depth = 9
    for i in [1]:
        make_historical_prediction(num_hold_days=i)
    slack_service.send_direct_message_to_chris("Done with make_historical_prediction")