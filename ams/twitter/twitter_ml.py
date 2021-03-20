from pathlib import Path
from statistics import mean

import pandas as pd

from ams.config import constants, logger_factory
from ams.config.constants import ensure_dir
from ams.services import slack_service, file_services, pickle_service
from ams.twitter import twitter_ml_utils
from ams.twitter.TrainAndPredictionParamFactory import TrainAndPredictionParamFactory
from ams.twitter.TrainAndPredictionParams import TrainAndPredictionParams, PredictionMode, PredictionMaxRounds
from ams.twitter.skip_day_predictor import get_every_nth_tweet_date

pd.set_option('display.max_rows', 5000)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

logger = logger_factory.create(__name__)

df_rec_quart_drop = None


def merge_with_stocks_all_days(tapp: TrainAndPredictionParams, output_parent_path: Path):
    rois = []
    is_at_end = False
    data_file_count = 0
    while not is_at_end:
        logger.info(f"\nMerging Tweets with stock data for tweet date {tapp.tweet_date_str} for purchase on {tapp.purchase_date_str} with {tapp.predict_num_rows} rows ...\n")

        is_valid, in_range = tapp.validate_tweet_date_str()
        if is_valid:
            is_at_end, rois, data_file_count = twitter_ml_utils.merge_with_stocks_for_day(tapp=tapp, data_file_count=data_file_count, output_parent_path=output_parent_path)
            if is_at_end:
                break

        is_at_end = tapp.subtract_day()
        if is_at_end:
            break

    if len(rois) > 0:
        logger.info(f"Overall mean roi: {mean(rois):.4f}")


def get_tweet_data():
    df_tweets = twitter_ml_utils.load_twitter_raw(proc_path=constants.REFINED_TWEETS_BUCKET_PATH)
    return twitter_ml_utils.get_stock_matchable(df=df_tweets)


def make_a_real_prediction(tweet_date_str: str, num_hold_days: int) -> str:
    df = get_tweet_data()
    logger.info(f"Num tweets: {df.shape[0]}")

    pred_params = TrainAndPredictionParams(df=df,
                                           prediction_mode=PredictionMode.RealMoneyStockRecommender,
                                           tweet_date_str=tweet_date_str,
                                           min_date_str=tweet_date_str,
                                           max_date_str=tweet_date_str,
                                           num_hold_days=num_hold_days)

    purchase_date_str = pred_params.purchase_date_str

    raise Exception("Not implemented yet.")
    # predict_in_range(pp=pred_params, output_parent_path=output_parent_path)

    return purchase_date_str


def get_tweet_data_alt():
    df_tweets = twitter_ml_utils.load_twitter_raw(proc_path=Path(constants.TEMP_PATH, "twitter", "end_drop"))
    return twitter_ml_utils.get_stock_matchable(df=df_tweets)


def make_historical_prediction(data_dirname: str, num_hold_days: int, max_date_str: str = None, min_date_str: str = None, num_revolutions: int = 1):
    df: pd.DataFrame = get_tweet_data()
    # df = df.sample(frac=.1)
    df = df[df["date"] <= max_date_str].copy()

    tapp = TrainAndPredictionParamFactory.create_generic_trainer(df=df, num_hold_days=num_hold_days,
                                                                 max_date_str=max_date_str, min_date_str=min_date_str,
                                                                 require_balance=True)

    history_parent_path = Path(constants.TWIT_STOCK_MERGE_DROP_PATH, data_dirname)
    ensure_dir(history_parent_path)

    logger.info(f"About to clean out {history_parent_path} ...")
    file_services.clean_dir(history_parent_path)

    pred_path_str = Path(history_parent_path, "pred_params.pkl")
    pickle_service.save(tapp, file_path=pred_path_str)

    # NOTE: 2021-03-09: chris.flesche: To avoid overlapping datespans which would cause data leakage,
    # create 3 different models that are trained with no overlap by themselves. Using ensemble/stacking
    # we can use all 3 models with one prediction set.
    for skip_days in range(num_revolutions):
        output_parent_path = Path(history_parent_path, f"revolution_{skip_days}")

        tweet_days_1 = get_every_nth_tweet_date(nth_sell_day=3, skip_start_days=skip_days)

        df_skipped = df[df["date"].isin(tweet_days_1)].copy()
        logger.info(f"Num tweets: {df_skipped.shape[0]}")

        df_skipped.sort_values(by=["date"], inplace=True)
        df_skipped.dropna(subset=["date"], inplace=True)

        tapp.df = df_skipped

        tapp.tweet_date_str = df_skipped["date"].max()

        logger.info(f"Will process data starting with {tapp.tweet_date_str} ...")

        merge_with_stocks_all_days(tapp=tapp, output_parent_path=output_parent_path)


if __name__ == '__main__':
    data_dirname = "twitter_ml"

    for i in [1]:
        make_historical_prediction(data_dirname=data_dirname, max_date_str="2021-03-04", num_hold_days=i, num_revolutions=3)

    slack_service.send_direct_message_to_chris("Done with make_historical_prediction")