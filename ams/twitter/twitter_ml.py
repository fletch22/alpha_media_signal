from statistics import mean

import pandas as pd

from ams.config import constants, logger_factory
from ams.twitter import pred_persistence
from ams.twitter import twitter_ml_utils
from ams.twitter.PredictionParamFactory import PredictionParamFactory
from ams.twitter.PredictionParams import TrainParams, PredictionParams, PredictionMode

pd.set_option('display.max_rows', 5000)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

logger = logger_factory.create(__name__)

df_rec_quart_drop = None


# df_tiny = df_tweets_ready.sample(frac=.4)

def predict_in_range(pp: PredictionParams):
    print("Cleaning prediction file...")
    pred_persistence.clean_prediction_file(pp=pp)

    rois = []
    while pp.validate_prediction_date_str():
        print(f"\nTraining and predicting for {pp.predict_date_str} with {pp.predict_num_rows} rows ...\n")

        is_complete, rois = twitter_ml_utils.predict_day(pp=pp)
        if is_complete:
            break

        pp.subtract_day()

    if len(rois) > 0:
        logger.info(f"Overall mean roi: {mean(rois):.4f}")


def get_tweet_data():
    df_tweets = twitter_ml_utils.load_twitter_raw(proc_path=constants.TWITTER_END_DROP)
    return twitter_ml_utils.get_stock_matchable(df=df_tweets)


def make_a_real_prediction(predict_date_str: str, num_hold_days: int):
    df = get_tweet_data()
    print(f"Num tweets: {df.shape[0]}")

    pred_params = PredictionParams()
    pred_params.prediction_mode = PredictionMode.RealMoneyStockRecommender
    pred_params.min_date_str = predict_date_str
    pred_params.predict_date_str = predict_date_str
    pred_params.max_date_str = pred_params.predict_date_str
    pred_params.num_hold_days = num_hold_days
    pred_params.df = df

    pred_params.clean_pure_run = False
    pred_params.train_params = TrainParams()

    predict_in_range(pp=pred_params)


def make_historical_prediction(num_hold_days: int, max_date_str: str = None, min_date_str: str = None):
    df = get_tweet_data()
    print(f"Num tweets: {df.shape[0]}")

    pred_params = PredictionParamFactory.create_generic_trainer(df=df, num_hold_days=num_hold_days, max_date_str=max_date_str, min_date_str=min_date_str)

    predict_in_range(pp=pred_params)


if __name__ == '__main__':
    make_historical_prediction(num_hold_days=5)