from datetime import datetime, timedelta
from pathlib import Path
from typing import Set

import pandas as pd

from ams.DateRange import DateRange
from ams.config import constants, logger_factory
from ams.config.constants import ensure_dir
from ams.pipes import batchy_bae
from ams.services import pickle_service, twitter_service, slack_service
from ams.services.twitter_service import EARLIEST_TWEET_DATE_STR
from ams.twitter import twitter_ml_utils
from ams.twitter.TrainAndPredictionParamFactory import TrainAndPredictionParamFactory
from ams.twitter.TrainAndPredictionParams import TrainAndPredictionParams, is_good_market_date
from ams.twitter.twitter_ml_utils import seal_label_leak, easy_convert_columns, get_stocks_based_on_tweets, \
    combine_with_quarterly_stock_data, merge_tweets_with_stock_data, add_calendar_info, one_hot
from ams.utils import date_utils, tipranks_utils
from ams.utils.date_utils import get_standard_ymd_format, get_market_holidays, get_next_market_date

logger = logger_factory.create(__name__)

STOCKS_MERGED_FILENAME = "stocked_merged.parquet"
PRED_PARAMS_FILENAME = "pred_params.pkl"


def get_tweet_data(src_path: Path):
    df_tweets = twitter_ml_utils.load_twitter_raw(src_path)
    return twitter_ml_utils.get_stock_matchable(df=df_tweets)


def is_stock_market_closed(dt: datetime):
    date_str = get_standard_ymd_format(dt)
    max_date = sorted(get_market_holidays())[-1]
    reached_end_of_data = False
    if date_str > max_date:
        reached_end_of_data = True
    is_closed = False
    if dt.weekday() > 4:
        is_closed = True
    else:
        if date_str in get_market_holidays():
            is_closed = True
    return is_closed, reached_end_of_data


def get_all_market_days_in_range(date_range: DateRange):
    current_date_str = date_range.from_date_str
    market_days = []
    while current_date_str < date_range.to_date_str:
        current_dt = date_utils.parse_std_datestring(current_date_str)
        is_closed, reached_end_of_data = is_stock_market_closed(dt=current_dt)
        if reached_end_of_data:
            raise Exception("Reached end of market data. Need to add calendar data.")
        if not is_closed:
            market_days.append(current_date_str)
        current_date_str = get_next_market_date(date_str=current_date_str)

    return market_days


def get_every_nth_tweet_date(nth_sell_day: int, skip_days: int = 0) -> Set[str]:
    now_dt_str = date_utils.get_standard_ymd_format(datetime.now())

    early_dt = date_utils.parse_std_datestring(EARLIEST_TWEET_DATE_STR)
    early_dt = early_dt + timedelta(days=skip_days)
    early_dt_str = date_utils.get_standard_ymd_format(early_dt)

    date_range = DateRange.from_date_strings(from_date_str=early_dt_str, to_date_str=now_dt_str)
    all_mrkt_days = get_all_market_days_in_range(date_range=date_range)
    sell_days = set()
    for ndx, md in enumerate(all_mrkt_days):
        # NOTE: 2021-02-24: chris.flesche: Adds the first date as a tweet date.
        if ndx == 0 or ndx % nth_sell_day == 0:
            sell_days.add(md)

    return sell_days


def add_buy_sell_and_omit(df: pd.DataFrame) -> pd.DataFrame:
    df = twitter_service.add_buy_sell(df=df)
    df = twitter_service.omit_columns(df=df)

    logger.info(f"Oldest date of prepared data (future_date): {df['future_date'].max()}")

    return df


def persist(df_combined: pd.DataFrame, output_parent_path: Path):
    train_path = Path(output_parent_path, STOCKS_MERGED_FILENAME)
    df_combined.to_parquet(train_path)

    logger.info("Persisted df_combined.")


def show_metrics(df: pd.DataFrame, tapp: TrainAndPredictionParams, msg: str):
    df_test = df[df["date"] == tapp.tweet_date_str]
    count = df_test.shape[0]
    logger.info(f"{msg}: {count}")


def merge_with_stocks_for_day(tapp: TrainAndPredictionParams, output_parent_path: Path):
    df = tapp.df
    df = df[df["date"] <= tapp.tweet_date_str].copy()

    df_twitter = easy_convert_columns(df=df)

    df_sd_futured = get_stocks_based_on_tweets(df_tweets=df_twitter, tweet_date_str=tapp.tweet_date_str,
                                               num_hold_days=tapp.num_hold_days, num_days_until_purchase=tapp.num_days_until_purchase)

    # NOTE: 2021-04-04: chris.flesche: .005
    # df_ranked = tipranks_utils.agg_tipranks(df_stocks=df_sd_futured)

    # show_metrics(df_sd_futured, tapp, "df_sd_futured rows")

    df_stock_and_quarter, columns_fundy = combine_with_quarterly_stock_data(df=df_sd_futured)

    # show_metrics(df_stock_and_quarter, tapp, "df_stock_and_quarter rows")

    df_merged = merge_tweets_with_stock_data(df_twitter=df_twitter, df_stock_and_quarter=df_stock_and_quarter)

    # show_metrics(df_merged, tapp, "df_merged rows")

    df_days_until = add_calendar_info(df=df_merged,
                                      tweet_date_str=tapp.tweet_date_str,
                                      columns_fundy=columns_fundy,
                                      num_hold_days=tapp.num_hold_days,
                                      oldest_tweet_date=tapp.oldest_tweet_date)

    # NOTE: 2021-04-07: chris.flesche: min_volume: Experimental. Negative correlation when min_volume == 100000
    min_volume = None
    df_refined = twitter_service.refine_pool(df=df_days_until, min_volume=min_volume, max_price=None)

    # df_ticker_hotted, narrow_cols = one_hot(df=df_refined)

    # NOTE: 2021-03-12: chris.flesche: Chop! Remove all rows after this date
    df_ready = df_refined[df_refined["date"] <= tapp.tweet_date_str].copy()

    if df_ready is not None and df_ready.shape[0] > 0:

        # NOTE: 2021-03-12: chris.flesche: Is tweet_date available in df?
        df_test = df_ready[df_ready["date"] == tapp.tweet_date_str].copy()
        if df_test is not None and df_test.shape[0] > 0:

            logger.info(f"df_test after prep_predict: {df_test.shape[0]}")

            # NOTE: 2021-03-12: chris.flesche: Guard against data leaks; modify all "future_date" columns to get information
            # from either present or past.
            df_sealed = seal_label_leak(df=df_ready, purchase_date_str=tapp.purchase_date_str)

            # NOTE: 2021-03-12: chris.flesche: Add buy_sell labels; drop leaky columns.
            df_combined = add_buy_sell_and_omit(df=df_sealed)

            # NOTE: 2021-03-12: chris.flesche: Persist
            persist(output_parent_path=output_parent_path, df_combined=df_combined)

        else:
            logger.info("Not enough df_prepped data after 'prep_predict'.")
    else:
        logger.info("Not enough training data after 'prep_predict'.")

    return


def process(src_dir_path: Path, dest_dir_path: Path,
            max_date_str: str = None,
            sample_fraction: float = None,
            num_hold_days: int = 1,
            min_price: float = 0.):
    logger.info(f"Getting tweet data from {src_dir_path}")
    df: pd.DataFrame = get_tweet_data(src_path=src_dir_path)
    if sample_fraction is not None:
        df = df.sample(frac=sample_fraction)

    col_new_date = "f22_tweet_applied_date"
    if col_new_date in df.columns:
        logger.info("Found new column 'f22_tweet_applied_date'! Renaming to 'date' ...")
        df.rename(columns={col_new_date: "date"}, inplace=True)

    df.dropna(subset=["date"], inplace=True)

    if max_date_str is None:
        max_date_str = df["date"].max()
        dt_tweet = date_utils.parse_std_datestring(max_date_str)
        is_good = is_good_market_date(dt_tweet)
        if not is_good:
            max_date_str = get_next_market_date(date_str=max_date_str, is_reverse=True)

        # TODO: 2021-03-24: chris.flesche: Is this needed?
        # max_date_str = get_next_market_date(max_date_str, is_reverse=True)

    logger.info(f"max_date: {max_date_str}")

    tapp = TrainAndPredictionParamFactory.create_generic_trainer(df=df,
                                                                 max_date_str=max_date_str,
                                                                 min_price=min_price,
                                                                 num_hold_days=num_hold_days,
                                                                 require_balance=False)

    pred_path_str = Path(dest_dir_path, PRED_PARAMS_FILENAME)
    pickle_service.save(tapp, file_path=pred_path_str)

    logger.info(f"Num tweets: {df.shape[0]}")

    df.sort_values(by=["date"], inplace=True)
    df.dropna(subset=["date"], inplace=True)

    tapp.df = df

    logger.info(f"Will process data starting with {tapp.tweet_date_str} ...")

    merge_with_stocks_for_day(tapp=tapp, output_parent_path=dest_dir_path)


def get_stock_merge_trainer_params(stock_merge_drop_path: Path) -> TrainAndPredictionParams:
    tran_and_pred_path = Path(stock_merge_drop_path, PRED_PARAMS_FILENAME)
    return pickle_service.load(tran_and_pred_path)


def start(src_dir_path: Path, dest_dir_path: Path,
          should_delete_leftovers: bool,
          sample_fraction: float = None,
          num_hold_days: int = 1,
          min_price: float = 0.):
    ensure_dir(dest_dir_path)

    batchy_bae.ensure_clean_output_path(dest_dir_path, should_delete_remaining=should_delete_leftovers)

    process(src_dir_path=src_dir_path,
            dest_dir_path=dest_dir_path,
            sample_fraction=sample_fraction,
            num_hold_days=num_hold_days,
            min_price=min_price)


if __name__ == '__main__':
    # twit_root_path = Path(constants.TEMP_PATH, "twitter") # Path(constants.TWITTER_OUTPUT_RAW_PATH)  #
    twit_root_path = Path(constants.TWITTER_OUTPUT_RAW_PATH)

    src_dir_path = Path(twit_root_path, "refined_tweets_bucket")
    dest_dir_path = Path(twit_root_path, "stock_merge_drop", "main")

    sample_frac = 1  # None  # .4
    num_hold_days = 1
    min_price = .0

    start(src_dir_path=src_dir_path,
          dest_dir_path=dest_dir_path,
          should_delete_leftovers=True,
          sample_fraction=sample_frac,
          num_hold_days=num_hold_days,
          min_price=min_price)

    slack_service.send_direct_message_to_chris("Stocks merged w tweets.")