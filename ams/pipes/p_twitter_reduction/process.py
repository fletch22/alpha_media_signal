import os
from datetime import timedelta
from pathlib import Path

import pandas as pd

from ams.config import constants
from ams.config.constants import ensure_dir
from ams.pipes import batchy_bae
from ams.services import file_services, twitter_service
from ams.utils import date_utils

ORG_COLS = ["f22_ticker", "date"]
fraction = 1.


def increment_day_if_tweet_after_hours(row: pd.Series):
    date_str = row["date"]
    is_tweet_after_hours = row["f22_is_tweet_after_hours"]

    if is_tweet_after_hours:
        dt = date_utils.parse_std_datestring(date_str)
        dt = dt + timedelta(days=1)
        date_str = date_utils.get_standard_ymd_format(dt)

    return date_str


def group_and_reduce(df: pd.DataFrame):
    df = twitter_service.add_is_tweet_after_hours(df=df)
    df["date"] = df.apply(increment_day_if_tweet_after_hours, axis=1)

    df_all = []
    df_g = df.groupby(ORG_COLS)
    for ndx, (group_name, df_group) in enumerate(df_g):
        # TODO: 2021-01-17: chris.flesche: Expermiment with summing instead for counts.
        df_group["favorite_count"] = df_group["favorite_count"].mean()
        df_group["user_listed_count"] = df_group["user_listed_count"].mean()
        df_group["user_friends_count"] = df_group["user_friends_count"].mean()
        df_group["retweet_count"] = df_group["retweet_count"].mean()
        df_group["user_followers_count"] = df_group["user_followers_count"].mean()
        df_group["f22_has_cashtag"] = df_group["f22_has_cashtag"].mean()
        df_group["f22_num_other_tickers_in_tweet"] = df_group["f22_num_other_tickers_in_tweet"].mean()
        df_group["f22_sentiment_pos"] = df_group["f22_sentiment_pos"].mean()
        df_group["f22_sentiment_neu"] = df_group["f22_sentiment_neu"].mean()
        df_group["f22_sentiment_neg"] = df_group["f22_sentiment_neg"].mean()
        df_group["f22_sentiment_compound"] = df_group["f22_sentiment_compound"].mean()
        df_group["f22_compound_score"] = df_group["f22_compound_score"].mean()
        # TODO: 2021-01-17: chris.flesche: Enable this when doing full recalc
        # df_group["f22_day_tweet_count"] = df_group.shape[0]

        df_all.append(df_group)

    df = None
    if len(df_all) > 0:
        df = pd.concat(df_all, axis=0).reset_index(drop=True)
        df = df.drop_duplicates(subset=ORG_COLS)

    return df


def process(source_dir_path: Path, output_dir_path: Path):
    all_dfs = []

    file_paths = file_services.list_files(parent_path=source_dir_path, ends_with=".parquet.in_transition", use_dir_recursion=True)

    num_files = len(file_paths)
    for ndx, f in enumerate(file_paths):
        print(f"Processing {ndx + 1} of {num_files}: '{f}'.")
        df = pd.read_parquet(f)

        df_reduced = group_and_reduce(df=df)
        if df_reduced is not None:
            all_dfs.append(df_reduced)

    df_twitter_raw = pd.concat(all_dfs, axis=0)
    df_twitter_raw = group_and_reduce(df=df_twitter_raw)

    output_path = file_services.create_unique_filename(parent_dir=str(output_dir_path), prefix="twitter_reduce", extension="parquet")
    df_twitter_raw.to_parquet(str(output_path))

    print(f"Total records processed: {df_twitter_raw.shape[0]}")


def get_output_dir(twitter_root_path: Path):
    output_dir = Path(twitter_root_path, "great_reduction", "main")
    ensure_dir(output_dir)
    return output_dir


def start(source_dir_path: Path, twitter_root_path: Path, snow_plow_stage: bool):
    output_dir_path = get_output_dir(twitter_root_path=twitter_root_path)
    ensure_dir(output_dir_path)

    batchy_bae.ensure_clean_output_path(output_dir_path)

    batchy_bae.start(source_path=source_dir_path, output_dir_path=output_dir_path, process_callback=process, should_archive=False, snow_plow_stage=snow_plow_stage)

    return output_dir_path


def start_old():
    source_dir_path = Path(constants.TWITTER_OUTPUT_RAW_PATH, "learning_prep_drop", "main")
    output_dir_path = Path(constants.TWITTER_OUTPUT_RAW_PATH, "great_reduction", "main")
    os.makedirs(output_dir_path, exist_ok=True)

    batchy_bae.start(source_path=source_dir_path, output_dir_path=output_dir_path, process_callback=process, should_archive=False)

    return output_dir_path