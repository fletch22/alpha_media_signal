import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytz
from pyspark.sql import DataFrame

from ams.config import constants
from ams.config import logger_factory
from ams.config.constants import ensure_dir
from ams.pipes import batchy_bae
from ams.services import file_services
from ams.utils import date_utils
from ams.utils.date_utils import TZ_AMERICA_NEW_YORK, STANDARD_DAY_FORMAT

pd.set_option('display.max_rows', 5000)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

logger = logger_factory.create(__name__)


def persist_parquet(df: DataFrame, parent_dir: str):
    file_path = file_services.create_unique_filename(parent_dir=parent_dir, prefix="lpd", extension="parquet")
    file_path_str = str(file_path)
    df.to_parquet(file_path_str)


def convert_to_date_string(utc_timestamp: int):
    dt_utc = datetime.fromtimestamp(utc_timestamp)
    dt_nyc = dt_utc.astimezone(pytz.timezone(TZ_AMERICA_NEW_YORK))
    return dt_nyc.strftime(STANDARD_DAY_FORMAT)


def calc_compound_score(row):
    return row["user_followers_count"] * row["f22_sentiment_compound"]


def add_ts(date_string: str):
    result = None
    try:
        dt = datetime.strptime(date_string, date_utils.TWITTER_LONG_FORMAT)
        result = int(dt.timestamp())
    except Exception as e:
        pass
    return result


def add_timestamp(df):
    logger.info(f'Count: {df.shape[0]}')

    df = df[df['created_at'].notnull()]

    df["created_at_timestamp"] = df["created_at"].apply(add_ts)

    return df


def get_fave_cols(df):
    return df[["created_at",
               "user_screen_name",
               "favorite_count",
               "in_reply_to_screen_name",
               "user_verified",
               "metadata_result_type",
               "user_listed_count",
               "user_time_zone",
               "user_has_extended_profile",
               "in_reply_to_status_id",
               "user_statuses_count",
               "user_protected",
               "user_is_translation_enabled",
               "user_location",
               "lang",
               "user_geo_enabled",
               "place_country",
               "place_name",
               "possibly_sensitive",
               "user_friends_count",
               "retweet_count",
               "user_follow_request_sent",
               "user_followers_count",
               "f22_ticker",
               "f22_has_cashtag",
               "f22_ticker_in_text",
               "f22_num_other_tickers_in_tweet",
               "f22_sentiment_pos",
               "f22_sentiment_neu",
               "f22_sentiment_neg",
               "f22_sentiment_compound",
               "f22_id"]]


def process(source_dir_path: Path, output_dir_path: Path):
    file_paths = file_services.list_files(source_dir_path, ends_with=".parquet.in_transition", use_dir_recursion=True)

    tot_files = len(file_paths)

    file_paths = sorted(file_paths)

    total_count = 0
    for f_ndx, f in enumerate(file_paths):
        logger.info(f"Processing {f_ndx + 1} of {tot_files}: {f}")
        df = pd.read_parquet(f)

        df = get_fave_cols(df)

        df = add_timestamp(df)

        df = df[df['created_at_timestamp'].notnull()]

        df['date'] = df['created_at_timestamp'].apply(convert_to_date_string)

        df = df[df['user_followers_count'].notnull()]
        df = df[df['f22_sentiment_compound'].notnull()]

        df['user_followers_count'] = df['user_followers_count'].astype("int")
        df['f22_sentiment_compound'] = df['f22_sentiment_compound'].astype("float64")

        # NOTE: 2021-02-06: chris.flesche: Mysteriously, setting this value here made the following line work.
        df['f22_compound_score'] = None

        df['f22_compound_score'] = df.apply(calc_compound_score, axis=1)

        df = df.drop_duplicates(subset=['f22_id'])

        total_count += df.shape[0]

        logger.info(f"Columns: {df.columns}")

        persist_parquet(df=df, parent_dir=str(output_dir_path))

    logger.info(f"Total records processed: {total_count}")


def start(source_dir_path: Path, twitter_root_path: Path, snow_plow_stage: bool):
    file_services.unnest_files(parent=source_dir_path, target_path=source_dir_path, filename_ends_with=".parquet")

    output_dir_path = Path(twitter_root_path, 'learning_prep_drop', "main")
    ensure_dir(output_dir_path)

    batchy_bae.ensure_clean_output_path(output_dir_path)

    batchy_bae.start(source_path=source_dir_path, output_dir_path=output_dir_path, process_callback=process, should_archive=False, snow_plow_stage=snow_plow_stage)

    return output_dir_path


def start_old():
    source_dir_path = Path(constants.TWITTER_OUTPUT_RAW_PATH, "sent_drop", "main")
    file_services.unnest_files(parent=source_dir_path, target_path=source_dir_path, filename_ends_with=".parquet")

    output_dir_path = Path(constants.TWITTER_OUTPUT_RAW_PATH, 'learning_prep_drop', "main")
    os.makedirs(output_dir_path, exist_ok=True)

    batchy_bae.ensure_clean_output_path(output_dir_path)

    batchy_bae.start(source_path=source_dir_path, output_dir_path=output_dir_path, process_callback=process, should_archive=False)