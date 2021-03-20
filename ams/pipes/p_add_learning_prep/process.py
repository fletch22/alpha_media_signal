from datetime import datetime
from pathlib import Path

import pandas as pd
import pytz
from pandas import DataFrame
from pyspark.sql import functions as F, types as T

from ams.config import logger_factory, constants
from ams.config.constants import ensure_dir
from ams.pipes import batchy_bae
from ams.services import file_services, spark_service, dataframe_services
from ams.utils import date_utils
from ams.utils.date_utils import TZ_AMERICA_NEW_YORK, STANDARD_DAY_FORMAT, ensure_market_date

pd.set_option('display.max_rows', 5000)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

logger = logger_factory.create(__name__)

FAVE_COLS = ["created_at",
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
             "f22_id"]


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
    except BaseException as e:
        pass
    return result


def add_timestamp(df):
    logger.info(f'Count: {df.shape[0]}')

    df = df[df['created_at'].notnull()]

    df["created_at_timestamp"] = df["created_at"].apply(add_ts)

    return df


def process_with_spark(source_dir_path: Path, output_dir_path: Path):
    spark = spark_service.get_or_create("twitter")

    file_paths = file_services.list_files(source_dir_path, ends_with=".parquet.in_transition", use_dir_recursion=True)

    f_path_strs = [str(f) for f in file_paths]

    logger.info(f"Processing {len(f_path_strs)} files")
    df = spark.read.parquet(*f_path_strs)
    df = df.drop_duplicates(subset=['f22_id'])

    df = df.select(FAVE_COLS)

    df = df.na.drop(subset=["created_at"])

    add_ts_udf = F.udf(add_ts, T.IntegerType())
    df = df.withColumn("created_at_timestamp", add_ts_udf(F.col("created_at")))
    df = df.na.drop(subset=["created_at_timestamp"])

    convert_to_date_string_udf = F.udf(convert_to_date_string, T.StringType())
    df = df.withColumn("date", convert_to_date_string_udf(F.col("created_at_timestamp")))

    # TODO: 2021-03-15: chris.flesche: Uncomment when ready to reprocess all data.
    # ensure_market_date_udf = F.udf(ensure_market_date, T.StringType())
    # df = df.withColumn("applied_date", ensure_market_date_udf(F.col("date")))

    df.na.fill(value=0, subset=["user_followers_count", "f22_sentiment_compound"])

    df = df.withColumn("user_followers_count", F.col("user_followers_count").cast(T.IntegerType()))
    df = df.withColumn("f22_sentiment_compound", F.col("f22_sentiment_compound").cast(T.FloatType()))
    df = df.withColumn("f22_compound_score", F.col("user_followers_count") * F.col("f22_sentiment_compound"))

    logger.info(f"All the columns: {df.columns}")

    dataframe_services.persist_dataframe(df=df, output_drop_folder_path=output_dir_path, prefix="add_learning_prep")


def start(source_dir_path: Path, dest_dir_path: Path, snow_plow_stage: bool, should_delete_leftovers: bool):
    file_services.unnest_files(parent=source_dir_path, target_path=source_dir_path, filename_ends_with=".parquet")

    ensure_dir(dest_dir_path)

    batchy_bae.ensure_clean_output_path(dest_dir_path, should_delete_remaining=should_delete_leftovers)

    batchy_bae.start_drop_processing(source_path=source_dir_path, out_dir_path=dest_dir_path,
                                     process_callback=process_with_spark, should_archive=False,
                                     snow_plow_stage=snow_plow_stage, should_delete_leftovers=should_delete_leftovers)


if __name__ == '__main__':
    twit_root_dir = Path(f"{constants.TEMP_PATH}2", "twitter")
    src_dir_path = Path(twit_root_dir, "sent_drop", "main")

    start(source_dir_path=src_dir_path,
          twitter_root_path=twit_root_dir,
          snow_plow_stage=False,
          should_delete_leftovers=False)