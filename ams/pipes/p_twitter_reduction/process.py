import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F, types as T

from ams.config import logger_factory, constants
from ams.config.constants import ensure_dir
from ams.pipes import batchy_bae
from ams.services import file_services, spark_service, dataframe_services
from ams.services.twitter_service import COL_AFTER_HOURS
from ams.utils import date_utils

logger = logger_factory.create(__name__)

ORG_COLS = ["f22_ticker", "date"]
fraction = 1.


def group_and_reduce_spark(df: DataFrame):
    def is_after_n_closed(created_at_timestamp: str):
        return date_utils.is_after_nasdaq_closed(utc_timestamp=int(created_at_timestamp))

    is_after_closed_udf = F.udf(is_after_n_closed, T.BooleanType())

    df = df.withColumn(COL_AFTER_HOURS, is_after_closed_udf(F.col("created_at_timestamp")))

    df = df.withColumn("f22_has_cashtag", F.col("f22_has_cashtag").cast(T.BooleanType()).cast(T.IntegerType()))
    df = df.withColumn("user_has_extended_profile", F.col("user_has_extended_profile").cast(T.BooleanType()).cast(T.IntegerType()))
    df = df.withColumn("user_verified", F.col("user_verified").cast(T.BooleanType()).cast(T.IntegerType()))
    df = df.withColumn("user_location", F.col("user_location").cast(T.BooleanType()).cast(T.IntegerType()))
    df = df.withColumn("user_geo_enabled", F.col("user_geo_enabled").cast(T.BooleanType()).cast(T.IntegerType()))
    df = df.withColumn("possibly_sensitive", F.col("possibly_sensitive").cast(T.BooleanType()).cast(T.IntegerType()))
    df = df.withColumn("f22_ticker_in_text", F.col("f22_ticker_in_text").cast(T.BooleanType()).cast(T.IntegerType()))
    df = df.withColumn("f22_is_tweet_after_hours", F.col("f22_is_tweet_after_hours").cast(T.BooleanType()).cast(T.IntegerType()))
    df = df.withColumn("user_follow_request_sent", F.col("user_follow_request_sent").cast(T.BooleanType()).cast(T.IntegerType()))

    logger.info(f"Columns before the groupBy: {df.columns}")
    df = df.groupBy(*ORG_COLS) \
        .agg(F.mean("created_at").alias("created_at"),
             F.mean("user_time_zone").alias("user_time_zone"),
             F.mean("user_verified").alias("user_verified"),
             F.mean("user_geo_enabled").alias("user_geo_enabled"),
             F.mean("user_location").alias("user_location"),
             F.mean("favorite_count").alias("favorite_count"),
             F.mean("user_has_extended_profile").alias("user_has_extended_profile"),
             F.mean("user_follow_request_sent").alias("user_follow_request_sent"),
             F.mean("user_listed_count").alias("user_listed_count"),
             F.mean("user_friends_count").alias("user_friends_count"),
             F.sum("retweet_count").alias("retweet_count"),
             F.mean("user_followers_count").alias("user_followers_count"),
             F.mean("user_statuses_count").alias("user_statuses_count"),
             F.mean("f22_ticker_in_text").alias("f22_ticker_in_text"),
             F.mean("f22_has_cashtag").alias("f22_has_cashtag"),
             F.mean("f22_num_other_tickers_in_tweet").alias("f22_num_other_tickers_in_tweet"),
             F.mean("f22_sentiment_pos").alias("f22_sentiment_pos"),
             F.mean("f22_sentiment_neu").alias("f22_sentiment_neu"),
             F.mean("f22_sentiment_neg").alias("f22_sentiment_neg"),
             F.mean("f22_sentiment_compound").alias("f22_sentiment_compound"),
             F.mean("f22_compound_score").alias("f22_compound_score"),
             F.mean("f22_is_tweet_after_hours").alias("f22_is_tweet_after_hours"),
             F.size(F.collect_list("f22_sentiment_compound")).alias("f22_day_tweet_count")
             )

    logger.info(f"Columns after the groupBy: {df.columns}")

    return df


def process_with_spark(source_dir_path: Path, output_dir_path: Path):
    spark = spark_service.get_or_create('twitter')

    file_paths = file_services.list_files(parent_path=source_dir_path, ends_with=".parquet.in_transition", use_dir_recursion=True)

    f_path_strs = [str(f) for f in file_paths]
    logger.info(f"Processing {len(f_path_strs)} files ...")
    df = spark.read.parquet(*f_path_strs)

    with TemporaryDirectory() as td:
        df = group_and_reduce_spark(df=df)

        output_parq_path = dataframe_services.persist_dataframe(df=df, output_drop_folder_path=Path(td), prefix="twitter_reduce", num_output_files=1)

        parq_list = file_services.list_files(parent_path=output_parq_path, ends_with=".parquet")
        for p in parq_list:
            target_path = Path(output_dir_path, p.name)
            logger.info(f"Moving {p} to {target_path}")
            shutil.move(p, target_path)


def get_output_dir(twitter_root_path: Path):
    output_dir = Path(twitter_root_path, "great_reduction", "main")
    ensure_dir(output_dir)
    return output_dir


def start(source_dir_path: Path, twitter_root_path: Path, snow_plow_stage: bool, should_delete_leftovers: bool):
    file_services.unnest_files(parent=source_dir_path, target_path=source_dir_path, filename_ends_with=".parquet")

    output_dir_path = get_output_dir(twitter_root_path=twitter_root_path)
    ensure_dir(output_dir_path)

    batchy_bae.ensure_clean_output_path(output_dir_path, should_delete_remaining=should_delete_leftovers)

    batchy_bae.start(source_path=source_dir_path, out_dir_path=output_dir_path,
                     process_callback=process_with_spark, should_archive=False,
                     snow_plow_stage=snow_plow_stage, should_delete_leftovers=should_delete_leftovers)

    return output_dir_path


if __name__ == '__main__':
    twit_root_dir = Path(f"{constants.TEMP_PATH}2", "twitter")
    src_dir_path = Path(twit_root_dir, "learning_prep_drop", "main")

    start(source_dir_path=src_dir_path,
          twitter_root_path=twit_root_dir,
          snow_plow_stage=False,
          should_delete_leftovers=False)