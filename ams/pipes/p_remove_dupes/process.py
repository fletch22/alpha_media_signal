import gc
from functools import reduce
from pathlib import Path
from typing import List

import pandas as pd
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import StringType

from ams.config import logger_factory
from ams.config.constants import ensure_dir
from ams.pipes import batchy_bae
from ams.services import file_services, spark_service, dataframe_services

logger = logger_factory.create(__name__)

def convert_to_nlp_text(tweet, ticker):
    token = "RT "
    tweet = tweet.strip()
    result = tweet
    if tweet.startswith(token):
        result = tweet[len(token) - 1:]
    return f"{ticker} {result}"


def process(source_dir_path: Path, output_dir_path: Path):
    trunc_udf = F.udf(convert_to_nlp_text, StringType())

    spark = spark_service.get_or_create('twitter')

    files = file_services.list_files(parent_path=source_dir_path, use_dir_recursion=False)
    df_all = []
    max_files = 8
    total_rows = 0
    for f in files:
        logger.info(f"File: {f}")
        df = spark.read.parquet(str(f))
        df = df.where(F.col("text").isNotNull())
        df = df.where(F.col("f22_ticker").isNotNull())
        df = df.withColumn("nlp_text", trunc_udf(F.col("text"), F.col("f22_ticker")))
        df = df.drop_duplicates(["nlp_text"]).drop("text")
        df = df.withColumn("place_full_name", F.col('place_full_name').cast(StringType()))

        # NOTE: 2021-02-28: chris.flesche: Attempted bug fix for "pyspark.sql.utils.AnalysisException: Cannot resolve column name "__index_level_0__"
        if "__index_level_0__" in df.columns:
            df = df.drop("__index_level_0__")

        df_all.append(df)

        if len(df_all) >= max_files:
            total_rows += combine_and_persist(df_all=df_all, output_dir_path=output_dir_path)
            df_all = []

        gc.collect()

    if len(df_all) > 0:
        total_rows += combine_and_persist(df_all=df_all, output_dir_path=output_dir_path)

    logger.info(f"Total records processed: {total_rows}")



def combine_and_persist(df_all: List[pd.DataFrame], output_dir_path: Path):
    df_combined = reduce(DataFrame.unionByName, df_all)

    df_combined = df_combined.drop_duplicates(["nlp_text"])

    dataframe_services.persist_dataframe(df=df_combined, output_drop_folder_path=output_dir_path, prefix='dedupe')

    return df_combined.count()


def start(source_dir_path: Path, dest_dir_path: Path, snow_plow_stage: bool, should_delete_leftovers: bool):

    ensure_dir(dest_dir_path)

    batchy_bae.ensure_clean_output_path(dest_dir_path, should_delete_remaining=should_delete_leftovers)

    batchy_bae.start_drop_processing(source_path=source_dir_path, out_dir_path=dest_dir_path,
                                     process_callback=process, should_archive=False,
                                     snow_plow_stage=snow_plow_stage, should_delete_leftovers=should_delete_leftovers)
