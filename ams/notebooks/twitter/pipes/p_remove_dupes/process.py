import gc
import os
from functools import reduce
from pathlib import Path
from typing import List

import findspark
import pandas as pd
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import StringType

from ams.config import constants
from ams.notebooks.twitter.pipes import batchy_bae
from ams.services import file_services, spark_service, dataframe_services
from ams.utils.Stopwatch import Stopwatch


def convert_to_nlp_text(tweet, ticker):
    token = "RT "
    tweet = tweet.strip()
    result = tweet
    if tweet.startswith(token):
        result = tweet[len(token) - 1:]
    return f"{ticker} {result}"


def process(source_dir_path: Path, output_dir_path: Path):
    trunc_udf = F.udf(convert_to_nlp_text, StringType())

    findspark.init()
    spark = spark_service.get_or_create('dedupe')

    files = file_services.list_files(parent_path=source_dir_path, use_dir_recursion=False)
    df_all = []
    max_files = 8
    total_rows = 0
    for f in files:
        print(f"File: {f}")
        df = spark.read.parquet(str(f))
        df = df.where(F.col("text").isNotNull())
        df = df.where(F.col("f22_ticker").isNotNull())
        df = df.withColumn("nlp_text", trunc_udf(F.col("text"), F.col("f22_ticker")))
        df = df.drop_duplicates(["nlp_text"]).drop("text")

        df_all.append(df)

        if len(df_all) >= max_files:
            total_rows += combine_and_persist(df_all=df_all, output_dir_path=output_dir_path)
            df_all = []

        gc.collect()

    if len(df_all) > 0:
        combine_and_persist(df_all=df_all, output_dir_path=output_dir_path)

    print(f"Total rows: {total_rows}")


def combine_and_persist(df_all: List[pd.DataFrame], output_dir_path: Path):
    df_combined = reduce(DataFrame.unionByName, df_all)

    df_combined = df_combined.drop_duplicates(["nlp_text"])

    dataframe_services.persist_dataframe(df=df_combined, output_drop_folder_path=output_dir_path, prefix='dedupe', num_output_files=1)

    return df_combined.count()


def start():
    source_dir_path = Path(constants.TWITTER_OUTPUT_RAW_PATH, "id_fixed", "main")
    output_dir_path = Path(constants.TWITTER_OUTPUT_RAW_PATH, "deduped", "main")
    os.makedirs(output_dir_path, exist_ok=True)

    if not file_services.is_empty(output_dir_path):
        raise Exception(f"Output folder '{output_dir_path}' is not empty.")

    batchy_bae.start(source_path=source_dir_path, output_dir_path=output_dir_path, process_callback=process, should_archive=False)

    return output_dir_path


if __name__ == '__main__':
    stopwatch = Stopwatch()
    start()
    stopwatch.end("remove_dupes")
