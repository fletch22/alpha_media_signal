import json

from ams.config import logger_factory
from ams.pipes.p_smallify_files.process import fix_naked_ticker, ensure_unwrap_tweet
from ams.services import spark_service

from pyspark.sql import functions as F, types as T
import pandas as pd

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

logger = logger_factory.create(__name__)

def test_spark():

    bad_token = r'\{"version":\s"0.9.1",\s"f22_ticker":\sticker,\s'
    good_token = '{"version": "0.9.1", "f22_ticker": "ticker", '

    rows = [
        ('{"version": "0.9.1", "f22_ticker": ticker, "fruit": "banana"}',),
        ('{foo ',),
        (good_token,),
        ('foo',),
        ('{"fruit": "apple"}',),
        ('{"version": "0.9.2", "f22_ticker": AAPL, "tweet": {"fruit": "banana"}}',),
        ('{"version": "1.0", "tweet": {"foo": "bar"}}', ),
    ]

    # for r, in rows:
    #     print(f"{ensure_unwrap_tweet(r)}")

    spark = spark_service.get_or_create_tiny("test_twitter")
    df = spark.createDataFrame(rows, ["value"])

    df = df.withColumn("value", F.regexp_replace("value", bad_token, good_token))

    fix_naked_ticker_udf = F.udf(fix_naked_ticker, T.StringType())

    df = df.withColumn("value", fix_naked_ticker_udf(F.col("value")))

    ensure_unwrap_tweet_udf = F.udf(ensure_unwrap_tweet, T.StringType())

    df = df.withColumn("value", ensure_unwrap_tweet_udf(F.col("value")))

    df = df.dropna()

    pdf = df.toPandas()
    logger.info(pdf.head(10))