import json
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType


def get_twitter_schema(spark: SparkSession, twitter_sample_path: Path):
    df_sample = (spark
                 .read
                 .option("multiLine", True)
                 .json(str(twitter_sample_path)))

    tweet_schema_tmp = df_sample.schema

    output_path = Path(twitter_sample_path.parent, "tweet_schema.json")
    with open(str(output_path), '+w') as w:
        w.writelines(tweet_schema_tmp.json())

    with open(output_path, 'r+') as r:
        json_str = r.readline()
        thing = json.loads(json_str)

        tweet_schema = StructType.fromJson(thing)
        return tweet_schema


def get_twitter_schema_json(schema_path: Path):
    with open(str(schema_path), 'r+') as r:
        json_str = r.readline()
        return json_str
