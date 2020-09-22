from pyspark.sql import SparkSession


def get_or_create(app_name: str):
    return SparkSession.builder \
        .master("local") \
        .appName(app_name) \
        .getOrCreate()
