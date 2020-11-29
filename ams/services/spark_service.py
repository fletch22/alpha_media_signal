import findspark
from pyspark.sql import SparkSession


def _get_or_create(app_name):
    spark_session = SparkSession.builder \
        .master("local[*]") \
        .appName(app_name) \
        .config("spark.executor.memory", "2g") \
        .getOrCreate()
    spark_session.sparkContext.setCheckpointDir("c://tmp")

    return spark_session


def get_or_create(app_name: str):
    findspark.init()
    spark_session = _get_or_create(app_name)
    spark_session.stop()
    spark_session = _get_or_create(app_name)

    return spark_session
