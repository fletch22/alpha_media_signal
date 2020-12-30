import findspark
from pyspark import SparkConf
from pyspark.sql import SparkSession


def _get_or_create(app_name):
    conf = SparkConf()
    conf.setAppName(app_name)
    conf.setMaster("local[*]")
    conf.set("spark.driver.cores", "20")
    conf.set("spark.driver.memory", "5g")
    conf.set("spark.cores.max", 20)

    spark_session = SparkSession.builder \
        .config(conf=conf) \
        .getOrCreate()
    spark_session.sparkContext.setCheckpointDir("c://tmp")

    return spark_session


def get_or_create(app_name: str):
    findspark.init()
    spark_session = _get_or_create(app_name)
    spark_session.stop()
    spark_session = _get_or_create(app_name)

    return spark_session
