import findspark
from pyspark import SparkConf
from pyspark.sql import SparkSession


def _get_or_create(app_name):
    max_cores = 10

    conf = SparkConf()
    conf.setAppName(app_name)
    conf.setMaster(f"local[{max_cores}]")
    # conf.set("spark.driver.cores", max_cores)
    conf.set("spark.driver.memory", "12g")
    # conf.set("spark.cores.max", max_cores)
    conf.set("spark.local.dir", "c:\\tmp\\spark-temp\\")
    conf.set("spark.executor.heartbeatInterval", "2400s")
    conf.set("spark.storage.blockManagerSlaveTimeoutMs", "2400s")
    conf.set("spark.network.timeout", "3600s")

    spark_session = SparkSession.builder \
        .config(conf=conf) \
        .getOrCreate()

    print(spark_session.sparkContext.uiWebUrl)

    return spark_session


def get_or_create(app_name: str):
    findspark.init()
    spark_session = _get_or_create(app_name)
    spark_session.stop()
    spark_session = _get_or_create(app_name)

    return spark_session
