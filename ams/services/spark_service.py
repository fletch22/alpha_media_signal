from pathlib import Path

from pyspark import SparkConf
from pyspark.sql import SparkSession

from ams.config import logger_factory, constants

logger = logger_factory.create(__name__)


def _get_or_create(app_name, spark_driver_memory: str = "32G", max_cores: int = 15):
    conf = SparkConf()
    conf.setAppName(app_name)
    conf.setMaster(f"local[{max_cores}]")
    conf.set("spark.executor.memory", "1g")
    conf.set("spark.driver.memory", spark_driver_memory)
    conf.set("spark.executor.heartbeatInterval", "2400s")
    conf.set("spark.storage.blockManagerSlaveTimeoutMs", "2400s")
    conf.set("spark.network.timeout", "3600s")
    conf.set("spark.sql.autoBroadcastJoinThreshold", -1)  # NOTE: 2021-02-26: chris.flesche: Attempt to fix "broadcast large binary task with size ..."
    conf.set("spark.sql.shuffle.partitions", 3 * max_cores)
    conf.set("spark.eventLog.enabled", "true")
    conf.set("spark.eventLog.dir", f'{Path(constants.TEMP_PATH, "logs")}')

    spark_session = SparkSession.builder \
        .config(conf=conf) \
        .getOrCreate()

    logger.info(spark_session.sparkContext.uiWebUrl)

    return spark_session


def get_or_create(app_name: str):
    spark_session = _get_or_create(app_name)

    return spark_session


def get_or_create_tiny(app_name: str):
    spark_session = _get_or_create(app_name, spark_driver_memory="1g", max_cores=2)

    return spark_session