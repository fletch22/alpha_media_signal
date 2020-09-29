from pyspark.sql import SparkSession
# import findspark

def get_or_create(app_name: str):
    # findspark.init()
    return SparkSession.builder \
        .master("local[15]") \
        .appName(app_name) \
        .config("spark.executor.memory", "2g") \
        .getOrCreate()

# .config("spark.executor.heartbeatInterval", 36000) \
        # .config("spark.network.timeout", 37000) \
