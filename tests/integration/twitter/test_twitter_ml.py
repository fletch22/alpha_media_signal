from pyspark.sql import Window
from pyspark.sql import functions as F, types as T

from ams.services import spark_service


def test_window_and_groupby():
    rows = [
        ("apple", "AAPL", "2020-10-01", 5),
        ("apple", "AAPL", "2020-10-01", 1),
    ]

    cols = ["fruit", "ticker", "date", "score"]
    spark = spark_service.get_or_create_tiny("test_twitter")
    df = spark.createDataFrame(rows, cols)

    df = df.withColumn("score", F.col("score").cast(T.IntegerType()))

    df = df.groupBy("ticker", "date").agg(F.first("fruit"),
                                          F.sum("score").alias("score"),
                                          F.size(F.collect_list("score")).alias("f22_score"))

    # df.groupBy("department") \
    # .agg(F.sum("salary").alias("sum_salary"), \
    #      F.avg("salary").alias("avg_salary"), \
    #      F.sum("bonus").alias("sum_bonus"), \
    #      F.max("bonus").alias("max_bonus"))

    print(df.select(cols).toPandas().head())