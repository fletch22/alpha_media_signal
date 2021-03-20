from pyspark.sql import SQLContext

def persist(sc, df_tickered):
    sql_context = SQLContext(sc)

    df_test = df_tickered.limit(10000)
    df_test.createOrReplaceTempView("df_test")

    columns = ",".join(df_test.columns)

    sql = """DROP TABLE IF EXISTS TestHiveTableCSV"""
    sql_context.sql(sql)

    sql = f"""CREATE TABLE TestHiveTableCSV ROW FORMAT DELIMITED FIELDS TERMINATED BY ',' LINES TERMINATED BY '\n' AS SELECT {columns} FROM df_test"""

    sql_context.sql(sql)