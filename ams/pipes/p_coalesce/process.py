from pathlib import Path

from ams.config import logger_factory
from ams.config.constants import ensure_dir
from ams.pipes import batchy_bae
from ams.services import file_services, spark_service

logger = logger_factory.create(__name__)


def process(source_dir_path: Path, output_dir_path: Path):
    spark = spark_service.get_or_create(app_name='twitter')
    sc = spark.sparkContext
    log4jLogger = sc._jvm.org.apache.log4j
    LOGGER = log4jLogger.LogManager.getLogger(__name__)
    LOGGER.info("pyspark script logger initialized")

    files = file_services.list_files(parent_path=source_dir_path, ends_with=".parquet.in_transition")
    files = [f for f in files if f.stat().st_size > 0]
    files = [str(f) for f in files]

    df = spark.read.parquet(*files)

    out_write_path = Path(output_dir_path, "out")
    num_records = df.count()
    num_coalesce = 100
    if num_records < 1000:
        num_coalesce = 1

    # NOTE: 2021-02-06: chris.flesche: This oddity seems necessary only when the file sizes are very small.
    # I suspect type inferences is going on here, and the data is dirty. So some data might be binary, some int, etc.
    from pyspark.sql import functions as F, types as T
    df = df.withColumn("place_full_name", F.col('place_full_name').cast(T.StringType()))

    df.coalesce(num_coalesce).write.format("parquet").mode("overwrite").save(str(out_write_path))


def start(source_dir_path: Path, dest_dir_path: Path, snow_plow_stage: bool, should_delete_leftovers: bool):
    file_services.unnest_files(parent=source_dir_path, target_path=source_dir_path, filename_ends_with=".parquet")

    ensure_dir(dest_dir_path)

    batchy_bae.ensure_clean_output_path(dest_dir_path, should_delete_remaining=should_delete_leftovers)

    batchy_bae.start_drop_processing(source_path=source_dir_path, out_dir_path=dest_dir_path,
                                     process_callback=process, should_archive=False,
                                     snow_plow_stage=snow_plow_stage, should_delete_leftovers=should_delete_leftovers)
