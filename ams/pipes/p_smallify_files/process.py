import json
from pathlib import Path

from ams.config import constants, logger_factory
from ams.config.constants import ensure_dir
from ams.pipes import batchy_bae
from ams.services import file_services, spark_service, dataframe_services
from ams.services.dataframe_services import PersistedDataFrameTypes

logger = logger_factory.create(__name__)
from pyspark.sql import functions as F, types as T


def fix_naked_ticker(json_str):
    token = '{"version": "0.9.2", "f22_ticker": '
    if json_str.find(token) == 0:
        start_pos = len(token)
        next_char = json_str[start_pos:start_pos + 1]
        if next_char != "\"":
            end_ticker_pos = json_str.index(",", start_pos)
            ticker = json_str[start_pos:end_ticker_pos]

            json_str = f"""{json_str[:start_pos]}\"{ticker}\"{json_str[end_ticker_pos:]}"""
    return json_str


def process(source_path: Path, output_dir_path: Path):
    process_without_spark(source_path=source_path, output_dir_path=output_dir_path)


def process_without_spark(source_path: Path, output_dir_path: Path):
    files = file_services.walk(source_path, use_dir_recursion=False)

    logger.info(f"Num files to process: {len(files)}")

    max_records_per_file = 50000
    total_records_processed = 0
    for f in files:
        count = 0
        with open(str(f), 'r+') as r:
            wf = None
            create_new = True
            while True:
                if create_new:
                    file_path = file_services.create_unique_filename(parent_dir=str(output_dir_path), prefix="smallified", extension="txt")
                    wf = open(str(file_path), 'a+')
                    create_new = False
                count += 1
                line = r.readline()
                if len(line) == 0:
                    if wf is not None:
                        logger.info("Closing file.")
                        wf.close()
                    break
                try:
                    line = line.replace('{"version": "0.9.1", "f22_ticker": ticker, ', '{"version": "0.9.1", "f22_ticker": "ticker", ')
                    line = fix_naked_ticker(line)
                    obj = json.loads(line)
                    if "version" in obj.keys():
                        obj = obj["tweet"]
                    line_alt = json.dumps(obj)
                    wf.write(line_alt + "\n")
                    total_records_processed += 1
                except Exception as e:
                    pass
                if count % 1000 == 0:
                    logger.info(count)
                if count >= max_records_per_file:
                    if wf is not None:
                        logger.info("Closing file.")
                        wf.close()
                    count = 0
                    create_new = True

    logger.info(f"Total records processed: {total_records_processed}")


def ensure_unwrap_tweet(line):
    try:
        obj = json.loads(line)
        if "version" in obj.keys():
            obj = obj["tweet"]
            line = json.dumps(obj, ensure_ascii=False)
        else:
            line = json.dumps(obj, ensure_ascii=False)
    except BaseException as be:
        line = None

    return line


def process_with_spark(source_path: Path, output_dir_path: Path):
    spark = spark_service.get_or_create("twitter")

    files = file_services.list_files(parent_path=source_path, use_dir_recursion=False)  # , ends_with=".txt.in_transition")
    files = [str(f) for f in files]
    logger.info(f"Num files to process: {len(files)}")

    df = spark.read.option("charset", "UTF-8").text(files)

    # FIXME: 2021-02-28: chris.flesche: Temporary
    # df = df.sample(fraction=.1)

    row_count_orig = df.count()

    bad_token = r'\{"version":\s"0.9.1",\s"f22_ticker":\sticker,\s'
    good_token = '{"version": "0.9.1", "f22_ticker": "ticker", '
    df = df.withColumn('value', F.regexp_replace('value', bad_token, good_token))

    fix_naked_ticker_udf = F.udf(fix_naked_ticker, T.StringType())
    df = df.withColumn("value", fix_naked_ticker_udf(F.col("value")))

    ensure_unwrap_tweet_udf = F.udf(ensure_unwrap_tweet, T.StringType())
    df = df.withColumn("value", ensure_unwrap_tweet_udf(F.col("value")))

    # NOTE: 2021-02-28: chris.flesche: Spark BUG here! Persist is necessary to prevent a crash on save to disk later on.
    df.persist()
    df = df.dropna()

    logger.info(f"Number rows removed during smallify process: {row_count_orig - df.count()}")

    dataframe_services.persist_dataframe(df=df, output_drop_folder_path=output_dir_path,
                                         prefix="smallified", file_type=PersistedDataFrameTypes.TXT)

    df.unpersist()


def start(src_dir_path: Path, dest_dir_path: Path, snow_plow_stage: bool, should_delete_leftovers: bool):

    ensure_dir(dest_dir_path)

    batchy_bae.ensure_clean_output_path(dest_dir_path, should_delete_remaining=should_delete_leftovers)

    batchy_bae.start_drop_processing(source_path=src_dir_path, out_dir_path=dest_dir_path,
                                     process_callback=process_with_spark, should_archive=False, snow_plow_stage=snow_plow_stage, should_delete_leftovers=should_delete_leftovers)


if __name__ == '__main__':
    twitter_output_path = Path(f"{constants.TEMP_PATH}2", "twitter")
    source_dir_path = Path(twitter_output_path, "raw_drop", "main")

    start(src_dir_path=source_dir_path,
          twitter_root_path=twitter_output_path,
          snow_plow_stage=False,
          should_delete_leftovers=False)