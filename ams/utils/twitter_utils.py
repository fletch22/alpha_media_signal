import os
from datetime import datetime
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
import pyspark
import pytz
from pyspark.sql import functions as F

from ams.config import constants, logger_factory
from ams.services import file_services
from ams.utils import date_utils
from ams.utils.date_utils import TZ_AMERICA_NEW_YORK, STANDARD_DAY_FORMAT, TWITTER_FORMAT, convert_timestamp_to_nyc_date_str_udf

logger = logger_factory.create(__name__)


def create_date_column(df: pyspark.sql.DataFrame):
    return df.withColumn('date', convert_timestamp_to_nyc_date_str_udf(F.col('created_at_timestamp')))


def get_time_from_json(json):
    # "tweet": {"created_at": "
    date_str = None
    raw_date_str = extract_raw_date_from_tweet(json)
    if raw_date_str is not None:
        try:
            dt = date_utils.parse_twitter_dt_str_to_dt(raw_date_str)
            date_str = date_utils.get_standard_ymd_format(dt)
        except Exception as e:
            pass

    return date_str


def extract_raw_date_from_tweet(json: str):
    token = "\"created_at\": \""
    raw_date_str = None
    try:
        if type(json) == bytes:
            json = json.decode("utf-8")

        token_ndx = json.index(token)
        if token_ndx > 0:
            json_pref = json[token_ndx + len(token):]
            end_token = "\", \""
            end_ndx = json_pref.index(end_token)
            raw_date_str = json_pref[:end_ndx]
    except Exception as e:
        pass

    return raw_date_str


def get_youngest_tweet_from_line(line: str, youngest_dt_str: str):
    if len(line) > 0:
        date_str = get_time_from_json(json=line)
        if date_str is not None and (youngest_dt_str is None or date_str > youngest_dt_str):
            youngest_dt_str = date_str
    return youngest_dt_str


def get_youngest_raw_textfile_tweet(source_path: Path):
    files = file_services.list_files(parent_path=source_path)
    youngest_dt_str = None
    logger.info(f"Number of files to search in source: {len(files)}")
    for ndx, f in enumerate(files):
        logger.info(f"Reading file {f} ({ndx + 1} of {len(files)})")
        with open(str(f), 'rb') as f:
            f.seek(-2, os.SEEK_END)
            while f.read(1) != b'\n':
                f.seek(-2, os.SEEK_CUR)
            last_line = f.readline().decode()
            youngest_dt_str = get_youngest_tweet_from_line(line=last_line, youngest_dt_str=youngest_dt_str)

    return youngest_dt_str


def get_max_date_from_all_lines(youngest_dt_str, r):
    count = 0
    # while True:
    #     rez = r.readline()
    #     if rez is None or len(rez.strip()) == 0:
    #         break
    line = ""
    for line in r:
        logger.info("reading line")
        pass
    last_line = line

    youngest_dt_str = get_youngest_tweet_from_line(line=last_line, youngest_dt_str=youngest_dt_str)
    count += 1
    return youngest_dt_str


def convert_to_date_string(utc_timestamp: int):
    dt_utc = pd.datetime.fromtimestamp(utc_timestamp)
    dt_nyc = dt_utc.astimezone(pytz.timezone(TZ_AMERICA_NEW_YORK))
    return dt_nyc.strftime(STANDARD_DAY_FORMAT)


def add_ts(date_string: str):
    result = None
    try:
        dt = datetime.strptime(date_string, TWITTER_FORMAT)
        result = int(dt.timestamp())
    except Exception as e:
        pass
    return result


def get_youngest_end_drop_tweet():
    files = file_services.list_files(constants.REFINED_TWEETS_BUCKET_PATH, ends_with=".parquet")
    yougest_date_str = ""
    for f in files:
        logger.info(f"Reading {f}")
        df = pd.read_parquet(f)

        created_at_str = df["date"].max()

        if created_at_str > yougest_date_str:
            yougest_date_str = created_at_str

    return yougest_date_str


def get_youngest_tweet_date_in_system():
    raw_drop_path = Path(constants.TWITTER_OUTPUT_RAW_PATH, "raw_drop", "main")
    youngest_raw_dt_str = get_youngest_raw_textfile_tweet(source_path=raw_drop_path)

    youngest_end_drop_dt_str = get_youngest_end_drop_tweet()

    youngest_tweet_dt_str = youngest_end_drop_dt_str
    if youngest_raw_dt_str is not None and youngest_raw_dt_str > youngest_end_drop_dt_str:
        youngest_tweet_dt_str = youngest_raw_dt_str

    return youngest_tweet_dt_str
