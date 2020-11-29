from datetime import datetime
from pathlib import Path

import pandas as pd
import pyspark
import pytz
from pyspark.sql import functions as F, types as T

from ams.config import constants
from ams.services import file_services
from ams.utils.date_utils import TZ_AMERICA_NEW_YORK, STANDARD_DAY_FORMAT, convert_timestamp_to_nyc_date_str


def create_date_column(df: pyspark.sql.DataFrame):
    def convert_to_date_string(utc_timestamp: int):
        # return date_utils.convert_timestamp_to_nyc_date_str(utc_timestamp=utc_timestamp)
        dt_utc = datetime.fromtimestamp(utc_timestamp)
        dt_nyc = dt_utc.astimezone(pytz.timezone(TZ_AMERICA_NEW_YORK))
        return dt_nyc.strftime(STANDARD_DAY_FORMAT)

    parse_udf = F.udf(convert_to_date_string, T.StringType())

    return df.withColumn('date', convert_timestamp_to_nyc_date_str(F.col('created_at_timestamp')))
