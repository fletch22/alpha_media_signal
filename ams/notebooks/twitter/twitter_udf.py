from datetime import timedelta

from pyspark.sql import functions as F, types as T

from ams.utils import date_utils


@F.udf(returnType=T.FloatType())
def calc_roi(close, future_close):
    result = 0
    if future_close is None or close is None:
        result = None
    elif close != 0.:
        result = (future_close - close) / close
    return result


@F.udf(returnType=T.IntegerType())
def get_days_from_epoch(date_str: str):
    result = None
    if date_str is not None and len(date_str.strip()) > 0:
        min_date = "1970-01-01"
        min_dt = date_utils.parse_std_datestring(min_date)
        dt = date_utils.parse_std_datestring(datestring=date_str)
        result = (dt - min_dt).days
    return result