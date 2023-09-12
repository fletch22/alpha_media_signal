from datetime import datetime, timedelta
from random import random
from typing import Tuple

import pandas as pd
import pytz
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType

from ams.config import constants, logger_factory

TZ_AMERICA_NEW_YORK = 'America/New_York'

STANDARD_DAY_FORMAT = '%Y-%m-%d'
US_DATE_FORMAT = '%m/%d/%Y'
STANDARD_DATE_WITH_SECONDS_FORMAT = '%Y-%m-%d_%H-%M-%S'
WTD_DATE_WITH_SECONDS_FORMAT = '%Y-%m-%d %H:%M:%S'
TWITTER_FORMAT = '%Y%m%D%H%M'

TWITTER_LONG_FORMAT = "%a %b %d %H:%M:%S %z %Y"

NASDAQ_CLOSE_AT_ONE_PM_DATES = ["2020-11-27", "2020-12-24"]

logger = logger_factory.create(__name__)

stock_market_holidays = None


def parse_twitter_date_string_as_timestamp(date_string: str):
    return parse_twitter_dt_str_to_dt(date_string, TWITTER_LONG_FORMAT).timestamp()


def parse_twitter_dt_str_to_dt(date_str: str):
    return datetime.strptime(date_str, TWITTER_LONG_FORMAT)


def parse_std_datestring(datestring):
    return datetime.strptime(datestring, STANDARD_DAY_FORMAT)


def get_us_mdy_format(date: datetime):
    return date.strftime(US_DATE_FORMAT)


def get_standard_ymd_format(date: datetime):
    return date.strftime(STANDARD_DAY_FORMAT)


def format_file_system_friendly_date(date: datetime):
    millis = round(date.microsecond / 1000, 2)
    return f"{date.strftime(STANDARD_DATE_WITH_SECONDS_FORMAT)}-{millis}"


def convert_wtd_nyc_date_to_std(date_str: str):
    dt_nyc = datetime.strptime(date_str, WTD_DATE_WITH_SECONDS_FORMAT)
    nyc_t = pytz.timezone(TZ_AMERICA_NEW_YORK)
    return nyc_t.localize(dt_nyc, is_dst=None)


def convert_wtd_nyc_date_to_utc(date_str: str):
    dt_nyz_tz = convert_wtd_nyc_date_to_std(date_str)
    dt_utc = dt_nyz_tz.astimezone(pytz.utc)

    return dt_utc


def convert_utc_timestamp_to_nyc(utc_timestamp):
    dt_utc = datetime.fromtimestamp(utc_timestamp)
    return dt_utc.astimezone(pytz.timezone(TZ_AMERICA_NEW_YORK))


def get_nasdaq_close_on_date(dt_utc: datetime):
    dtc_nyc = dt_utc.astimezone(pytz.timezone(TZ_AMERICA_NEW_YORK))

    # NOTE: 2020-09-26: chris.flesche: String time off - make midnight
    date_str = get_standard_ymd_format(dtc_nyc)
    dt_closed = parse_std_datestring(date_str)

    # NOTE: 2020-09-26: chris.flesche: Add timezone because above did not.
    tz = pytz.timezone(str(dtc_nyc.tzinfo))
    dt_closed = tz.localize(dt_closed)

    # NOTE: 2020-09-26: chris.flesche: Adjust if today's date has a special close date.
    hours_after_midnight = 13 if date_str in NASDAQ_CLOSE_AT_ONE_PM_DATES else 16
    dt_closed = dt_closed + timedelta(hours=hours_after_midnight)

    # NOTE: 2020-09-26: chris.flesche: Always convert back to utc
    return dt_closed.astimezone(pytz.utc)


def is_after_nasdaq_closed(utc_timestamp: int):
    dt_utc_tz_unaware = datetime.fromtimestamp(utc_timestamp)
    dt_utc = dt_utc_tz_unaware.astimezone(pytz.utc)

    dt_close = get_nasdaq_close_on_date(dt_utc=dt_utc)
    return dt_utc > dt_close


def convert_timestamp_to_nyc_date_str(utc_timestamp):
    dt_nyc = convert_utc_timestamp_to_nyc(utc_timestamp=utc_timestamp)
    return get_standard_ymd_format(dt_nyc)


@udf(returnType=StringType())
def convert_timestamp_to_nyc_date_str_udf(utc_timestamp):
    return convert_timestamp_to_nyc_date_str(utc_timestamp=utc_timestamp)


def skip_market_days(dt: datetime, num_market_days_to_skip: int):
    is_reverse = num_market_days_to_skip < 0
    for i in range(abs(num_market_days_to_skip)):
        dt = find_next_market_open_day(dt=dt, is_reverse=is_reverse)
    return dt


def ensure_market_date(date_str) -> Tuple[str, bool]:
    dt = parse_std_datestring(date_str)
    is_closed_date = is_stock_market_closed(dt=dt)
    valid_dt_found = True
    if is_closed_date:
        date_str, valid_dt_found = get_next_market_date(date_str, is_reverse=True)

    return date_str, valid_dt_found


def get_next_market_date(date_str: str, is_reverse: bool = False) -> Tuple[str, bool]:
    dt = parse_std_datestring(date_str)
    dt, valid_dt_found = find_next_market_open_day(dt, is_reverse=is_reverse)
    dt_str = None
    if valid_dt_found:
        dt_str = get_standard_ymd_format(dt)
    return dt_str, valid_dt_found


def find_next_market_open_day(dt: datetime, is_reverse: bool = False) -> Tuple[datetime, bool]:
    day = -1 if is_reverse else 1
    valid_dt_found = True
    while True:
        dt = dt + timedelta(days=day)
        is_closed, reached_end_of_data = is_stock_market_closed(dt)
        if reached_end_of_data:
            valid_dt_found = False
            break
        if not is_closed:
            break
    return dt, valid_dt_found


def is_stock_market_closed(dt: datetime):
    date_str = get_standard_ymd_format(dt)
    max_date = sorted(get_market_holidays())[-1]
    reached_end_of_data = False
    if date_str > max_date:
        reached_end_of_data = True
    is_closed = False
    if dt.weekday() > 4:
        is_closed = True
    else:
        if date_str in get_market_holidays():
            is_closed = True
    return is_closed, reached_end_of_data


def get_market_holidays() -> str:
    global stock_market_holidays
    if stock_market_holidays is None:
        df = pd.read_csv(constants.US_MARKET_HOLIDAYS_PATH, header=0)
        df['date'] = pd.to_datetime(df['date'], format='%m/%d/%y')
        df['date'] = df['date'].dt.strftime(STANDARD_DAY_FORMAT)
        stock_market_holidays = df["date"].to_list()

    return stock_market_holidays


def get_next_market_day_no_count_closed_days(date_str: str, num_days: int) -> str:
    dt = parse_std_datestring(date_str)
    return get_standard_ymd_format(skip_market_days(dt, num_days))


def get_days_between(date_str_1: str, date_str_2: str):
    dt_1 = parse_std_datestring(date_str_1)
    dt_2 = parse_std_datestring(date_str_2)
    return abs((dt_2 - dt_1).days)


def get_random_past_date(from_str: str = "2012-01-01",
                         to_str: str = "2020-12-31",
                         end_buffer_days: int = 10):
    import random
    from_dt = datetime.strptime(from_str, STANDARD_DAY_FORMAT)
    to_dt = datetime.strptime(to_str, STANDARD_DAY_FORMAT)
    diff_days = (to_dt - from_dt).days - end_buffer_days
    while True:
        random_number_of_days = random.randrange(0, diff_days)
        rnd_date = from_dt + timedelta(days=random_number_of_days)

        is_closed, _ = is_stock_market_closed(rnd_date)
        if not is_closed:
            return rnd_date
