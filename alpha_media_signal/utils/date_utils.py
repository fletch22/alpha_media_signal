from datetime import datetime

import pytz

STANDARD_DAY_FORMAT = '%Y-%m-%d'
STANDARD_DATE_WITH_SECONDS_FORMAT = '%Y-%m-%d_%H-%M-%S'
WTD_DATE_WITH_SECONDS_FORMAT = '%Y-%m-%d %H:%M:%S'
TWITTER_FORMAT = '%Y%m%D%H%M'


def parse_std_datestring(datestring):
    return datetime.strptime(datestring, STANDARD_DAY_FORMAT)


def get_standard_ymd_format(date: datetime):
    return date.strftime(STANDARD_DAY_FORMAT)


def format_file_system_friendly_date(date: datetime):
    millis = round(date.microsecond / 1000, 2)
    return f"{date.strftime(STANDARD_DATE_WITH_SECONDS_FORMAT)}-{millis}"


def convert_wtd_nyc_date_to_std(date_str: str):
    dt_nyc = datetime.strptime(date_str, WTD_DATE_WITH_SECONDS_FORMAT)
    nyc_t = pytz.timezone('America/New_York')
    return nyc_t.localize(dt_nyc, is_dst=None)


def convert_wtd_nyc_date_to_utc(date_str: str):
    dt_nyz_tz = convert_wtd_nyc_date_to_std(date_str)
    dt_utc = dt_nyz_tz.astimezone(pytz.timezone('UTC'))

    return dt_utc

