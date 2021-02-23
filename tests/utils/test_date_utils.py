import random
from datetime import timedelta, datetime

import pytest
import pytz

from ams.config import logger_factory
from ams.utils import date_utils
from ams.utils.date_utils import TZ_AMERICA_NEW_YORK

logger = logger_factory.create(__name__)

def test_dt():
    date_max = "2020-11-01"
    date_min = "2020-11-30"

    timestamp_max = date_utils.parse_std_datestring(date_max)
    timestamp_min = date_utils.parse_std_datestring(date_min)

    total_days = (timestamp_max - timestamp_min).days
    rnd_days = int(random.randint(total_days, 10))

    rand_dt = timestamp_min + timedelta(days=rnd_days)
    rand_date_string = date_utils.get_standard_ymd_format(rand_dt)


def test_return():
    success_rt = .53

    min = 40

    rnd_amts = []
    for i in range(100):
        rand_cents = random.randint(0, 100)
        rnd_amts.append(float(rand_cents / 100))

    success_pct = success_rt * 100

    up_or_downs = [True if i < success_pct else False for i in range(100)]
    random.shuffle(up_or_downs)

    tots = 40
    for ndx, ud in enumerate(up_or_downs):
        amt = rnd_amts[ndx]
        if ud:
            tots += amt
        else:
            tots += -amt


def test_convert_to_nyc():
    # Arrange
    orig_date_str = "2020-07-08_00-15-00"
    dt_here = datetime.strptime(orig_date_str, date_utils.STANDARD_DATE_WITH_SECONDS_FORMAT)
    dt_utc = pytz.utc.localize(dt_here)

    # Act
    dt_nyc = date_utils.convert_utc_timestamp_to_nyc(dt_utc.timestamp())
    result = dt_nyc.strftime(date_utils.STANDARD_DATE_WITH_SECONDS_FORMAT)

    # Assert
    assert (result == "2020-07-07_20-15-00")


def test_is_after_close():
    # Arrange
    orig_date_str = "2020-11-27_13-15-00"
    dt_here = datetime.strptime(orig_date_str, date_utils.STANDARD_DATE_WITH_SECONDS_FORMAT)
    tz_nyc = pytz.timezone(TZ_AMERICA_NEW_YORK)
    dt_nyc = tz_nyc.localize(dt_here)
    dt_utc = dt_nyc.astimezone(pytz.utc)
    logger.info(f"\ntest function: {dt_utc}")

    # Act
    is_closed = date_utils.is_after_nasdaq_closed(utc_timestamp=dt_utc.timestamp())

    # Assert
    assert (is_closed)


def test_get_prev_nasdaq_dt():
    # Arrange
    dt_sept = date_utils.parse_std_datestring("2020-09-10")

    logger.info(list(range(-3)))

    # Act
    dt_prev = date_utils.find_next_market_open_day(dt=dt_sept, num_days_to_skip=-7)

    # Assert
    assert (abs((dt_prev - dt_sept).days) > 7)


def test_foo():
    import pandas as pd

    df = pd.DataFrame([{"foo": "2019-04-12"}, {"foo": "2019-04-16"}])

    def day_of_week(date_str):
        return pd.Timestamp(date_str).dayofweek

    df["fd_day_of_week"] = df.apply(lambda x: day_of_week(x["foo"]), axis=1)

    def day_of_year(date_str):
        return pd.Timestamp(date_str).dayofyear

    df["fd_day_of_year"] = df.apply(lambda x: day_of_year(x["foo"]), axis=1)

    def day_of_month(date_str):
        return int(date_str.split("-")[2])

    df["fd_day_of_month"] = df.apply(lambda x: day_of_month(x["foo"]), axis=1)

    logger.info(df.head())

    # pd.Timestamp("2019-04-12").dayofweek


def test_is_stock_market_closed():
    # Arrange
    dt = date_utils.parse_std_datestring("2021-07-05")

    # Act
    is_closed, reached_end_of_data = date_utils.is_stock_market_closed(dt=dt)

    # Assert
    assert (is_closed)


def test_is_stock_market_closed_raises():
    # Arrange
    dt = date_utils.parse_std_datestring("9999-07-05")

    # Act
    with pytest.raises(Exception):
        # Assert
        is_closed, reached_end_of_data = date_utils.is_stock_market_closed(dt=dt)


def test_tuple():
    a = [123, 234, 345]
    a1, a2, a3 = tuple(a)