from datetime import datetime, timedelta
from typing import Set

from ams.DateRange import DateRange
from ams.services.twitter_service import EARLIEST_TWEET_DATE_STR
from ams.utils import date_utils
from ams.utils.date_utils import is_stock_market_closed, get_next_market_date


def get_all_market_days_in_range(date_range: DateRange):
    current_date_str = date_range.start_date_str
    market_days = []
    while current_date_str < date_range.end_date_str:
        current_dt = date_utils.parse_std_datestring(current_date_str)
        is_closed, reached_end_of_data = is_stock_market_closed(dt=current_dt)
        if reached_end_of_data:
            raise Exception("Reached end of market data. Need to add calendar data.")
        if not is_closed:
            market_days.append(current_date_str)
        current_date_str = get_next_market_date(date_str=current_date_str)

    return market_days


def get_every_nth_sell_date(nth_sell_day: int) -> Set[str]:
    now_dt_str = date_utils.get_standard_ymd_format(datetime.now())
    date_range = DateRange.from_date_strings(from_date_str=EARLIEST_TWEET_DATE_STR, to_date_str=now_dt_str)
    all_mrkt_days = get_all_market_days_in_range(date_range=date_range)
    sell_days = set()
    for ndx, md in enumerate(all_mrkt_days):
        if ndx % nth_sell_day == 0:
            sell_days.add(md)

    return sell_days


def get_every_nth_tweet_date(nth_sell_day: int, skip_start_days: int = 0) -> Set[str]:
    now_dt_str = date_utils.get_standard_ymd_format(datetime.now())

    early_dt_str = EARLIEST_TWEET_DATE_STR
    early_dt = date_utils.parse_std_datestring(early_dt_str)
    early_dt = early_dt + timedelta(days=skip_start_days)
    early_dt_str = date_utils.get_standard_ymd_format(early_dt)

    date_range = DateRange.from_date_strings(from_date_str=early_dt_str, to_date_str=now_dt_str)
    all_mrkt_days = get_all_market_days_in_range(date_range=date_range)
    sell_days = set()
    for ndx, md in enumerate(all_mrkt_days):
        # NOTE: 2021-02-24: chris.flesche: Adds the first date as a tweet date.
        if ndx == 0 or ndx % nth_sell_day == 0:
            sell_days.add(md)

    return sell_days


if __name__ == '__main__':
    tweet_days_1 = get_every_nth_tweet_date(nth_sell_day=3, skip_start_days=0)
    tweet_days_2 = get_every_nth_tweet_date(nth_sell_day=3, skip_start_days=1)
    tweet_days_3 = get_every_nth_tweet_date(nth_sell_day=3, skip_start_days=2)

    assert (len(tweet_days_1.intersection(tweet_days_2)) == 0)
    assert (len(tweet_days_1.intersection(tweet_days_3)) == 0)
    assert (len(tweet_days_2.intersection(tweet_days_3)) == 0)

