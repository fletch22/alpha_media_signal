from datetime import datetime
from typing import Set

from ams.DateRange import DateRange
from ams.services.twitter_service import EARLIEST_TWEET_DATE_STR
from ams.twitter.twitter_ml_utils import get_next_market_date
from ams.utils import date_utils
from ams.utils.date_utils import is_stock_market_closed


def get_all_market_days_in_range(date_range: DateRange):
    current_date_str = date_range.from_date_str
    market_days = []
    while current_date_str < date_range.to_date_str:
        current_dt = date_utils.parse_std_datestring(current_date_str)
        is_closed, reached_end_of_data = is_stock_market_closed(dt=current_dt)
        if reached_end_of_data:
            raise Exception("Reached end of market data. Need to add calendar data.")
        if not is_closed:
            market_days.append(current_date_str)
        current_date_str = get_next_market_date(date_str=current_date_str, num_days=1)

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


if __name__ == '__main__':
    sell_days = get_every_nth_sell_date(nth_sell_day=3)

    print(sell_days)