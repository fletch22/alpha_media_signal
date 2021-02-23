from statistics import mean
from typing import List

import pandas as pd
from datetime import datetime

from ams.config import logger_factory
from ams.services import ticker_service
from ams.services.ticker_service import get_nasdaq_info
from ams.utils import date_utils

logger = logger_factory.create(__name__)

def calc_and_persist_nasdaq_roi(date_from: datetime, date_to: datetime, days_hold_stock: int, min_price: float=None) -> (pd.DataFrame, List[str]):
    df_tickers = get_nasdaq_info()
    tickers = df_tickers["ticker"].to_list()
    tickers = sorted(tickers)

    # Act
    df, tickers_gathered = ticker_service.calc_and_persist_equity_daily_roi(date_from=date_from,
                                                                            date_to=date_to,
                                                                            min_price=min_price,
                                                                            max_price=None,
                                                                            tickers=tickers,
                                                                            days_hold_stock=days_hold_stock)

    return df, tickers_gathered


def start():
    from_str = f"2020-08-10"
    to_str = date_utils.get_standard_ymd_format(datetime.now())
    days_hold_stock = 1

    date_from = date_utils.parse_std_datestring(from_str)
    date_to = date_utils.parse_std_datestring(to_str)

    df, _ = calc_and_persist_nasdaq_roi(date_from=date_from, date_to=date_to, days_hold_stock=days_hold_stock)

    logger.info(df.head())

    df = df[df["date"] >= "2020-08-10"].copy()
    # df.sort_values(by=["date"], inplace=True)
    roi_mean = df["roi"].mean()

    logger.info(roi_mean)


if __name__ == '__main__':
    start()