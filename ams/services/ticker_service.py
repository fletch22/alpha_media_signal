from datetime import datetime, timedelta
from typing import List, Dict

import pandas as pd
from pandas import DataFrame

from ams.DateRange import DateRange
from ams.services import file_services
from ams.utils import date_utils

ticker_cache = {}
ticker_date_price_cache = {}


def get_ticker_cache():
    return ticker_cache


def set_ticker_cache(tc: Dict):
    ticker_cache = tc


def get_ticker_eod_data(ticker: str) -> DataFrame:
    ticker_path = file_services.get_eod_ticker_file_path(ticker)
    df = None
    if ticker_path.exists():
        df = pd.read_csv(str(ticker_path))

    return df


def get_ticker_price_on_date(ticker: str, dt: datetime):
    max = 4
    tc = get_ticker_cache()
    if ticker in tc.keys():
        df = tc[ticker]
    else:
        df = get_ticker_eod_data(ticker)
        if len(tc.keys()) > max:
            first_key = list(tc.keys())[0]
            del tc[first_key]
        tc[ticker] = df

    high = None
    if df is not None:
        date_string = date_utils.get_standard_ymd_format(dt)
        df_filtered = df[df["date"] == date_string]
        highs = df_filtered["high"].values.tolist()
        high = None if len(highs) == 0 else highs[0]

    return high


def get_ticker_price_from_date(ticker: str, date_str: str, days_ahead: int):
    dt = date_utils.parse_std_datestring(date_str)
    dt_new = dt + timedelta(days=days_ahead)
    date_str = date_utils.get_standard_ymd_format(dt_new)

    price = None
    if ticker in ticker_date_price_cache.keys():
        tick_data = ticker_date_price_cache[ticker]
        if date_str in tick_data.keys():
            price = tick_data[date_str]
        else:
            dt = date_utils.parse_std_datestring(date_str)
            price = get_ticker_price_on_date(ticker, dt=dt)
            if price is not None:
                tick_data[date_str] = price
    else:
        dt = date_utils.parse_std_datestring(date_str)
        price = get_ticker_price_on_date(ticker, dt=dt)
        if price is not None:
            tick_data = {date_str: price}
            ticker_date_price_cache[ticker] = tick_data

    return price


def get_ticker_from_date(ticker: str, dt: datetime, days_from: int):
    dt_new = dt + timedelta(days=days_from)
    price = get_ticker_price_on_date(ticker=ticker, dt=dt_new)
    return price


def get_next_trading_day_high(ticker: str, date_str: str, max_days_head: int = 5):
    high = None
    for i in range(1, max_days_head):
        high = get_ticker_price_from_date(ticker=ticker, date_str=date_str, days_ahead=i)
        if high is not None:
            break

    return high


def get_tickers_in_range(tickers: List[str], date_range: DateRange) -> DataFrame:
    start_date_str = date_utils.get_standard_ymd_format(date_range.from_date)
    end_date_str = date_utils.get_standard_ymd_format(date_range.to_date)

    all_dfs = []
    for t in tickers:
        df = get_ticker_eod_data(ticker=t)
        if df is not None:
            df_dated = df[(df['date'] > start_date_str) & (df['date'] < end_date_str)]
            all_dfs.append(df_dated)

    return pd.concat(all_dfs)
