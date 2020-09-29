from datetime import datetime, timedelta
from typing import List, Dict
import time

import pandas as pd
from pandas import DataFrame

from ams.DateRange import DateRange
from ams.services import file_services
from ams.services.EquityFields import EquityFields
from ams.utils import date_utils

ticker_cache = {}
ticker_date_row_cache = {}


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


def get_ticker_attribute_on_date(ticker: str, dt: datetime, equity_fields: List[EquityFields] = [EquityFields.high]):
    date_str = date_utils.get_standard_ymd_format(dt)
    cache_key = f"{ticker}_{date_str}"

    row = {}
    if cache_key in ticker_date_row_cache.keys():
        row = ticker_date_row_cache[cache_key]
    else:
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

        if df is not None:
            df_filtered = df[df["date"] == date_str]

            records = df_filtered.to_dict('records')
            if len(records) > 0:
                row = records[0]
                ticker_date_row_cache[cache_key] = row

    if row is not None and bool(row):
        values = [row[ef.value] for ef in equity_fields]
    else:
        values = [None for ef in equity_fields]

    return tuple(values)


def get_ticker_attr_from_date(ticker: str, date_str: str, equity_fields: List[EquityFields] = [EquityFields.high], days_ahead: int = 5):
    dt = date_utils.parse_std_datestring(date_str)
    dt_new = dt + timedelta(days=days_ahead)

    return get_ticker_attribute_on_date(ticker, dt=dt_new, equity_fields=equity_fields)


def get_start_end_dates(date_strs: List[str]):
    date_strs = sorted(date_strs)
    start_date = date_strs[0]
    end_date = date_strs[len(date_strs) - 1]

    dt = date_utils.parse_std_datestring(end_date)
    dt_end_adjust = dt + timedelta(days=4)
    end_date_adj = date_utils.get_standard_ymd_format(dt_end_adjust)

    return start_date, end_date_adj


def get_ticker_on_dates(tick_dates: Dict[str, List[str]]) -> pd.DataFrame:
    df_list = []
    for ticker, date_strs in tick_dates.items():
        df = get_equity_on_dates(ticker=ticker, date_strs=date_strs)
        if df is None:
            print(f"No data for ticker '{ticker}' in {date_strs}")
        else:
            df_list.append(df)
    return pd.concat(df_list).dropna(subset=["future_open", "future_low", "future_high", "future_close", "future_date"])


def get_equity_on_dates(ticker: str, date_strs: List[str]) -> pd.DataFrame:
    df = get_ticker_eod_data(ticker)
    df_in_dates = None
    if df is not None:
        start, end = get_start_end_dates(date_strs)
        df_in_range = df[(df["date"] >= start) & (df["date"] <= end)].sort_values(by="date")
        df_in_range["future_open"] = df_in_range["open"]
        df_in_range["future_low"] = df_in_range["low"]
        df_in_range["future_high"] = df_in_range["high"]
        df_in_range["future_close"] = df_in_range["close"]
        df_in_range["future_date"] = df_in_range["date"]
        cols = ["future_open", "future_low", "future_high", "future_close", "future_date"]
        df_in_range[cols] = df_in_range[cols].shift(-1)

        df_in_dates = df_in_range[df_in_range["date"].isin(date_strs)]
    return df_in_dates


def get_next_date(df_stocks: pd.DataFrame, ticker: str, date_str: str):
    df_found = df_stocks[(df_stocks["ticker"] == ticker) & (df_stocks["date"] > date_str)]

    result = (None, None, None, None)
    if df_found.shape[0] > 0:
        df_found.sort_values(["ticker", "date"], inplace=True)
        row = dict(df_found.iloc[0])
        result = row[EquityFields.open.value], row[EquityFields.low.value], row[EquityFields.high.value], row[EquityFields.close.value]

    return result


def pull_in_next_trading_day_info(df_tweets: pd.DataFrame):
    ttd = extract_ticker_tweet_dates(df_tweets)
    df_list = []
    start = time.time()
    for ticker, date_strs in ttd.items():
        df = get_next_trading_day(ticker=ticker, trading_days=date_strs)
        if df is not None:
            df_thin = df[[EquityFields.close.value, EquityFields.open.value, EquityFields.low.value, EquityFields.high.value, EquityFields.ticker.value, EquityFields.date.value]]
            df_list.append(df_thin)
    df_stocks = pd.concat(df_list)
    end = time.time()
    elapsed = end - start
    print(f"Got all stock data: {elapsed}")

    col_tmp = "tmp_future_values"
    df_tweets[col_tmp] = df_tweets.apply(lambda x: get_next_date(df_stocks=df_stocks, ticker=x["f22_ticker"], date_str=x["date"]), axis=1)

    new_cols = ["future_open", "future_low", "future_high", "future_close"]
    df_tweets[new_cols] = pd.DataFrame(df_tweets[col_tmp].tolist(), index=df_tweets.index)
    return df_tweets.drop(columns=col_tmp).dropna(subset=new_cols)


def get_next_trading_day(ticker: str, trading_days: List[str]):
    trading_days = sorted(trading_days)
    df = get_ticker_eod_data(ticker=ticker)

    df_dated = None
    if df is None:
        print(f"Equity ticker {ticker} not found.")
    else:
        start_date = trading_days[0]
        end_date = trading_days[len(trading_days) - 1]

        dt = date_utils.parse_std_datestring(end_date)
        dt_end_adjust = dt + timedelta(days=4)
        end_date_adj = date_utils.get_standard_ymd_format(dt_end_adjust)

        df_dated = df[(df["date"] >= start_date) & (df["date"] <= end_date_adj)]

    return df_dated


def get_next_trading_day_attr(ticker: str, date_str: str, equity_fields: List[EquityFields] = [EquityFields.high], max_days_head: int = 5):
    result = None,
    for i in range(1, max_days_head):
        result = get_ticker_attr_from_date(ticker=ticker, date_str=date_str, equity_fields=equity_fields, days_ahead=i)
        if result[0] is not None:
            break

    return result


def get_tickers_in_range(tickers: List[str], date_range: DateRange) -> DataFrame:
    start_date_str = date_utils.get_standard_ymd_format(date_range.from_date)
    end_date_str = date_utils.get_standard_ymd_format(date_range.to_date)

    all_dfs = []
    for t in tickers:
        df = get_ticker_eod_data(ticker=t)
        if df is None:
            print(f"Ticker {t} does not exist.")
        else:
            df_dated = df[(df['date'] > start_date_str) & (df['date'] < end_date_str)]
            all_dfs.append(df_dated)

    return pd.concat(all_dfs)


def extract_ticker_tweet_dates(df_tweets: pd.DataFrame):
    df_distinct_tweet_dts = df_tweets[["f22_ticker", "date"]].drop_duplicates()
    df_g_stocks = df_distinct_tweet_dts.groupby(by=["f22_ticker"])

    stock_days = {}
    for group_name, df_group in df_g_stocks:
        ticker = group_name
        dates = df_group["date"].values.tolist()
        stock_days[ticker] = dates

    return stock_days
