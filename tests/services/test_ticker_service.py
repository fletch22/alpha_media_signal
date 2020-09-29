import time
from datetime import datetime
from typing import Tuple

import pandas as pd

from ams.DateRange import DateRange
from ams.services import ticker_service
from ams.services.EquityFields import EquityFields
from ams.utils import date_utils


def test_get_ticker_data():
    # Arrange
    ticker = "AAPL"

    # Act
    df = ticker_service.get_ticker_eod_data(ticker=ticker)
    df_sorted = df.sort_values(by='date')

    # Assert
    assert (df_sorted.shape[0] > 190)
    assert (df_sorted.columns[0] == 'ticker')


def test_get_ticker_data_prepped():
    ticker = "AAPL"

    # Act
    df = ticker_service.get_ticker_eod_data(ticker=ticker)
    df_sorted = df.sort_values(by='date')

    start_date = '2020-08-01'
    end_date = '2020-08-27'

    df_dated = df_sorted[(df['date'] > start_date) & (df['date'] < end_date)]

    df_dated.groupby()

    from sklearn.model_selection import train_test_split
    train_test_split

    from sklearn.svm import SVC
    SVC

    print(f'Num in date range: {df_dated.shape[0]}')


def test_ticker_in_range():
    # Arrange
    tickers = ['AAPL']

    date_range = DateRange(from_date=date_utils.parse_std_datestring("2020-08-01"),
                           to_date=date_utils.parse_std_datestring("2020-08-30")
                           )

    # Act
    ticker_service.get_tickers_in_range(tickers=tickers, date_range=date_range)

    # Assert


def test_ticker_on_date():
    # Arrange
    date_string = "2020-07-09"
    dt: datetime = date_utils.parse_std_datestring(date_string)

    high_price = ticker_service.get_ticker_attribute_on_date(ticker="IBM", dt=dt)
    high_price = ticker_service.get_ticker_attribute_on_date(ticker="NVDA", dt=dt)
    high_price = ticker_service.get_ticker_attribute_on_date(ticker="ALXN", dt=dt)
    high_price = ticker_service.get_ticker_attribute_on_date(ticker="GOOGL", dt=dt)
    high_price = ticker_service.get_ticker_attribute_on_date(ticker="ADI", dt=dt)

    # Act
    start = time.time()
    high_price = ticker_service.get_ticker_attribute_on_date(ticker="AAPL", dt=dt)
    end = time.time()

    print(f"1st Elapsed: {end - start} seconds")

    start = time.time()
    high_price = ticker_service.get_ticker_attribute_on_date(ticker="AAPL", dt=dt)
    end = time.time()

    print(f"2nd Elapsed: {end - start} seconds")

    # Assert
    assert (high_price == 393.91)


def test_get_next_high_days():
    # Arrange
    ticker = 'AAPL'

    high_price, = ticker_service.get_next_trading_day_attr(ticker=ticker, date_str="2020-08-07")
    assert (high_price == 455.1)

    high_price, = ticker_service.get_next_trading_day_attr(ticker=ticker, date_str="2020-08-08")
    assert (high_price == 455.1)

    high_price, = ticker_service.get_next_trading_day_attr(ticker=ticker, date_str="2020-08-09")
    assert (high_price == 455.1)

    high_price, = ticker_service.get_next_trading_day_attr(ticker=ticker, date_str="2020-08-10")
    assert (high_price == 449.93)

    close, = ticker_service.get_next_trading_day_attr(ticker=ticker, equity_fields=[EquityFields.close], date_str="2020-08-10")
    assert (close == 437.5)

    close, high = ticker_service.get_next_trading_day_attr(ticker=ticker, equity_fields=[EquityFields.close, EquityFields.high], date_str="2020-08-08")
    assert (close == 450.91)
    assert (high == 455.1)


def get_test_tweets_and_stocks() -> Tuple[pd.DataFrame, pd.DataFrame]:
    tweet_rows = [
        {"f22_ticker": "AAA", "date": "2020-01-01", "close": ".11"},
        {"f22_ticker": "AAA", "date": "2020-01-02", "close": ".22"},
        {"f22_ticker": "BBB", "date": "2020-01-01", "close": ".33"},
        {"f22_ticker": "CCC", "date": "2020-01-01", "close": ".44"},
        {"f22_ticker": "CCC", "date": "2020-01-02", "close": ".55"},
    ]
    df_tweets = pd.DataFrame(tweet_rows)
    df_tweets = df_tweets.sample(frac=1.0)

    stock_rows = [
        {"ticker": "AAA", "date": "2020-01-01", "close": "1.11"},
        {"ticker": "AAA", "date": "2020-01-02", "close": "2.22"},
        {"ticker": "AAA", "date": "2020-01-03", "close": "21.02"},
        {"ticker": "BBB", "date": "2020-01-01", "close": "3.33"},
        {"ticker": "BBB", "date": "2020-01-02", "close": "31.01"},
        {"ticker": "CCC", "date": "2020-01-01", "close": "4.44"},
        {"ticker": "CCC", "date": "2020-01-02", "close": "5.55"},
        {"ticker": "CCC", "date": "2020-01-03", "close": "6.66"},
    ]
    df_stocks = pd.DataFrame(stock_rows)
    df_stocks = df_stocks.sample(frac=1.0)

    return df_tweets, df_stocks


def test_merge_future_price():
    # Arrange
    df_tweets, df_stocks = get_test_tweets_and_stocks()

    ttd = ticker_service.extract_ticker_tweet_dates(df_tweets)

    print(ttd)

    # Act

    # Assert
    assert (df_tweets.shape[0] == 5)
    assert (df_tweets.columns.all(["f22_ticker", "date", "close"]))


def test_get_equity_on_dates():
    tweet_rows = [
        {"f22_ticker": "AAPL", "date": "2020-09-08", "close": ".11"},
        {"f22_ticker": "AAPL", "date": "2020-09-09", "close": ".22"},
        {"f22_ticker": "MSFT", "date": "2020-09-15", "close": ".33"},
        {"f22_ticker": "ATVI", "date": "2020-09-17", "close": ".44"},
        {"f22_ticker": "ATVI", "date": "2020-09-18", "close": ".55"},
    ]
    df_tweets = pd.DataFrame(tweet_rows)
    df_tweets = df_tweets.sample(frac=1.0)

    ttd = ticker_service.extract_ticker_tweet_dates(df_tweets)
    df = ticker_service.get_ticker_on_dates(ttd)

    print(df.head())


def test_pull_in_next_trading_day_info():
    # Arrange
    tweet_rows = [
        {"f22_ticker": "AAPL", "date": "2020-09-08", "close": ".11"},
        {"f22_ticker": "AAPL", "date": "2020-09-09", "close": ".22"},
        {"f22_ticker": "MSFT", "date": "2020-09-15", "close": ".33"},
        {"f22_ticker": "ATVI", "date": "2020-09-17", "close": ".44"},
        {"f22_ticker": "ATVI", "date": "2020-09-18", "close": ".55"},
    ]
    df_tweets = pd.DataFrame(tweet_rows)
    df_tweets = df_tweets.sample(frac=1.0)

    # Act
    df_twt_exp = ticker_service.pull_in_next_trading_day_info(df_tweets=df_tweets)

    # Assert
    df_aapl = df_twt_exp[(df_twt_exp["f22_ticker"] == "AAPL") & (df_twt_exp["date"] == "2020-09-08")]
    assert (df_aapl.shape[0] == 1)

    row_dict = dict(df_aapl.iloc[0])
    assert (row_dict["future_open"] == 117.26)
    assert (row_dict["future_low"] == 115.26)
    assert (row_dict["future_high"] == 119.14)
    assert (row_dict["future_close"] == 117.32)


def test_get_thing():
    df_tweets, df_stocks = get_test_tweets_and_stocks()
