import time
from datetime import datetime

from fastai.tabular.core import TabularPandas
from pandas import DataFrame

from ams.DateRange import DateRange
from ams.services import ticker_service
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

    high_price = ticker_service.get_ticker_price_on_date(ticker="IBM", dt=dt)
    high_price = ticker_service.get_ticker_price_on_date(ticker="NVDA", dt=dt)
    high_price = ticker_service.get_ticker_price_on_date(ticker="ALXN", dt=dt)
    high_price = ticker_service.get_ticker_price_on_date(ticker="GOOGL", dt=dt)
    high_price = ticker_service.get_ticker_price_on_date(ticker="ADI", dt=dt)

    # Act
    start = time.time()
    high_price = ticker_service.get_ticker_price_on_date(ticker="AAPL", dt=dt)
    end = time.time()

    print(f"1st Elapsed: {end - start} seconds")

    start = time.time()
    high_price = ticker_service.get_ticker_price_on_date(ticker="AAPL", dt=dt)
    end = time.time()

    print(f"2nd Elapsed: {end - start} seconds")

    # Assert
    assert (high_price == 393.91)


def test_get_next_high_days():
    # Arrange
    ticker = 'AAPL'

    # Act
    high_price = ticker_service.get_next_trading_day_high(ticker=ticker, date_str="2020-08-07")
    assert (high_price == 455.1)

    # Assert
    high_price = ticker_service.get_next_trading_day_high(ticker=ticker, date_str="2020-08-08")
    assert (high_price == 455.1)

    high_price = ticker_service.get_next_trading_day_high(ticker=ticker, date_str="2020-08-09")
    assert (high_price == 455.1)

    high_price = ticker_service.get_next_trading_day_high(ticker=ticker, date_str="2020-08-10")
    assert (high_price != 455.1)

