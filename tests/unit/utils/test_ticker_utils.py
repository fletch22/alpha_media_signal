import pandas as pd

from ams.DateRange import DateRange

from ams.services import ticker_service
from ams.utils import ticker_utils

from ams.config import logger_factory

logger = logger_factory.create(__name__)

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def test_get_sma():
    # Arrange
    tweet_rows = [
        {"f22_ticker": "AAPL", "date": "2020-09-08", "close": .11},
        {"f22_ticker": "AAPL", "date": "2020-09-09", "close": .22},
        {"f22_ticker": "MSFT", "date": "2020-09-15", "close": .33},
        {"f22_ticker": "ATVI", "date": "2020-09-17", "close": .44},
        {"f22_ticker": "ATVI", "date": "2020-09-18", "close": .55},
        {"f22_ticker": "AAPL", "date": "2020-09-08", "close": .11},
        {"f22_ticker": "AAPL", "date": "2020-09-09", "close": .22},
        {"f22_ticker": "MSFT", "date": "2020-09-15", "close": .33},
        {"f22_ticker": "ATVI", "date": "2020-09-17", "close": .44},
        {"f22_ticker": "ATVI", "date": "2020-09-18", "close": .55},
    ]
    df_tweets = pd.DataFrame(tweet_rows)
    window_list = [15, 20, 50, 200]

    # Act
    df_tweets_new = ticker_utils.add_simple_moving_averages(df=df_tweets, target_column="close", windows=[2, 15, 20, 50, 200])

    # Assert
    assert (df_tweets_new["SMA_2"].fillna(0).mean() > 0)
    assert (df_tweets_new["SMA_200"].fillna(0).mean() == 0)
    logger.info(df_tweets_new.head(20))


def test_add_sma_history():
    # Arrange
    date_range = DateRange.from_date_strings("2020-09-01", "2020-10-01")
    df_equities = ticker_service.get_tickers_in_range(tickers=["NVDA", "MOMO"], date_range=date_range)
    df_equities = df_equities.rename(columns={"ticker": "f22_ticker"})

    # Act
    df = ticker_utils.add_sma_history(df=df_equities, target_column="close", windows=[20, 200])

    # Assert
    assert ("close_SMA_200" in df.columns)
    # assert(df[df["close_SMA_200"].isnull()].shape[0] == 0)

    logger.info(list(df.columns))


def test_set_num_days_under_sma():
    # Arrange
    date_range = DateRange.from_date_strings("2020-09-01", "2020-10-01")
    df_equities = ticker_service.get_tickers_in_range(tickers=["NVDA", "MOMO"], date_range=date_range)
    df_equities = df_equities.rename(columns={"ticker": "f22_ticker"})

    df = ticker_utils.add_sma_history(df=df_equities, target_column="close", windows=[20, 200])

    # Act
    df["close_SMA_200_diff"] = df["close"] - df["close_SMA_200"]


def test_ams():
    # Arrange
    date_range = DateRange.from_date_strings("2009-09-01", "2021-10-01")
    df_equities = ticker_service.get_tickers_in_range(tickers=["NVDA", "MOMO"], date_range=date_range)
    df_equities = df_equities.rename(columns={"ticker": "f22_ticker"})
    df = ticker_utils.add_sma_history(df=df_equities, target_column="close", windows=[20, 200])

    # Act
    df_ungrouped = ticker_utils.add_days_since_under_sma_many_tickers(df=df, col_sma="close_SMA_200", close_col="close")

    # Assert
    num_under = df_ungrouped[df_ungrouped["close_SMA_200_days_since_under"] > 0].shape[0]
    assert (num_under > 1)

def test_up_or_down():
    # Arrange
    df = pd.DataFrame([
        {"close": 1.0},
        {"close": 0.0},
        {"close": 3.0},
        {"close": 3.0},
        {"close": 8.0},
        {"close": 2.0}
    ])

    # Act
    df = ticker_service.prev_up_or_down(df=df)

    # Assert
    logger.info(df.head())