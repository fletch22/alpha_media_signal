import pandas as pd
from numpy import ravel
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ams.DateRange import DateRange
from ams.services import ticker_service
from ams.services.equities import equity_fundy_service

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def test_get_most_recent_quarter_data():
    df = equity_fundy_service.get_most_recent_quarter_data()

    tickers = ["AAPL", "IBM"]
    df_ticker = df[df["ticker"].isin(tickers)]

    print(df_ticker.head(100))


def test_max():
    all = [1, 2, 3, 4]

    print(max(all))


def test_get_tickers_in_range():
    # Arrange
    equity_fundy_service.get_nasdaq_tickers_std_and_cat()


    # print(df.head(20))










    # Assert
