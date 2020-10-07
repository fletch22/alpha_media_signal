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
    df_nasdaq = ticker_service.get_nasdaq_info()

    df_dropped = df_nasdaq.drop(columns=["firstpricedate", "lastpricedate", "firstquarter", "lastquarter",
                                         "secfilings", "companysite", "lastupdated", "cusips",
                                         "isdelisted", "name", "exchange", "firstadded", "permaticker", "sicindustry", "relatedtickers"])

    for c in df_dropped.columns:
        print(f"{c}: {len(df_dropped[c].unique().tolist())}")

    # print(df.head(20))

    # Assert