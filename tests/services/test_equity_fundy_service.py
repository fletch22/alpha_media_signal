import pandas as pd

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


def test_get_top_100_market_cap():
    # Arrange
    df_nasdaq = ticker_service.get_nasdaq_info()

    df_nasdaq.sort_values(by=["scalemarketcap"], ascending=False, inplace=True)

    tickers = df_nasdaq.loc[:100, "ticker"].unique().tolist()
    # Act
    print(tickers)


    # Assert


def test_most_rec_quarter_integration():
    # Arrange
    df_equity_funds = equity_fundy_service.get_equity_fundies()

    date_range = DateRange.from_date_strings(from_date_str="2018-10-01", to_date_str="2020-10-10")

    df_ticker = ticker_service.get_tickers_in_range(tickers=["AAPL", "MOMO"], date_range=date_range)

    df_result = pd.merge(df_ticker, df_equity_funds, on="ticker").sort_values(by=["calendardate"])

    df_drop_future = df_result[df_result["date"] > df_result["calendardate"]]

    df_dd = df_drop_future.drop_duplicates(subset=["ticker"], keep="last")

    assert (df_dd.shape[0] == 2)


def test_most_rec_quarter_join():
    # Arrange
    # df = equity_fundy_service.get_equity_fundies()
    #
    # date_strs = ["2019-10-01", "2020-10-10"]
    # df_ticker = ticker_service.get_equity_on_dates("AAPL", date_strs=date_strs)

    rows_funda = [
        {"date": "2019-10-01", "ticker": "FOO"},
        {"date": "2020-10-01", "ticker": "FOO"},
        {"date": "2021-10-01", "ticker": "FOO"},
        {"date": "2019-10-01", "ticker": "BAR"},
        {"date": "2020-10-01", "ticker": "BAR"},
        {"date": "2021-10-01", "ticker": "BAR"}
    ]

    df_equity_funds = pd.DataFrame(rows_funda)

    rows_tickers = [
        {"date": "2019-11-01", "ticker": "FOO"},
        {"date": "2020-12-01", "ticker": "FOO"},
        {"date": "2020-12-01", "ticker": "FOO"},
        {"date": "2020-12-02", "ticker": "FOO"},
        {"date": "2020-12-02", "ticker": "FOO"},
        {"date": "2019-11-15", "ticker": "BAR"},
        {"date": "2020-09-15", "ticker": "BAR"},
        {"date": "2020-10-02", "ticker": "BAR"}
    ]

    df_ticker = pd.DataFrame(rows_tickers)

    df_ticker['id'] = range(1, len(df_ticker.index) + 1)

    df_result = pd.merge(df_ticker, df_equity_funds, on="ticker", suffixes=[None, "_ef"]).sort_values(by=["date"])

    print(df_result.head(20))

    df_drop_future = df_result[df_result["date"] > df_result["date_ef"]]

    print(df_drop_future.head(20))

    df_dd = df_drop_future.sort_values(by=["date_ef"]).drop_duplicates(subset=["id"], keep="last").sort_values(by=["date"])

    print(df_dd.head(20))

    # Act

    # Assert
    assert (df_dd.shape[0] == 8)
