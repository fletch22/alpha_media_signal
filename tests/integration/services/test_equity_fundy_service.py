import pandas as pd
from pyspark.sql import functions as F

from ams.DateRange import DateRange
from ams.config import logger_factory
from ams.services import ticker_service
from ams.services.equities import equity_fundy_service as efs
from ams.services.equities.EquityFundaDimension import EquityFundaDimension
from ams.utils import ticker_utils

logger = logger_factory.create(__name__)

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def test_get_most_recent_quarter_data():
    df = efs.get_most_recent_quarter_data()

    tickers = ["AAPL", "IBM"]
    df_ticker = df[df["ticker"].isin(tickers)]

    logger.info(df_ticker.head(100))


def test_max():
    all = [1, 2, 3, 4]

    logger.info(max(all))


def test_get_tickers_in_range():
    # Arrange
    df_nasdaq = ticker_service.get_nasdaq_info()

    df_dropped = df_nasdaq.drop(columns=["firstpricedate", "lastpricedate", "firstquarter", "lastquarter",
                                         "secfilings", "companysite", "lastupdated", "cusips",
                                         "isdelisted", "name", "exchange", "firstadded", "permaticker", "sicindustry", "relatedtickers"])

    for c in df_dropped.columns:
        logger.info(f"{c}: {len(df_dropped[c].unique().tolist())}")

    # Assert


def test_get_top_100_market_cap():
    # Arrange
    df_nasdaq = ticker_service.get_nasdaq_info()

    df_nasdaq.sort_values(by=["scalemarketcap"], ascending=False, inplace=True)

    tickers = df_nasdaq.loc[:100, "ticker"].unique().tolist()
    # Act
    logger.info(tickers)

    # Assert


def test_most_rec_quarter_integration():
    # Arrange
    df_equity_funds = efs.get_equity_fundies()

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

    logger.info(df_result.head(20))

    df_drop_future = df_result[df_result["date"] > df_result["date_ef"]]

    logger.info(df_drop_future.head(20))

    df_dd = df_drop_future.sort_values(by=["date_ef"]).drop_duplicates(subset=["id"], keep="last").sort_values(by=["date"])

    logger.info(df_dd.head(20))

    # Act

    # Assert
    assert (df_dd.shape[0] == 8)


def test_get_top_prev():
    # tickers = ticker_service.get_ticker_info()["ticker"].unique()
    # tickers = ["GOOG", "NVDA", "FB", "MSFT", "AMZN", "AAPL", "DDOG"]
    df_tickers = ticker_service.get_nasdaq_tickers()
    tickers = df_tickers["ticker_drop"].unique().tolist()

    logger.info(f"Number of tickers: {len(tickers)}")

    # prev_days = int((365 * 1) / 2)
    prev_days = 365 * 5
    for year in range(2011, 2022):
        to_date_str = f"{year}-01-01"
        top_tickers = ticker_utils.get_top_roi_tickers(tickers=tickers, prev_days=prev_days, to_date_str=to_date_str, top_n=20)
        print(f"{year}: {top_tickers}")


def test_get_top_by_attribute():
    df = efs.get_equity_fundies()

    df = efs.filter_by_dimension(df=df, efd=EquityFundaDimension.MostRecentAnnual)

    assert df.shape[0] > 0

    # logger.info()
    # return

    top_n = 100
    attr_interest = "pe"
    is_low_good = False
    easy_trade = True

    year_range = list(range(2015, 2021))
    for year in year_range:
        start_rep_year = f"{year}-01-01"
        end_rep_year = f"{year}-12-31"

        df_year = df[(df["reportperiod"] >= start_rep_year) & (df["reportperiod"] <= end_rep_year)]

        tickers = df_year["ticker"].unique()

        dr = DateRange.from_date_strings(from_date_str=start_rep_year, to_date_str=end_rep_year)
        df_tickers = ticker_service.get_tickers_in_range(tickers, date_range=dr)

        # df_tickers["mean_vol"] = df_tickers["volume"].mean()
        # des_cols = ["ticker", "mean_vol"]
        # df_tickers = df_tickers.drop_duplicates(subset=des_cols)[des_cols].copy()

        df_tickers = df_tickers.set_index(["ticker"])
        mean_thing = df_tickers.groupby("ticker")["volume"].mean()
        df_tickers["mean_vol"] = mean_thing
        df_tickers = df_tickers.reset_index()

        df_tickers = df_tickers.sort_values(["ticker", "date"])
        des_cols = ["ticker"]
        df_tickers = df_tickers.drop_duplicates(subset=des_cols, keep="first")[["ticker", "mean_vol"]].copy()

        # logger.info(df_tickers[["ticker", "mean_vol"]].head())

        df_year = df_year.sort_values(["ticker", "reportperiod"])
        df_year = df_year.drop_duplicates(subset=des_cols, keep="last").copy()

        df_enh = pd.merge(left=df_year, right=df_tickers, on="ticker")

        if easy_trade:
            df_enh = df_enh[(df_enh[attr_interest] >= 30)]
            df_enh = df_enh[(df_enh["price"] * df_enh["mean_vol"]) > (10 * 250000)]

        tickers = df_enh.sort_values(by=[attr_interest], ascending=is_low_good)["ticker"].values.tolist()

        print(f"_{year + 1}={tickers[:top_n]},")