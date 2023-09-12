import re
import statistics
from pathlib import Path

import pandas as pd

from ams.config import constants
from ams.config import logger_factory
from ams.services import ticker_service as ts

logger = logger_factory.create(__name__)
sp500_list_hist_ref_path = str(constants.SP500_LIST_HIST_REF_PATH)


def get_sp500_list_hist() -> pd.DataFrame:
    return pd.read_csv(str(constants.SP500_LIST_HIST_PATH))


def append_year(df_old, df_new) -> pd.DataFrame:
    df = pd.concat([df_old, df_new], axis=0)
    df = df.sort_values("year")

    return df


def get_early_years():
    df = get_sp500_list_hist()

    assert (df.shape[0] > 0)

    years = range(2008, 2013)
    df_all = []
    for y in years:
        df_year = df[(df["t1"] <= f"{y}-01-01") & (df["t2"] >= f"{y}-01-02")]
        if df_year.shape[0] > 0:
            df_year["year"] = y
            df_all.append(df_year[["ticker", "year"]])

    return pd.concat(df_all)


def get_later_years():
    token_middle_years = "<input type=\"hidden\" name=\"symbol\" value=\""
    years_info = [
        (2012, token_middle_years),
        (2013, token_middle_years),
        (2014, token_middle_years),
        (2015, token_middle_years),
        (2016, token_middle_years),
        (2017, token_middle_years),
        (2018, "<td><a href=\"https://web.archive.org/web/20180703193826/https://www.slickcharts.com/symbol/"),
        (2019, "<td><a href=\"https://web.archive.org/web/20190102075237/https://www.slickcharts.com/symbol/"),
        (2020, "<td><a href=\"https://web.archive.org/web/20200109175226/https://www.slickcharts.com/symbol/"),
        (2021, "<td><a href=\"https://web.archive.org/web/20210129194252/https://www.slickcharts.com/symbol/"),
        (2022, "<td><a href=\"https://web.archive.org/web/20220123141358/https://www.slickcharts.com/symbol/")
    ]

    df = None
    for year, token in years_info:
        year_html_path = str(Path(constants.SP500_HTML_PATH, f"SP500_{year}.html"))

        end_token = "\""

        with open(year_html_path, "r") as fr:
            content = fr.read()
            instances = [m.start() for m in re.finditer(token, content)]
            ticker_info = []
            for i in instances:
                start_pos = i + len(token)
                end_pos = start_pos + content[start_pos:start_pos + 10].find(end_token)
                ticker_info.append({"ticker": content[start_pos:end_pos], "year": year})

            df_this_year = pd.DataFrame(ticker_info)

            df_this_year = df_this_year.drop_duplicates()

            if df is not None:
                df = append_year(df, df_this_year)
            else:
                df = df_this_year

            df = df.sort_values("year", ascending=False)

    return df


ticker_cache = {}


def process():
    # NOTE: 2022-03-13: chris.flesche: Gets the top_n roi tickers every year, then invests in the top_n the following year, then calculates the roi per year.
    top_n = 12
    month_buy = 4
    month_sell = month_buy + 1
    df = pd.read_parquet(sp500_list_hist_ref_path)

    df_all = []
    for year in range(2012, 2022):
        df_year = df[df["year"] == year]

        tickers = df_year["ticker"].values.tolist()

        print(f"Num tickers: {len(tickers)}")

        single_year_rois = []
        for t in tickers:
            df_ticker = get_years_data(ticker=t, year=year, month_buy=month_buy, month_sell=month_sell)

            if df_ticker is None or df_ticker.shape[0] == 0:
                pass
                # print(f"Ticker '{t}' year {year} has no data.")
            else:
                df_ticker = df_ticker.sort_values("date")

                start_price = df_ticker.iloc[0]["close"]
                close_price = df_ticker.iloc[-1]["close"]

                roi = (close_price - start_price) / start_price
                single_year_rois.append({"ticker": t, "roi": roi, "year": year, "start": start_price, "end": close_price})

        single_year_rois = sorted(single_year_rois, key=lambda i: i['roi'], reverse=True)
        single_year_rois = single_year_rois[:top_n]

        df_top_n = pd.DataFrame(single_year_rois)
        df_all.append(df_top_n)

    years_returns = []
    single_year_rois = []

    df_years = pd.concat(df_all, axis=0)
    df_years.to_csv(constants.SP_TOP_N_PATH)

    for df in df_all:
        print("")
        print(df[["ticker", "year", "start", "end", "roi"]].head(top_n))

    for df in df_all:
        year = df.iloc[0]["year"]
        tickers = df["ticker"].values.tolist()

        for t in tickers:
            fetch_year = year + 1
            df_ticker = get_years_data(ticker=t, year=fetch_year)

            if df_ticker is None or df_ticker.shape[0] == 0:
                # print(f"Ticker '{t}' year {fetch_year} has no data.")
                break

            df_ticker = df_ticker.sort_values("date")

            start_price = df_ticker.iloc[0]["close"]
            close_price = df_ticker.iloc[-1]["close"]

            roi = (close_price - start_price) / start_price
            single_year_rois.append(roi)

        if len(single_year_rois) > 0:
            year_roi = statistics.mean(single_year_rois)
            single_year_rois = []
            print(f"{fetch_year} roi: {year_roi}")

            years_returns.append(year_roi)

    total_roi = statistics.mean(years_returns)

    print(f"Mean roi: {total_roi}")

    initial = 1000
    inv = initial
    for roi in years_returns:
        inv = inv + (inv * roi)
        print(inv)

    inv_ret = (inv - initial) / initial
    print(f"Investment return: {inv_ret}")


def get_years_data(ticker, year, month_buy: int = 1, month_sell: int = 2):
    if ticker in ticker_cache.keys():
        df = ticker_cache[ticker]
    else:
        df = ts.get_ticker_eod_data(ticker=ticker)
        ticker_cache[ticker] = df

    if df is not None:
        df = df[(df["date"] >= f"{year}-{month_buy:02d}-01") & (df["date"] < f"{year + 1}-{month_sell:02d}-01")]

    return df


if __name__ == '__main__':
    process()
    # df = pd.read_csv(constants.SP_TOP_N_PATH)
    #
    # df = df[df["year"] == 2021].copy()
    #
    # print(df.head(20))
