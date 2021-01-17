from typing import List

import pandas as pd

from ams.services import ticker_service


def add_simple_moving_averages(df: pd.DataFrame, target_column: str, windows: List[int]):
    df_copy = df.copy()
    for w in windows:
        with pd.option_context('mode.chained_assignment', None):
            df_copy[f"{target_column}_SMA_{w}"] = df_copy.loc[:, target_column].rolling(window=w).mean().astype("float64")
    return df_copy


def add_sma_history(df: pd.DataFrame, target_column: str, windows: List[int]):
    max_window = max(windows)
    ticker_list = df['f22_ticker'].unique().tolist()
    dt_oldest_str = min(df["date"])

    all_dataframes = []

    for t in ticker_list:
        df_equity = ticker_service.get_ticker_eod_data(t)
        # NOTE: 2020-10-14: chris.flesche: This technique helps us find the simple moving avg. If we try to calculate an SMA by getting yesterdays
        # date we might find that yesterday the market was closed. That's bad because we can't count that day while calculating the SMA. So we rely on
        # counting backwards through the equity trading history to get the nth prior trading day
        if df_equity is not None:
            df_equity.sort_values(by="date", ascending=True, inplace=True)

            dt_equity_oldest_str = min(df_equity["date"])
            dt_oldest_str = dt_oldest_str if dt_oldest_str > dt_equity_oldest_str else dt_equity_oldest_str

            df_olded = df_equity[df_equity["date"] < dt_oldest_str]
            if df_olded.shape[0] > max_window:
                with pd.option_context('mode.chained_assignment', None):
                    dt_start_str = df_olded.iloc[-max_window:, :]["date"].values.tolist()[0]
            else:
                if df_olded.shape[0] > 0:
                    dt_start_str = min(df_olded["date"])

            df_dated = df_equity[df_equity["date"] > dt_start_str]

            df_sma = add_simple_moving_averages(df=df_dated, target_column=target_column, windows=windows)
            all_dataframes.append(df_sma)

    df_all = pd.concat(all_dataframes)

    df_merged = pd.merge(df, df_all, how='inner', left_on=["f22_ticker", "date"], right_on=["ticker", "date"], suffixes=[None, "_drop"])
    df_dropped = df_merged.drop(columns=[c for c in df_merged.columns if c.endswith("_drop")]).drop(columns=['ticker'])

    return df_dropped


days_under_sma = 0


def add_days_since_under_sma_many_tickers(df: pd.DataFrame, col_sma: str, close_col: str):
    df_g = df.groupby(by=["f22_ticker"])

    new_groups = []
    for _, df_group in df_g:
        df_group = add_days_since_under_sma_to_ticker(df_one_ticker=df_group, col_sma=col_sma, close_col=close_col)
        new_groups.append(df_group)

    return pd.concat(new_groups)


def get_count_days(row: pd.Series, col_sma: str, close_col: str):
    global days_under_sma

    close = row[close_col]
    sma = row[col_sma]

    if close < sma:
        if days_under_sma < 0:
            days_under_sma = 0
        elif days_under_sma >= 0:
            days_under_sma += 1
    else:
        if days_under_sma > 0:
            days_under_sma = 0
        elif days_under_sma <= 0:
            days_under_sma -= 1

    return days_under_sma


def add_days_since_under_sma_to_ticker(df_one_ticker: pd.DataFrame, col_sma: str, close_col: str):
    global days_under_sma
    days_under_sma = 0

    df_one_ticker.sort_values(by=["date"], inplace=True)
    df_one_ticker[f"{col_sma}_days_since_under"] = df_one_ticker.apply(lambda x: get_count_days(x, close_col=close_col, col_sma=col_sma), axis=1)

    return df_one_ticker
