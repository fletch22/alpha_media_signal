from typing import List

import pandas as pd

from ams.services import ticker_service


def add_simple_moving_averages(df: pd.DataFrame, target_column: str, windows: List[int]):
    df_copy = df.copy()
    for w in windows:
        with pd.option_context('mode.chained_assignment', None):
            df_copy[f"{target_column}_SMA_{w}"] = df_copy[target_column].rolling(window=w).mean().astype("float64")
    return df_copy


def add_sma_history(df: pd.DataFrame, target_column: str, windows: List[int], tweet_date_str: str, sma_day_before: bool):
    max_window = max(windows)

    all_dataframes = []

    date_col = "date"

    df_g = df.groupby(by=["f22_ticker"])

    for ticker, df_group in df_g:
        df_equity = ticker_service.get_ticker_eod_data(ticker)
        dt_oldest_tweet_str = min(df_group["date"])

        if df_equity is not None:
            df_equity = df_equity[df_equity["date"] <= tweet_date_str].copy()

            df_equity.dropna(subset=[date_col], inplace=True)

            df_equity.sort_values(by=date_col, ascending=True, inplace=True)

            if df_equity is not None and df_equity.shape[0] > 0:
                # dt_youngest_ticker_str = max(df_equity["date"])
                # dt_oldest_tweet_str = dt_oldest_tweet_str if dt_oldest_tweet_str > dt_youngest_ticker_str else dt_oldest_ticker_str

                df_hist = df_equity[df_equity[date_col] < dt_oldest_tweet_str].copy()
                dt_start_str = None
                if df_hist.shape[0] > max_window:
                    with pd.option_context('mode.chained_assignment', None):
                        dt_start_str = df_hist.iloc[-max_window:][date_col].values.tolist()[0]
                elif df_hist.shape[0] > 0:
                        dt_start_str = df_hist[date_col].min()

                if dt_start_str is not None:
                    df_dated = df_equity[df_equity[date_col] >= dt_start_str].copy()

                    df_sma = add_simple_moving_averages(df=df_dated, target_column=target_column, windows=windows)

                    all_dataframes.append(df_sma)

    df_all = pd.concat(all_dataframes, axis=0)

    if sma_day_before:
        df_all = df_all.rename(columns={date_col: "prev_date", "ticker": "f22_ticker"})
        df_merged = pd.merge(df, df_all, how='inner', on=["f22_ticker", "prev_date"], suffixes=[None, "_drop"])
    else:
        df_all = df_all.rename(columns={"ticker": "f22_ticker"})
        df_merged = pd.merge(df, df_all, how='inner', left_on=["f22_ticker", "date"], right_on=["f22_ticker", date_col], suffixes=[None, "_drop"])

    df_dropped = df_merged.drop(columns=[c for c in df_merged.columns if c.endswith("_drop")])

    return df_dropped


days_under_sma = 0


def add_days_since_under_sma_many_tickers(df: pd.DataFrame, col_sma: str, close_col: str):
    df_g = df.groupby(by=["f22_ticker"])

    new_groups = []
    for _, df_group in df_g:
        df_group = add_days_since_under_sma_to_ticker(df_one_ticker=df_group, col_sma=col_sma, close_col=close_col)
        new_groups.append(df_group)

    df_result = df
    if len(new_groups) > 0:
        df_result = pd.concat(new_groups).reset_index(drop=True)

    return df_result


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
