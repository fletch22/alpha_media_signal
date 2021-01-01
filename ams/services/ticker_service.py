import time
from datetime import datetime, timedelta
from typing import List, Dict

import pandas as pd
from pandas import DataFrame, CategoricalDtype
from sklearn.preprocessing import StandardScaler

from ams.DateRange import DateRange
from ams.config import constants
from ams.services import file_services, pickle_service
from ams.services.EquityFields import EquityFields
from ams.utils import date_utils

ticker_cache = {}
ticker_date_row_cache = {}

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def get_ticker_cache():
    return ticker_cache


def set_ticker_cache(tc: Dict):
    ticker_cache = tc


def get_ticker_eod_data(ticker: str) -> DataFrame:
    ticker_path = file_services.get_eod_ticker_file_path(ticker)
    df = None
    if ticker_path.exists():
        df = pd.read_csv(str(ticker_path))

    return df


def get_ticker_attribute_on_date(ticker: str, dt: datetime,
                                 equity_fields: List[EquityFields] = [EquityFields.high]):
    date_str = date_utils.get_standard_ymd_format(dt)
    cache_key = f"{ticker}_{date_str}"

    row = {}
    if cache_key in ticker_date_row_cache.keys():
        row = ticker_date_row_cache[cache_key]
    else:
        max = 4
        tc = get_ticker_cache()
        if ticker in tc.keys():
            df = tc[ticker]
        else:
            df = get_ticker_eod_data(ticker)
            if len(tc.keys()) > max:
                first_key = list(tc.keys())[0]
                del tc[first_key]
            tc[ticker] = df

        if df is not None:
            df_filtered = df[df["date"] == date_str]

            records = df_filtered.to_dict('records')
            if len(records) > 0:
                row = records[0]
                ticker_date_row_cache[cache_key] = row

    if row is not None and bool(row):
        values = [row[ef.value] for ef in equity_fields]
    else:
        values = [None for ef in equity_fields]

    return tuple(values)


def get_ticker_attr_from_date(ticker: str, date_str: str,
                              equity_fields: List[EquityFields] = [EquityFields.high],
                              days_ahead: int = 5):
    dt = date_utils.parse_std_datestring(date_str)
    dt_new = dt + timedelta(days=days_ahead)

    return get_ticker_attribute_on_date(ticker, dt=dt_new, equity_fields=equity_fields)


def get_start_end_dates(date_strs: List[str]):
    date_strs = sorted(date_strs)
    start_date = date_strs[0]
    end_date = date_strs[len(date_strs) - 1]

    dt = date_utils.parse_std_datestring(end_date)
    dt_end_adjust = dt + timedelta(days=4)
    end_date_adj = date_utils.get_standard_ymd_format(dt_end_adjust)

    return start_date, end_date_adj


def get_ticker_on_dates(tick_dates: Dict[str, List[str]], num_days_in_future: int = 1, should_drop_missing_future_dates: bool=True) -> pd.DataFrame:

    all_dfs = []
    for ticker, date_strs in tick_dates.items():
        df_equity = get_equity_on_dates(ticker=ticker, date_strs=date_strs,
                                        num_days_in_future=num_days_in_future)
        if df_equity is not None:
            all_dfs.append(df_equity)

    df_ticker = pd.concat(all_dfs)

    if should_drop_missing_future_dates:
        df_ticker = df_ticker.dropna(subset=["future_open", "future_low", "future_high", "future_close", "future_date"])

    return df_ticker


def get_equity_on_dates(ticker: str, date_strs: List[str],
                        num_days_in_future: int = 1) -> pd.DataFrame:
    df = get_ticker_eod_data(ticker)
    df_in_dates = None
    if df is not None:
        start, end = get_start_end_dates(date_strs)
        df_in_range = df[(df["date"] >= start) & (df["date"] <= end)].sort_values(by="date")
        df_in_range["future_open"] = df_in_range["open"]
        df_in_range["future_low"] = df_in_range["low"]
        df_in_range["future_high"] = df_in_range["high"]
        df_in_range["future_close"] = df_in_range["close"]
        df_in_range["future_date"] = df_in_range["date"]
        cols = ["future_open", "future_low", "future_high", "future_close", "future_date"]
        df_in_range[cols] = df_in_range[cols].shift(-num_days_in_future)

        df_in_dates = df_in_range[df_in_range["date"].isin(date_strs)]
    return df_in_dates


def get_next_date(df_stocks: pd.DataFrame, ticker: str, date_str: str):
    df_found = df_stocks[(df_stocks["ticker"] == ticker) & (df_stocks["date"] > date_str)]

    result = (None, None, None, None)
    if df_found.shape[0] > 0:
        df_found.sort_values(["ticker", "date"], inplace=True)
        row = dict(df_found.iloc[0])
        result = row[EquityFields.open.value], row[EquityFields.low.value], row[
            EquityFields.high.value], row[EquityFields.close.value]

    return result


def pull_in_next_trading_day_info(df_tweets: pd.DataFrame):
    ttd = extract_ticker_tweet_dates(df_tweets)
    df_list = []
    start = time.time()
    for ticker, date_strs in ttd.items():
        df = get_next_trading_day(ticker=ticker, trading_days=date_strs)
        if df is not None:
            df_thin = df[[EquityFields.close.value, EquityFields.open.value, EquityFields.low.value,
                          EquityFields.high.value, EquityFields.ticker.value,
                          EquityFields.date.value]]
            df_list.append(df_thin)
    df_stocks = pd.concat(df_list)
    end = time.time()
    elapsed = end - start
    print(f"Got all stock data: {elapsed}")

    col_tmp = "tmp_future_values"
    df_tweets[col_tmp] = df_tweets.apply(
        lambda x: get_next_date(df_stocks=df_stocks, ticker=x["f22_ticker"], date_str=x["date"]),
        axis=1)

    new_cols = ["future_open", "future_low", "future_high", "future_close"]
    df_tweets[new_cols] = pd.DataFrame(df_tweets[col_tmp].tolist(), index=df_tweets.index)
    return df_tweets.drop(columns=col_tmp).dropna(subset=new_cols)


def get_next_trading_day(ticker: str, trading_days: List[str]):
    trading_days = sorted(trading_days)
    df = get_ticker_eod_data(ticker=ticker)

    df_dated = None
    if df is None:
        print(f"Equity ticker {ticker} not found.")
    else:
        start_date = trading_days[0]
        end_date = trading_days[len(trading_days) - 1]

        dt = date_utils.parse_std_datestring(end_date)
        dt_end_adjust = dt + timedelta(days=4)
        end_date_adj = date_utils.get_standard_ymd_format(dt_end_adjust)

        df_dated = df[(df["date"] >= start_date) & (df["date"] <= end_date_adj)]

    return df_dated


def get_next_trading_day_attr(ticker: str, date_str: str,
                              equity_fields: List[EquityFields] = [EquityFields.high],
                              max_days_head: int = 5):
    result = None,
    for i in range(1, max_days_head):
        result = get_ticker_attr_from_date(ticker=ticker, date_str=date_str,
                                           equity_fields=equity_fields, days_ahead=i)
        if result[0] is not None:
            break

    return result


def get_tickers_in_range(tickers: List[str], date_range: DateRange) -> DataFrame:
    start_date_str = date_utils.get_standard_ymd_format(date_range.from_date)
    end_date_str = date_utils.get_standard_ymd_format(date_range.to_date)

    all_dfs = []
    for t in tickers:
        df = get_ticker_eod_data(ticker=t)
        if df is not None:
            df_dated = df[(df['date'] > start_date_str) & (df['date'] < end_date_str)]
            all_dfs.append(df_dated)

    return pd.concat(all_dfs)


def extract_ticker_tweet_dates(df_tweets: pd.DataFrame):
    df_distinct_tweet_dts = df_tweets[["f22_ticker", "date"]].drop_duplicates()
    df_g_stocks = df_distinct_tweet_dts.groupby(by=["f22_ticker"])

    stock_days = {}
    for group_name, df_group in df_g_stocks:
        ticker = group_name
        dates = df_group["date"].values.tolist()
        stock_days[ticker] = dates

    return stock_days


def get_all_tickers():
    da_paths = file_services.walk(constants.SHAR_SPLIT_EQUITY_EOD_DIR)

    tickers = []
    for d in da_paths:
        tickers.append(d.stem)

    return tickers


def get_tickers_w_filters(min_price: float = 5.0, min_volume: int = 100000):
    da_paths = file_services.walk(constants.SHAR_SPLIT_EQUITY_EOD_DIR)

    tickers = []
    for d in da_paths:
        ticker = d.stem
        print(f"Inspecting '{ticker}'")
        df = get_ticker_eod_data(ticker)
        row = df.iloc[-1]
        price = row["close"]
        volume = row["volume"]
        if price > min_price and volume > min_volume:
            tickers.append(ticker)

    return tickers


def get_ticker_info():
    return pd.read_csv(constants.SHAR_TICKER_DETAIL_INFO_PATH)


def get_nasdaq_info():
    df = get_ticker_info()
    return df[df["exchange"] == "NASDAQ"]


def make_one_hotted(df: pd.DataFrame, df_all_tickers: pd.DataFrame, cols: List[str]):
    df_one_hots = []
    for c in cols:
        df_all_tickers[c] = df_all_tickers[c].fillna("<unknown>")
        uniques = df_all_tickers[c].unique().tolist()
        uniques.append("<unknown>")
        uniques = list(set(uniques))

        df[c] = df[c].fillna("<unknown>")
        df[c] = df[c].astype(CategoricalDtype(uniques))
        df_new_cols = pd.get_dummies(df[c], prefix=c)
        df_one_hots.append(df_new_cols)

    df_one_dropped = df.drop(columns=cols)
    df_one_hots.append(df_one_dropped)

    return pd.concat(df_one_hots, axis=1)


def make_one_hotted_for_one_column(df: pd.DataFrame, unique_values: List[str], col: str):
    df[col] = df[col].fillna("<unknown>")
    df[col] = df[col].astype(CategoricalDtype(unique_values))
    df_new_cols = pd.get_dummies(df[col], prefix=col)

    df_dropped = df.drop(columns=[col])

    return pd.concat([df_dropped, df_new_cols], axis=1)


def get_nasdaq_tickers():
    df_nasdaq = get_nasdaq_info()

    df_dropped = df_nasdaq.drop(
        columns=["firstpricedate", "lastpricedate", "firstquarter", "lastquarter",
                 "secfilings", "companysite", "lastupdated", "cusips",
                 "isdelisted", "name", "exchange", "firstadded", "permaticker", "sicindustry",
                 "relatedtickers"
                 ])

    df_all_tickers = get_ticker_info()
    df_rem = df_all_tickers[df_dropped.columns]

    columns = [c for c in df_rem.columns if str(df_rem[c].dtype) == "object"]
    columns.remove("ticker")

    df_one_hotted = make_one_hotted(df=df_rem, df_all_tickers=df_all_tickers, cols=columns)

    df_ren = df_one_hotted.rename(columns={"ticker": "ticker_drop"})

    return df_ren


def std_dataframe(df_train: pd.DataFrame, df_test: pd.DataFrame, df_val: pd.DataFrame):
    df_train = df_train.copy()
    df_test = df_test.copy()
    df_val = df_val.copy()
    num_cols = [c for c in df_train.columns if
                str(df_train[c].dtype) == "float64"]  # need logic to get numeric

    for c in num_cols:
        standard_scaler = StandardScaler()

        df_train = fillna_column(df=df_train, col=c)
        df_test = fillna_column(df=df_test, col=c)
        df_val = fillna_column(df=df_val, col=c)

        # NOTE: 2020-10-11: chris.flesche: Consider adding another column here
        # col_new = f"{c}_na_filled"
        # df_train[col_new] = False
        # df_train.loc[df_train[c] == None, col_new] = True

        with pd.option_context('mode.chained_assignment', None):
            df_train.loc[:, c] = standard_scaler.fit_transform(df_train[[c]])
            df_test.loc[:, c] = standard_scaler.transform(df_test[[c]])
            df_val.loc[:, c] = standard_scaler.transform(df_val[[c]])

    return df_train, df_test, df_val


def fillna_column(df: pd.DataFrame, col: str):
    median = df[col].median()
    median = 0 if str(median) == 'nan' else median
    with pd.option_context('mode.chained_assignment', None):
        df.loc[df[col].isnull(), col] = median
        # df[col] = df[col].fillna(median)

    return df


def get_single_attr(df: pd.DataFrame, col: str):
    col_values = df[col].tolist()
    result = None
    if len(col_values) > 0:
        result = col_values[0]
    return result


def get_stock_info(df: pd.DataFrame, ticker: str, date_str: str):
    df_ticker_on_date = df[(df["f22_ticker"] == ticker) & (df["purchase_date"] == date_str)]

    original_open = get_single_attr(df_ticker_on_date, "open")
    if original_open is None:
        raise Exception("original_open is None.")

    close = get_single_attr(df_ticker_on_date, "original_close_price")
    if close is None:
        raise Exception("close is None.")

    split_share_multiplier = get_single_attr(df_ticker_on_date, "split_share_multiplier")
    if split_share_multiplier is None:
        raise Exception("split_share_multiplier is None.")
    split_share_multiplier = 1 / split_share_multiplier

    future_close = get_single_attr(df_ticker_on_date, "future_close")
    if future_close is None:
        raise Exception("future_close is None.")

    future_high = get_single_attr(df_ticker_on_date, "future_high")
    if future_high is None:
        raise Exception("future_high is None.")

    future_open = get_single_attr(df_ticker_on_date, "future_open")
    if future_open is None:
        raise Exception("future_open is None.")

    future_date = get_single_attr(df_ticker_on_date, "future_date")
    if future_open is None:
        raise Exception("future_date is None.")

    return close, future_close, future_high, future_open, future_date, split_share_multiplier


def get_roi(close_price: float, target_roi_frac: float, future_high: float, future_close: float):
    target_price = close_price * (1 + target_roi_frac)

    if target_price < future_high:
        roi = target_roi_frac
    else:
        roi = (future_close - close_price) / close_price

    return roi


def calculate_roi(target_roi: float, close_price: float, future_high: float, future_close: float,
                  calc_dict: Dict[str, List[float]], zero_in: bool = False):
    last_roi = None
    num_repeats = 0
    max_repeats = 3
    if zero_in:
        calc_tar_roi = target_roi
        calc_roi = get_roi(close_price=close_price, target_roi_frac=calc_tar_roi,
                           future_high=future_high, future_close=future_close)
        calc_key = str(round(calc_tar_roi, 6))

        if calc_key not in calc_dict.keys():
            calc_dict[calc_key] = []
        calc_list = calc_dict[calc_key]
        calc_list.append(calc_roi)
        # print(f"calc_key: {calc_key}: calc_roi: {calc_roi}: {statistics.mean(calc_list):.4f}%")
    else:
        for i in range(800):
            calc_tar_roi = target_roi + (i * .001)
            calc_roi = get_roi(close_price=close_price, target_roi_frac=calc_tar_roi,
                               future_high=future_high, future_close=future_close)
            calc_key = str(round(calc_tar_roi, 6))

            if calc_key not in calc_dict.keys():
                calc_dict[calc_key] = []
            calc_list = calc_dict[calc_key]
            calc_list.append(calc_roi)
            # print(f"calc_key: {calc_key}: calc_roi: {calc_roi}: {statistics.mean(calc_list):.4f}%")

            if calc_roi == last_roi:
                num_repeats += 1
                if num_repeats > max_repeats:
                    break
            else:
                num_repeats = 0

            last_roi = calc_roi


def add_days_until_sale(df: pd.DataFrame):
    def days_between(row):
        date_str = row["purchase_date"]
        future_date_str = row["future_date"]

        dt_date = date_utils.parse_std_datestring(date_str)
        dt_fut_date = date_utils.parse_std_datestring(future_date_str)
        days_bet = (dt_fut_date - dt_date).days

        return days_bet

    df["days_util_sale"] = df.apply(days_between, axis=1)

    return df


def get_nasdaq_perf(date_from: datetime,
                    tickers: List[str],
                    date_to: datetime = None,
                    min_price: float = None,
                    max_price: float = None,
                    days_hold_stock: int = 1) -> (pd.DataFrame, List[str]):
    date_from_str = date_utils.get_standard_ymd_format(date_from)
    date_to_str = date_utils.get_standard_ymd_format(date_to)

    all_df = list()
    cols = ["future_open", "future_close", "future_high", "future_low"]

    ticks_gathered = set()
    for ndx, t in enumerate(tickers):
        df = get_ticker_eod_data(ticker=t)
        if df is not None:
            print(f"Processing ticker: {t}")
            df = df[df["date"] >= date_from_str]
            if date_to is not None:
                df = df[df["date"] <= date_to_str]
            if min_price is not None:
                df = df[df["open"] > min_price]
            if max_price is not None:
                df = df[df["open"] < max_price]
            if df is not None and df.shape[0] > 0:
                df.sort_values(by=["date"], inplace=True)
                df["future_open"] = df["open"]
                df["future_low"] = df["low"]
                df["future_high"] = df["high"]
                df["future_close"] = df["close"]
                df[cols] = df[cols].shift(-days_hold_stock)
                ticks_gathered.add(t)
                all_df.append(df)

    num_tickers = len(all_df)
    df_nas = pd.concat(all_df, axis=0).dropna(subset=cols)

    min_date = df_nas["date"].min()
    max_date = df_nas["date"].max()

    dt_min = date_utils.parse_std_datestring(min_date)
    dt_max = date_utils.parse_std_datestring(max_date)

    dt_current = dt_min + timedelta(days=1)
    dt_max = dt_max + timedelta(days=1)

    df_nas.sort_values(by=["ticker", "date"], inplace=True)

    df_nas["roi"] = (df_nas["future_close"] - df_nas["close"]) / df_nas["close"]

    results = []
    while dt_current < dt_max:
        dt_curr_str = date_utils.get_standard_ymd_format(dt_current)
        df_day = df_nas[df_nas["date"] == dt_curr_str]
        if df_day is not None and df_day.shape[0] > 0:
            roi_mean = df_day["roi"].mean()
            results.append({"date": dt_curr_str, "roi": roi_mean})
        dt_current = dt_current + timedelta(days=1)

    df = pd.DataFrame(results, columns=["date", "roi"])
    print(f"DF cols: {df.columns}")
    df.to_parquet(constants.DAILY_ROI_NASDAQ_PATH)

    return df, ticks_gathered


def create_tickers_available_on_day():
    df = get_nasdaq_info()
    all_tickers = df["ticker"].to_list()

    sorted(all_tickers)

    t_on_d = dict()
    for t_ndx, t in enumerate(all_tickers):
        print(f"Processing {t}.")
        df = get_ticker_eod_data(ticker=t)
        if df is None or df.shape[0] == 0:
            print(f"\tNo values for {t}.")
            continue
        dates = df["date"].to_list()
        close_prices = df["close"].to_list()
        for ndx, d in enumerate(dates):
            if d not in t_on_d.keys():
                t_on_d[d] = {t: close_prices[ndx]}
            else:
                t_on_d[d][t] = close_prices[ndx]
        # if t_ndx > 10:
        #     break

    print("About to pickle t_on_d.")
    pickle_service.save(t_on_d, file_path=constants.TOD_PICKLE_PATH)

    return t_on_d


def load_tickers_on_day() -> Dict:
    return pickle_service.load(file_path=constants.TOD_PICKLE_PATH)



