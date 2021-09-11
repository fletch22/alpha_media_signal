import statistics
from datetime import timedelta
from typing import List, Dict

import pandas as pd

from ams.DateRange import DateRange
from ams.config import logger_factory
from ams.monte_carlo.ma_bottom_feed_data import stock_dict_high_divyield, stock_dict_30pe_tradeable_1
from ams.services import realtime_quote_service, slack_service
from ams.utils import ticker_utils, date_utils

logger = logger_factory.create(__name__)

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def get_roi(min_days_under_sought: int, stocks: Dict[str, List[str]]):
    # tickers = ticker_service.get_ticker_info()["ticker"].unique()

    num_days_under = 40
    num_hold_days = 1
    all_rois = []
    for _year in sorted(stocks.keys()):
        year = int(_year[1:])
        tickers = stocks[_year]
        year_rois = []

        start_dt_str = f"{year}-01-01"
        start_dt = date_utils.parse_std_datestring(start_dt_str)

        end_dt = start_dt + timedelta(days=365)
        dr = DateRange(from_date=start_dt, to_date=end_dt)

        df = ticker_utils.get_maos(tickers=tickers, dr=dr, num_days_under=num_days_under)

        tar_dts = []
        for k in range(365):
            target_dt = start_dt + timedelta(days=k) + timedelta(hours=11)

            is_closed, _ = date_utils.is_stock_market_closed(target_dt)
            if is_closed:
                continue
            else:
                tar_dts.append(target_dt)

        for target_dt in tar_dts:
            target_dt_str = date_utils.get_standard_ymd_format(target_dt)

            df_one_day = df[df["date"] == target_dt_str]

            df_all = []
            col_days_down = "days_down_in_a_row"
            df_one_day[col_days_down] = -1
            for x in range(num_days_under - 1, 0, -1):
                col = f"is_long_down_{x}"

                df_true = df_one_day[df_one_day[col] == True].copy()
                if df_true.shape[0] > 0:
                    df_true[col_days_down] = x
                    df_all.append(df_true)

                df_one_day = df_one_day[df_one_day[col] == False].copy()

            if len(df_all) > 0:
                df_one_day = pd.concat(df_all)

                df_one_day = df_one_day.sort_values(col_days_down, ascending=False)

                mean_under_today = df_one_day[col_days_down].mean()

                df_one_day = df_one_day[df_one_day[col_days_down] >= min_days_under_sought].copy()

                import numpy as np
                for ndx, row in df_one_day.iterrows():
                    ticker = row["ticker"]
                    roi = row[f"fut_day_{num_hold_days}_roi"]
                    if not np.isnan(roi) and roi is not None:
                        mean_roi = 0
                        if len(year_rois) > 0:
                            mean_roi = statistics.mean(year_rois)
                        logger.info(f"{target_dt_str}: {ticker} roi: {roi:.4f}: mean_under_today: {mean_under_today:.1f}; mean roi so far: {mean_roi:.4f}")
                        year_rois.append(roi)

        if len(year_rois) > 0:
            logger.info(f"{year}: Num inv: {len(year_rois)}; Mean roi: {statistics.mean(year_rois):.4f}")
            all_rois += year_rois

    if len(all_rois) > 0:
        init_inv = 100
        real_ret = get_real_return(all_rois, init_inv)
        logger.info(f"\n\nInitial: 100: End with: {real_ret:,.2f}: overall mean roi: {statistics.mean(all_rois):.4f}\n\n")


def get_real_return(all_rois: List[float], init_inv: float):
    inv = init_inv
    for r in all_rois:
        inv = inv + (inv * r)
    return inv


def get_recommendations():
    # tickers = ['IBM', 'NVDA', 'GOOGL', 'ABC', 'GE', 'FB']

    # NOTE: 2021-09-09: chris.flesche: # 39th day; above 30 pe + tradeable; 0.0078
    tickers = stock_dict_30pe_tradeable_1["_2020"]

    date_str = "2021-09-09"
    start_dt = date_utils.parse_std_datestring(date_str)
    start_dt = start_dt + timedelta(hours=10)

    num_days_to_test = 1
    num_days_under = 40
    ma_days = 20
    target_days_down = [29, 39]
    all_msg = []
    for i in range(num_days_to_test):
        start_dt_str = date_utils.get_standard_ymd_format(start_dt)
        is_closed, _ = date_utils.is_stock_market_closed(start_dt)
        if is_closed:
            logger.info(f"Stock market closed {start_dt_str}")
        else:
            end_dt = start_dt + timedelta(days=1)
            end_dt_str = date_utils.get_standard_ymd_format(end_dt)
            dr = DateRange.from_date_strings(from_date_str=start_dt_str, to_date_str=end_dt_str)

            logger.info(f"Looking at {start_dt_str} to {end_dt_str}")

            df = ticker_utils.get_maos(tickers=tickers, ma_days=ma_days, dr=dr, num_days_under=num_days_under, add_future_cols=False)

            if df is None:
                continue

            # logger.info(f"Size of df: {df.shape[0]}")

            df = df.sort_values(by=["ticker", "date"])

            # logger.info(df[["ticker", "date"]].head(100))

            # cols = [f"is_long_down_{x}" for x in range(num_days_under)]
            # cols.append("ticker")
            # logger.info(df[cols].head())

            df_all = []
            col_days_down = "days_down_in_a_row"
            df[col_days_down] = -1
            for x in range(num_days_under - 1, 0, -1):
                col = f"is_long_down_{x}"

                df_true = df[df[col] == True].copy()
                if df_true.shape[0] > 0:
                    df_true[col_days_down] = x
                    df_all.append(df_true)

                df = df[df[col] == False].copy()

            if len(df_all) > 0:
                df = pd.concat(df_all)

            df = df.sort_values([col_days_down, "ticker"], ascending=False)

            logger.info(df[["ticker", col_days_down]].head())

            df = df[df["days_down_in_a_row"].isin(target_days_down)]

            for ndx, row in df.iterrows():
                ticker = row["ticker"]
                days_down = row["days_down_in_a_row"]
                close = row["close"]
                close_ma = row["close_ma"]
                raw_stock_info = realtime_quote_service.get_raw_stock_price(ticker=ticker)
                curr_price = raw_stock_info['c']
                ma_1_d_before = row["close_ma_1_day_before"]

                ma_proj = ((ma_1_d_before * (ma_days - 1)) + curr_price) / ma_days

                pred_buy = "Yes" if (ma_proj > curr_price) else "No"

                msg = f"{ticker}; buy: {pred_buy}; dd: {days_down}; last close: {close:.2f}; last ma: {close_ma:.2f}; current_price: {curr_price:.2f}; ma projected: {ma_proj:.2f}: "
                all_msg.append(msg)

                logger.info(msg)

        start_dt = start_dt + timedelta(days=1)

    msgs = ", \n".join(all_msg)
    if msgs is not None:
        slack_service.send_direct_message_to_chris(message=msgs)
