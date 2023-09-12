import statistics
from datetime import timedelta
from random import shuffle
from typing import List, Dict

import pandas as pd

from ams.DateRange import DateRange
from ams.config import logger_factory
from ams.monte_carlo.ma_bottom_feed_data import stock_dict_high_pe_above_30, stock_dict_low_evebitda
from ams.services import realtime_quote_service, slack_service
from ams.utils import ticker_utils, date_utils

logger = logger_factory.create(__name__)

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def get_roi(min_days_under_sought: int, stocks: Dict[str, Dict], use_max_num_invs: bool = False):
    num_days_under = 50
    num_hold_days = 1
    all_rois = []
    num_years = len(stocks.keys())
    for key, info in stocks.items():
        year_rois = []

        start_dt_str = info["start_dt"]
        end_dt_str = info["end_dt"]
        tickers = info["tickers"]
        dr = DateRange.from_date_strings(from_date_str=start_dt_str, to_date_str=end_dt_str)
        start_dt = dr.from_date

        df = ticker_utils.get_maos(tickers=tickers, dr=dr, num_days_under=num_days_under)

        tar_dts = []
        target_dt = start_dt + timedelta(hours=11)

        while target_dt < dr.to_date:
            target_dt = target_dt + timedelta(days=1)

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
                        if roi > .5:
                            print(f"\n\nXX-Large: {ticker}: {roi}\n\n")

                        # NOTE: 2021-09-11: chris.flesche: The data on this date is suspiciously high
                        if target_dt_str != "2019-11-15" and ticker != "GOVX":
                            logger.info(f"{target_dt_str}: {ticker} roi: {roi:.4f}: mean_under_today: {mean_under_today:.1f}; mean roi so far: {mean_roi:.4f}")
                            year_rois.append(roi)

        if len(year_rois) > 0:
            year = start_dt.year
            period = info["period"]
            logger.info(f"{year}_{period}: Num inv: {len(year_rois)}; Mean roi: {statistics.mean(year_rois):.4f}")
            all_rois += year_rois

    if use_max_num_invs:
        shuffle(all_rois)
        max_num_invs = 250 * num_years
        if len(all_rois) > max_num_invs:
            all_rois = all_rois[:max_num_invs]

    return all_rois, min_days_under_sought


def get_real_return(all_rois: List[float], init_inv: float):
    inv = init_inv
    for r in all_rois:
        inv = inv + (inv * r)
    return inv


def get_recommendations():
    tickers = ['GOOGL', "AMZN"]
    # tickers = ['IBM', 'NVDA', 'GOOGL', 'ABC', 'GE', 'FB']
    # tickers = ["AMZN", "AAPL", "COIN", "ADBE", "CRWD"]

    # NOTE: 2021-09-09: chris.flesche: # 39th day; above 30 pe + tradeable; 0.0078
    # tickers = stock_dict_low_evebitda["_2021_p0"]["tickers"]

    # tickers = ['QUOT', 'RNG', 'SDC', 'FEAC', 'HUBS', 'MNCL', 'MODN', 'OAC', 'AGNC', 'HYAC', 'ROKU', 'BYND', 'TDOC', 'ZG', 'PEN', 'RDFN', 'SITM', 'TCMD', 'WTRE', 'HCAC', 'PENN', 'ALGT', 'STAR', 'BYFC', 'FVRR', 'AVLR', 'AWI', 'SPOT', 'AAXN', 'SRCL', 'AMTX', 'LFAC', 'BDC', 'NET', 'WMG', 'CHEF', 'CNTY', 'SDGR', 'FSLY', 'OPES', 'BILL', 'TNDM', 'MUR', 'LMPX', 'TRNE', 'CMO', 'EVER', 'IRTC', 'DMYT', 'NLY', 'UPWK', 'WKHS', 'TEAM', 'TRHC', 'NEWR', 'APPN', 'SHOO', 'CDNA', 'AMRN', 'DK', 'TWLO', 'ZEN', 'EVBG', 'PDFS', 'BLFS', 'EFC', 'MXL', 'CYRX', 'MSGS', 'CVLT', 'TXG', 'EXTR', 'DKNG', 'MDP', 'GNMK', 'QTWO', 'PDD', 'CCXI', 'WIX', 'SVMK', 'PLUG', 'PDCE', 'INCY', 'CSII', 'GPMT', 'HLIT', 'RPD', 'NFE', 'NVRO', 'PEGA', 'PINS', 'HSII', 'GPRE', 'TRS', 'SILK', 'WK', 'OPRX', 'IMAX', 'LASR', 'GH']

    date_str = "2021-09-17"
    start_dt = date_utils.parse_std_datestring(date_str)
    start_dt = start_dt + timedelta(hours=10)

    num_days_to_test = 10
    num_days_under = 40
    ma_days = 20
    target_days_down = list(range(20,31))
    all_msg = []
    tickers_ided = set()
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

                if ticker not in tickers_ided:
                    tickers_ided.add(ticker)
                    msg = f"{ticker}; buy: {pred_buy}; dd: {days_down}; last close: {close:.2f}; last ma: {close_ma:.2f}; current_price: {curr_price:.2f}; ma projected: {ma_proj:.2f}: "
                    all_msg.append(msg)

        start_dt = start_dt + timedelta(days=1)

    msgs = ", \n".join(all_msg)
    if msgs is not None:
        logger.info(msgs)
        # slack_service.send_direct_message_to_chris(message=msgs)

if __name__ == '__main__':
    get_recommendations()