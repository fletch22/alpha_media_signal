from datetime import timedelta, datetime
from random import shuffle
from statistics import mean

import pandas as pd

from ams.config import constants, logger_factory
from ams.services import ticker_service
from ams.utils import date_utils

# import matplotlib
# # matplotlib.use("TkAgg")  # Do this before importing pyplot!
# import matplotlib.pyplot as plt


logger = logger_factory.create(__name__)


def start(start_dt: datetime, num_hold_days: int, num_days_perf: int, end_date_str: str = None, min_price: float = 0, size_buy_lot: int = None):
    df_preds = pd.read_csv(constants.TWITTER_PREDICTIONS_PATH)

    all_days_rois = []

    for day_ndx in range(num_days_perf):
        dt = start_dt + timedelta(days=day_ndx)
        date_str = date_utils.get_standard_ymd_format(dt)
        if end_date_str is not None and date_str > end_date_str:
            break
        roi = get_days_roi_from_prediction_table(df_preds, date_str, num_hold_days, min_price, size_buy_lot)
        if roi is not None:
            all_days_rois.append(roi)

    if len(all_days_rois) > 0:
        print(f"Overall roi: {mean(all_days_rois):.4f}")


def get_days_roi_from_prediction_table(df_preds: pd.DataFrame, date_str: str, num_hold_days: int, min_price: float = None, size_buy_lot: int = None):
    df = df_preds[df_preds["purchase_date"] == date_str]

    tickers = df["f22_ticker"].to_list()
    if size_buy_lot is not None and size_buy_lot < len(tickers):
        shuffle(tickers)
        tickers = tickers[:size_buy_lot]
    rois = []

    for t in tickers:
        df_tick = ticker_service.get_ticker_eod_data(t)
        df_tick = df_tick[df_tick["date"] >= date_str]
        df_tick.sort_values(by=["date"], ascending=True, inplace=True)

        row_tick = df_tick.iloc[0]
        purchase_price = row_tick["close"]
        if min_price is None or purchase_price > min_price:
            if df_tick.shape[0] == 0:
                logger.info(f"No EOD stock data for {date_str}.")
                continue

            num_days = df_tick.shape[0]
            row = None
            if num_days > num_hold_days:
                row = df_tick.iloc[num_hold_days]
            elif num_days > 1:
                row = df_tick.iloc[num_days - 1]

            if row is None:
                roi = 0
            else:
                sell_price = row["close"]
                roi = (sell_price - purchase_price) / purchase_price

            rois.append(roi)

    result = None
    if len(rois) > 0:
        result = mean(rois)
        print(f"{date_str}: roi: {result}")
    else:
        print(f"No data found on {date_str}.")

    return result


# Assert

if __name__ == '__main__':
    start_date_str = "2020-08-01"
    end_date_str = "2021-01-14"
    min_price = 5.
    start_dt = date_utils.parse_std_datestring(start_date_str)

    start(start_dt=start_dt, num_hold_days=1, num_days_perf=171, end_date_str=end_date_str, min_price=min_price, size_buy_lot=None)