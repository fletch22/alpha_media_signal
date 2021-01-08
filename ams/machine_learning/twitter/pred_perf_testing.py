from datetime import timedelta, datetime
from random import shuffle
from statistics import mean

import pandas as pd

from ams.config import constants, logger_factory
from ams.services import ticker_service
from ams.utils import date_utils

import matplotlib
matplotlib.use("TkAgg")  # Do this before importing pyplot!
import matplotlib.pyplot as plt


logger = logger_factory.create(__name__)


def start(start_dt: datetime, num_hold_days: int, num_days_perf: int, min_price: float = 0, size_buy_lot: int = None):
    df_preds = pd.read_csv(constants.TWITTER_PREDICTIONS_PATH)

    all_days_rois = []

    # FIXME: 2021-01-04: chris.flesche: Temp
    num_hold_days_tmp = num_hold_days
    # num_hold_days_tmp += 1

    for day_ndx in range(num_days_perf):
        dt = start_dt + timedelta(days=day_ndx)
        date_str = date_utils.get_standard_ymd_format(dt)
        df = df_preds[(df_preds["purchase_date"] == date_str) & (df_preds["num_hold_days"] == num_hold_days)]

        # Act
        tickers = df["f22_ticker"].to_list()
        if size_buy_lot is not None and size_buy_lot < len(tickers):
            shuffle(tickers)
            tickers = tickers[:size_buy_lot]

        rois = []
        for t in tickers:
            df_tick = ticker_service.get_ticker_eod_data(t)
            df_tick = df_tick[df_tick["date"] >= date_str]
            df_tick.sort_values(by=["date"], inplace=True)
            # FIXME: 2021-01-04: chris.flesche: Temp
            if df_tick.shape[0] < num_hold_days:
            # if df_tick.shape[0] < num_hold_days_tmp:
                print(f"Not enough tick {t} data: {df_tick.shape[0]}; Hold days: {num_hold_days}")
                continue
            # FIXME: 2021-01-04: chris.flesche: Temp
            if min_price is None or (df_tick.iloc[0]["close"] > min_price):
            # if min_price is None or (df_tick.iloc[1]["open"] > min_price):
                if df_tick.shape[0] == 0:
                    logger.info(f"No EOD stock data for {date_str}.")
                    continue
                # FIXME: 2021-01-04: chris.flesche: Temp
                purchase_price = df_tick.iloc[0]["close"]
                # purchase_price = df_tick.iloc[1]["open"]
                if df_tick.shape[0] > num_hold_days_tmp:
                    sell_price = df_tick.iloc[num_hold_days_tmp]["close"]
                    rois.append((sell_price - purchase_price) / purchase_price)

        if len(rois) > 0:
            day_roi = mean(rois)
            print(f"{date_str} roi: {day_roi:.4f}")
            all_days_rois.append(day_roi)
        else:
            print(f"No data found on {date_str}.")

    if len(all_days_rois) > 0:
        print(f"Overall roi: {mean(all_days_rois):.4f}")


# Assert

if __name__ == '__main__':
    start_date_str = "2020-08-10"
    start_dt = date_utils.parse_std_datestring(start_date_str)

    start(start_dt=start_dt, num_hold_days=5, num_days_perf=150, min_price=5., size_buy_lot=None)