from datetime import timedelta, datetime
from enum import Enum
from random import shuffle
from statistics import mean

import pandas as pd

from ams.config import constants, logger_factory
from ams.services import ticker_service
from ams.utils import date_utils

logger = logger_factory.create(__name__)


class TrainingOrReal(Enum):
    Training = "training"
    Real = "real"


def start(start_dt: datetime, num_hold_days: int, num_days_perf: int,
          end_date_str: str = None, min_price: float = 0, size_buy_lot: int = None,
          verbose: bool = False, addtl_hold_days: int = 0, training_or_real: TrainingOrReal = TrainingOrReal.Training):

    file_path = constants.TWITTER_TRAINING_PREDICTIONS_FILE_PATH
    if training_or_real == TrainingOrReal.Real:
        file_path = constants.TWITTER_REAL_MONEY_PREDICTIONS_FILE_PATH

    df_preds = pd.read_csv(file_path)

    all_days_rois = []

    for day_ndx in range(num_days_perf):
        dt = start_dt + timedelta(days=day_ndx)
        date_str = date_utils.get_standard_ymd_format(dt)
        if end_date_str is not None and date_str > end_date_str:
            break
        roi = get_days_roi_from_prediction_table(df_preds=df_preds,
                                                 purchase_date_str=date_str,
                                                 num_hold_days=num_hold_days,
                                                 min_price=min_price,
                                                 size_buy_lot=size_buy_lot,
                                                 verbose=verbose, addtl_hold_days=addtl_hold_days)
        if roi is not None:
            all_days_rois.append(roi)

    if len(all_days_rois) > 0:
        logger.info(f"Overall roi: {mean(all_days_rois):.4f}")


def get_days_roi_from_prediction_table(df_preds: pd.DataFrame,
                                       purchase_date_str: str,
                                       num_hold_days: int,
                                       min_price: float = None,
                                       size_buy_lot: int = None,
                                       verbose: bool = False,
                                       addtl_hold_days: int = 0):
    df = df_preds[(df_preds["purchase_date"] == purchase_date_str) & (df_preds["num_hold_days"] == num_hold_days)]

    tickers = df["f22_ticker"].to_list()
    shuffle(tickers)
    rois = []

    for t in tickers:
        df_tick = ticker_service.get_ticker_eod_data(t)
        df_tick = df_tick[df_tick["date"] >= purchase_date_str]
        df_tick.sort_values(by=["date"], ascending=True, inplace=True)

        if df_tick.shape[0] > 0:
            row_tick = df_tick.iloc[0]
            purchase_price = row_tick["close"]
            if min_price is None or purchase_price > min_price:
                if df_tick.shape[0] == 0:
                    logger.info(f"No EOD stock data for {purchase_date_str}.")
                    continue

                num_days = df_tick.shape[0]
                row = None
                lookahead_days = num_hold_days + addtl_hold_days
                if num_days > lookahead_days:
                    row = df_tick.iloc[lookahead_days]
                elif num_days > 1:
                    row = df_tick.iloc[num_days - 1]

                if row is None:
                    roi = 0
                else:
                    sell_price = row["close"]
                    roi = (sell_price - purchase_price) / purchase_price

                rois.append(roi)

                if size_buy_lot is not None and len(rois) >= size_buy_lot:
                    break

    result = None
    if len(rois) > 0:
        result = mean(rois)
        suffix = ""
        if verbose:
            suffix = f": {sorted(tickers)}"
        logger.info(f"{purchase_date_str}: roi: {result}: {len(rois)} tickers{suffix}")
    else:
        logger.info(f"No data found on {purchase_date_str}.")

    return result


# Assert

if __name__ == '__main__':
    # start_date_str = "2020-08-10"
    # end_date_str = "2021-02-16"
    start_date_str = "2021-02-03" # "2020-08-10"
    end_date_str = "2021-02-16"
    min_price = 5
    num_hold_days = 5
    addtl_hold_days = 0
    start_dt = date_utils.parse_std_datestring(start_date_str)
    training_or_real = TrainingOrReal.Real

    start(start_dt=start_dt, num_hold_days=num_hold_days, num_days_perf=255,
          end_date_str=end_date_str, min_price=min_price, size_buy_lot=None,
          verbose=True,
          addtl_hold_days=addtl_hold_days,
          training_or_real=training_or_real)