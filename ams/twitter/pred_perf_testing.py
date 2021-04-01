from datetime import timedelta, datetime
from enum import Enum
from pathlib import Path
from random import shuffle
from statistics import mean

import pandas as pd

from ams.config import constants, logger_factory
from ams.services import ticker_service
from ams.utils import date_utils
from ams.utils.date_utils import get_next_market_date

logger = logger_factory.create(__name__)


class TrainingOrReal(Enum):
    Training = "training"
    Real = "real"


def start(src_path: Path, start_dt: datetime, num_hold_days: int, num_days_perf: int,
          end_date_str: str = None, min_price: float = 0, size_buy_lot: int = None,
          verbose: bool = False, addtl_hold_days: int = 0, training_or_real: TrainingOrReal = TrainingOrReal.Training):
    from ams.pipes.p_make_prediction.mp_process import PREDICTIONS_CSV
    from ams.pipes.p_make_prediction.mp_process import MONEY_PREDICTIONS_CSV

    file_path = Path(src_path, PREDICTIONS_CSV)
    if training_or_real == TrainingOrReal.Real:
        file_path = Path(src_path, MONEY_PREDICTIONS_CSV)

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

    matched_tickers = []
    for t in tickers:
        df_tick = ticker_service.get_ticker_eod_data(t)
        df_tick.sort_values(by=["date"], ascending=True, inplace=True)

        df_tweet_date =  df_tick[df_tick["date"] < purchase_date_str].copy()
        df_tweet_date.sort_values(by=["date"], ascending=True, inplace=True)
        df_tick = df_tick[df_tick["date"] >= purchase_date_str].copy()

        if df_tick.shape[0] > 0 and df_tweet_date.shape[0] > 0:
            tweet_date_tick = df_tweet_date.iloc[-1]
            tweet_close = tweet_date_tick["close"]
            purchase_date_tick = df_tick.iloc[0]
            purchase_price = purchase_date_tick["close"]

            # NOTE: 2021-03-30: chris.flesche: Testing indicates that we should avoid stocks where price < -1% below prev
            # close or > 0% above prev close date. Perhaps this should be disabled
            # until I can determine how to implement this; the naive solution is to manually check the price at close
            # time; but large EOD price fluxes may alter the equity's buy eligibility or I might not be able execute a
            # trade close to the close date. We'll see.
            pre_purch_inc = (purchase_price - tweet_close) / tweet_close
            if pre_purch_inc > 0.:
                continue

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
                matched_tickers.append(t)

                if size_buy_lot is not None and len(rois) >= size_buy_lot:
                    break

    result = None
    if len(rois) > 0:
        result = mean(rois)
        suffix = ""
        if verbose:
            suffix = f": {sorted(matched_tickers)}"
        logger.info(f"{purchase_date_str}: roi: {result}: {len(rois)} tickers{suffix}")
    else:
        logger.info(f"No data found on {purchase_date_str}.")

    return result


# Assert

if __name__ == '__main__':
    # start_date_str = "2020-08-10"
    # end_date_str = "2021-02-16"
    start_date_str = "2020-08-10"
    end_date_str = "2021-03-30"
    min_price = 5.0
    num_hold_days = 1
    addtl_hold_days = 1
    start_dt = date_utils.parse_std_datestring(start_date_str)
    training_or_real = TrainingOrReal.Training

    src_path = Path(constants.TWITTER_OUTPUT_RAW_PATH, "prediction_bucket")

    start(src_path=src_path, start_dt=start_dt, num_hold_days=num_hold_days, num_days_perf=255,
          end_date_str=end_date_str, min_price=min_price, size_buy_lot=None,
          verbose=True,
          addtl_hold_days=addtl_hold_days,
          training_or_real=training_or_real)