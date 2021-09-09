import math
from random import shuffle
from statistics import mean

import numpy as np

from ams.config import logger_factory
from ams.services import ticker_service
from ams.utils import ticker_utils

logger = logger_factory.create(__name__)


def get_ticker_vol(ticker: str, vol_thresh: float = None) -> (float, float):
    df = ticker_service.get_ticker_eod_data(ticker)

    if df is None:
        return None, None

    df = df[df["date"] > "2021-04-10"].copy()

    if df.shape[0] < 2:
        return None, None

    first_day = df.iloc[0]
    init_price = first_day["open"]
    close_price = first_day["closeadj"]

    pur_price = init_price

    init_investment = 1000
    investment = init_investment
    num_shares = math.floor(init_investment / pur_price)
    cash_acct_tot = init_investment % pur_price

    sell_thresh_rat = .1
    buy_thresh_rat = -.1

    rng = df.shape[0] - 1
    sell_price = init_price
    last_close = close_price

    df = ticker_utils.add_price_volatility(df=df, col_price="close")

    for i in range(rng):

        today_row = df.iloc[i + 1]
        low = today_row["low"]
        high = today_row["high"]
        last_close = today_row["closeadj"]
        vola = today_row["price_volatility"]

        if vol_thresh is not None:
            if np.isnan(vola) or vola > vol_thresh:
                continue

        if num_shares > 0:
            max_inc_roi = (high - pur_price) / pur_price
            if max_inc_roi > sell_thresh_rat:
                sell_price = pur_price + (pur_price * sell_thresh_rat)
                investment = (sell_price * num_shares) + cash_acct_tot
                num_shares = 0
        else:
            max_dec_rate = (low - sell_price) / sell_price
            if max_dec_rate < buy_thresh_rat:
                pur_price = sell_price + (sell_price * buy_thresh_rat)
                num_shares = math.floor(investment / pur_price)
                cash_acct_tot = init_investment % pur_price

    if num_shares > 0:
        investment = (last_close * num_shares) + cash_acct_tot
        num_shares = 0

    inv_roi = (investment - init_investment) / init_investment
    hodl_roi = (last_close - init_price) / init_price

    return inv_roi, hodl_roi


def test_volatility_trading():
    # Arrange
    n_ticks = ticker_service.get_nasdaq_info()["ticker"].unique()
    shuffle(n_ticks)

    # n_ticks = ['EYESW', 'CABA', 'MDIA', 'VXRT', 'GOVXW', 'CTXRW', 'NDRAW', 'LGHLW', 'BRPAW', 'RKDA', 'CREXW', 'CLRBZ', 'BRMK.WS', 'RETO']
    arkk = ['TSLA', 'TDOC', 'ROKU', 'SHOP', 'SQ', 'ZM', 'TWLO', 'U', 'SPOT', 'COIN']
    best_guys = ['PAVMW', 'GEC', 'CIDM', 'VMACW', 'BRQS', 'ECOLW', 'GXGXW', 'PHAT', 'JRJC', 'AYLA', 'VRME']
    n_ticks = best_guys

    inv_roi_list = []
    hodl_roi_list = []
    best_ticks = []
    for ndx, ticker in enumerate(n_ticks):
        inv_roi, hodl_roi = get_ticker_vol(ticker, vol_thresh=None)

        if hodl_roi is None:
            continue

        if not np.isnan(inv_roi):
            if inv_roi > 1.0:
                best_ticks.append(ticker)

            inv_roi_list.append(inv_roi)

        if not np.isnan(hodl_roi):
            hodl_roi_list.append(hodl_roi)

        logger.info(f"Investment roi: {inv_roi:.4f}; hodl: {hodl_roi:.4f}; diff: {inv_roi - hodl_roi:.4f}")

        if ndx > 2000:
            break

    mean_inv_roi = 0
    if len(inv_roi_list) > 0:
        mean_inv_roi = mean(inv_roi_list)

    mean_hodl_roi = 0
    if len(hodl_roi_list) > 0:
        mean_hodl_roi = mean(hodl_roi_list)

    logger.info(f"Mean inv roi: {mean_inv_roi:.4f}; mean hodl: {mean_hodl_roi:.4f}; diff: {mean_inv_roi - mean_hodl_roi:.4f}")

    print(best_ticks)

    # Act

    # Assert