from datetime import timedelta
from typing import List

import pandas as pd
from diskcache import Cache

from config import logger_factory
from config.constants import INSIDER_BUYS_DATA_PATH, INS_BUYS_SUBMISSIONS_DATA_PATH
from services.ticker_service import get_all_tickers, get_equity_on_dates
from ams.services import ticker_service as tick_svc
from utils import date_utils
from utils.Stopwatch import Stopwatch
import numpy as np

logger = logger_factory.create(__name__)

COL_TICKER = 'ticker'
CACHE_KEY_INSIDE_BUYS_W_TICKER = 'inside_buys_w_ticker'
cache = Cache('/tmp')


def process():
    # cache.clear()
    sw = Stopwatch()
    df_ins = get_inside_buys_w_ticker()
    num_hold_days = 30
    num_days_until_purchase = 4
    sock_amt = 200000
    reduce_amt = 100000
    price_min = 5.0

    initial_princ = 1000
    investment = initial_princ
    bank = 0
    df_ins.sort_values(by=['TRANS_DATE_STD'], inplace=True)
    logger.info(f'Got {df_ins.shape[0]:,} rows.')
    for ndx, row in df_ins.iterrows():
        ann_dt = row['TRANS_DATE_STD']
        mkt_dt, valid_dt_found = date_utils.ensure_market_date(ann_dt)
        if not valid_dt_found:
            continue
        ticker = row[COL_TICKER]
        earl_dt = get_earliest_ticker_date(ticker=ticker)
        if mkt_dt < earl_dt:
            continue

        dt = date_utils.parse_std_datestring(mkt_dt)
        dt = dt + timedelta(days=num_days_until_purchase)
        dt = dt + timedelta(days=num_days_until_purchase + num_hold_days + 10)
        end_dt_str = date_utils.get_standard_ymd_format(dt)
        date_strs = [mkt_dt, end_dt_str]

        df_equity = get_eq_on_dt(ticker=ticker,
                                 date_strs=date_strs,
                                 num_hold_days=num_hold_days,
                                 num_days_until_purchase=num_days_until_purchase)

        if df_equity is None or df_equity.shape[0] < 1:
            continue
        row = df_equity.iloc[0]
        open_price = row['purchase_open']
        if price_min is not None and open_price < price_min:
            continue
        future_close = row['future_close']
        if not np.isnan(future_close):
            logger.info(f'{ticker} on {mkt_dt}: open: {open_price:,.2f}, future_close: {future_close:,.2f}')
            roi = (future_close - open_price) / open_price
            investment = investment * (1 + roi)
            if investment > sock_amt:
                investment -= reduce_amt
                bank += reduce_amt
            ongoing_roi = (investment - initial_princ) / initial_princ
            logger.info(f'Investment: {investment:,.2f}:  Ongoing ROI: {ongoing_roi:,.3f}: Total: {investment + bank:,.2f}')
            if investment < 5:
                if bank < 1000:
                    break
                bank -= 1000
                investment += 1000
        # break
    final_roi = ((bank + investment) - initial_princ) / initial_princ
    logger.info(f'Final ROI: {final_roi:,.3f}: Final cash: {investment + bank:,.2f}')
    sw.end()


@cache.memoize(expire=6000, name='get_eq_on_dt')
def get_eq_on_dt(ticker: str, date_strs: List[str], num_hold_days: int, num_days_until_purchase: int):
    return get_equity_on_dates(ticker=ticker, date_strs=date_strs,
                               num_hold_days=num_hold_days, num_days_until_purchase=num_days_until_purchase)


@cache.memoize(expire=6000, name='get_earliest_ticker_date')
def get_earliest_ticker_date(ticker: str):
    return tick_svc.get_earliest_date(ticker=ticker)


def filter_to_avail(df_ins):
    ins_tickers = df_ins[COL_TICKER].unique()
    all_tickers = get_all_tickers()
    common_ticks = set(all_tickers).intersection(set(ins_tickers))
    logger.info(f'Got {len(common_ticks):,} common tickers.')
    df_ins = df_ins[df_ins[COL_TICKER].isin(common_ticks)]
    return df_ins


@cache.memoize(expire=6000, name=CACHE_KEY_INSIDE_BUYS_W_TICKER)
def get_inside_buys_w_ticker() -> pd.DataFrame:
    assert INSIDER_BUYS_DATA_PATH.exists()
    assert INS_BUYS_SUBMISSIONS_DATA_PATH.exists()
    acc_col = 'ACCESSION_NUMBER'
    symbol_col = 'ISSUERTRADINGSYMBOL'
    df_ins = pd.read_csv(INSIDER_BUYS_DATA_PATH, low_memory=False, index_col=acc_col)
    df_sub = pd.read_csv(INS_BUYS_SUBMISSIONS_DATA_PATH, low_memory=False, index_col=acc_col)
    df_sub = df_sub[[symbol_col]]
    df_ins = df_ins.merge(df_sub, on=acc_col, how='inner')
    logger.info(f'Got {df_ins.shape[0]:,} rows: columns: {df_ins.columns}')

    df_ins.rename(columns={symbol_col: COL_TICKER}, inplace=True)

    df_ins = filter_to_avail(df_ins)

    return df_ins


if __name__ == '__main__':
    process()
