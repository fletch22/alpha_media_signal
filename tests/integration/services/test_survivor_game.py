import datetime
import json
import logging
import shutil
import time
from pathlib import Path
from typing import Set, List

import pytest

from ams.monte_carlo.survivor import surv_strat
from ams.services import ticker_service as ts
from ams.monte_carlo.survivor.surv_strat import SurvivorRun, ConditionalOperator, EqCond
from ams.services import pickle_service
from ams.utils import date_utils
from ams.config import logger_factory
from ams.config import constants

ALL_EQUITIES = ts.get_all_tickers()

HOLD_DAYS = 1
NUM_TRAD_DAYS_YEAR = 252

TECH = ['MSFT', 'GE', 'DELL', 'AMD', 'INTC', 'AAPL', 'IBM', 'HPQ', 'CSCO', 'ORCL', 'GOOG', 'AMZN', 'FB', 'TSLA', 'NVDA']


def test_vanilla():
    # Arrange
    eq_conds = []
    eq_conds = [EqCond(field_name='vol_price_prod', cond_op=ConditionalOperator.GTE, field_val=(100000 * 5))]
    # eq_conds = [EqCond(field_name='volume', cond_op=ConditionalOperator.GTE, field_val=10000),
    # eq_conds = [EqCond(field_name='close', cond_op=ConditionalOperator.GTE, field_val=5),
    #             EqCond(field_name='close', cond_op=ConditionalOperator.LTE, field_val=10)]
    # eq_conds = [EqCond(field_name='close', cond_op=ConditionalOperator.LTE, field_val=5)]

    # Act
    for i in range(1, 8):
        surv_strat.process(sample_size=20,
                           up_thresh=i / 8,
                           exam_period_days=30,
                           num_loops=1000,
                           eq_conds=eq_conds,
                           turbo_mode=False)
    # surv_strat.process(sample_size=20,
    #                    up_thresh=7 / 8,
    #                    exam_period_days=30,
    #                    num_loops=1000,
    #                    eq_conds=eq_conds,
    #                    turbo_mode=False)
    # Assert


def test_get_all_equities():
    eqs = surv_strat.BASE_EQUITIES

    for eq in eqs:
        print(eq)


def test_logging():
    logger = logger_factory.create(__name__)
    logger.info("test XXXXXX")


def test_get_survivor_change_samp_sz():
    # Arrange
    # Act
    start_time = time.time()
    for samp_sz in [10, 20, 30, 40, 50, 60]:
        surv_strat.process(sample_size=samp_sz,
                           up_thresh=7 / 8,
                           exam_period_days=5,
                           num_loops=2000)
    end_time = time.time()
    print(f'Elapsed time: {end_time - start_time} seconds')
    # Assert


def test_get_survivor_iterate_1():
    # Arrange
    # Act
    start_time = time.time()
    for samp_sz in range(6):
        surv_strat.process(sample_size=50,
                           up_thresh=7 / 8,
                           exam_period_days=5,
                           num_loops=2000)
    end_time = time.time()
    print(f'Elapsed time: {end_time - start_time} seconds')
    # Assert


def test_get_all_files():
    files = SurvivorRun.get_all_pickled_files()

    for f in files:
        stemp_name = Path(f).stem

        if stemp_name == 'survivor_run_2023_07_12_03_57_28':
            sr: SurvivorRun = pickle_service.load(f)
            print(
                f'{stemp_name}: ann_roi: {sr.get_annualized_roi(sr.num_years):,.2f}%; sample_size: {sr.sample_size}; threshold: {sr.up_thresh}; Num iterations: {sr.num_loops}')

            print(sr.purchase_set_df.head(5))


@pytest.mark.parametrize(
    "use_prev_month_equities,eq_conds", [
        (True, []),
        (False, [EqCond(field_name='volume', cond_op=ConditionalOperator.GT, field_val=1000000)]),
    ]
)
def test_get_sample_equities(use_prev_month_equities: bool, eq_conds: List[EqCond]):
    # Arrange
    samp_sz = 50
    purch_dt = date_utils.get_random_past_date()
    prev_month_equities = None
    if use_prev_month_equities:
        prev_month_equities = surv_strat.get_sample_equities(purchase_date=purch_dt,
                                                             sample_size=samp_sz)

    # Act
    samp_acts = surv_strat.get_sample_equities(purchase_date=purch_dt,
                                               sample_size=samp_sz,
                                               prev_month_equities=prev_month_equities,
                                               eq_conds=eq_conds)

    # Assert
    assert len(samp_acts) == samp_sz
    if prev_month_equities is not None:
        assert len(set(samp_acts).intersection(prev_month_equities)) == 0


def test_get_random_date():
    # Arrange

    # Act
    dt = date_utils.get_random_past_date()

    # Assert
    is_closed, _ = date_utils.is_stock_market_closed(dt)
    assert not is_closed


def test_get_market_holidays():
    # Arrange
    # Act
    hols = date_utils.get_market_holidays()
    # Assert
    assert len(hols) == 79


def test_get_matching_tickers():
    ti_df = ts.get_ticker_info()
    us_stocks = set(ti_df[ti_df['exchange'].isin(['NYSE', 'NASDAQ'])]['ticker'].to_list())

    cand_tickers = set(ts.get_tickers_w_filters(min_price=0, max_price=5))
    el_stocks = us_stocks.intersection(cand_tickers)

    with open(constants.STOCKS_US_LOW_PRICE_PATH, 'w') as f:
        f.write(json.dumps(list(el_stocks)))


def test_get_technology_tickers():
    # shutil.move(constants.STOCKS_US_LOW_PRICE_PATH.stem + '.json', constants.STOCKS_US_LOW_PRICE_PATH)

    with open(constants.STOCKS_US_LOW_PRICE_PATH, 'r') as f:
        us_stocks = set(json.loads(f.read()))

    ti_df = ts.get_ticker_info()
    tech_stocks = set(ti_df[ti_df['sector'] == 'Technology']['ticker'].to_list())
    us_tech_stocks = us_stocks.intersection(tech_stocks)

    with open(constants.STOCKS_US_TECH_STOCKS_PATH, 'w') as f:
        f.write(json.dumps(list(us_tech_stocks)))


def test_show_low_price_us_stocks():
    with open(constants.STOCKS_US_LOW_PRICE_PATH, 'r') as f:
        print(json.loads(f.read()))


def test_show_tech_us_stocks():
    print(len(ts.get_all_technology_stocks()))




