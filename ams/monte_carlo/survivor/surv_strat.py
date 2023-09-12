import dataclasses
import logging
import math
import random
import statistics
import time
from datetime import timedelta, datetime
from enum import Enum
from pathlib import Path
from typing import List, Dict, Set, Tuple

from ams.services import ticker_service as ts
import pandas as pd

from ams.config import constants
from ams.config.constants import SURVIVOR_RESULTS_DIR
from ams.services import pickle_service
from ams.services import file_services
from ams.utils import date_utils
from ams.config import logger_factory

BASE_EQUITIES = ts.get_all_technology_stocks()  # ts.get_all_us_stocks()

HOLD_DAYS = 5
NUM_TRAD_DAYS_YEAR = 252

EQUITY_CACHE = dict()

OUTPUT_RESULT_PATH = constants.SURVIVOR_RESULTS_PATH

logger = logger_factory.create(__name__)


def calc_amt(amt_map: Dict[str, Tuple[float, float]]):
    vals = [act_roi for (_, act_roi) in amt_map.values()]
    return statistics.mean(vals)


@dataclasses.dataclass
class SurvivorRun:
    sample_size: int
    up_thresh: float
    exam_period_days: int
    samples_in_year: int = dataclasses.field(init=False)
    num_years: float = dataclasses.field(init=False)
    num_loops: int
    initial_amt: float = 10 ** 6
    total: float = dataclasses.field(init=False)
    results_df: pd.DataFrame = dataclasses.field(init=False)
    purchase_set_df: pd.DataFrame = dataclasses.field(init=False)
    output_path: Path = dataclasses.field(init=False)
    elapsed_run_time: float = dataclasses.field(init=False)

    def __post_init__(self):
        self.samples_in_year = int(NUM_TRAD_DAYS_YEAR / self.exam_period_days)
        self.total = self.initial_amt
        self.num_years = self.num_loops / self.samples_in_year
        self.output_path = SURVIVOR_RESULTS_DIR / f'survivor_run_{time.strftime("%Y_%m_%d_%H_%M_%S")}.pkl'
        self.results_df = pd.DataFrame(columns=['iteration', 'ann_roi'])
        self.purchase_set_df = pd.DataFrame(columns=['ticker',
                                                     'actual_purchase_date',
                                                     'actual_sell_date',
                                                     'guiding_roi',
                                                     'actual_roi'])

    def to_pickle(self):
        pickle_service.save(self, str(self.output_path))

    def append_results(self, iteration: int, ann_roi: float):
        self.results_df = self.results_df.append({'iteration': iteration, 'ann_roi': ann_roi}, ignore_index=True)

    def get_annualized_roi(self, num_years: float):
        return (((self.total / self.initial_amt) ** (1 / num_years)) - 1) * 100

    def append_purchase_set(self, actual_purchase_date: datetime,
                            actual_sell_date: datetime,
                            amt_map: Dict[str, Tuple[str, str]]):
        for ticker, (guiding_roi, actual_roi) in amt_map.items():
            self.purchase_set_df = self.purchase_set_df.append({'actual_purchase_date': actual_purchase_date,
                                                           'actual_sell_date': actual_sell_date,
                                                           'ticker': ticker, 'guiding_roi': guiding_roi,
                                                           'actual_roi': actual_roi}, ignore_index=True)

    @staticmethod
    def get_all_pickled_files():
        return file_services.list_files(SURVIVOR_RESULTS_DIR, ends_with='.pkl')


class ConditionalOperator(Enum):
    EQ = 'eq'
    GT = 'gt'
    GTE = 'gte'
    LT = 'lt'
    LTE = 'lte'


@dataclasses.dataclass
class EqCond:
    field_name: str
    cond_op: ConditionalOperator
    field_val: float


def get_shuffled_equities():
    avail_equities = BASE_EQUITIES.copy()
    random.shuffle(avail_equities)
    return set(avail_equities)


def combine_prev_month_equities(prev_month_equities):
    return set(item for sublist in prev_month_equities for item in sublist)


# NOTE: chris.flesche: 2023-07-09: In any given month we want to avoid buying/selling the same equity to avoid
# the wash sale rule.
def get_sample_equities(purchase_date: datetime,
                        sample_size: int,
                        prev_month_equities: List[Set] = None,
                        end_date_buffer: int = 10,
                        eq_conds: List[EqCond] = None, ) -> Dict[str, pd.DataFrame]:
    avail_equities = get_shuffled_equities()
    purch_dt_str = purchase_date.strftime(date_utils.STANDARD_DAY_FORMAT)
    if prev_month_equities is None:
        no_go_equities = set()
    else:
        no_go_equities = combine_prev_month_equities(prev_month_equities)
    rnd_eq_dfs = dict()
    while True:
        if len(avail_equities) == 0:
            logger.warning(f'Returning with {len(rnd_eq_dfs)} equities. Not enough to match.')
            return rnd_eq_dfs
        else:
            ticker = avail_equities.pop()

        if ticker in EQUITY_CACHE:
            eq_df = EQUITY_CACHE[ticker]
        else:
            eq_df = ts.get_ticker_eod_data(ticker=ticker)
            if eq_df is None or eq_df.shape[0] == 0:
                no_go_equities.add(ticker)
                continue
            eq_df.rename(columns={'date': 'date_original'}, inplace=True)
            eq_df['date'] = pd.to_datetime(eq_df['date_original'], format="%Y-%m-%d")
            eq_df['vol_price_prod'] = eq_df['volume'] * eq_df['close']
            eq_df['mean_price'] = (eq_df['low'] + eq_df['high']) / 2
            eq_df.sort_values('date', inplace=True)
            EQUITY_CACHE[ticker] = eq_df

        eq_df_test: pd.DataFrame = eq_df.iloc[:-end_date_buffer, :]

        failed_conditions = False
        if eq_conds is not None:
            for ec in eq_conds:
                eq_df_test = eq_df_test[eq_df_test['date_original'] == purch_dt_str]
                if ec.cond_op == ConditionalOperator.EQ:
                    if eq_df_test[eq_df_test[ec.field_name] == ec.field_val].shape[0] == 0:
                        failed_conditions = True
                        break
                elif ec.cond_op == ConditionalOperator.GT:
                    if eq_df_test[eq_df_test[ec.field_name] > ec.field_val].shape[0] == 0:
                        failed_conditions = True
                        break
                elif ec.cond_op == ConditionalOperator.LT:
                    if eq_df_test[eq_df_test[ec.field_name] < ec.field_val].shape[0] == 0:
                        failed_conditions = True
                        break
                elif ec.cond_op == ConditionalOperator.GTE:
                    if eq_df_test[eq_df_test[ec.field_name] >= ec.field_val].shape[0] == 0:
                        failed_conditions = True
                        break
                elif ec.cond_op == ConditionalOperator.LTE:
                    if eq_df_test[eq_df_test[ec.field_name] <= ec.field_val].shape[0] == 0:
                        failed_conditions = True
                        break
            if failed_conditions:
                no_go_equities.add(ticker)
                continue

        rnd_eq_dfs[ticker] = eq_df

        if len(rnd_eq_dfs) == sample_size:
            return rnd_eq_dfs


def process(sample_size: int,
            up_thresh: float,
            exam_period_days: int,
            num_loops: int,
            initial_amt: float = 10 ** 6,
            eq_conds: List[EqCond] = None,
            turbo_mode: bool = False):
    start_time = time.time()
    sr = SurvivorRun(sample_size=sample_size,
                     up_thresh=up_thresh,
                     exam_period_days=exam_period_days,
                     num_loops=num_loops,
                     initial_amt=initial_amt)
    print(f'Will use {sr.num_loops} purchase events.')
    past_equity_sets: List[Set] = []
    num_weeks_in_month = 5

    for loop_ndx in range(sr.num_loops):
        purchase_date = date_utils.get_random_past_date()

        rnd_eq_dfs = get_sample_equities(purchase_date=purchase_date,
                                         sample_size=sample_size,
                                         prev_month_equities=past_equity_sets,
                                         eq_conds=eq_conds,
                                         end_date_buffer=exam_period_days)
        if len(past_equity_sets) > num_weeks_in_month:
            past_equity_sets.pop(0)
        past_equity_sets.append(set(rnd_eq_dfs.keys()))

        date_exam = []
        for j in range(math.floor(sr.exam_period_days / HOLD_DAYS)):
            spike_date = purchase_date + timedelta(days=j * HOLD_DAYS)
            date_exam.append(spike_date)

        survivor_map = rnd_eq_dfs.copy()

        mean_mean_roi = []

        for dt in date_exam:
            if not date_utils.is_stock_market_closed(dt):
                continue
            dt_future = dt + timedelta(days=HOLD_DAYS + 10)
            del_ticks = []
            amt_map = dict()
            for ticker, df in survivor_map.items():
                if df.shape[0] == 0:
                    del_ticks.append(ticker)
                    continue
                df = df[(df['date'] >= dt) & (df['date'] <= dt_future)].copy()
                df.sort_values('date', inplace=True)

                # This price will be the earliest we could practically sell it. Using trading software
                # we might be able to sell earlier.
                df['actual_sell_date'] = df['date'].shift(-HOLD_DAYS)
                df['actual_sell_price'] = df['open'].shift(-HOLD_DAYS)
                df = df[df['date'] == dt]
                if df.shape[0] == 0:
                    del_ticks.append(ticker)
                    continue

                # This price will tell us in the evening if we should sell in the morning.
                guiding_sell_price = df['close'].values[0]

                actual_sell_price = df['actual_sell_price'].values[0]
                actual_sell_date = df['actual_sell_date'].values[0]
                purchase_price = df['open'].values[0]

                if math.isnan(guiding_sell_price) or math.isnan(actual_sell_price):
                    del_ticks.append(ticker)
                    continue

                # We place an order before the market opens.
                guide_roi = (guiding_sell_price - purchase_price) / purchase_price
                act_roi = (actual_sell_price - purchase_price) / purchase_price
                amt_map[ticker] = (guide_roi, act_roi)

                if not turbo_mode:
                    sr.append_purchase_set(actual_purchase_date=dt, actual_sell_date=actual_sell_date, amt_map=amt_map)

            if len(amt_map) == 0:
                continue

            up_count = sum([1 for _, (guide_roi, _) in amt_map.items() if guide_roi >= 0])
            if up_count / len(amt_map) >= sr.up_thresh:
                for t, (guide_roi, act_roi) in amt_map.items():
                    if guide_roi <= 0:
                        survivor_map.pop(t)
            mean_roi = calc_amt(amt_map)
            sr.total += mean_roi * sr.total
        curr_num_years = loop_ndx / sr.samples_in_year
        if curr_num_years > 0.000001:
            curr_ann_roi = sr.get_annualized_roi(num_years=curr_num_years)
            mean_mean_roi.append(curr_ann_roi)
            print(f'Loop {loop_ndx}: Ann ROI: {curr_ann_roi:,.2f}')
            if not turbo_mode:
                sr.append_results(iteration=loop_ndx, ann_roi=curr_ann_roi)
                sr.to_pickle()

    ann_roi = sr.get_annualized_roi(num_years=sr.num_years)
    print(f'ann_roi: {ann_roi:,.2f}%')
    sr.elapsed_run_time = time.time() - start_time
    print(f'Elapsed time: {sr.elapsed_run_time:.2f}')
    sr.to_pickle()
