from pathlib import Path
from typing import Dict

import pandas as pd

from ams.config import constants
from ams.config.constants import ensure_dir
from ams.notebooks.twitter.pipes import batchy_bae
from ams.services import file_services
from ams.utils import date_utils

df_tip_ranks = None
results_cache = dict()


def get_lpd_files(parent_path: Path):
    return file_services.list_files(parent_path=parent_path, ends_with=".parquet.in_transition", use_dir_recursion=True)


def get_date_and_ticks(source_path: Path) -> Dict[str, set]:
    files = get_lpd_files(parent_path=source_path)

    print(f"Num data files: {len(files)}")

    all_ticks = {}
    for f_ndx, f in enumerate(files):
        print(f"Reading file '{f}'.")
        df = pd.read_parquet(f)

        df_g = df.groupby(["date", "f22_ticker"])

        for ndx, (group_info, df_g_td) in enumerate(df_g):
            date_str, ticker = group_info
            if date_str in all_ticks.keys():
                all_ticks[date_str].add(ticker)
            else:
                all_ticks[date_str] = {ticker}

        if f_ndx > 1:
            break

    return all_ticks


def get_tip_ranks():
    global df_tip_ranks
    if df_tip_ranks is None:
        print("Reading Tip Ranks Scraped Data ...")
        df_tip_ranks = pd.read_parquet(constants.TIP_RANKS_STOCK_DATA_PATH)
    return df_tip_ranks


def transform_and_persist(tip_results: Dict[str, Dict[str, object]], output_path: Path):
    rows = []
    for dt_str, day_results in tip_results.items():
        sr = day_results.copy()
        for t, ticker_results in sr.items():
            ticker_row = ticker_results.copy()
            ticker_row["ticker"] = t
            ticker_row["date"] = dt_str
            rows.append(ticker_row)

    df = pd.DataFrame(rows)

    file_path = file_services.create_unique_filename(parent_dir=str(output_path), prefix="tip_rank", extension="parquet")
    df.to_parquet(file_path)

    return file_path


def process_tip_rank(date_of_purchase: str, tickers: set):
    dt_purchase = date_utils.parse_std_datestring(date_of_purchase)

    df_tipped = get_tip_ranks()
    df = df_tipped[df_tipped["rating_date"] < date_of_purchase]

    dop = {}
    results = {
        date_of_purchase: dop
    }

    for t in tickers:
        if t in dop.keys():
            continue

        df_tick = df[df["ticker"] == t].copy()

        if df_tick is None or df_tick.shape[0] == 0:
            continue

        # Mean rating
        rating = df_tick["rating"].mean()

        # Mean rank
        rank = df_tick["rank"].mean()

        # Mean age of tip
        def get_age_in_days(value):
            dt_rating = date_utils.parse_std_datestring(value)
            return (dt_purchase - dt_rating).days

        df_tick["rating_age_days"] = df_tick["rating_date"].apply(get_age_in_days)
        rating_age_days = df_tick["rating_age_days"].mean()

        # Mean rating initiating. init gets special "new"=True column/value, upgrade = 1, downgrade=-1, reiter=0
        # Mean off_target_price
        target_price = df_tick["target_price"].mean()

        # df_ticker = ticker_service.get_ticker_eod_data(ticker=t)
        # purchase_close_price = df_tick[df_tick["trade_date"] == date_of_purchase].iloc[0]["close"]
        # rank_roi = (target_price - purchase_close_price)/purchase_close_price

        result = {
            "rating": rating,
            "rank": rank,
            "rating_age_days": rating_age_days,
            "target_price": target_price,
        }

        dop[t] = result

    return results


def process(source_path: Path, output_dir_path: Path):
    global df_tip_ranks

    # NOTE: 2020-12-13: chris.flesche: replace with function iterates through every calendar day
    # gets all the ticks for that day and then creates a dictionary of {"date_str": {"ticker": results}}
    tick_dates = get_date_and_ticks(source_path=source_path)

    results = dict()
    count = 0
    print(tick_dates.keys())
    for date_str, ticker_set in tick_dates.items():
        print(f"Processing date {date_str}...")
        tr_results = process_tip_rank(date_of_purchase=date_str, tickers=ticker_set)
        if results is None:
            results = tr_results
        else:
            results.update(tr_results)
        # count += 1
        # if count > 2:
        #     break

    # print(f"Results: {results}")
    output_file_path = transform_and_persist(tip_results=results, output_path=output_dir_path)

    print(f"Wrote output file '{output_file_path}'.")


def start():
    source_dir_path = Path(constants.TWITTER_OUTPUT_RAW_PATH, "learning_prep_drop", "lpd_2020-12-06_14-00-19-584.44")
    output_dir_path = Path(constants.TWITTER_OUTPUT_RAW_PATH, "tip_ranked", "main")
    ensure_dir(output_dir_path)

    batchy_bae.start(source_path=source_dir_path, output_dir_path=output_dir_path, process_callback=process, should_archive=False)

    return output_dir_path


if __name__ == '__main__':
    start()