import collections
import random
from pathlib import Path
from statistics import mean
from typing import List, Dict, Tuple, Union, Set

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

from ams.config import constants, logger_factory
from ams.services import twitter_service, ticker_service, pickle_service, file_services
from ams.services.csv_service import write_dicts_as_csv
from ams.services.equities.EquityFundaDimension import EquityFundaDimension
from ams.services.ticker_service import get_ticker_eod_data
from ams.twitter import pred_perf_testing
from ams.twitter.PredictionParams import PredictionParams, PredictionMode
from ams.twitter.pred_persistence import save_predictions
from ams.utils import date_utils, ticker_utils
from ams.utils.SplitData import SplitData

num_iterations = 1

EquityHolding = collections.namedtuple('EquityHolding', 'ticker purchase_price num_shares purchase_dt, expiry_dt')
TwitterModelPackage = collections.namedtuple("TwitterModelPackage", "model scaler")

BATCH_SIZE = 1
LEARNING_RATE = .0001
initial_cash = 10000
num_trades_at_once = 1

logger = logger_factory.create(__name__)

df_rec_quart_drop = None


def group_and_mean_preds_buy_sell(df: pd.DataFrame, model: object, train_cols: List[str], standard_scaler: StandardScaler, is_model_torch=False):
    df_g_holdout = df.groupby(['f22_ticker', 'purchase_date'])

    group_count = 0
    acc_acc = []
    group_preds = {}
    for group_name, df_group in df_g_holdout:
        X_holdout_raw = np.array(df_group[train_cols])
        X_holdout = standard_scaler.transform(X_holdout_raw)

        y_holdout = np.array(df_group['buy_sell'])

        if is_model_torch:
            desired_dtype = 'float64'
            X_holdout_con = X_holdout.astype(desired_dtype)
            X_torch = torch.FloatTensor(X_holdout_con)

            y_holdout = np.where(y_holdout == -1, 0, y_holdout)
            pre_y_ho = sum(y_holdout) / len(y_holdout)
            pre_y_ho = 0 if pre_y_ho < .5 else 1

            y_pred_tag = model_torch_predict(X_torch, model)
            pred_mean = sum(y_pred_tag.numpy()) / len(y_holdout)

            pred_mean = -1 if pred_mean < .5 else 1
        else:
            prediction = model.predict(X_holdout)

            pred_mean = sum(prediction) / len(prediction)
            pred_mean = -1 if pred_mean < 0 else 1

            pre_y_ho = sum(y_holdout) / len(y_holdout)
            pre_y_ho = -1 if pre_y_ho < 0 else 1

        # NOTE: This means that we are predicting only when buys are successful. And we only buy.
        pred_buy_success = 0
        if pred_mean == 1:
            if pre_y_ho == pred_mean:
                pred_buy_success = 1

            # Of the buys what's is the success rate?
            acc_acc.append(pred_buy_success)

        ticker = group_name[0]
        date_str = group_name[1]

        if ticker in group_preds.keys():
            info = group_preds[ticker]
        else:
            info = {}
            group_preds[ticker] = info
        info[date_str] = pred_mean

    if len(acc_acc) > 0:
        logger.info(f"Mean: {mean(acc_acc)}")
    else:
        logger.info("No mean on grouped mean - no rows?")

    g_tickers = []
    for group_name, df_group in df_g_holdout:
        g_tickers.append(group_name[0])

    return g_tickers, group_preds


def model_torch_predict(X_torch, model):
    model.eval()
    with torch.no_grad():
        model.cpu()
        raw_out = model(X_torch)
    y_pred_tag = torch.round(torch.sigmoid(raw_out.data))
    return y_pred_tag


def transform_to_numpy(df: pd.DataFrame, narrow_cols: List[str], require_balance: bool = True) -> Tuple[any, any, any]:
    standard_scaler = StandardScaler()

    train_cols = twitter_service.get_feature_columns(narrow_cols)

    X_train_raw, y_train = twitter_service.split_df_for_learning(df=df, train_cols=train_cols, require_balance=require_balance)

    if X_train_raw is None or X_train_raw.shape[0] == 0:
        return None, None, None

    standard_scaler = standard_scaler.fit(X_train_raw)
    X_train = standard_scaler.transform(X_train_raw)

    return X_train, y_train, standard_scaler


def split_off_data(df: pd.DataFrame,
                   narrow_cols: List[str],
                   use_recent_for_holdout=True,
                   split_off_test: bool = True) -> SplitData:
    standard_scaler = StandardScaler()
    df_samp, df_val_raw = twitter_service.ho_split_by_days(df,
                                                           small_data_days_to_pull=None,
                                                           small_data_frac=.025,
                                                           use_only_recent_for_holdout=use_recent_for_holdout)

    X_train = None
    y_train = None
    X_test = None
    y_test = None
    min_rows_enough = 200

    has_enough_data = df_samp is not None and df_val_raw is not None and df_val_raw.shape[0] > min_rows_enough
    if has_enough_data:
        train_cols = twitter_service.get_feature_columns(narrow_cols)

        if split_off_test:
            df_train_raw, df_test_raw = twitter_service.ho_split_by_days(df_samp, small_data_days_to_pull=None, small_data_frac=.2,
                                                                         use_only_recent_for_holdout=use_recent_for_holdout)

            has_enough_data = df_train_raw is not None and df_test_raw is not None and df_test_raw.shape[0] > min_rows_enough
            if has_enough_data:
                logger.info(f"Original: {df.shape[0]}; train_set: {df_train_raw.shape[0]}; test_set: {df_test_raw.shape[0]}")

                X_train_raw, y_train = twitter_service.split_df_for_learning(df=df_train_raw, train_cols=train_cols)
                standard_scaler = standard_scaler.fit(X_train_raw)
                X_train = standard_scaler.transform(X_train_raw)

                X_test_raw, y_test = twitter_service.split_df_for_learning(df=df_test_raw, train_cols=train_cols)
                X_test = standard_scaler.transform(X_test_raw)
        else:
            df_test_raw = None

            df_train_raw = df_samp
            has_enough_data = df_train_raw is not None and df_train_raw.shape[0] > min_rows_enough

            if has_enough_data:
                logger.info(f"Original: {df.shape[0]}; train_set: {df_train_raw.shape[0]}; val_set: {df_val_raw.shape[0]}")

                X_train_raw, y_train = twitter_service.split_df_for_learning(df=df_train_raw, train_cols=train_cols)
                standard_scaler = standard_scaler.fit(X_train_raw)
                X_train = standard_scaler.transform(X_train_raw)

        return SplitData(X_train=X_train,
                         y_train=y_train,
                         X_test=X_test,
                         y_test=y_test,
                         df_test_raw=df_test_raw,
                         df_val_raw=df_val_raw,
                         train_cols=train_cols,
                         has_enough_data=has_enough_data,
                         standard_scaler=standard_scaler)
    if not has_enough_data:
        logger.info("Not enough data.")


TwitterTrade = collections.namedtuple('TwitterTrade', 'ticker purchase_price purchase_dt sell_dt sell_price')

CalcProfitResults = collections.namedtuple('CalcProfitResults', 'total num_trades acc calc_dict roi_list pot_roi_list total_invested trade_history')


def calc_profit(target_roi: float, df_helper: pd.DataFrame, group_preds: object, zero_in: bool = False) -> CalcProfitResults:
    roi_list = []
    trade_cost = 0
    pred_correct = 0
    pot_roi_list = []
    num_trades = 0
    cash = initial_cash
    calc_dict = {}
    total = 0
    total_invested = 0
    fc_num_trades = 0
    trade_history: List[TwitterTrade] = []
    for ticker, info in group_preds.items():
        num_shares = 0

        for date_str, pred_mean in info.items():
            should_buy = (pred_mean == 1)

            close, future_close, future_high, future_open, future_date = ticker_service.get_stock_info(df=df_helper,
                                                                                                       ticker=ticker,
                                                                                                       date_str=date_str)
            pot_roi = (future_high - close) / close
            pot_roi_list.append(pot_roi)

            shares_price = num_trades_at_once * close

            if should_buy:
                ticker_service.calculate_roi(target_roi=target_roi, close_price=close, future_high=future_high,
                                             future_close=future_close, calc_dict=calc_dict, zero_in=zero_in)

                trade_history.append(TwitterTrade(ticker=ticker,
                                                  purchase_price=close,
                                                  purchase_dt=date_utils.parse_std_datestring(date_str),
                                                  sell_dt=date_utils.parse_std_datestring(future_date),
                                                  sell_price=future_close
                                                  ))

                roi = (future_close - close) / close
                roi_list.append(roi)

                if future_close > close:
                    pred_correct += 1
                fc_num_trades += 1

                if cash > shares_price:
                    cash -= shares_price + trade_cost
                    num_shares += num_trades_at_once
                    num_trades += 1

                    total_invested += shares_price + trade_cost
                    cashed_out = ((split_share_multiplier * num_shares) * future_close) - trade_cost
                    total = cashed_out + cash
                    cash = total
                    num_shares = 0


                else:
                    logger.info("Not enough cash for purchase.")

    acc = -1
    if fc_num_trades > 0:
        acc = pred_correct / fc_num_trades

    calcProfitResults = CalcProfitResults(total=total, num_trades=num_trades, acc=acc, calc_dict=calc_dict, roi_list=roi_list,
                                          pot_roi_list=pot_roi_list, total_invested=total_invested, trade_history=trade_history)
    return calcProfitResults


def write_twitter_trade_history(rows: List[Dict], overwrite: bool = False):
    if len(rows) == 0:
        raise Exception("Nothing to write. Rows are empty.")

    write_dicts_as_csv(output_file_path=constants.TWITTER_TRADE_HISTORY_FILE_PATH, overwrite=overwrite, rows=rows)


def persist_trade_history(overwrite_existing: bool, twitter_trades: List[TwitterTrade]):
    trades = [tt._asdict() for tt in twitter_trades]
    for t in trades:
        t["purchase_dt"] = date_utils.get_standard_ymd_format(t["purchase_dt"])
        t["sell_dt"] = date_utils.get_standard_ymd_format(t["sell_dt"])
    write_twitter_trade_history(rows=trades, overwrite=overwrite_existing)


def get_roi_matrix(df: pd.DataFrame, group_preds: object, target_roi_frac: float, zero_in: bool = False):
    cpr = calc_profit(target_roi=target_roi_frac, df_helper=df, group_preds=group_preds, zero_in=zero_in)

    sac_roi = None
    if len(cpr.roi_list) > 0:
        sac_roi = mean(cpr.roi_list)

    profit = cpr.total - initial_cash

    if cpr.num_trades > 0:
        profit_per_trade = round(profit / cpr.num_trades, 6)
        investment_per_trade = cpr.total_invested / cpr.num_trades
        roi_per_trade = round(profit_per_trade / investment_per_trade, 4)
        logger.info(f"Avg investment per trade: {investment_per_trade}")
        logger.info(f"Roi per trade: {roi_per_trade}")

    pot = round(sum(cpr.pot_roi_list) / len(cpr.pot_roi_list), 6)
    if cpr.acc > 0:
        acc = round(cpr.acc, 5)
    else:
        acc = "<NA>"

    logger.info(f"Num trades: {cpr.num_trades} | acc: {acc} | s@close roi: {sac_roi} | s@high roi: {pot}")

    avg_list = []
    for roi_calc, roi_calc_list in cpr.calc_dict.items():
        if len(roi_calc_list) > 0:
            avg_roi = mean(roi_calc_list)
            num_list = len(roi_calc_list)
            logger.info(f"Sell high/close roi@{roi_calc}: {round(avg_roi, 6)}; weight: {num_list * avg_roi}")
            avg_list.append(avg_roi)

    return avg_list, cpr.roi_list, cpr.trade_history


# def calc_nn_roi(
#     df_val_raw: pd.DataFrame,
#     model: object,
#     train_cols: List[str],
#     target_roi_frac: float,
#     zero_in: bool,
#     standard_scaler: StandardScaler,
#     is_model_torch: bool = True
# ):
#     if df_val_raw.shape[0] == 0:
#         raise Exception("No holdout data.")
#
#     g_tickers, group_preds = group_and_mean_preds_buy_sell(df_val_raw, model, train_cols, is_model_torch=is_model_torch, standard_scaler=standard_scaler)
#
#     _, sac_roi_list, trade_history = get_roi_matrix(df=df_val_raw, group_preds=group_preds, target_roi_frac=target_roi_frac, zero_in=zero_in)
#
#     mean_sac = None
#     if len(sac_roi_list) > 0:
#         persist_trade_history(twitter_trades=trade_history, overwrite_existing=False)
#         mean_sac = mean(sac_roi_list)
#         logger.info(f"Mean sac_roi: {mean_sac}")
#
#     return mean_sac


def save_twitter_stock_join(df: pd.DataFrame):
    sorted(list(df.columns))

    file_path_str = str(Path(constants.OVERFLOW_DATA_PATH, "twitter", "inference_model_drop", "twitter_id_with_label.parquet"))

    df.to_parquet(file_path_str)


def rank_roied(row: pd.Series):
    close = row["close"]
    target_price = row["target_price"]

    if target_price is None:
        target_price = close + .01
    else:
        target_price = row["target_price"]

    rank_roi = (target_price - close) / close

    return rank_roi


def add_tip_ranks(df: pd.DataFrame, tr_file_path: Path):
    df_tip_ranks = pd.read_parquet(str(tr_file_path))
    df_tip_ranks = df_tip_ranks.rename(columns={"ticker": "f22_ticker"})

    df_ranked = pd.merge(df, df_tip_ranks, on=["date", "f22_ticker"], how="left")

    rows_are_null = df_ranked['target_price'].isnull()
    df_ranked.loc[rows_are_null, "target_price"] = df_ranked["prev_close"] + .01

    df_ranked.loc[:, "rank_roi"] = df_ranked.apply(rank_roied, axis=1)

    rank_mean = df_tip_ranks["rank"].mean()

    df_ranked.loc[:, "rating"] = df_ranked["rating"].fillna(0)
    df_ranked.loc[:, "rank"] = df_ranked["rank"].fillna(rank_mean)
    df_ranked.loc[:, "rating_age_days"] = df_ranked["rating_age_days"].fillna(1000)

    return df_ranked


def add_tip_ranks_2(df: pd.DataFrame, tr_file_path: Path):
    df_tip_ranks = pd.read_parquet(str(tr_file_path))
    df_tip_ranks = df_tip_ranks.rename(columns={"ticker": "f22_ticker"})

    df_ranked = pd.merge(df, df_tip_ranks, on=["date", "f22_ticker"], how="left")

    rows_are_null = df_ranked['target_price'].isnull()
    df_ranked.loc[rows_are_null, "target_price"] = df_ranked["close"] + .01

    df_ranked.loc[:, "rank_roi"] = df_ranked.apply(rank_roied, axis=1)

    rank_mean = df_tip_ranks["rank"].mean()

    df_ranked.loc[:, "rating"] = df_ranked["rating"].fillna(0)
    df_ranked.loc[:, "rank"] = df_ranked["rank"].fillna(rank_mean)
    df_ranked.loc[:, "rating_age_days"] = df_ranked["rating_age_days"].fillna(1000)

    return df_ranked


def show_distribution(df: pd.DataFrame, group_column_name: str = "date"):
    df.sort_values(by=[group_column_name], inplace=True)

    day_groups = df.groupby(df[group_column_name])[group_column_name].count()

    day_groups.plot(kind='bar', figsize=(10, 5), legend=None)


def truncate_avail_columns(df: pd.DataFrame):
    cols = list(df.columns)

    # cols = [c for c in cols if not c.startswith("location_")]
    # cols = [c for c in cols if not c.startswith("currency_")]
    # cols = [c for c in cols if not c.startswith("industry_")]
    # cols = [c for c in cols if not c.startswith("famaindustry_")]
    # cols = [c for c in cols if not c.startswith("category_")]
    # cols = [c for c in cols if not c.startswith("sector_")]
    # cols = [c for c in cols if not c.startswith("scalerevenue_")]
    # cols = [c for c in cols if not c.startswith("table_")]
    # cols = [c for c in cols if not c.startswith("sicsector_")]
    # cols = [c for c in cols if not c.startswith("scalemarketcap_")]

    # cols = list(set(cols) - {'sharesbas', 'sps', 'ps', 'receivables', 'debtnc', 'invcap', 'sbcomp', 'workingcapital',
    #                         'taxliabilities', 'ebt', 'retearn', 'accoci', 'invcapavg', 'liabilitiesnc', 'pb', 'taxassets',
    #                          'revenueusd', 'price', 'netincdis', 'sharefactor', 'netmargin', 'ncfcommon', 'investmentsc', 'opinc',
    #                          'inventory', 'eps', 'de', 'sgna', 'siccode', 'fxusd', 'revenue', 'opex', 'cashnequsd', 'tbvps', 'shareswa',
    #                          'ros', 'evebitda', 'ncfdebt', 'consolinc', 'ncfinv', 'deposits', 'marketcap', 'ev', 'roe', 'payoutratio',
    #                          'investmentsnc', 'equity', 'roa', 'divyield', 'investmentsnc', 'equity', 'roa', 'divyield', 'ps1',
    #                          'shareswadil', 'liabilitiesc', 'gp', 'tangibles', 'epsusd', 'assetsnc', 'ppnenet', 'epsdil', 'ncfdiv',
    #                           'ncfi', 'payables', 'fcfps', 'investments', 'cashneq', 'roic', 'currentratio', 'ebit', 'ebitda',
    #                          'volume', 'ncfo', 'netinc', 'netinccmn', 'debt', 'pe', 'debtc', 'rnd', 'evebit', 'ebitusd', 'netincnci',
    #                          'assetsc', 'assetsavg', 'assetturnover', 'taxexp', 'ebitdausd', 'liabilities', 'capex', 'prefdivis',
    #                          'netinccmnusd', 'depamor', 'famasector', 'dps', 'assets', 'fcf', 'ebitdamargin', 'equityusd', 'ncfx',
    #                          'ncfbus', 'equityavg', 'dividends', 'cor', 'grossmargin', 'ncff', 'intangibles', 'debtusd', 'bvps', 'pe1',
    #                          'intexp', 'ncf'
    #                         })
    # logger.info(f"Total cols: {len(cols)}")

    col_drop = set(df.columns) - set(cols)

    return df.drop(columns=col_drop).reset_index(drop=True)


def coerce_convert_to_numeric(df: pd.DataFrame, col: str):
    df[col] = pd.to_numeric(df[col], errors='coerce').copy()
    df[col] = df[col].fillna(0).copy()
    return df


def convert_twitter_to_numeric(df: pd.DataFrame):
    df = coerce_convert_to_numeric(df=df, col="user_followers_count")
    df = coerce_convert_to_numeric(df=df, col="f22_sentiment_compound")
    return coerce_convert_to_numeric(df=df, col="f22_compound_score")


def merge_fundies_with_stock(df_stock_data: pd.DataFrame):
    from ams.services.equities import equity_fundy_service as efs
    df_equity_fundies = efs.get_equity_fundies()

    df_eq_fun_quarters = df_equity_fundies[df_equity_fundies["dimension"] == EquityFundaDimension.AsReportedQuarterly.value].copy()

    return pd.merge(df_eq_fun_quarters, df_stock_data, on=["ticker"], how='outer', suffixes=[None, "_eq_fun"])


def num_days_from(from_date: str, to_date: str):
    from_dt = date_utils.parse_std_datestring(from_date)
    to_dt = date_utils.parse_std_datestring(to_date)

    return (to_dt - from_dt).days


def add_days_since_quarter_results(df: pd.DataFrame, should_drop_missing_future_date: bool = True):
    df = df.dropna(axis="rows", subset=["datekey", "future_date"]).copy()

    # df["days_since"] = df.apply(lambda x: num_days_from(x["datekey"], x["future_date"]), axis=1)
    df.loc[:, "days_since"] = df.apply(lambda x: num_days_from(x["datekey"], x["future_date"]), axis=1)

    return df


def add_days_since_earliest_date(df: pd.DataFrame, earliest_date_str: str):
    df = df.dropna(axis="rows", subset=["datekey", "future_date"]).copy()

    df.loc[:, "days_since_earliest_date"] = df.apply(lambda x: num_days_from(x["datekey"], earliest_date_str), axis=1)

    return df


def add_calendar_days(df: pd.DataFrame):
    def day_of_week(date_str):
        return pd.Timestamp(date_str).dayofweek

    df.loc[:, "fd_day_of_week"] = df.apply(lambda x: day_of_week(x["future_date"]), axis=1)

    def day_of_year(date_str):
        return pd.Timestamp(date_str).dayofyear

    df.loc[:, "fd_day_of_year"] = df.apply(lambda x: day_of_year(x["future_date"]), axis=1)

    def day_of_month(date_str):
        return int(date_str.split("-")[2])

    df.loc[:, "fd_day_of_month"] = df.apply(lambda x: day_of_month(x["future_date"]), axis=1)

    return df


def add_nasdaq_roi(df: pd.DataFrame):
    # Add num hold days to function
    # Add new column: shift forward num_hold_days
    # add moving average column where target_column == new_column, window == num_hold_days
    #    with pd.option_context('mode.chained_assignment', None):
    #    df_copy[f"mov_avg"] = df_copy.loc[:, target_column].rolling(window=w).mean().astype("float64")
    # add new column = mov_avg * num_hold_days

    df_roi_nasdaq = pd.read_parquet(str(constants.DAILY_ROI_NASDAQ_PATH))
    df_roi_nasdaq = df_roi_nasdaq.rename(columns={"roi": "nasdaq_day_roi"})

    df = pd.merge(df_roi_nasdaq, df, on=["date"], how="right")

    return df.drop_duplicates(subset=["f22_ticker", "date"])


def add_nasdaq_roi_new(df: pd.DataFrame, num_hold_days: int = 1):
    df_roi_nasdaq = pd.read_parquet(str(constants.DAILY_ROI_NASDAQ_PATH))
    df_roi_nasdaq = df_roi_nasdaq.rename(columns={"roi": "nasdaq_day_roi"})

    df = pd.merge(df_roi_nasdaq, df, on=["date"], how="right")

    # logger.info(f"Filthy: {df.shape[0]}")

    if num_hold_days > 1:
        df.loc[:, "roi_sell_date"] = df["nasdaq_day_roi"]
        df["roi_sell_date"] = df["roi_sell_date"].shift(-num_hold_days)
        with pd.option_context('mode.chained_assignment', None):
            df[f"mov_avg"] = df.loc[:, "roi_sell_date"].rolling(window=num_hold_days).mean().astype("float64")
        df.loc[:, "nasdaq_day_roi"] = np.subtract(np.power(np.add(1, df["mov_avg"]), num_hold_days), 1)
        df = df.drop(columns=["roi_sell_date", "mov_avg"])

    # logger.info(f"after rolling mov avg: {df.shape[0]}")

    return df.drop_duplicates(subset=["f22_ticker", "date"])


def add_sma_stuff(df: pd.DataFrame, tweet_date_str: str):
    df = ticker_utils.add_sma_history(df=df, target_column="close", windows=[15, 20, 50, 100, 200], tweet_date_str=tweet_date_str)

    df = ticker_utils.add_days_since_under_sma_many_tickers(df=df, col_sma="close_SMA_200", close_col="close")
    df = ticker_utils.add_days_since_under_sma_many_tickers(df=df, col_sma="close_SMA_15", close_col="close")
    df = ticker_utils.add_days_since_under_sma_many_tickers(df=df, col_sma="close_SMA_20", close_col="close")
    df = ticker_utils.add_days_since_under_sma_many_tickers(df=df, col_sma="close_SMA_50", close_col="close")

    return ticker_utils.add_days_since_under_sma_many_tickers(df=df, col_sma="close_SMA_100", close_col="close")


def get_data_for_predictions(df: pd.DataFrame,
                             narrow_cols: List[str],
                             standard_scaler: StandardScaler):
    feature_cols = twitter_service.get_feature_columns(narrow_cols)

    df_features = df[feature_cols]

    X_array_raw = np.array(df_features)
    return standard_scaler.transform(X_array_raw)


def get_next_market_date(date_str: str, num_days: int):
    dt = date_utils.parse_std_datestring(date_str)
    return date_utils.get_standard_ymd_format(date_utils.find_next_market_open_day(dt, num_days))


def add_future_date_for_nan(df: pd.DataFrame, num_days_in_future: int):
    df.loc[df["future_date"].isnull(), "future_date"] = df.loc[df["future_date"].isnull(), "date"].apply(lambda fd: get_next_market_date(fd, num_days_in_future))

    return df


def load_twitter_raw(proc_path: Path):
    file_paths = file_services.list_files(parent_path=proc_path, ends_with=".parquet", use_dir_recursion=True)
    all_dfs = []
    for f in file_paths:
        df = pd.read_parquet(f)
        all_dfs.append(df)

    return pd.concat(all_dfs, axis=0)


def load_model_for_prediction():
    model_xgb: TwitterModelPackage = pickle_service.load(constants.TWITTER_XGB_MODEL_PATH)
    return model_xgb


def get_twitter_stock_data(df_tweets: pd.DataFrame, num_hold_days: int):
    df_stock_data = twitter_service.get_stock_data_for_twitter_companies(df_tweets=df_tweets,
                                                                         num_days_in_future=num_hold_days)

    df_stock_data = add_future_date_for_nan(df=df_stock_data, num_days_in_future=num_hold_days)

    return df_stock_data


def get_twitter_stock_data_2(df_tweets: pd.DataFrame, num_hold_days: int, num_days_until_purchase: int):
    df_stock_data = twitter_service.get_stock_data_for_twitter_companies_2(df_tweets=df_tweets,
                                                                           num_hold_days=num_hold_days,
                                                                           num_days_until_purchase=num_days_until_purchase)
    num_days_in_future = num_hold_days + num_days_until_purchase
    df_stock_data = add_future_date_for_nan(df=df_stock_data, num_days_in_future=num_days_in_future)

    return df_stock_data


def split_train_predict(df: pd.DataFrame, tweet_date_str: str, num_days_until_purchase: int, num_hold_days: int) -> Tuple[
    Union[None, pd.DataFrame], Union[None, pd.DataFrame]]:
    df_th_train = df[df["date"] < tweet_date_str].copy()

    if df_th_train is None or df_th_train.shape[0] == 0:
        return None, None

    df_train = twitter_service.add_buy_sell(df=df_th_train)

    # FIXME: 2021-02-19: chris.flesche: Find out why I need this.
    # num_days_in_future = num_hold_days + num_days_until_purchase
    # future_date_limit = get_next_market_date(tweet_date_str, num_days_in_future)
    # df_predict = df[(df["date"] == tweet_date_str) & (df["future_date"] <= future_date_limit)].copy()

    df_predict = df[df["date"] == tweet_date_str].copy()

    df_train = twitter_service.omit_columns(df=df_train)
    df_predict = twitter_service.omit_columns(df=df_predict)

    logger.info(f"Num rows of prepared data: {df_train.shape[0]}")
    logger.info(f"Oldest date of prepared data (future_date): {df_train['future_date'].max()}")
    logger.info(f"Num prepped: {df_predict.shape[0]}")

    return df_train, df_predict


def get_stock_matchable(df):
    tickers = list(set(df["f22_ticker"].to_list()))

    good_tickers = []
    for t in tickers:
        if ticker_service.does_ticker_data_exist(ticker=t):
            good_tickers.append(t)

    return df[df["f22_ticker"].isin(good_tickers)].copy()


def easy_convert_columns(df):
    df_booled = twitter_service.convert_to_bool(df=df)
    return convert_twitter_to_numeric(df=df_booled)


def get_stocks_based_on_tweets(df_tweets: pd.DataFrame, tweet_date_str: str, num_hold_days: int):
    df_tweets = df_tweets[df_tweets["date"] <= tweet_date_str].copy()

    if df_tweets.shape[0] == 0:
        logger.info("No data before prediction date.")
        return None

    df_stock_data = get_twitter_stock_data(df_tweets=df_tweets,
                                           num_hold_days=num_hold_days)
    return df_stock_data


def get_stocks_based_on_tweets_2(df_tweets: pd.DataFrame, tweet_date_str: str, num_hold_days: int, num_days_until_purchase: int):
    df_tweets = df_tweets[df_tweets["date"] <= tweet_date_str].copy()

    if df_tweets.shape[0] == 0:
        logger.info("No data before prediction date.")
        return None

    df_stock_data = get_twitter_stock_data_2(df_tweets=df_tweets,
                                             num_hold_days=num_hold_days,
                                             num_days_until_purchase=num_days_until_purchase)
    return df_stock_data


def get_quarterly_data():
    global df_rec_quart_drop
    if df_rec_quart_drop is None:
        df_rec_quart_drop = twitter_service.get_all_quarterly_data_for_twitter()
    return df_rec_quart_drop.copy()


def combine_with_quarterly_stock_data(df):
    # NOTE: 2021-02-13: chris.flesche: This is a tricky way to merge tweet + daily stock data + stock fundamentals
    # without stock fundamentals from the future leaking data into the daily tweet + daily stock data.

    df_rec_quart_drop = get_quarterly_data()
    columns_fundy = list(df_rec_quart_drop.columns)
    df_result = merge_fundies_with_stock(df_stock_data=df).copy()
    df_drop_init = df_result.dropna(subset=["date"]).copy().drop(columns="lastupdated_eq_fun").copy()
    df_drop_future = df_drop_init[df_drop_init["date"] > df_drop_init["calendardate"]].copy()
    df_drop_future = df_drop_future.sort_values(by=["ticker", "date", "calendardate"], ascending=False).copy()
    df_stock_and_quarter = df_drop_future.drop_duplicates(subset=["ticker", "date"], keep="first").copy()
    logger.info("Finished merging in quarterly stock data.")

    return df_stock_and_quarter, columns_fundy


def merge_tweets_with_stock_data(df_twitter, df_stock_and_quarter):
    df_nas_tickers_info = ticker_service.get_nasdaq_tickers().copy()

    col_ticker = "ticker_drop"

    df_stock_quart_info = pd.merge(df_stock_and_quarter, df_nas_tickers_info, how='inner', left_on=["ticker"], right_on=[col_ticker])
    df_stock_quart_info.drop(columns=[col_ticker], inplace=True)

    df_stock_renamed = df_stock_quart_info.rename(columns={"ticker": "f22_ticker"})

    if 'None' in df_stock_renamed.columns:
        df_stock_renamed.drop(columns=['None'], inplace=True)

    df_merged = pd.merge(df_twitter, df_stock_renamed, how='inner', on=["f22_ticker", "date"])

    df_merged.sort_values(by=["f22_ticker", "date"], inplace=True)

    if df_merged.shape[0] == 0:
        logger.info("Not enough data after merge.")

    df_ranked = add_tip_ranks(df=df_merged, tr_file_path=constants.TIP_RANKED_DATA_PATH)

    return df_ranked


def merge_tweets_with_stock_data_2(df_twitter, df_stock_and_quarter):
    df_nas_tickers_info = ticker_service.get_nasdaq_tickers().copy()

    col_ticker = "ticker_drop"

    df_stock_quart_info = pd.merge(df_stock_and_quarter, df_nas_tickers_info, how='inner', left_on=["ticker"], right_on=[col_ticker])
    df_stock_quart_info.drop(columns=[col_ticker], inplace=True)

    df_stock_renamed = df_stock_quart_info.rename(columns={"ticker": "f22_ticker"})

    if 'None' in df_stock_renamed.columns:
        df_stock_renamed.drop(columns=['None'], inplace=True)

    df_merged = pd.merge(df_twitter, df_stock_renamed, how='inner', on=["f22_ticker", "date"])

    df_merged.sort_values(by=["f22_ticker", "date"], inplace=True)

    if df_merged.shape[0] == 0:
        logger.info("Not enough data after merge.")

    df_ranked = add_tip_ranks_2(df=df_merged, tr_file_path=constants.TIP_RANKED_DATA_PATH)

    return df_ranked


def add_calendar_info(df: pd.DataFrame, columns_fundy: List[str], predict_date_str: str, num_hold_days: int, oldest_tweet_date):
    cols_fundy_numeric = list(set(columns_fundy) - {"ticker", 'calendardate', 'datekey', 'reportperiod'})

    df = add_days_since_quarter_results(df=df)
    df = add_calendar_days(df=df)
    df = add_nasdaq_roi_new(df=df, num_hold_days=num_hold_days)
    df = add_days_since_earliest_date(df=df, earliest_date_str=oldest_tweet_date)

    # FIXME: 2021-01-15: chris.flesche: "close" should be approximated for when predicting (?)
    df.loc[:, "original_close_price"] = df["close"]

    df = twitter_service.fill_null_numeric(df=df, cols_fundy_numeric=cols_fundy_numeric)

    df.loc[:, "purchase_date"] = df["date"]

    df = ticker_service.add_days_until_sale(df=df)

    df = add_sma_stuff(df=df, tweet_date_str=predict_date_str)

    return df


def add_calendar_info_2(df: pd.DataFrame, columns_fundy: List[str], tweet_date_str: str, num_hold_days: int, oldest_tweet_date):
    cols_fundy_numeric = list(set(columns_fundy) - {"ticker", 'calendardate', 'datekey', 'reportperiod'})

    df = add_days_since_quarter_results(df=df)
    df = add_calendar_days(df=df)
    df = add_nasdaq_roi_new(df=df, num_hold_days=num_hold_days)
    df = add_days_since_earliest_date(df=df, earliest_date_str=oldest_tweet_date)

    df = twitter_service.fill_null_numeric(df=df, cols_fundy_numeric=cols_fundy_numeric)

    df = ticker_service.add_days_until_sale(df=df)

    df = add_sma_stuff(df=df, tweet_date_str=tweet_date_str)

    return df


def one_hot(df):
    df_ticker_hotted, _ = ticker_service.make_f22_ticker_one_hotted(df_ranked=df)

    narrow_cols = list(df_ticker_hotted.columns)

    return df_ticker_hotted, narrow_cols


def prep_predict(df, tweet_date_str: str):
    df_train = df[df["date"] < tweet_date_str].copy()
    df_pred = df[df["date"] == tweet_date_str].copy()

    if df_pred.shape[0] == 0:
        logger.info("Not enough prediction data.")
        return None

    df_pred = df_pred.drop(columns=[c for c in df_pred.columns if "_SMA_" in c]).copy()

    df_pred = add_sma_stuff(df=df_pred, tweet_date_str=tweet_date_str)

    return pd.concat([df_pred, df_train], axis=0)


def remove_fundy_category_cols(all_columns):
    clean_cols = []
    for c in all_columns:
        if not c.startswith("f22_ticker") \
            and not c.startswith("industry_") \
            and not c.startswith("location_") \
            and not c.startswith("category_") \
            and not c.startswith("sector_") \
            and not c.startswith("sicsector_") \
            and not c.startswith("table_") \
            and not c.startswith("currency_") \
            and not c.startswith("scalerevenue_") \
            and not c.startswith("scalemarketcap_") \
            and not c.startswith("famaindustry_"):
            clean_cols.append(c)

    return clean_cols


def train_predict(df_train, df_predict, narrow_cols):
    import warnings
    require_balance = True

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="invalid value encountered in true_divide")
        X_train, y_train, standard_scaler = transform_to_numpy(df=df_train,
                                                               narrow_cols=narrow_cols,
                                                               require_balance=require_balance)

        if X_train is None or X_train.shape[0] == 0 or y_train is None:
            logger.info("Not enough training data.")
            return None

        if require_balance:
            logger.info(f"XGB Max depth: {constants.xgb.defaults.max_depth}")
            model = xgb.XGBClassifier(max_depth=constants.xgb.defaults.max_depth)
        else:
            num_buy = df_train[df_train["buy_sell"] == 1].shape[0]
            num_sell = df_train[df_train["buy_sell"] != 1].shape[0]

            balance_ratio = num_buy / num_sell

            logger.info(f"Buy: {num_buy}; Sell: {num_sell}; balance_ratio: {balance_ratio}")

            model = xgb.XGBClassifier(max_depth=4, scale_pos_weight=balance_ratio)

        model.fit(X_train, y_train)

    X_predict = get_data_for_predictions(df=df_predict, narrow_cols=narrow_cols, standard_scaler=standard_scaler)

    logger.info("Invoking model prediction ...")
    prediction = model.predict(X_predict)

    df_predict.loc[:, "prediction"] = prediction

    return df_predict


def remove_purchase_cols(cols: Set[str]) -> List[str]:
    p_cols = {"purchase_open", "purchase_low", "purchase_high", "purchase_close", "closeunadj"}

    return list(cols - p_cols)


def get_train_columns(all_columns: List):
    return remove_purchase_cols(cols=set(all_columns))
    # clean_cols = remove_fundy_category_cols(all_columns)
#     misc_1 = set(['f22_num_other_tickers_in_tweet', 'high', 'days_util_sale', 'evebitda', 'user_screen_name', 'close_SMA_200_days_since_under', 'days_since', 'debtc', 'shareswa', 'debtnc', 'ebitdausd', 'receivables', 'currentratio', 'ebitda', 'future_date', 'de', 'date', 'netincdis', 'price', 'investmentsnc', 'shareswadil', 'closeunadj', 'assetsavg', 'ppnenet', 'f22_sentiment_neg', 'f22_has_cashtag', 'inventory', 'depamor', 'table_SFP', 'epsusd', 'fd_day_of_year', 'siccode', 'fcfps', 'f22_sentiment_pos', 'close_SMA_50_days_since_under', 'tbvps', 'dps', 'taxliabilities', 'ebitusd', 'evebit', 'liabilitiesc', 'pe', 'eps', 'fxusd', 'sbcomp', 'sps', 'ebit', 'close_SMA_20', 'user_listed_count', 'pe1', 'revenueusd', 'close_SMA_15', 'tangibles', 'ncfdebt', 'investments', 'equityavg', 'netinccmn'])
#     misc_2 = set(['ncf', 'ps', 'opex', 'intangibles', 'ros', 'liabilitiesnc', 'fd_day_of_week', 'prev_volume', 'rnd', 'ebt', 'user_verified', 'buy_sell', 'nasdaq_day_roi', 'stock_val_change_ex', 'capex', 'original_close_price', 'deposits', 'favorite_count', 'ncfi', 'investmentsc', 'ps1', 'f22_sentiment_compound', 'sgna', 'grossmargin', 'rank_roi', 'invcap', 'ncfdiv', 'close_SMA_100_days_since_under', 'ebitdamargin', 'fd_day_of_month', 'volume', 'user_friends_count', 'f22_sentiment_neu', 'accoci', 'sharefactor', 'assets', 'low', 'deferredrev', 'ncfinv', 'cashneq', 'assetsc', 'prev_high'])
#     add_back = ['f22_id', 'dimension', 'in_reply_to_status_id', 'created_at', 'calendardate', 'prev_date', 'datekey', 'place_country', 'user_location', 'lastupdated', 'user_time_zone', 'famasector', 'place_name', 'reportperiod', 'lang', 'created_at_timestamp', 'metadata_result_type', 'in_reply_to_screen_name']
#     clean_cols = list(set(list(set(clean_cols) - misc_1 - misc_2)))
#     return clean_cols


def seal_label_leak(df: pd.DataFrame, purchase_date_str: str, num_hold_days: int, num_days_until_purchase: int):
    df.sort_values(by=["date"], ascending=False, inplace=True)

    # NOTE: 2021-02-03: chris.flesche: This neatly erases any prediction day that could possibly fall on or after our prediction date.
    df.loc[df["future_date"] >= purchase_date_str, "buy_sell"] = -1

    return df

def seal_label_leak_2(df: pd.DataFrame, purchase_date_str: str, num_hold_days: int, num_days_until_purchase: int):
    df_grouped = df.groupby(by=["f22_ticker"])

    df_all = []
    for ticker, df_g in df_grouped:
        df_ticker = get_ticker_eod_data(ticker)

        df_dated = df_ticker[df_ticker["date"] < purchase_date_str].copy()
        df_dated.sort_values(by=["date"], inplace=True)
        if df_dated.shape[0] > 0:
            row = df_dated.iloc[-1]
            df_g.loc[df_g["future_date"] >= purchase_date_str, "future_open"] = row["open"]
            df_g.loc[df_g["future_date"] >= purchase_date_str, "future_low"] = row["low"]
            df_g.loc[df_g["future_date"] >= purchase_date_str, "future_high"] = row["high"]
            df_g.loc[df_g["future_date"] >= purchase_date_str, "future_close"] = row["close"]
            df_g.loc[df_g["future_date"] >= purchase_date_str, "future_date"] = row["date"]
        else:
            df_g.loc[df_g["future_date"] >= purchase_date_str, "buy_sell"] = -1

        df_all.append(df_g)

    df_concatted = pd.concat(df_all, axis=0)

    return df_concatted


def has_pred_rows_on_date(df: pd.DataFrame, date_str: str, tag: str):
    result = True
    if not has_rows_on_date(df=df, tag=tag):
        result = False
    else:
        df_predict = df[df["date"] == date_str]
        if df_predict is None or df_predict.shape[0] == 0:
            logger.info(f"Not enough prediction data on {date_str} after '{tag}'.")
            result = False

    return result


def has_rows_on_date(df: pd.DataFrame, tag: str):
    result = True
    if df is None or df.shape[0] == 0:
        logger.info(f"Not enough data after '{tag}'.")
        result = False
    return result


def predict_day(pp: PredictionParams):
    rois = []
    df = pp.df.copy()

    df_twitter = easy_convert_columns(df=df)

    # NOTE: 2021-02-22: chris.flesche: Ascribe after hours tweets to previous market date.

    df_sd_futured = get_stocks_based_on_tweets_2(df_tweets=df_twitter, tweet_date_str=pp.tweet_date_str,
                                                 num_hold_days=pp.num_hold_days, num_days_until_purchase=pp.num_days_until_purchase)

    if not has_pred_rows_on_date(df=df_sd_futured, date_str=pp.tweet_date_str, tag="get_stocks_based_on_tweets"):
        return False, rois

    df_stock_and_quarter, columns_fundy = combine_with_quarterly_stock_data(df=df_sd_futured)

    if not has_pred_rows_on_date(df=df_stock_and_quarter, date_str=pp.tweet_date_str, tag="combine_with_quarterly_stock_data"):
        return False, rois

    df_merged = merge_tweets_with_stock_data_2(df_twitter=df_twitter, df_stock_and_quarter=df_stock_and_quarter)

    if not has_pred_rows_on_date(df=df_merged, date_str=pp.tweet_date_str, tag="merge_tweets_with_stock_data"):
        return False, rois

    df_days_until = add_calendar_info_2(df=df_merged,
                                        tweet_date_str=pp.tweet_date_str,
                                        columns_fundy=columns_fundy,
                                        num_hold_days=pp.num_hold_days,
                                        oldest_tweet_date=pp.oldest_tweet_date)

    if not has_pred_rows_on_date(df=df_days_until, date_str=pp.tweet_date_str, tag="add_calendar_info"):
        return False, rois
    else:
        df_refined = twitter_service.refine_pool(df=df_days_until, min_volume=None, min_price=None, max_price=None)
        if not has_rows_on_date(df=df_refined, tag="refine_pool"):
            return False, rois

        df_ticker_hotted, narrow_cols = one_hot(df=df_refined)

        if not has_rows_on_date(df=df_ticker_hotted, tag="one_hot"):
            return False, rois

    while True:

        df_ready = df_ticker_hotted[df_ticker_hotted["date"] <= pp.tweet_date_str].copy()
        df_prepped = prep_predict(df=df_ready, tweet_date_str=pp.tweet_date_str)
        # df_prepped = df_ready

        if df_prepped is not None and df_prepped.shape[0] > 0:

            df_predict = df_prepped[df_prepped["date"] == pp.tweet_date_str].copy()
            if df_predict is not None and df_predict.shape[0] > 0:

                logger.info(f"df_predict after prep_predict: {df_predict.shape[0]}")

                df_sealed = seal_label_leak_2(df=df_prepped, purchase_date_str=pp.purchase_date_str, num_hold_days=pp.num_hold_days,
                                           num_days_until_purchase=pp.num_days_until_purchase)

                df_train, df_predict = split_train_predict(df=df_sealed, tweet_date_str=pp.tweet_date_str, num_days_until_purchase=pp.num_days_until_purchase,
                                                           num_hold_days=pp.num_hold_days)

                if df_train is not None and df_predict is not None and df_predict.shape[0] > 0 and df_train.shape[0] > 0:
                    logger.info(f"df_predict after split_train_predict: {df_predict.shape[0]}")

                    narrow_cols = get_train_columns(all_columns=list(df_train.columns))

                    # df_train = seal_label_leak(df=df_train, purchase_date_str=pp.purchase_date_str, num_hold_days=pp.num_hold_days, num_days_until_purchase=pp.num_days_until_purchase)

                    df_predict = train_predict(df_train=df_train,
                                               df_predict=df_predict,
                                               narrow_cols=narrow_cols)
                    if df_predict is not None and df_predict.shape[0] > 0:
                        num_buys = df_predict[df_predict["prediction"] == 1].shape[0]
                        logger.info(f"df_predict: {df_predict.shape[0]}: num buy predictions: {num_buys}")

                        df_buy = save_predictions(df_predict=df_predict, pp=pp)

                        if df_buy is not None and df_buy.shape[0] > 0 and pp.prediction_mode == PredictionMode.DevelopmentAndTraining:
                            days_roi_1 = pred_perf_testing.get_days_roi_from_prediction_table(df_preds=df_buy, purchase_date_str=pp.purchase_date_str,
                                                                                              num_hold_days=pp.num_hold_days)
                            days_roi_5 = pred_perf_testing.get_days_roi_from_prediction_table(df_preds=df_buy, purchase_date_str=pp.purchase_date_str, num_hold_days=5)

                            logger.info(f"Tweet: {pp.tweet_date_str}; Purchase: {pp.purchase_date_str}: roi {pp.num_hold_days} day: {days_roi_1}: 5: {days_roi_5}")

                            if days_roi_1 is not None:
                                rois.append(days_roi_1)
                    else:
                        logger.info("Training or prediction data not big enough after 'train_predict'.")
                else:
                    logger.info("Training or prediction data not big enough after 'split_train_predict'.")
            else:
                logger.info("Not enough df_prepped data after 'prep_predict'.")
        else:
            logger.info("Not enough training data after 'prep_predict'.")

        if len(rois) > 0:
            logger.info(f"Roi so far: {mean(rois):.4f}")

        # NOTE If this is True, this return will force the core data set to re-refetch with more date appropriate data for each
        # historical daily model build. We get more accurate data but the whole cycle takes 9 hours of crunching.
        if pp.clean_pure_run:
            return False, rois

        is_at_end = pp.subtract_day()
        if is_at_end:
            break

    if len(rois) > 0:
        logger.info(f"Ongoing roi: {mean(rois)}")

    return True, rois


def get_real_predictions(sample_size: int, num_hold_days: int, purchase_date_str: str, min_price: float = 0):
    file_path = constants.TWITTER_REAL_MONEY_PREDICTIONS_FILE_PATH
    df = pd.read_csv(str(file_path))

    logger.info(f"Searching for real predictions on {purchase_date_str} ...")

    df = df[(df["purchase_date"] == purchase_date_str) & (df["num_hold_days"] == num_hold_days)].copy()

    tickers = df["f22_ticker"].to_list()
    random.shuffle(tickers)

    if min_price > 0:
        tick_filtered = []
        for t in tickers:
            df_eod = ticker_service.get_ticker_eod_data(t)
            df_eod = df_eod[df_eod["date"] < purchase_date_str].copy()
            df_eod.sort_values(by=["date"], ascending=False, inplace=True)

            price = df_eod.iloc[0]["close"]
            if price >= min_price:
                tick_filtered.append(t)
                if len(tick_filtered) >= sample_size:
                    break

        tickers = tick_filtered
    else:
        tickers = tickers[:sample_size]

    return tickers