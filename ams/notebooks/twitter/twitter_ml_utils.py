import collections
from statistics import mean
from typing import List, Dict

import numpy as np
import pandas as pd
import torch

from ams.config import constants
from ams.services.csv_service import write_dicts_as_csv
from ams.utils import date_utils

EquityHolding = collections.namedtuple('EquityHolding', 'ticker purchase_price num_shares purchase_dt, expiry_dt')

from ams.services import twitter_service, ticker_service
from ams.utils.SplitData import SplitData

initial_cash = 10000
num_trades_at_once = 1


def group_and_mean_preds_buy_sell(df: pd.DataFrame, model: object, train_cols: List[str], is_model_torch=False):
    df_g_holdout = df.groupby(['f22_ticker', 'purchase_date'])

    group_count = 0
    acc_acc = []
    group_preds = {}
    for group_name, df_group in df_g_holdout:
        X_holdout = np.array(df_group[train_cols])
        y_holdout = np.array(df_group['buy_sell'])

        if is_model_torch:
            desired_dtype = 'float64'
            X_holdout_con = X_holdout.astype(desired_dtype)
            X_torch = torch.FloatTensor(X_holdout_con)

            y_holdout = np.where(y_holdout == -1, 0, y_holdout)
            pre_y_ho = sum(y_holdout) / len(y_holdout)
            pre_y_ho = 0 if pre_y_ho < .5 else 1

            model.eval()

            with torch.no_grad():
                model.cpu()
                raw_out = model(X_torch)

            prediction = raw_out.data.numpy()
            y_pred_tag = torch.round(torch.sigmoid(raw_out.data))
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
        print(f"Mean: {mean(acc_acc)}")
    else:
        print("No mean on grouped mean - no rows?")

    g_tickers = []
    for group_name, df_group in df_g_holdout:
        g_tickers.append(group_name[0])

    return g_tickers, group_preds


def group_and_mean_preds_regress(df: pd.DataFrame, model: object, train_cols: List[str], is_model_torch=False):
    df_g_holdout = df.groupby(['f22_ticker', 'purchase_date'])

    group_count = 0
    acc_acc = []
    group_preds = {}
    for group_name, df_group in df_g_holdout:
        X_holdout = np.array(df_group[train_cols])
        #         y_holdout = np.array(df_group['buy_sell'])
        y_holdout = np.array(df_group['stock_val_change_ex'])

        if is_model_torch:
            #             desired_dtype = 'float64'
            #             X_holdout_con = X_holdout.astype(desired_dtype)
            #             X_torch = torch.FloatTensor(X_holdout_con)

            #             y_holdout = np.where(y_holdout==-1, 0, y_holdout)
            #             pre_y_ho = sum(y_holdout)/len(y_holdout)
            #             pre_y_ho = 0 if pre_y_ho < .5 else 1

            #             model.eval()

            #             with torch.no_grad():
            #                 model.cpu()
            #                 raw_out = model(X_torch)

            #             prediction = raw_out.data.numpy()
            #             y_pred_tag = torch.round(torch.sigmoid(raw_out.data))
            #             pred_mean = sum(y_pred_tag.numpy()) / len(y_holdout)

            #             pred_mean = -1 if pred_mean < .5 else 1
            raise Exception("Not implemented yet.")
        else:
            prediction = model.predict(X_holdout)
            pred_mean = sum(prediction) / len(prediction)

            print(pred_mean)
            threshold = 0

            pred_mean = -1 if pred_mean < threshold else 1

            pre_y_ho = sum(y_holdout) / len(y_holdout)
            pre_y_ho = -1 if pre_y_ho < threshold else 1

        # NOTE: This means that we are predicting only when buys are successfull. And we only buy.
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
        print(f"Mean: {mean(acc_acc)}")
    else:
        print("No mean on grouped mean - no rows?")

    g_tickers = []
    for group_name, df_group in df_g_holdout:
        g_tickers.append(group_name[0])

    return g_tickers, group_preds


def has_f22_column(df: pd.DataFrame):
    cols = [c for c in df.columns if c.startswith("f22")]
    print(cols)


def split_off_data(df: pd.DataFrame, narrow_cols: List[str], tickers: List[str] = None, use_recent_for_holdout=True) -> SplitData:
    df_samp, df_val_raw = twitter_service.ho_split_by_days(df,
                                                           small_data_days_to_pull=None,
                                                           small_data_frac=.025,
                                                           use_only_recent_for_holdout=use_recent_for_holdout)

    X_train = None
    y_train = None
    X_test = None
    y_test = None
    df_test_std = None
    df_val_std = None
    train_cols = None

    has_enough_data = df_samp is not None and df_val_raw is not None and df_val_raw.shape[0] > 500
    if has_enough_data:

        df_shuff = df_samp.sample(frac=1.0)

        df_train_raw, df_test_raw = twitter_service.ho_split_by_days(df_shuff, small_data_days_to_pull=None, small_data_frac=.2,
                                                                     use_only_recent_for_holdout=use_recent_for_holdout)  #

        has_enough_data = df_train_raw is not None and df_test_raw is not None and df_test_raw.shape[0] > 500
        if has_enough_data:
            print(f"Original: {df.shape[0]}; train_set: {df_train_raw.shape[0]}; test_set: {df_test_raw.shape[0]}")

            df_train_std, df_test_std, df_val_std = ticker_service.std_dataframe(df_train=df_train_raw, df_test=df_test_raw, df_val=df_val_raw)

            X_train, y_train, X_test, y_test, train_cols = twitter_service.split_train_test(train_set=df_train_std, test_set=df_test_std,
                                                                                            narrow_cols=narrow_cols)

        return SplitData(X_train=X_train,
                         y_train=y_train,
                         X_test=X_test,
                         y_test=y_test,
                         df_test_raw=df_test_raw,
                         df_test_std=df_test_std,
                         df_val_raw=df_val_raw,
                         df_val_std=df_val_std,
                         train_cols=train_cols,
                         has_enough_data=has_enough_data)
    if not has_enough_data:
        print("Not enough data.")


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
        action = "<not specified yet>"

        for date_str, pred_mean in info.items():
            should_buy = (pred_mean == 1)

            close, future_close, future_high, future_open, future_date, split_share_multiplier = ticker_service.get_stock_info(df=df_helper,
                                                                                                                               ticker=ticker,
                                                                                                                               date_str=date_str)
            num_shares = split_share_multiplier * num_shares

            pot_roi = (future_high - close) / close
            pot_roi_list.append(pot_roi)

            shares_price = num_trades_at_once * close

            trade_history.append(TwitterTrade(ticker=ticker,
                                              purchase_price=close,
                                              purchase_dt=date_utils.parse_std_datestring(date_str),
                                              sell_dt=date_utils.parse_std_datestring(future_date),
                                              sell_price=future_close
                                              ))

            if should_buy:
                ticker_service.calculate_roi(target_roi=target_roi, close_price=close, future_high=future_high,
                                             future_close=future_close, calc_dict=calc_dict, zero_in=zero_in)

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
                    print("Not enough cash for purchase.")

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
        print(f"Avg investment per trade: {investment_per_trade}")
        print(f"Roi per trade: {roi_per_trade}")

    pot = round(sum(cpr.pot_roi_list) / len(cpr.pot_roi_list), 6)
    if cpr.acc > 0:
        acc = round(cpr.acc, 5)
    else:
        acc = "<NA>"

    print(f"Num trades: {cpr.num_trades} | acc: {acc} | s@close roi: {sac_roi} | s@high roi: {pot}")

    avg_list = []
    for roi_calc, roi_calc_list in cpr.calc_dict.items():
        if len(roi_calc_list) > 0:
            avg_roi = mean(roi_calc_list)
            num_list = len(roi_calc_list)
            print(f"Sell high/close roi@{roi_calc}: {round(avg_roi, 6)}; weight: {num_list * avg_roi}")
            avg_list.append(avg_roi)

    return avg_list, sac_roi, cpr.trade_history