import collections
import copy
import random
from pathlib import Path
from statistics import mean
from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xgboost as xgb
from matplotlib.legend_handler import HandlerLine2D
from pandas import CategoricalDtype
from sklearn.ensemble import RandomForestClassifier
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from ams.config import constants
from ams.notebooks.twitter import twitter_ml_utils
from ams.services import twitter_service, ticker_service, pickle_service
from ams.services.csv_service import write_dicts_as_csv
from ams.services.equities.EquityFundaDimension import EquityFundaDimension
from ams.utils import date_utils, ticker_utils
from ams.utils.SplitData import SplitData
from ams.utils.Stopwatch import Stopwatch

num_iterations = 1

EquityHolding = collections.namedtuple('EquityHolding', 'ticker purchase_price num_shares purchase_dt, expiry_dt')

BATCH_SIZE = 1
LEARNING_RATE = .0001
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
        print(f"Mean: {mean(acc_acc)}")
    else:
        print("No mean on grouped mean - no rows?")

    g_tickers = []
    for group_name, df_group in df_g_holdout:
        g_tickers.append(group_name[0])

    return g_tickers, group_preds


def model_torch_predict(X_torch, model):
    model.eval()
    with torch.no_grad():
        model.cpu()
        raw_out = model(X_torch)
    prediction = raw_out.data.numpy()
    y_pred_tag = torch.round(torch.sigmoid(raw_out.data))
    return y_pred_tag


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

            threshold = 1
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
    min_rows_enough = 200

    has_enough_data = df_samp is not None and df_val_raw is not None and df_val_raw.shape[0] > min_rows_enough
    if has_enough_data:

        df_shuff = df_samp.sample(frac=1.0)

        df_train_raw, df_test_raw = twitter_service.ho_split_by_days(df_shuff, small_data_days_to_pull=None, small_data_frac=.2,
                                                                     use_only_recent_for_holdout=use_recent_for_holdout)  #

        has_enough_data = df_train_raw is not None and df_test_raw is not None and df_test_raw.shape[0] > min_rows_enough
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

                    trade_history.append(TwitterTrade(ticker=ticker,
                                                      purchase_price=close,
                                                      purchase_dt=date_utils.parse_std_datestring(date_str),
                                                      sell_dt=date_utils.parse_std_datestring(future_date),
                                                      sell_price=future_close
                                                      ))
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

    return avg_list, cpr.roi_list, cpr.trade_history


class trainData(Dataset):

    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)


## test data
class testData(Dataset):

    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)


class BinaryClassification(nn.Module):
    #     def __init__(self, num_input_features: int):
    #         super(BinaryClassification, self).__init__()
    #         #Our network consists of 3 layers. 1 input, 1 hidden and 1 output layer
    #         #This applies Linear transformation to input data.
    #         self.fc1 = nn.Linear(num_input_features, num_input_features - 10)
    #         #This applies linear transformation to produce output data
    #         self.fc2 = nn.Linear(num_input_features - 10, num_input_features)
    #         self.layer_out = nn.Linear(num_input_features, 1)
    #         self.relu = nn.ReLU()

    #     #This must be implemented
    #     def forward(self, inputs):
    #         #Output of the first layer
    #         x = self.relu(self.fc1(inputs))
    #         x = self.relu(self.fc2(x))
    #         #This produces output
    #         x = self.layer_out(x)

    #         return x

    # .003 daily roi
    def __init__(self, num_input_features: int):
        super(BinaryClassification, self).__init__()
        self.layer_1 = nn.Linear(num_input_features, num_input_features - 10)
        self.layer_2 = nn.Linear(num_input_features - 10, num_input_features)
        self.layer_out = nn.Linear(num_input_features, 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.batchnorm1 = nn.BatchNorm1d(num_input_features - 10, )
        self.batchnorm2 = nn.BatchNorm1d(num_input_features)

    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.batchnorm1(x)
        x = self.relu(self.layer_2(x))
        x = self.batchnorm2(x)
        x = self.dropout(x)
        x = self.layer_out(x)

        return x


def initialize_gpu():
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        device = torch.device('cuda:0')
        print("Using GPU")
    else:
        print("Using CPU")
        device = torch.device('cpu')

    return device


def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0]
    acc = torch.round(acc * 100)

    return acc


def train(model, data_loader, epoch_ndx, criterion: object, optimizer: object, device: object):
    model.train()
    train_loss = 0
    epoch_loss = 0
    epoch_acc = 0
    start = Stopwatch(start_now=True)
    for batch_idx, (X_batch, y_batch) in enumerate(data_loader):
        optimizer.zero_grad()

        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        y_pred = model(X_batch)

        loss = criterion(y_pred, y_batch.unsqueeze(1))
        acc = binary_acc(y_pred, y_batch.unsqueeze(1))

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    num_items = len(data_loader)
    avg_loss = epoch_loss / num_items
    avg_acc = epoch_acc / num_items
    print(f'Epoch {epoch_ndx + 0:03}: | Loss: {avg_loss:.5f} | Acc: {avg_acc:.3f}')

    start.end(msg="Epoch time")

    return avg_loss, avg_acc


def test(model, data_loader, criterion: object, device: object):
    model.eval()
    test_loss = 0
    correct = 0
    total_loss = 0
    total_acc = 0
    with torch.no_grad():
        count_loops = 0
        for X_batch, y_batch in data_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            y_pred = model(X_batch)

            loss = criterion(y_pred, y_batch.unsqueeze(1))
            acc = binary_acc(y_pred, y_batch.unsqueeze(1))

            total_loss += loss.item()
            total_acc += acc.item()
            count_loops += 1

    num_items = len(data_loader.dataset)
    avg_loss = total_loss / num_items
    avg_acc = total_acc / count_loops

    print(f'\nTest loss: {avg_loss:.4f} | Acc: {avg_acc:.2f}% | Count_loops: {count_loops})\n')

    return avg_loss, avg_acc


def train_model(model: object, train_loader: object, test_loader: object, criterion: object, optimizer: object, device: object):
    model.train()
    EPOCHS = 200

    last_t_loss = 9999999999
    best_test_loss = 9999999999
    best_test_acc = 0
    countdown = 0
    max_wait_countdown = 3
    for e in range(1, EPOCHS + 1):
        loss, acc = train(model, train_loader, e, criterion=criterion, optimizer=optimizer, device=device)
        t_loss, t_acc = test(model, test_loader, criterion=criterion, device=device)

        # Overfitting guard
        #         if t_loss < best_test_loss:
        if t_acc > best_test_acc:
            best_test_loss = t_loss
            best_test_acc = t_acc

            best_model = copy.deepcopy(model)
            countdown = 0
        else:
            countdown += 1

        if countdown >= max_wait_countdown:
            break

    print(f"Best loss: {loss} | best acc: {acc}% | best test loss: {best_test_loss} | best test acc: {best_test_acc}%")

    return best_model


def get_initialized_model(num_input_features: int, device: object):
    model = BinaryClassification(num_input_features)
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    return model, criterion, optimizer


def calc_nn_roi(
    df_val_raw: pd.DataFrame,
    df_val_std: pd.DataFrame,
    model: object,
    train_cols: List[str],
    target_roi_frac: float,
    zero_in: bool,
    is_model_torch: bool = True,
    is_batch_first_run: bool = False
):
    if df_val_raw.shape[0] == 0:
        raise Exception("No holdout data.")

    g_tickers, group_preds = group_and_mean_preds_buy_sell(df_val_std, model, train_cols, is_model_torch=is_model_torch)

    df_splitted = twitter_service.join_with_stock_splits(df=df_val_raw)

    _, sac_roi_list, trade_history = get_roi_matrix(df=df_splitted, group_preds=group_preds, target_roi_frac=target_roi_frac, zero_in=zero_in)

    persist_trade_history(twitter_trades=trade_history, overwrite_existing=is_batch_first_run)

    if len(sac_roi_list) > 0:
        print(f"Mean sac_roi: {mean(sac_roi_list)}")

    return mean(sac_roi_list)


def torch_non_linear(df: pd.DataFrame, narrow_cols: List[str]):
    device = initialize_gpu()
    desired_dtype = 'float64'

    num_iterations = 1000

    zero_in = True
    if zero_in:
        target_roi_frac = 0.07
    else:
        target_roi_frac = 0.001

    sac_by_bucket = []
    sac_list = []
    for i in range(num_iterations):
        sd = split_off_data(df=df, narrow_cols=narrow_cols)

        if sd and sd.has_enough_data:
            X_train = sd.X_train
            y_train = sd.y_train
            X_test = sd.X_test
            y_test = sd.y_test
            df_test_raw = sd.df_test_raw
            df_test_std = sd.df_test_std
            df_val_raw = sd.df_val_raw
            df_val_std = sd.df_val_std
            train_cols = sd.train_cols
            has_enough_data = sd.has_enough_data

            if df_val_raw.shape[0] > 1000 or has_enough_data:
                X_train_con = X_train.astype(desired_dtype)

                y_train = np.where(y_train == -1, 0, y_train)
                y_train_con = y_train.astype(desired_dtype)

                train_dataset = trainData(torch.FloatTensor(X_train_con),
                                          torch.FloatTensor(y_train_con))

                X_test_con = X_test.astype(desired_dtype)

                y_test = np.where(y_test == -1, 0, y_test)
                y_test_con = y_test.astype(desired_dtype)

                test_dataset = testData(torch.FloatTensor(X_test_con),
                                        torch.FloatTensor(y_test_con))

                train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
                test_loader = DataLoader(dataset=test_dataset, batch_size=1)

                model, criterion, optimizer = get_initialized_model(num_input_features=X_train.shape[1], device=device)

                best_model = train_model(model=model, train_loader=train_loader, test_loader=test_loader, criterion=criterion, optimizer=optimizer, device=device)

                sac_mean_list = calc_nn_roi(df_val_raw=df_val_raw,
                                            df_val_std=df_val_std,
                                            model=best_model,
                                            train_cols=train_cols,
                                            target_roi_frac=target_roi_frac,
                                            zero_in=zero_in,
                                            is_batch_first_run=(i == 0)
                                            )

                sac_list += sac_mean_list

                sac_by_bucket.append(sac_mean_list)

                print(f"\nmultiday: s@close_mean: {mean(sac_list)}\n")
        else:
            print("No data from split_off_data.")

        df, has_remaining_days = twitter_service.remove_last_days(df=df, num_days=1)

        if not has_remaining_days:
            print("No more remaining days to test.")
            break

    return sac_list


def randy_forest(X_train: np.array, y_train: np.array, X_test: np.array, y_test: np.array):
    n_estimators = [1, 2, 4, 8, 16, 32, 64, 100, 200]
    train_results = []
    test_results = []

    for estimator in n_estimators:
        rf = RandomForestClassifier(n_estimators=estimator, n_jobs=-1)
        rf.fit(X_train, y_train)
        train_pred = rf.predict(X_train)
        from sklearn.metrics import roc_curve
        false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
        from sklearn.metrics import auc
        roc_auc = auc(false_positive_rate, true_positive_rate)
        train_results.append(roc_auc)
        y_pred = rf.predict(X_test)
        false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        test_results.append(roc_auc)

    line1, = plt.plot(n_estimators, train_results, 'b', label="Train AUC")
    line2, = plt.plot(n_estimators, test_results, 'r', label="Test AUC")
    plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
    plt.ylabel('AUC score')
    plt.xlabel('n_estimators')
    plt.show()


def rnd_forest_clf(X_train: np.array, y_train: np.array, X_test: np.array, y_test: np.array):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score

    rf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                                max_depth=None, max_features='auto', max_leaf_nodes=None,
                                min_impurity_split=1e-07, min_samples_leaf=1,
                                min_samples_split=2, min_weight_fraction_leaf=0.0,
                                n_estimators=1, n_jobs=1, oob_score=False, random_state=None,
                                verbose=0, warm_start=False)
    rf.fit(X=X_train, y=y_train)

    y_test_pred = rf.predict(X_test)

    print(accuracy_score(y_test, y_test_pred))

    return rf


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

    df_ranked["rank_roi"] = df_ranked.apply(rank_roied, axis=1)

    rank_mean = df_tip_ranks["rank"].mean()

    df_ranked["rating"] = df_ranked["rating"].fillna(0)
    df_ranked["rank"] = df_ranked["rank"].fillna(rank_mean)
    df_ranked["rating_age_days"] = df_ranked["rating_age_days"].fillna(1000)

    return df_ranked


def make_f22_ticker_one_hotted(df_ranked: pd.DataFrame):
    cols = ["f22_ticker"]
    df = df_ranked[cols].copy()
    df_one_hots = []
    for c in cols:
        df[c] = df[c].fillna("<unknown>")
        uniques = df[c].unique().tolist()
        uniques.append("<unknown>")
        uniques = list(set(uniques))

        df[c] = df[c].astype(CategoricalDtype(uniques))
        df_new_cols = pd.get_dummies(df[c], prefix=c)
        df_one_hots.append(df_new_cols)

    df_hot_cols = pd.concat(df_one_hots, axis=1)

    return pd.concat([df_ranked, df_hot_cols], axis=1)


def show_distribution(df: pd.DataFrame, group_column_name: str = "date"):
    df.sort_values(by=[group_column_name], inplace=True)

    day_groups = df.groupby(df[group_column_name])[group_column_name].count()

    day_groups.plot(kind='bar', figsize=(10, 5), legend=None)


def find_ml_pred_perf(df: pd.DataFrame, narrow_cols: List[str]):
    zero_in = True
    if zero_in:
        target_roi_frac = 0.06
    else:
        target_roi_frac = .001

    print(f"Count: {df.shape[0]}")

    use_rnd_forest = True
    use_regression = False

    all_avgs = []

    tickers = df["f22_ticker"].unique().tolist()
    random.shuffle(tickers)

    sac_roi_list = []

    count = 0
    while True:
        sd = split_off_data(df=df, narrow_cols=narrow_cols, use_recent_for_holdout=True)
        if sd and sd.has_enough_data:
            X_train = sd.X_train
            y_train = sd.y_train
            X_test = sd.X_test
            y_test = sd.y_test
            df_test_raw = sd.df_test_raw
            df_test_std = sd.df_test_std
            df_val_raw = sd.df_val_raw
            df_val_std = sd.df_val_std
            train_cols = sd.train_cols
            has_enough_data = sd.has_enough_data

            if df_test_raw.shape[0] == 0:
                raise Exception("No holdout data.")

            if use_rnd_forest:

                if use_regression:
                    model = twitter_service.dec_tree_regressor(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, max_depth=16)
                else:
                    # randy_forest(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
                    # model = rnd_forest_clf(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
                    print(f"Size of X_train: {X_train.shape[0]}")

                    model = twitter_service.dec_tree(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, max_depth=16)

                acc_model = model
            else:
                acc_model = twitter_service.train_mlp(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

            df_splitted = twitter_service.join_with_stock_splits(df=df_val_raw)

            if use_regression:
                g_tickers, group_preds = group_and_mean_preds_regress(df_val_std, acc_model, train_cols)
            else:
                g_tickers, group_preds = group_and_mean_preds_buy_sell(df_val_std, acc_model, train_cols)

            roi_list, sac_roi, trade_history = get_roi_matrix(df=df_splitted, group_preds=group_preds, target_roi_frac=target_roi_frac, zero_in=zero_in)

            overwrite_existing = True if count == 0 else False
            persist_trade_history(twitter_trades=trade_history, overwrite_existing=overwrite_existing)

            if len(roi_list) > 0:
                max_avg = max(roi_list)
                all_avgs.append(max_avg)

            if sac_roi is not None:
                sac_roi_list.append(sac_roi)

            if len(all_avgs) > 0 and len(sac_roi_list) > 0:
                print(f"Cumulative Metrics: Avg Best Sell@High ROI rate: {mean(all_avgs)} | Avg Sell@Close ROI: {mean(sac_roi_list)} ")

        df, has_remaining_days = twitter_service.remove_last_days(df=df, num_days=1)

        if not has_remaining_days:
            print("No more remaining days to test.")
            break

        count += 1

    return sac_roi_list


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
    # print(f"Total cols: {len(cols)}")

    col_drop = set(df.columns) - set(cols)

    return df.drop(columns=col_drop).reset_index(drop=True)


def coerce_convert_to_numeric(df: pd.DataFrame, col: str):
    df[col] = pd.to_numeric(df[col], errors='coerce')
    df[col] = df[col].fillna(0)
    return df


def convert_twitter_to_numeric(df: pd.DataFrame):
    df = coerce_convert_to_numeric(df=df, col="user_followers_count")
    df = coerce_convert_to_numeric(df=df, col="f22_sentiment_compound")
    return coerce_convert_to_numeric(df=df, col="f22_compound_score")


def merge_fundies_with_stock(df_stock_data: pd.DataFrame):
    from ams.services.equities import equity_fundy_service as efs
    df_equity_fundies = efs.get_equity_fundies()

    df_eq_fun_quarters = df_equity_fundies[df_equity_fundies["dimension"] == EquityFundaDimension.AsReportedQuarterly.value]

    return pd.merge(df_eq_fun_quarters, df_stock_data, on=["ticker"], how='outer', suffixes=[None, "_eq_fun"])


def num_days_from(from_date: str, to_date: str):
    from_dt = date_utils.parse_std_datestring(from_date)
    to_dt = date_utils.parse_std_datestring(to_date)

    return (to_dt - from_dt).days


def add_days_since_quarter_results(df: pd.DataFrame, should_drop_missing_future_date: bool=True):
    df = df.dropna(axis="rows", subset=["datekey", "future_date"])

    df["days_since"] = df.apply(lambda x: num_days_from(x["datekey"], x["future_date"]), axis=1)

    return df


def add_calendar_days(df: pd.DataFrame):
    def day_of_week(date_str):
        return pd.Timestamp(date_str).dayofweek

    df["fd_day_of_week"] = df.apply(lambda x: day_of_week(x["future_date"]), axis=1)

    def day_of_year(date_str):
        return pd.Timestamp(date_str).dayofyear

    df["fd_day_of_year"] = df.apply(lambda x: day_of_year(x["future_date"]), axis=1)

    def day_of_month(date_str):
        return int(date_str.split("-")[2])

    df["fd_day_of_month"] = df.apply(lambda x: day_of_month(x["future_date"]), axis=1)

    return df


def add_nasdaq_roi(df: pd.DataFrame):
    df_roi_nasdaq = pd.read_parquet(str(constants.DAILY_ROI_NASDAQ_PATH))
    df_roi_nasdaq = df_roi_nasdaq.rename(columns={"roi": "nasdaq_day_roi"})

    df = pd.merge(df_roi_nasdaq, df, on=["date"], how="right")

    return df.drop_duplicates(subset=["f22_ticker", "date"])


def add_sma_stuff(df: pd.DataFrame):
    df = ticker_utils.add_sma_history(df=df, target_column="close", windows=[15, 20, 50, 100, 200])

    df = ticker_utils.add_days_since_under_sma_many_tickers(df=df, col_sma="close_SMA_200", close_col="close")
    df = ticker_utils.add_days_since_under_sma_many_tickers(df=df, col_sma="close_SMA_15", close_col="close")
    df = ticker_utils.add_days_since_under_sma_many_tickers(df=df, col_sma="close_SMA_20", close_col="close")
    df = ticker_utils.add_days_since_under_sma_many_tickers(df=df, col_sma="close_SMA_50", close_col="close")

    return ticker_utils.add_days_since_under_sma_many_tickers(df=df, col_sma="close_SMA_100", close_col="close")


def xgb_learning(df: pd.DataFrame, narrow_cols: List[str]):
    min_train_rows = 50
    num_iterations = 500

    zero_in = True
    if zero_in:
        target_roi_frac = 0.07
    else:
        target_roi_frac = 0.001

    sac_list = []
    for i in range(num_iterations):
        sd = twitter_ml_utils.split_off_data(df=df, narrow_cols=narrow_cols)

        if sd and sd.has_enough_data:
            X_train = sd.X_train
            y_train = sd.y_train
            X_test = sd.X_test
            y_test = sd.y_test
            df_val_raw = sd.df_val_raw
            df_val_std = sd.df_val_std
            train_cols = sd.train_cols
            has_enough_data = sd.has_enough_data

            if df_val_raw.shape[0] > min_train_rows and has_enough_data:
                model = xgb.XGBClassifier(max_depth=4)
                model.fit(X_train, y_train)

                if i == 0:
                    pickle_service.save(model, str(constants.TWITTER_XGB_MODEL_PATH))

                best_model = model

                sac_mean = twitter_ml_utils.calc_nn_roi(df_val_raw=df_val_raw,
                                                             df_val_std=df_val_std,
                                                             model=best_model,
                                                             train_cols=train_cols,
                                                             target_roi_frac=target_roi_frac,
                                                             zero_in=zero_in,
                                                             is_model_torch=False,
                                                             is_batch_first_run=(i == 0)
                                                             )

                sac_list.append(sac_mean)

                mean_sac_list = mean(sac_list)

                print(f"\nOverall mean s@close: {mean_sac_list}\n")
            else:
                print("No data from split_off_data.")

                df, has_remaining_days = twitter_service.remove_last_days(df=df, num_days=1)

                if not has_remaining_days:
                    print("No more remaining days to test.")

    return sac_list