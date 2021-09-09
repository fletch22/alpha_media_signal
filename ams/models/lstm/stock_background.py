import random
import time
from math import sqrt
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
# multivariate data preparation
from numpy import array
from sklearn.preprocessing import MinMaxScaler

from ams.config import logger_factory, constants
from ams.services import ticker_service, twitter_service
from ams.twitter import twitter_ml_utils
from ams.twitter.twitter_ml_utils import merge_with_stock_details

logger = logger_factory.create(__name__)


class Model(torch.nn.Module):
    def __init__(self, input_size, seq_len, hidden_size, output_size):
        super(Model, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lstm = torch.nn.LSTMCell(self.input_size, self.hidden_size)
        self.linear = torch.nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, future=0, y=None):
        outputs = []

        # reset the state of LSTM
        # the state is kept till the end of the sequence
        h_t = torch.zeros(input.size(0), self.hidden_size, dtype=torch.float32)
        c_t = torch.zeros(input.size(0), self.hidden_size, dtype=torch.float32)

        for i, input_t in enumerate(input.chunk(input.size(1), dim=1)):
            h_t, c_t = self.lstm(input_t, (h_t, c_t))
            output = self.linear(h_t)
            outputs += [output]

        for i in range(future):
            if y is not None and random.random() > 0.5:
                output = y[:, [i]]  # teacher forcing
            h_t, c_t = self.lstm(output, (h_t, c_t))
            output = self.linear(h_t)
            outputs += [output]
        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs


class MV_LSTM(torch.nn.Module):
    def __init__(self, n_features, seq_length):
        super(MV_LSTM, self).__init__()
        self.n_features = n_features
        self.seq_len = seq_length
        self.n_hidden = 20  # number of hidden states
        self.n_layers = 1  # number of LSTM layers (stacked)
        dropout = 0

        self.l_lstm = torch.nn.LSTM(input_size=n_features,
                                    hidden_size=self.n_hidden,
                                    num_layers=self.n_layers,
                                    batch_first=True,
                                    dropout=dropout)
        # according to pytorch docs LSTM output is
        # (batch_size,seq_len, num_directions * hidden_size)
        # when considering batch_first = True
        self.l_linear = torch.nn.Linear(self.n_hidden * self.seq_len, 1)

    def init_hidden(self, batch_size):
        # even with batch_first = True this remains same as docs
        hidden_state = torch.zeros(self.n_layers, batch_size, self.n_hidden)
        cell_state = torch.zeros(self.n_layers, batch_size, self.n_hidden)
        self.hidden = (hidden_state, cell_state)

    def forward(self, x, future=0):
        batch_size, seq_len, _ = x.size()

        lstm_out, self.hidden = self.l_lstm(x, self.hidden)
        # lstm_out(with batch_first = True) is
        # (batch_size,seq_len,num_directions * hidden_size)
        # for following linear layer we want to keep batch_size dimension and merge rest
        # .contiguous() -> solves tensor compatibility error
        x = lstm_out.contiguous().view(batch_size, -1)
        return self.l_linear(x)


def generate_batch_data(x, y, batch_size: int):
    for batch_ndx, i in enumerate(range(0, len(x) - batch_size, batch_size)):
        x_batch = x[i:i + batch_size, :, :]
        y_batch = y[i:i + batch_size]

        x_batch = torch.tensor(x_batch, dtype=torch.float32)
        y_batch = torch.tensor(y_batch, dtype=torch.float32)

        yield x_batch, y_batch, batch_ndx


class Optimization:
    """ A helper class to train, test and diagnose the LSTM"""

    def __init__(self, model, loss_fn, optimizer, scheduler):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_losses = []
        self.val_losses = []
        self.futures = []

    def train(
        self,
        x_train,
        y_train,
        x_val=None,
        y_val=None,
        batch_size=16,
        n_epochs=120,
        do_teacher_forcing=None,
    ):
        seq_len = x_train.shape[1]
        for epoch in range(n_epochs):
            train_loss = 0
            start_time = time.time()
            for x_batch, y_batch, batch_ndx in generate_batch_data(x_train, y_train, batch_size):
                self.model.train()
                self.model.init_hidden(x_batch.size(0))

                y_pred = self._predict(x_batch, y_batch, seq_len, do_teacher_forcing)
                loss = self.loss_fn(y_pred.view(-1), y_batch)

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                train_loss += loss.item()

            train_loss /= batch_ndx
            elapsed = time.time() - start_time

            self._validation(x_val, y_val, batch_size)
            self.scheduler.step(self.val_losses[-1])

            print(
                "Epoch %d Train loss: %.8f. Validation loss: %.8f. Avg future: %.2f. Elapsed time: %.2fs."
                % (epoch + 1, train_loss, self.val_losses[-1], np.average(self.futures), elapsed)
            )

    def _predict(self, x_batch, y_batch, seq_len, do_teacher_forcing):
        if do_teacher_forcing:
            future = random.randint(1, int(seq_len) / 2)
            limit = x_batch.size(1) - future
            y_pred = self.model(x_batch[:, :limit], future=future, y=y_batch[:, limit:])
        else:
            future = 0
            y_pred = self.model(x_batch)
        self.futures.append(future)
        return y_pred

    def _validation(self, x_val, y_val, batch_size):
        if x_val is None or y_val is None:
            return
        with torch.no_grad():
            val_loss = 0
            batch_ndx = 0
            self.model.eval()
            for x_batch, y_batch, batch_ndx in generate_batch_data(x_val, y_val, batch_size):
                output = self.model(x_batch)
                loss = self.loss_fn(output.view(-1), y_batch)
                val_loss += loss.item()

            val_loss /= batch_ndx
            self.val_losses.append(val_loss)

    def evaluate(self, x_test, y_test, batch_size, future=1):
        with torch.no_grad():
            test_loss = 0
            batch_ndx = 0
            actual, predicted = [], []
            self.model.eval()
            for x_batch, y_batch, batch_ndx in generate_batch_data(x_test, y_test, batch_size):
                y_pred = self.model(x_batch, future=future)

                y_batch = y_batch.reshape(-1, 1)

                y_pred = (
                    y_pred[:, -len(y_batch)] if y_pred.shape[1] > y_batch.shape[1] else y_pred
                )

                loss = self.loss_fn(y_pred, y_batch)
                test_loss += loss.item()
                actual += torch.squeeze(y_batch[:, -1]).data.cpu().numpy().tolist()
                predicted += torch.squeeze(y_pred[:, -1]).data.cpu().numpy().tolist()
            test_loss /= batch_ndx
            return actual, predicted, test_loss

    def plot_losses(self):
        plt.plot(self.train_losses, label="Training loss")
        plt.plot(self.val_losses, label="Validation loss")
        plt.legend()
        plt.title("Losses")


# def generate_sequence(scaler, model, x_sample, future=0):
#     """ Generate future values for x_sample with the model """
#     y_pred_tensor = model(x_sample, future=future)
#     y_pred = y_pred_tensor.cpu().tolist()
#     y_pred = scaler.inverse_transform(y_pred)
#     return y_pred


def to_dataframe(actual, predicted):
    return pd.DataFrame({"actual": actual, "predicted": predicted})


def inverse_transform(scaler, df: pd.DataFrame, columns: List[str]):
    for col in columns:
        values = df[col].values.reshape(-1, 1)
        df[col] = scaler.inverse_transform(values)
    return df


def add_future_info(df: pd.DataFrame, num_hold_days: int, num_days_until_purchase: int):
    df.loc[:, "future_close"] = df["close"]
    df.loc[:, "future_date"] = df["date"]
    cols = ["future_close", "future_date"]
    df.loc[:, (cols)] = df[cols].shift(-(num_hold_days + num_days_until_purchase))

    df.loc[:, "purchase_close"] = df["close"]
    df.loc[:, "purchase_date"] = df["date"]
    cols = ["purchase_close", "purchase_date"]
    df.loc[:, (cols)] = df[cols].shift(-(num_days_until_purchase))

    return df


def merge_stock_data_w_details(df: pd.DataFrame, industries: List[str] = None):
    df_merged = merge_with_stock_details(df)

    df_merged.sort_values(by=["ticker", "date"], inplace=True)

    if df_merged.shape[0] == 0:
        logger.info("Not enough data after merge.")

    df_filtered = df_merged
    if industries is not None:
        df_filtered = df_merged[df_merged["industry"].isin(industries)].copy()

    return df_filtered


# split a multivariate sequence into samples
def split_sequences(sequences, n_steps):
    X, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if end_ix > len(sequences):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix - 1, -1]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


def drop_non_unique_cols(df: pd.DataFrame):
    cols = list(df.columns)
    col_drops = []
    for c in cols:
        uniques = list(df[c].unique())
        if len(uniques) == 1:
            col_drops.append(c)

    return df.drop(columns=col_drops)


def get_evaluation(optimization: Optimization, label_scaler: object, x_test: np.array, y_test: np.array, batch_size: int):
    actual_1, predicted_1, test_loss_1 = optimization.evaluate(x_test, y_test, future=1, batch_size=batch_size)
    df_result = to_dataframe(actual_1, predicted_1)

    return inverse_transform(label_scaler, df_result, ['actual', 'predicted'])


def merge_with_ticker_info(tickers: List[str],
                           max_tickers=200,
                           num_hold_days: int = 1,
                           num_days_until_purchase: int = 1):
    df_all = []

    for ndx, ticker in enumerate(tickers):
        df_eod = ticker_service.get_ticker_eod_data(ticker=ticker)
        if df_eod is not None:
            df_all.append(df_eod)

        if max_tickers is not None and ndx >= max_tickers:
            break

    if len(df_all) == 0:
        return None

    df_eod_all = pd.concat(df_all, axis=0)

    df_futured = add_future_info(df=df_eod_all, num_hold_days=num_hold_days, num_days_until_purchase=num_days_until_purchase)

    df_w_fundy, _ = twitter_ml_utils.combine_with_quarterly_stock_data(df=df_futured)

    df_merged = merge_stock_data_w_details(df=df_w_fundy)

    df_merged["ticker"].unique()
    df_merged.fillna(method="ffill", inplace=True)

    df_agg = pd.read_parquet(constants.STOCK_AGG_DATAFILE)
    df_agg = df_agg.rename(columns={"1_day_roi": "nasdaq_day_roi"})

    df_roied = pd.merge(df_merged, df_agg, on=["ticker", "date"])

    df_bs = twitter_service.add_buy_sell(df=df_roied)
    df_bs.loc[:, 'stock_val_change'] = df_bs["stock_val_change"].apply(lambda svc: -1 * sqrt(abs(svc)) if svc < 0 else sqrt(svc))

    return df_bs


def fix_columns(df: pd.DataFrame):
    df_dropped = drop_non_unique_cols(df=df)

    label_col = "future_close"
    col_drops = {'calendardate', 'datekey', 'reportperiod', "ticker", "purchase_date", "future_date", "table", "lastupdated",
                 'stock_val_change', "location", "scalerevenue", "scalemarketcap", 'accoci',
                 'assets', 'assetsc', 'assetsnc', 'bvps', 'capex', 'cashneq', 'cashnequsd', 'cor', 'consolinc', 'currentratio', 'de',
                 'debt', 'debtc', 'debtnc', 'debtusd', 'deferredrev', 'depamor', 'divyield', 'dps', 'ebit', 'ebitda', 'ebitdamargin',
                 'ebitdausd', 'dimension', 'ebitusd', 'ebt', 'eps', 'epsdil', 'epsusd', 'equity', 'equityusd', 'ev', 'evebit', 'evebitda',
                 'fcf', 'fcfps', 'gp', 'grossmargin', 'intangibles', 'intexp', 'invcap', 'inventory', 'investments', 'investmentsc',
                 'investmentsnc', 'liabilities', 'liabilitiesc', 'liabilitiesnc', 'marketcap', 'ncf', 'ncfbus', 'ncfcommon', 'ncfdebt',
                 'ncfdiv', 'ncff', 'ncfi', 'ncfinv', 'ncfo', 'netinc', 'netinccmn', 'netinccmnusd', 'netincnci', 'netmargin', 'opex',
                 'opinc', 'payables', 'payoutratio', 'pb', 'pe', 'pe1', 'ppnenet', 'price', 'ps', 'ps1', 'receivables', 'retearn',
                 'revenue', 'revenueusd', 'sbcomp', 'sgna', 'sharesbas', 'shareswa', 'shareswadil', 'sps', 'tangibles', 'taxassets',
                 'taxexp', 'taxliabilities', 'tbvps', 'workingcapital', 'close_x', 'dividends', 'purchase_close', "2_day_roi", "3_day_roi",
                 "4_day_roi", "5_day_roi", "f22_ticker", "nasdaq_day_roi"}

    dr_cols = set(df_dropped.columns)

    rem_cols = dr_cols - col_drops
    df_hot_ready = df_dropped[rem_cols].copy()

    df_all_tickers = ticker_service.get_ticker_info()
    col_objs = [c for c in df_hot_ready.columns if str(df_hot_ready[c].dtype) == "object"]
    col_objs = list(set(col_objs) - {"date", "future_date", "purchase_date"})

    df_hotted = ticker_service.make_one_hotted(df=df_hot_ready, df_all_tickers=df_all_tickers, cols=col_objs)

    last_cols = [label_col, "date"]
    rem_cols = list(set(df_hotted.columns) - set(last_cols))

    df_col_ordered = df_hotted[rem_cols + last_cols].copy()
    df_col_ordered = df_col_ordered.copy()

    df_col_ordered.fillna(0, inplace=True)
    df_col_ordered.drop(columns=["buy_sell"], inplace=True)

    df_flight = df_col_ordered.copy()

    return df_flight


def scale(ds_train_x, ds_val_x, ds_test_x, ds_train_y, ds_val_y, ds_test_y):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    train_arr_x = scaler.fit_transform(ds_train_x)
    val_arr_x = scaler.transform(ds_val_x)
    test_arr_x = scaler.transform(ds_test_x)

    label_scaler = MinMaxScaler(feature_range=(-1, 1))
    train_arr_y = label_scaler.fit_transform(ds_train_y)
    val_arr_y = label_scaler.transform(ds_val_y)
    test_arr_y = label_scaler.transform(ds_test_y)

    return train_arr_x, val_arr_x, test_arr_x, train_arr_y, val_arr_y, test_arr_y, scaler, label_scaler


def split_train_val_test(df: pd.DataFrame, label_col: str):
    dates = sorted(df["date"].unique())

    train_frac = 0.7

    total_size = len(dates)
    train_size = int(total_size * train_frac)
    train_dates = dates[:train_size]
    val_test_dates = dates[-train_size:]

    test_frac = .50
    test_size = int(len(val_test_dates) * test_frac)
    val_dates = val_test_dates[:test_size]
    test_dates = val_test_dates[-test_size:]

    df_train = df.loc[df["date"].isin(train_dates)]
    df_val = df.loc[df["date"].isin(val_dates)]
    df_test = df.loc[df["date"].isin(test_dates)]

    df_train.drop(columns=["date"], inplace=True)
    df_val.drop(columns=["date"], inplace=True)
    df_test.drop(columns=["date"], inplace=True)

    df_train_x = df_train.loc[:, df_train.columns != label_col].copy()
    df_val_x = df_val.loc[:, df_val.columns != label_col].copy()
    df_test_x = df_test.loc[:, df_test.columns != label_col].copy()

    df_train_y = df_train.loc[:, df_train.columns == label_col].copy()
    df_val_y = df_val.loc[:, df_val.columns == label_col].copy()
    df_test_y = df_test.loc[:, df_test.columns == label_col].copy()

    logger.info(list(df_test.columns)[-1])

    ds_train_x = df_train_x.astype('float32').to_numpy()
    ds_val_x = df_val_x.astype('float32').to_numpy()
    ds_test_x = df_test_x.astype('float32').to_numpy()

    ds_train_y = df_train_y.astype('float32').to_numpy()
    ds_val_y = df_val_y.astype('float32').to_numpy()
    ds_test_y = df_test_y.astype('float32').to_numpy()

    return ds_train_x, ds_val_x, ds_test_x, ds_train_y, ds_val_y, ds_test_y


def split_all_sequences(train_arr_x, train_arr_y, val_arr_x, val_arr_y, test_arr_x, test_arr_y, n_timesteps: int):
    train_arr = np.append(train_arr_x, train_arr_y, axis=1)
    val_arr = np.append(val_arr_x, val_arr_y, axis=1)
    test_arr = np.append(test_arr_x, test_arr_y, axis=1)

    x_train, y_train = split_sequences(train_arr, n_timesteps)
    x_val, y_val = split_sequences(val_arr, n_timesteps)
    x_test, y_test = split_sequences(test_arr, n_timesteps)

    return x_train, y_train, x_val, y_val, x_test, y_test


def train(x_train, y_train, x_val, y_val, n_epochs: int, batch_size: int, n_features: int, n_timesteps: int, lr: float, patience: int):
    model_1 = MV_LSTM(n_features, n_timesteps)
    loss_fn_1 = torch.nn.MSELoss()
    optimizer_1 = torch.optim.Adam(model_1.parameters(), lr=lr)
    # scheduler_1 = torch.optim.lr_scheduler.StepLR(optimizer_1, step_size=10, gamma=.01)
    scheduler_1 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_1, 'min', patience=patience)

    optimization = Optimization(model_1, loss_fn_1, optimizer_1, scheduler_1)

    optimization.train(x_train, y_train, x_val, y_val, do_teacher_forcing=False, n_epochs=n_epochs, batch_size=batch_size)

    return optimization


def show_roi(roi_results, n_epochs: int):
    all_roi = []
    bh_roi = []
    for key in roi_results.keys():
        all_roi.append(roi_results[key]["pred_roi"])
        bh_roi.append(roi_results[key]["buy_hold"])

    from statistics import mean

    if len(all_roi) > 0 and len(bh_roi) > 0:
        logger.info(f"n_epochs: {n_epochs}; bh mean roi: {mean(bh_roi)}; mean roi: {mean(all_roi)}")

# # create NN
# mv_net = MV_LSTM(n_features, n_timesteps)
# criterion = torch.nn.MSELoss() # reduction='sum' created huge loss value
# optimizer = torch.optim.Adam(mv_net.parameters(), lr=1e-5)

# train_episodes = 120
# batch_size = 16

# mv_net.train()
# X = x_train
# y = y_train
# for t in range(train_episodes):
#     for b in range(0,len(X),batch_size):
#         inpt = X[b:b+batch_size,:,:]
#         target = y[b:b+batch_size]

#         x_batch = torch.tensor(inpt, dtype=torch.float32)
#         y_batch = torch.tensor(target, dtype=torch.float32)

#         mv_net.init_hidden(x_batch.size(0))
#     #    lstm_out, _ = mv_net.l_lstm(x_batch,nnet.hidden)
#     #    lstm_out.contiguous().view(x_batch.size(0),-1)
#         output = mv_net(x_batch)
#         loss = criterion(output.view(-1), y_batch)

#         loss.backward()
#         optimizer.step()
#         optimizer.zero_grad()
#     print('step : ' , t , 'loss : ' , loss.item())