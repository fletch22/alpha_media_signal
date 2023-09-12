import statistics
import time

import pandas
import matplotlib.pyplot as plt
from ams.config import logger_factory
from ams.services import ticker_service as ts
from ams.utils import ticker_utils
import pandas as pd

logger = logger_factory.create(__name__)


def process():
    all_tickers = ['GOOGL', 'IBM', 'NVDA', 'ADBE', 'F', 'MSFT']  # "GOOGL"

    start_dt_str = None  # "2000-01-01"
    end_dt_str = None  # "2023-06-01"

    start_dt_str = None # "2017-01-01"
    end_dt_str = "2024-06-01"

    date_col = 'date'
    val_col = 'close'

    all_roi = []
    for ticker in all_tickers:
        df = ts.get_ticker_eod_data(ticker=ticker).sort_values(date_col)

        if df is None:
            continue

        df[date_col] = pd.to_datetime(df[date_col], format="%Y-%m-%d")

        print(f"Ticker: {ticker}; Num rows: {len(df)}")
        if start_dt_str is not None:
            df = df.loc[df[date_col] > start_dt_str].copy()
        if end_dt_str is not None:
            df = df.loc[df[date_col] < end_dt_str].copy()
        print(f"Ticker: {ticker}; Num rows: {len(df)}")



        # last_val_col = f'last_{val_col}'
        # df[last_val_col] = df[val_col].shift(1)
        #
        # val_pct_col = f'{val_col}_pct'
        # df[val_pct_col] = ((df[val_col] - df[last_val_col]) / df[last_val_col]) * 100
        #
        # # show_stock_price(df=df, ticker=ticker, col_x=date_col, col_y=val_pct_col)
        #
        # first_price = df.iloc[1][val_col]
        # first_date = df.iloc[0][date_col]
        # last_price = df.iloc[-1][val_col]
        # last_date = df.iloc[-1][date_col]

        logger.info(f"{first_date}: {first_price}; last_date: {last_date}; {last_price}")

        # roi = pul_sel_thresh(df=df, val_col=val_col, last_price=last_price, first_price=first_price)
        # roi = move_avg(df=df, last_price=last_price, first_price=first_price)
        roi = move_avg_basement_living(df=df, last_price=last_price, first_price=first_price)
        all_roi.append(roi)

    print(f'Mean roi: {statistics.mean(all_roi):.3f}; Median roi: {statistics.median(all_roi):.3f}')


def move_avg_basement_living(df: pandas.DataFrame, last_price: float, first_price: float):
    holding = False
    min_num_d_under_ma = 3
    principle = 100000
    cash = principle
    num_purchases = 0
    pur_price = None
    val_col = 'close'

    df = ticker_utils.add_simple_moving_averages(df=df, target_column=val_col, windows=[20])

    consec_days_under = 0
    for ndx, row in df.iterrows():
        close = row[val_col]

        if not holding and consec_days_under >= min_num_d_under_ma:
            pur_price = close
            cash = cash - pur_price
            holding = True
            continue

        if holding:
            roi = (close - pur_price) / pur_price
            if roi > .01:
                cash += close - pur_price
                holding = False
                num_purchases += 1

        if row["close_SMA_20"] > close:
            consec_days_under += 1
        else:
            consec_days_under = 0

    if holding:
        cash += close - pur_price
        if cash <= 0:
            raise Exception('You broke as a dog.')

    alpha_roi = (cash - principle) / principle
    buy_hold_roi = (last_price - first_price) / first_price

    logger.info(f"buy_hold roi: {buy_hold_roi:.3f}: buy_dip roi: {alpha_roi:.3f}; num_purchases: {num_purchases}")

    return alpha_roi


def show_stock_price(df, ticker, col_y, col_x):
    plt.title(ticker)
    plt.plot(df[col_x], df[col_y])
    plt.xlabel(col_x)
    plt.ylabel(col_y)

    plt.xticks(rotation=45)
    plt.show()


if __name__ == '__main__':
    process()
