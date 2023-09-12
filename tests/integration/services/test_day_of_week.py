import math
import random
from datetime import datetime, timedelta
from random import shuffle
from statistics import mean, median

import pandas as pd

from ams.services.equities import equity_fundy_service
from ams.services.ticker_service import get_ticker_eod_data

EQUITIES = ['GOOG', 'COST', 'FB', 'MSFT', 'NKE', 'NVDA', 'AAPL']
TOP_2014 = ['NFX', 'GMCR', 'LUV', 'EA', 'NBR', 'AA', 'WMB', 'UA', 'AVGO', 'MNK']
TOP_2015 = ['NFX', 'AMZN', 'ATVI', 'NVDA', 'ALTUS', 'HRL', 'VRSN', 'RAI', 'SBUX', 'FSLR']
TOP_2016 = ['NVDA', 'FCX', 'NEM', 'SEP', 'AMAT', 'PWR', 'CMA', 'MLM', 'HAL', 'OKE']
TOP_EQ = TOP_2016
EXAMINE_YEAR = 2018

YEAR_SPAN = 1
FROM_YEAR = f'{EXAMINE_YEAR}-01-01'
TO_YEAR = f'{EXAMINE_YEAR + YEAR_SPAN}-01-01'
EXAMINE_AFTER = f'{EXAMINE_YEAR + 1}-01-01'
HOLD_DAYS = 1


def test_get_roi_for_any_day_of_week():
    # Arrange
    all_ticks = []
    for e in EQUITIES:
        df = get_ticker_eod_data(e)
        # df = df[df['date'].str[:4] == str(EXAMINE_YEAR)]

        if df is None:
            continue
        # Drop rows with date null
        df = df.dropna(subset=['date'])
        df['date'] = pd.to_datetime(df['date'], format="%Y-%m-%d")

        df['close-prev'] = df['close'].shift(HOLD_DAYS)
        df['roi'] = (df['close'] - df['close-prev']) / df['close-prev']

        df['prev-1-days'] = df['close'].shift(1)
        df['prev-2-days'] = df['close'].shift(2)
        df['roi-prev'] = (df['prev-2-days'] - df['prev-1-days']) / df['prev-1-days']
        df['did_fall'] = df['roi-prev'] < 0
        # df = df[df['did_fall'] == True]

        df['date_of_week'] = df['date'].dt.day_name()

        df_g = df.groupby(['date_of_week', 'ticker']).agg({'roi': 'mean'})
        df = df_g.reset_index()

        all_ticks.append(df)

    df = pd.concat(all_ticks)

    print(f'Total rows: {df.shape[0]}')

    df_g = df.groupby(['date_of_week']).agg({'roi': 'mean'})
    print(df_g.head())


def test_get_roi_for_day_of_week():
    # Arrange
    all_ticks = []
    for e in TOP_EQ:
        df = get_ticker_eod_data(e)
        if FROM_YEAR:
            df = df[df['date'] >= FROM_YEAR]
        if TO_YEAR:
            df = df[df['date'] < TO_YEAR]
        if df is None:
            continue
        df = df.dropna(subset=['date', 'close'])

        df['prev'] = df['open'].shift(HOLD_DAYS)
        df = df.dropna(subset=['close', 'prev'])

        df['roi'] = (df['close'] - df['prev']) / df['prev']

        df['date_of_week'] = pd.to_datetime(df['date']).dt.day_name()

        df = df[df['date_of_week'] == 'Monday']

        all_ticks.append(df)

    df = pd.concat(all_ticks)

    print(f'Total rows: {df.shape[0]}')

    num_years = 10

    roi = shuffle_and_calc_roi(df)

    print(f'ROI: {roi * 100:.3f}%')


def shuffle_and_calc_roi(df: pd.DataFrame):
    df = df.sample(frac=1).reset_index(drop=True)

    init_invest = 100
    current_amt = init_invest
    for ndx, row in df.iterrows():
        roi = row['roi']
        current_amt = current_amt * (1 + roi)
        if current_amt <= 0:
            break

    # print(f'Current amount: {current_amt}')

    roi = (current_amt - init_invest) / init_invest
    print(f'ROI: {roi}')

    return roi


def test_ef():
    df_equity_funds = equity_fundy_service.get_equity_fundies()

    print(df_equity_funds.head())


def test_tuesday():
    # Arrange
    equities = ['AU', 'GFI', 'GRD', 'FNV']  # , 'JWN', 'COST', 'DELL', 'DDOG', 'F', 'FANG', 'FSLY', 'GME', 'HPE', 'IBM', 'INTC', 'JPM', 'KO', 'LMT', 'MCD', 'MSFT', 'NKE', 'NVDA', 'PFE', 'PG', 'PLTR', 'PYPL', 'QCOM', 'SBUX', 'TSLA', 'TWTR', 'V', 'WMT', 'XOM']
    equities = ['AG', 'SVM', 'EXK', 'MUX']
    equities = ['TGB', 'ERO', 'SCCO', 'HBM']
    equities = ['CLF', 'BHP', 'MSB', 'VALE']
    # equities = ['AGRO', 'BG', 'SEB', 'TAGS']
    # equities = ['CENX', 'RIO', 'VEDL', 'AA']

    # Gold stocks in 2014
    # equities = ['ABX', 'AU', 'AUY', 'GG', 'NEM', 'RGLD'] # 11%

    # silver stock symbols 2014
    equities = ['SILV', 'ASM', 'AG', 'HL', 'SVM', 'PAAS', 'CDE', 'FSM', 'EXK'] # 14.63

    for y in range(2014, 2021):
        calc(year_start=y, equities=equities)


def calc(year_start: int, equities: list):
    all_ticks = []
    for e in equities:
        df = get_ticker_eod_data(e)
        if df is None:
            continue

        # filter df between 2 dates
        df = df[df['date'] >= f'{year_start}-01-01'].copy()
        df = df[df['date'] <= f'{year_start + 2}-01-01'].copy()

        df['close-prev'] = df['close'].shift(2)
        df['roi'] = (df['close'] - df['close-prev']) / df['close-prev']

        df.dropna(subset=['roi', 'date'], inplace=True)
        if df is None:
            continue

        df['date_of_week'] = pd.to_datetime(df['date']).dt.day_name()

        df = df[df['date_of_week'] == 'Wednesday']

        all_ticks.append(df)

    df = pd.concat(all_ticks)

    print(f'Total rows: {df.shape[0]}')

    init_amount = 100
    curr_amt = init_amount
    df = df.sample(frac=1)
    total_rows = df.shape[0]
    for _, r in df.iterrows():
        roi = r['roi']
        curr_amt = curr_amt * (1 + roi)
        if curr_amt <= 0:
            raise Exception('You\'re now broke!')
    total_ret = curr_amt - init_amount
    years = total_rows / 52

    if curr_amt > init_amount:
        ann_ror = ((1 + total_ret) ** (1 / years)) - 1
        print(f'Num years: {years}: Annualized ROR: {(ann_ror * 100)}%')

    print(f'Year start: {year_start}: Total return: {total_ret:.2f}')


def test_chart():
    import matplotlib.pyplot as plt

    equities = ['FSLR']
        # , 'JWN', 'COST', 'DELL', 'DDOG', 'F', 'FANG', 'FSLY', 'GME', 'HPE', 'IBM',
        #         'INTC', 'JPM', 'KO', 'LMT', 'MCD', 'MSFT', 'NKE', 'NVDA', 'PFE', 'PG', 'PLTR', 'PYPL',
        #         'QCOM', 'SBUX', 'TSLA', 'TWTR', 'V', 'WMT', 'XOM']

    # return
    shuffle(equities)
    start_date = 2017
    num_months = 15
    moving_average_span = 19
    num_samples = 10000
    inv_total = 100
    all_roi_vals = []

    for ndx in range(1):
        eq = equities[random.choice(range(len(equities)))]
        df = get_ticker_eod_data(eq)
        df.sort_values('date', inplace=True)

        if df.shape[0] == 0:
            continue

        first_date = df['date'].iloc[:1].values.tolist()[0]
        first_dt = datetime.strptime(first_date, '%Y-%m-%d')
        start_dt = first_dt + timedelta(days=random.choice(range(365 * 6)))

        start_val_dt = start_dt + timedelta(days=(num_months * 30) - int((num_months / 3) * 30))
        end_val_dt = start_dt + timedelta(days=(num_months * 30))

        df = df[(df['date'] >= start_dt.strftime('%Y-%m-%d')) & (df['date'] <= start_val_dt.strftime('%Y-%m-%d'))]
        df['date'] = pd.to_datetime(df['date'])

        # Create a line chart
        label = 'Close - Diff 20DMA'
        plt.plot(df['date'], df['close'], label='ClosePrice')
        # plt.plot(df['date'], df[diff_col], label=label)
        #
        # Add labels and title
        plt.xlabel('Date')
        plt.ylabel(label)
        plt.title(f'{eq} {label}')

        # Add legend
        plt.legend()

        # Display the chart
        plt.show()


def test_show_nice_chart():
    from mplchart.chart import Chart
    from mplchart.helper import get_prices
    from mplchart.primitives import Candlesticks, Volume
    from mplchart.indicators import ROC, SMA, EMA, RSI, MACD

    ticker = 'AAPL'
    freq = 'daily'
    df = get_ticker_eod_data(ticker)
    df.sort_values('date', inplace=True)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    # prices = get_prices(ticker, freq=freq)
    prices = df[['open', 'high', 'low', 'close', 'volume']]

    max_bars = 250

    indicators = [
        Candlesticks(), SMA(50), SMA(200), Volume(),
        RSI(),
        MACD(),
    ]

    chart = Chart(title=ticker, max_bars=max_bars)
    chart.plot(prices, indicators)
    chart.show()
