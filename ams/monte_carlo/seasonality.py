import pandas as pd

from ams.services import ticker_service as ts
from learntools.time_series.utils import plot_periodogram, seasonal_plot

def process():
    ticker = "gld"  # "GOOGL"

    df = ts.get_ticker_eod_data(ticker=ticker).sort_values("date")

    # start_dt_str = "2020-01-01"
    # end_dt_str = "2021-01-01"
    # df = df[(df["date"] >= start_dt_str) & (df["date"] <= end_dt_str)]

    print(df.shape[0])

    df = df.dropna(subset=['close'])
    df = df[['close', 'date']]
    # df['close'] = 1.0

    df['date'] = pd.to_datetime(df['date'])
    df['date'] = df['date'].dt.to_period('D')
    df = df.set_index(['date']).sort_index()

    # print(df.groupby('date').mean().squeeze())
    # res = df.groupby('date').mean().squeeze().loc['2020']
    # print(res.head())

    print(list(df.columns))

    average_close = (
        df
        .groupby('date').mean()
        .squeeze()
    )

    X = average_close.to_frame()
    # X = average_close
    X["week"] = X.index.week
    X["day"] = X.index.dayofweek
    ax = seasonal_plot(X, y='close', period='week', freq='day')
    import matplotlib.pyplot as plt
    plt.show()


    plot_periodogram(average_close)
    plt.show()

    #
    # print(df.head())

    # df.index.to_period('M')

if __name__ == '__main__':
    process()