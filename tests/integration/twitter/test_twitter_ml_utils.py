import pandas as pd

from ams.config import logger_factory
from ams.twitter import twitter_ml_utils as tmu
from ams.twitter.twitter_ml import get_tweet_data
from ams.twitter.twitter_ml_utils import get_next_market_date
from ams.utils import date_utils
from ams.utils.date_utils import is_stock_market_closed

logger = logger_factory.create(__name__)


def test_get_real_predictions():
    # Arrange
    sample_size = 10
    purchase_date_str = "2021-02-04"

    # Act
    tickers = tmu.get_real_predictions(sample_size=sample_size,
                                       purchase_date_str=purchase_date_str,
                                       num_hold_days=2,
                                       min_price=5)

    # Assert
    assert (len(tickers) == sample_size)


def ensure_market_date(row: pd.Series):
    date_str = row["date"]
    dt = date_utils.parse_std_datestring(date_str)
    is_closed_date = is_stock_market_closed(dt=dt)
    if is_closed_date:
        date_str = get_next_market_date(date_str, -1)

    return date_str


def adjust_tweet_dt_to_market_dt(df: pd.DataFrame):
    df_grouped = df.groupby(by=["date"])

    all_dfs = []
    for date_str, df_g in df_grouped:
        df_g.loc[:, "prev_market_date"] = df_g.apply(ensure_market_date, axis=1)
        all_dfs.append(df_g)

    df = pd.concat(all_dfs, axis=0)

    return df


def test_assign_tweets_to_previous_date():
    df = get_tweet_data()
    df = df.sample(frac=.1)

    df = adjust_tweet_dt_to_market_dt(df=df)

    print(df[["date", "prev_market_date"]].head(5))
