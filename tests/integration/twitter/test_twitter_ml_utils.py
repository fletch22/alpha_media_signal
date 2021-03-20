import pandas as pd

from ams.config import logger_factory
from ams.twitter import twitter_ml_utils as tmu
from ams.twitter.twitter_ml import get_tweet_data
from ams.utils.date_utils import get_next_market_date

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


def test_tw_():
    df = get_tweet_data()

    max_dt_str = df["date"].max()
    min_dt_str = "2020-08-10"

    all_dts = []
    current_dt_str = min_dt_str
    while current_dt_str <= max_dt_str:
        all_dts.append(current_dt_str)
        current_dt_str = get_next_market_date(current_dt_str)

    df_all_dts = pd.DataFrame(all_dts, columns=["date"])

    df_g = df.groupby(by=["date"]).size().reset_index(name="counts")
    df_joined = pd.merge(df_g, df_all_dts, how="left", on="date")
    df_joined.loc[df_joined["counts"].isnull(), "counts"] = 0

    df_joined.sort_values("date", inplace=True)

    print(df_joined.head(250))

    return

    logger.info(f"Prediction start from {min_dt_str} \n")
    logger.info(f"Prediction end at {max_dt_str} \n")

    datenums = all_dts
    value_raw = np.array(df['counts'])

    plt.figure()
    plt.subplots_adjust(bottom=0.2)
    plt.xticks(rotation=60)
    ax = plt.gca()
    ax.xaxis_date()
    plt.xlabel("date range")
    plt.ylabel("num tickers with tweets")
    plt.title("Num tickers with Tweets on Date")
    plt.grid()
    plt.plot(datenums, value_raw, linestyle='-', marker='o', markersize=5, color='r', linewidth=2, label="raw temp C")
    # plt.plot(datenums, value_predict, linestyle='-', marker='o', markersize=5, color='b', linewidth=2, label="predict temp C")
    # plt.legend(loc="best")
    #
    # plt.show()