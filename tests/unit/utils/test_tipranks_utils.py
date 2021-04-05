import pandas as pd

from ams.config import logger_factory
from ams.services import ticker_service
from ams.utils import tipranks_utils

pd.set_option('display.max_rows', 5000)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

logger = logger_factory.create(__name__)

def test_agg_tip_ranks():
    # Arrange
    date_str = "2020-07-01"

    df_apple = ticker_service.get_ticker_eod_data("AAPL")
    df_nvda = ticker_service.get_ticker_eod_data("NVDA")

    df_stocks = pd.concat([df_apple, df_nvda], axis=0)

    df_stocks = df_stocks[df_stocks["date"] > date_str]

    # Act
    df = tipranks_utils.agg_tipranks(df_stocks=df_stocks)

    df = df[["ticker", "date", "rating_momentum", "tr_rating_roi", "rank", "rating", "target_price"]]

    # Assert
    logger.info(df.head(1000))