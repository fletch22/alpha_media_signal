import pandas as pd

from ams.config import logger_factory
from ams.twitter import twitter_ml_utils

logger = logger_factory.create(__name__)


def test_add_future_date_for_nan():
    # Arrange
    df = pd.DataFrame([
        {'date': '2021-03-12', 'future_date': None},
    ])
    # Act
    df_new = twitter_ml_utils.add_future_date_for_nan(df=df, num_days_in_future=2)

    # Assert
    logger.info(df_new["future_date"].to_list())