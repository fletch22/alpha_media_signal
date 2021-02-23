from ams.config import logger_factory
from ams.services import ticker_service

logger = logger_factory.create(__name__)

def test_get_equity_on_dates():
    # Arrange
    date_strs = ["2021-01-04", "2021-01-11"]
    # Act
    df = ticker_service.get_equity_on_dates(ticker="AAPL", date_strs=date_strs)

    # Assert
    logger.info(df["future_open"].to_list())