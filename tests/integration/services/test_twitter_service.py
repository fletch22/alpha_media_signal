from ams.services import twitter_service as ts
from ams.config import logger_factory

logger = logger_factory.create(__name__)


def test_get_search_date_range():
    # Arrange
    # Act
    date_range = ts.get_search_date_range()

    # Assert
    logger.info(f"from {date_range.from_date}")
    logger.info(f"to {date_range.to_date}")