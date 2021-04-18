from ams.config import logger_factory
from ams.utils import twitter_utils

logger = logger_factory.create(__name__)


def test_get_oldest_tweet():
    youngest_tweet_dt_str = twitter_utils.get_youngest_tweet_date_in_system()

    logger.info(youngest_tweet_dt_str)
