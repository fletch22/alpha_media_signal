import finnhub
from finnhub import FinnhubAPIException
from retry_decorator import retry

from ams.config import constants, logger_factory

logger = logger_factory.create(__name__)

last_call = None


# define Python user-defined exceptions
class ApiLimitReachedException(Exception):
    """Base class for other exceptions"""
    pass


@retry(ApiLimitReachedException, tries=3, timeout_secs=62)
def get_raw_stock_price(ticker: str):
    api_key = constants.FINNHUB_API_KEY

    finnhub_client = finnhub.Client(api_key=api_key)
    response = None
    try:
        response = finnhub_client.quote(ticker)
    except FinnhubAPIException as fae:
        if "API limit reached" in fae.message:
            logger.error("API Limit reached. Might retry.")
            raise ApiLimitReachedException()

    return response


if __name__ == "__main__":
    ticker = "AAPL"

    for i in range(62):
        print(get_raw_stock_price(ticker=ticker))
