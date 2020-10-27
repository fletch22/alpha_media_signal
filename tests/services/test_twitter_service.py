from pathlib import Path

from ams.DateRange import DateRange
from ams.config import constants
from ams.services import twitter_service


def test_tuples():
    tickers_tuples = twitter_service.get_ticker_searchable_tuples()

    print(len(list(set(tickers_tuples))))

    print(f"Num ticker tuples: {len(tickers_tuples)}")
    sample = tickers_tuples[:4]

    print(sample)


def test_get_cashtags():
    # Arrange
    ticker = "AAPL"
    text = "Buy buy buy $AAPL stock!"

    # Act
    # result = re.search(f'\${ticker}', text)
    index = text.find(f'${ticker}')

    # Assert
    # assert(result is not None)
    print(index)


def test_twitter_service():
    from ams.services import file_services
    # Arrange
    query = "AAPL"
    date_range = DateRange.from_date_strings("2020-10-04", "2020-10-06")
    output_path = file_services.create_unique_filename(constants.TWITTER_TRASH_OUTPUT, prefix="search_twitter_test")

    # Act
    twitter_service.search_standard(query=query, tweet_raw_output_path=output_path, date_range=date_range)
    # Assert

def test_dict():
    ticker = "AAA"
    group_preds = {}
    info = {}
    group_preds[ticker] = info
    info["foo"] = 1

    print(group_preds)