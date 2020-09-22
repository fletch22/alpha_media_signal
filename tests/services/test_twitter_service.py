import re

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
