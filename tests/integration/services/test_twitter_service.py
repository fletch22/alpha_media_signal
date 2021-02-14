from ams.services import twitter_service as ts


def test_get_search_date_range():
    # Arrange
    # Act
    date_range = ts.get_search_date_range()

    # Assert
    print(f"from {date_range.from_date}")
    print(f"to {date_range.to_date}")