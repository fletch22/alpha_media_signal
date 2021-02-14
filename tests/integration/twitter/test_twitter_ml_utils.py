from ams.twitter import twitter_ml_utils as tmu


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
    print(tickers)