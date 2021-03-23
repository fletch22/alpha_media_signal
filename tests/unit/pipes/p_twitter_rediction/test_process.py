from datetime import datetime, timezone

from ams.pipes.p_twitter_reduction import process as tr_process


def test_minutes_from_tweet_eod():
    # Arrange
    dt = datetime(2021, 1, 1, 5)
    timestamp = dt.replace(tzinfo=timezone.utc).timestamp()

    # Act
    minutes = tr_process.minutes_from_tweet_eod(created_at_timestamp=timestamp, applied_date_str="2020-12-31")

    # Assert
    assert minutes == 1440