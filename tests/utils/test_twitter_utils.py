import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

from ams.config import constants
from ams.config.constants import ensure_dir
from ams.utils import twitter_utils

tweet_json = """{"version": "0.9.2", "f22_ticker": "AACG", "tweet": {"created_at": "Thu Feb 04 23:24:52 +0000 2021", "id": 1357470207548227589, "id_str": "1357470207548227589", "text": "$AACG ATA Creativity Global - A Top stock up 2034% from low. Close: 14.30 VolvsAvg: 9.98 Liq: $3546M https://t.co/G6KiEaz77f", "truncated": false, "entities": {"hashtags": [], "symbols": [{"text": "AACG", "indices": [0, 5]}], "user_mentions": [], "urls": [{"url": "https://t.co/G6KiEaz77f", "expanded_url": "http://stockcharts.com/h-sc/ui?s=AACG&p=D&yr=1&mn=0&dy=0&id=p14565816076", "display_url": "stockcharts.com/h-sc/ui?s=AACG\u2026", "indices": [101, 124]}]}, "metadata": {"iso_language_code": "en", "result_type": "recent"}, "source": "<a href=\"http://www.google.com\" rel=\"nofollow\">StockPoster</a>", "in_reply_to_status_id": null, "in_reply_to_status_id_str": null, "in_reply_to_user_id": null, "in_reply_to_user_id_str": null, "in_reply_to_screen_name": null, "user": {"id": 914214567152160768, "id_str": "914214567152160768", "name": "Stock Market Genius", "screen_name": "stockmktgenius", "location": "Ohio, USA", "description": "AI scanner for stock market leaders. New posts Mon-Sat. Locked accounts will be blocked. Stocks Market Money Wealth. $AMAT $NTGR $GPRO $NET #Bitcoin", "url": null, "entities": {"description": {"urls": []}}, "protected": false, "followers_count": 12692, "friends_count": 26, "listed_count": 117, "created_at": "Sat Sep 30 19:45:12 +0000 2017", "favourites_count": 4875, "utc_offset": null, "time_zone": null, "geo_enabled": true, "verified": false, "statuses_count": 101776, "lang": null, "contributors_enabled": false, "is_translator": false, "is_translation_enabled": false, "profile_background_color": "F5F8FA", "profile_background_image_url": null, "profile_background_image_url_https": null, "profile_background_tile": false, "profile_image_url": "http://pbs.twimg.com/profile_images/1102029146404859905/QGwQ63r0_normal.jpg", "profile_image_url_https": "https://pbs.twimg.com/profile_images/1102029146404859905/QGwQ63r0_normal.jpg", "profile_banner_url": "https://pbs.twimg.com/profile_banners/914214567152160768/1582256159", "profile_link_color": "1DA1F2", "profile_sidebar_border_color": "C0DEED", "profile_sidebar_fill_color": "DDEEF6", "profile_text_color": "333333", "profile_use_background_image": true, "has_extended_profile": false, "default_profile": true, "default_profile_image": false, "following": null, "follow_request_sent": null, "notifications": null, "translator_type": "none"}, "geo": null, "coordinates": null, "place": null, "contributors": null, "is_quote_status": false, "retweet_count": 0, "favorite_count": 0, "favorited": false, "retweeted": false, "possibly_sensitive": false, "lang": "en"}}"""


def test_find_youngest_record():
    # Arrange
    sample_file_path = Path(constants.TESTS_RESOURCES, "sample_raw_drop.txt")

    with TemporaryDirectory() as td:
        samp_dest_path = Path(td, "raw_drop", "main", sample_file_path.name)
        ensure_dir(samp_dest_path.parent)
        shutil.copy(str(sample_file_path), str(samp_dest_path))

        max_end_drop_dt_str = twitter_utils.get_youngest_raw_textfile_tweet(source_path=samp_dest_path.parent)

        # Assert
        assert (max_end_drop_dt_str == "2021-02-04")


def test_extract_raw_date_str():
    # Arrange
    # Act
    raw_date_str = twitter_utils.extract_raw_date_from_tweet(tweet_json)

    # Assert
    assert (raw_date_str == "Thu Feb 04 23:24:52 +0000 2021")


def test_get_time_from_json():
    # Arrange
    # Act
    date_str = twitter_utils.get_time_from_json(tweet_json)

    # Assert
    assert ("2021-02-04" == date_str)


def test_get_oldest_tweet():
    youngest_tweet_dt_str = twitter_utils.get_youngest_tweet_date_in_system()

    print(youngest_tweet_dt_str)

def test_get_oldest_tweet_2():
    youngest_tweet_dt_str = twitter_utils.get_youngest_tweet_date_in_system_2()

    print(youngest_tweet_dt_str)

