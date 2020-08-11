import json
import urllib
from pathlib import Path
from typing import Dict, List, Callable, Union

from searchtweets import load_credentials, gen_rule_payload, ResultStream

from alpha_media_signal import config
from alpha_media_signal.DateRange import DateRange
from alpha_media_signal.config import logger_factory
from alpha_media_signal.services import file_services
from alpha_media_signal.utils import date_utils

logger = logger_factory.create(__name__)


def _get_credentials():
    yaml_key = 'search_tweets_fullarchive_development'

    return load_credentials(filename=config.TWITTER_CREDS_PATH,
                            yaml_key=yaml_key,
                            env_overwrite=False)


def search(query: str, date_range: DateRange = None):
    # query_esc = query.replace('$', '\$')
    query_esc = f'NVDA lang:en' #  urllib.parse.urlencode(query) # query.replace('$', '\$')

    kwargs = {}
    if date_range is not None:
        from_date = date_utils.get_standard_ymd_format(date_range.from_date)
        to_date = date_utils.get_standard_ymd_format(date_range.to_date)
        kwargs = dict(from_date=from_date, to_date=to_date)

    rule = gen_rule_payload(pt_rule=query_esc, results_per_call=10, **kwargs)  # testing with a sandbox account

    dev_search_args = _get_credentials()

    rs = ResultStream(rule_payload=rule,
                      max_results=500,
                      max_pages=1,
                      **dev_search_args)

    tweet_generator = rs.stream()
    cache_length = 9

    tweets = []
    tweet_raw_output_path = file_services.create_unique_file_system_name(config.TWITTER_OUTPUT_RAW_PATH, prefix=config.TWITTER_RAW_TWEETS_PREFIX, extension='txt')
    for tweet in tweet_generator:
        tweets.append(tweet)
        if len(tweets) >= cache_length:
            print(f'Writing {len(tweets)} tweets.')
            append_tweets_to_output_file(output_path=tweet_raw_output_path, tweets=tweets)
            tweets = []

    if len(tweets) > 0:
        append_tweets_to_output_file(output_path=tweet_raw_output_path, tweets=tweets)

    return tweet_raw_output_path


def append_tweets_to_output_file(output_path: Path, tweets: List[Dict]):
    json_lines = [json.dumps(t) for t in tweets]
    print(f'Writing to {output_path}.')
    with open(str(output_path), 'a+') as f:
        json_lines_nl = [f'{j}\n' for j in json_lines]
        f.writelines(json_lines_nl)
