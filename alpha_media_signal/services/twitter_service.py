import json
import operator
import re
import time
import urllib
from pathlib import Path
from typing import Dict, List

import pandas as pd
import requests
from searchtweets import load_credentials, gen_rule_payload, ResultStream

from alpha_media_signal import utils
from alpha_media_signal.DateRange import DateRange
from alpha_media_signal.config import logger_factory, constants
from alpha_media_signal.services import file_services
from alpha_media_signal.services.equities.ExchangeType import ExchangeType
from alpha_media_signal.services.equities.TickerService import TickerService
from alpha_media_signal.steaming.BearerTokenAuth import BearerTokenAuth
from alpha_media_signal.utils import date_utils, equity_utils

logger = logger_factory.create(__name__)


def _get_credentials():
    yaml_key = 'search_tweets_fullarchive_development'

    return load_credentials(filename=constants.TWITTER_CREDS_PATH,
                            yaml_key=yaml_key,
                            env_overwrite=False)


def search(query: str, date_range: DateRange = None):
    language = 'lang:en'
    query_esc = f'{query} {language}'

    print(query_esc)

    kwargs = {}
    if date_range is not None:
        from_date = date_utils.get_standard_ymd_format(date_range.from_date)
        to_date = date_utils.get_standard_ymd_format(date_range.to_date)
        kwargs = dict(from_date=from_date, to_date=to_date)

    rule = gen_rule_payload(pt_rule=query_esc, results_per_call=500, **kwargs)

    dev_search_args = _get_credentials()

    rs = ResultStream(rule_payload=rule,
                      max_results=270000,
                      max_pages=1,
                      **dev_search_args)

    return process_tweets_stream(rs)


def process_tweets_stream(rs):
    tweet_generator = rs.stream()
    cache_length = 9
    count = 0
    tweets = []
    tweet_raw_output_path = file_services.create_unique_file_system_name(constants.TWITTER_OUTPUT_RAW_PATH, prefix=constants.TWITTER_RAW_TWEETS_PREFIX, extension='txt')
    for tweet in tweet_generator:
        tweets.append(tweet)
        if len(tweets) >= cache_length:
            append_tweets_to_output_file(output_path=tweet_raw_output_path, tweets=tweets)
            count += len(tweets)
            tweets = []
    if len(tweets) > 0:
        append_tweets_to_output_file(output_path=tweet_raw_output_path, tweets=tweets)
        count += len(tweets)
    return tweet_raw_output_path if count > 0 else None, count


def append_tweets_to_output_file(output_path: Path, tweets: List[Dict]):
    json_lines = [json.dumps(t) for t in tweets]
    if len(json_lines):
        with open(str(output_path), 'a+') as f:
            json_lines_nl = [f'{j}\n' for j in json_lines]
            f.writelines(json_lines_nl)


def search_standard(query: str, tweet_raw_output_path: Path, date_range: DateRange):
    language = 'lang:en'
    query_esc = urllib.parse.quote(f'{query} {language}')

    endpoint = constants.STANDARD_CREDS.endpoint

    bearer_token_auth = BearerTokenAuth(constants.STANDARD_CREDS.api_key, constants.STANDARD_CREDS.api_secret_key)

    start_date_str = date_utils.get_standard_ymd_format(date_range.from_date)
    end_date_str = date_utils.get_standard_ymd_format(date_range.to_date)

    # "https://api.twitter.com/1.1/search/tweets.json?q=Vodafone VOD lang:en&result_type=recent&since=2020-08-10&until=2020-08-11"
    url = f'{endpoint}?q={query_esc}&result_type=recent&since={start_date_str}&until={end_date_str}'

    next_results_token = 'next_results'
    status_key = 'statuses'

    count = 0
    while True:
        try:
            response = requests.get(url, auth=bearer_token_auth)

            search_results = response.json()
        except Exception:
            time.sleep(60)
            continue

        if 'errors' in search_results:
            # '{'errors': [{'message': 'Rate limit exceeded', 'code': 88}]}
            print(search_results['errors'])
            print('Pausing...')
            time.sleep(60)
            continue
        if status_key in search_results:
            tweets = search_results[status_key]
            count += len(tweets)
            append_tweets_to_output_file(output_path=tweet_raw_output_path, tweets=tweets)
        else:
            break
        if count > 5000:
            break
        search_metadata = search_results["search_metadata"]
        if next_results_token in search_metadata:
            query = search_metadata[next_results_token]
            url = f'{endpoint}{query}'
        else:
            break

    return count


def fix_common(name: str, ticker: str, common_words: List[str]):
    if name in common_words and ticker in common_words:
        name = f'{name} stock'

    return name


def create_colloquial_twitter_stock_search_tokens():
    ticker_tuples = TickerService.get_list_of_tickers_by_exchange(cols=['ticker', 'name'], exchange_types=[ExchangeType.NASDAQ])

    ticker_tuples.sort(key=operator.itemgetter(0))

    common_words = utils.load_common_words()

    ticker_data = []
    for t in ticker_tuples:
        ticker = t[0]
        name = t[1]

        name = equity_utils.convert_equity_name_to_common(name)
        name = fix_common(name, ticker, common_words=common_words)
        ticker_data.append({'name': name, 'ticker': ticker})

    df = pd.DataFrame(ticker_data)
    df.to_csv(constants.TICKER_NAME_SEARCHABLE_PATH, index=False)


def remove_items(ticker_tuples, ticker_to_flag: str, delete_before: bool):
    num = 0
    for num, tuple in enumerate(ticker_tuples):
        ticker = tuple[0]
        if ticker == ticker_to_flag:
            break

    return ticker_tuples[num + 1:] if delete_before is True else ticker_tuples[:num + 1]


def get_ticker_searchable_tuples():
    df = pd.read_csv(constants.TICKER_NAME_SEARCHABLE_PATH)
    ticker_tuples = list(map(tuple, df[['ticker', 'name']].to_numpy()))
    return ticker_tuples


def save_nasdaq_tweets():
    # Arrange
    from_date = date_utils.parse_std_datestring('2020-08-17')
    to_date = date_utils.parse_std_datestring('2020-08-18')
    date_range = DateRange(from_date=from_date, to_date=to_date)

    tweet_raw_output_path = file_services.create_unique_file_system_name(constants.TWITTER_OUTPUT_RAW_PATH, prefix=constants.TWITTER_RAW_TWEETS_PREFIX, extension='txt')
    print(f'Output path: {str(tweet_raw_output_path)}')

    ticker_tuples = get_ticker_searchable_tuples()
    # ticker_tuples = remove_items(ticker_tuples=ticker_tuples, ticker_to_flag='PALM', delete_before=True)

    # Act
    count = 0
    for t in ticker_tuples:
        ticker = t[0]
        name = t[1]

        print(f'{ticker}: {name}')
        # continue

        query = f'\"{ticker}\" {name}'

        count += search_standard(query=query, tweet_raw_output_path=tweet_raw_output_path, date_range=date_range)

    print(f'Number of tweets: {count}.')


def get_cashtag_info(ticker: str, has_cashtag: bool) -> Dict:
    return {"ticker": ticker, 'has_cashtag': has_cashtag}


def find_cashtag(raw_line: str, search_tuples: List) -> List[str]:
    tweet = json.loads(raw_line)
    text = tweet['text']
    cashtags_stock = []
    for s in search_tuples:
        ticker = s[0].strip()
        name = s[1].strip()

        if re.search(f'${ticker}', text) and re.search(name, text, re.IGNORECASE):
            cashtags_stock.append(get_cashtag_info(ticker=ticker, has_cashtag=True))

    if len(cashtags_stock) == 0:
        for s in search_tuples:
            ticker = s[0].strip()
            name = s[1].strip()

            if re.search(ticker, text) and re.search(name, text, re.IGNORECASE):
                cashtags_stock.append(ticker)
                get_cashtag_info(ticker=ticker, has_cashtag=False)

    if len(cashtags_stock) == 0:
        for s in search_tuples:
            ticker = s[0]
            name = s[1]
            if re.search(ticker, raw_line) and re.search(name, raw_line, re.IGNORECASE):
                cashtags_stock.append(ticker)

    print(cashtags_stock)

    tweet['flagged_stocks'] = cashtags_stock
    return tweet


if __name__ == '__main__':
    save_nasdaq_tweets()
