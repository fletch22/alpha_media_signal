import json
import time
from pathlib import Path
from pprint import pprint
from typing import List, Generator

import requests

from alpha_media_signal import config
from alpha_media_signal.services import file_services
from alpha_media_signal.steaming.BearerTokenAuth import BearerTokenAuth
from alpha_media_signal.twitter import append_tweets_to_output_file

stream_url = "https://api.twitter.com/labs/1/tweets/stream/filter"
rules_url = "https://api.twitter.com/labs/1/tweets/stream/filter/rules"

sample_rules = [
    {'value': 'dog has:images', 'tag': 'dog pictures'},
    {'value': 'cat has:images -grumpy', 'tag': 'cat pictures'},
]

bearer_token = BearerTokenAuth(config.constants.FLETCH22_CREDS.api_key, config.FLETCH22_CREDS.api_secret_key)


def get_all_rules(auth):
    response = requests.get(rules_url, auth=auth)

    if response.status_code is not 200:
        raise Exception(f"Cannot get rules (HTTP %d): %s" % (response.status_code, response.text))

    return response.json()


def delete_all_rules(rules, auth):
    if rules is None or 'data' not in rules:
        return None

    ids = list(map(lambda rule: rule['id'], rules['data']))

    payload = {
        'delete': {
            'ids': ids
        }
    }

    response = requests.post(rules_url, auth=auth, json=payload)

    if response.status_code is not 200:
        raise Exception(f"Cannot delete rules (HTTP %d): %s" % (response.status_code, response.text))


def set_rules(rules, auth):
    if rules is None:
        return

    payload = {
        'add': rules
    }

    response = requests.post(rules_url, auth=auth, json=payload)

    if response.status_code is not 201:
        raise Exception(f"Cannot create rules (HTTP %d): %s" % (response.status_code, response.text))


def process_tweets(it: Generator, output_path: Path):
    count = 0

    for response_line in it:
        if response_line:
            tweet = json.loads(response_line)
            pprint(tweet)
            append_tweets_to_output_file(output_path=output_path, tweets=[tweet])
            count += 1
            if count % 10 == 0:
                print(f'Captured {count} total tweets.')

    return count


def stream_connect(auth, output_path: Path):
    response = requests.get(stream_url, auth=auth, stream=True)
    process_tweets(response.iter_lines(), output_path=output_path)


def setup_rules(auth, rules: List):
    current_rules = get_all_rules(auth)
    delete_all_rules(current_rules, auth)
    set_rules(rules, auth)

    current_rules = get_all_rules(auth)
    pprint(current_rules)


def run(rules: List):
    # Comment this line if you already setup rules and want to keep them
    setup_rules(bearer_token, rules)

    tweet_raw_output_path = file_services.create_unique_file_system_name(config.TWITTER_OUTPUT_RAW_PATH, prefix=config.TWITTER_RAW_TWEETS_PREFIX, extension='txt')

    print(f'Will write tweets to {tweet_raw_output_path}')

    # Listen to the stream.
    # This reconnection logic will attempt to reconnect when a disconnection is detected.
    # To avoid rate limits, this logic implements exponential backoff, so the wait time
    # will increase if the client cannot reconnect to the stream.
    timeout = 0
    while True:
        stream_connect(bearer_token, output_path=tweet_raw_output_path)
        time.sleep(2 ** timeout)
        timeout += 1


if __name__ == '__main__':

    # vod = {'value': 'Vodaphone OR #VOD lang:en'}
    tsla = {'value': 'Tesla tsla stock lang:en'}

    stock_rules = [
        tsla
    ]

    run(rules=stock_rules)
