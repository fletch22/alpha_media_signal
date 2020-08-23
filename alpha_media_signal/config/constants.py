import os
from pathlib import Path

import yaml

from alpha_media_signal.config.Credentials import Credentials

PROJECT_ROOT = Path(__file__).parent.parent

import os

if os.name == 'nt':
    WORKSPACE_PATH = Path('C:\\Users\\Chris\\workspaces\\')
else:
    WORKSPACE_PATH = Path('/home', 'jovyan', 'work')

DATA_PATH = Path(WORKSPACE_PATH, 'data')
LOGGING_PATH = Path(DATA_PATH, 'logs', 'alpha_media_signal')

RESOURCES_PATH = Path(DATA_PATH, 'credentials')
TWITTER_CREDS_PATH = Path(RESOURCES_PATH, "search_tweets_creds.yaml")

TWITTER_OUTPUT_RAW_PATH = Path(DATA_PATH, 'twitter')
TWITTER_RAW_TWEETS_PREFIX = 'tweets_raw'

FIN_DATA = Path(DATA_PATH, 'financial')

TWITTER_OUTPUT = Path(FIN_DATA, 'twitter')
TWITTER_OUTPUT.mkdir(exist_ok=True)
TICKER_NAME_SEARCHABLE_PATH = Path(TWITTER_OUTPUT, 'ticker_names_searchable.csv')

YAHOO_OUTPUT_PATH = Path(FIN_DATA, 'yahoo')
YAHOO_COMPANY_INFO = Path(YAHOO_OUTPUT_PATH, 'company_info')
YAHOO_COMPANY_INFO.mkdir(exist_ok=True)

QUANDL_TBLS_DIR = Path(FIN_DATA, 'quandl', 'tables')
SHAR_TICKERS = os.path.join(QUANDL_TBLS_DIR, "shar_tickers.csv")

TOP_100K_WORDS_PATH = Path(DATA_PATH, 'english', 'top100KWords.txt')

with open(TWITTER_CREDS_PATH) as file:
    # The FullLoader parameter handles the conversion from YAML
    # scalar values to Python the dictionary format
    creds_yaml = yaml.load(file, Loader=yaml.FullLoader)

fletch22_key = 'search_tweets_fullarchive_development'
standard_search_key = 'standard_search_tweets'

FLETCH22_CREDS = Credentials(creds_yaml, fletch22_key)
STANDARD_CREDS = Credentials(creds_yaml, standard_search_key)

