from pathlib import Path

import yaml

from ams.config.Credentials import Credentials

PROJECT_ROOT = Path(__file__).parent.parent

import os


def make_dir(dir):
    os.makedirs(dir, exist_ok=True)


if os.name == 'nt':
    WORKSPACE_PATH = Path('C:\\Users\\Chris\\workspaces\\')
    OVERFLOW_WORKSPACE_PATH = Path('D:\\workspaces\\')
else:
    WORKSPACE_PATH = Path('/home', 'jovyan', 'work')
    OVERFLOW_WORKSPACE_PATH = Path('/home', 'jovyan', 'overflow_workspace')

DATA_PATH = Path(WORKSPACE_PATH, 'data')
LOGGING_PATH = Path(DATA_PATH, 'logs', 'alpha_media_signal')

RESOURCES_PATH = Path(DATA_PATH, 'credentials')
TWITTER_CREDS_PATH = Path(RESOURCES_PATH, "search_tweets_creds.yaml")

TWITTER_OUTPUT_RAW_PATH = Path(DATA_PATH, 'twitter')
TWITTER_RAW_TWEETS_PREFIX = 'tweets_raw'

DATA_PATH = Path(WORKSPACE_PATH, 'data')
OVERFLOW_DATA_PATH = Path(OVERFLOW_WORKSPACE_PATH, 'data')

WEATHER_DATA_DIR = Path(OVERFLOW_DATA_PATH, "weather")
WEATHER_DATA_PATH = Path(WEATHER_DATA_DIR, "station_1.txt")

FIN_DATA = Path(OVERFLOW_DATA_PATH, 'financial')

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

QUANDL_DIR = os.path.join(FIN_DATA, "quandl")
QUANDL_TBLS_DIR = os.path.join(QUANDL_DIR, "tables")

SHARADAR_ACTIONS_DIR = os.path.join(QUANDL_TBLS_DIR, "shar_actions")
make_dir(SHARADAR_ACTIONS_DIR)
SHARADAR_ACTIONS_FILEPATH = Path(SHARADAR_ACTIONS_DIR, "actions.csv")

SHAR_SPLIT_EQUITY_EOD_DIR = Path(QUANDL_TBLS_DIR, "splits_eod")

KAFKA_URL = "localhost:9092"
