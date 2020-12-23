import os
from pathlib import Path

import yaml

from ams.config.Credentials import Credentials

PROJECT_ROOT = Path(__file__).parent.parent.parent


def ensure_dir(dir):
    os.makedirs(dir, exist_ok=True)


if os.name == 'nt':
    WORKSPACE_PATH = Path('C:\\Users\\Chris\\workspaces\\')
    OVERFLOW_WORKSPACE_PATH = Path('D:\\workspaces\\')
else:
    # /home/jupyter/alpha_media_signal/
    WORKSPACE_PATH = Path('/home', 'jupyter')
    OVERFLOW_WORKSPACE_PATH = Path('/home', 'jupyter', 'overflow_workspace')
ensure_dir(str(OVERFLOW_WORKSPACE_PATH))

DATA_PATH = Path(WORKSPACE_PATH, 'data')
LOGGING_PATH = Path(DATA_PATH, 'logs', 'alpha_media_signal')
ensure_dir(LOGGING_PATH)

if os.name == 'nt':
    RESOURCES_PATH = Path(DATA_PATH, 'credentials')
    TWITTER_CREDS_PATH = Path(RESOURCES_PATH, "search_tweets_creds.yaml")

    with open(TWITTER_CREDS_PATH) as file:
        # The FullLoader parameter handles the conversion from YAML
        # scalar values to Python the dictionary format
        creds_yaml = yaml.load(file, Loader=yaml.FullLoader)

    fletch22_key = 'search_tweets_fullarchive_development'
    standard_search_key = 'standard_search_tweets'
    rogd_key = "standard_search_tweets_rogd"

    FLETCH22_CREDS = Credentials(creds_yaml, fletch22_key)
    STANDARD_CREDS = Credentials(creds_yaml, standard_search_key)
    ROGD_CREDS = Credentials(creds_yaml, standard_search_key)

    CURRENT_CREDS = ROGD_CREDS

TWITTER_OUTPUT_RAW_PATH = Path(DATA_PATH, 'twitter')
TWITTER_RAW_TWEETS_PREFIX = 'tweets_raw'

DATA_PATH = Path(WORKSPACE_PATH, 'data')
OVERFLOW_DATA_PATH = Path(OVERFLOW_WORKSPACE_PATH, 'data')

WEATHER_DATA_DIR = Path(OVERFLOW_DATA_PATH, "weather")
WEATHER_DATA_PATH = Path(WEATHER_DATA_DIR, "station_1.txt")

FIN_DATA = Path(OVERFLOW_DATA_PATH, 'financial')

TWITTER_OUTPUT = Path(OVERFLOW_DATA_PATH, 'twitter')
TWITTER_OUTPUT.mkdir(exist_ok=True)
TICKER_NAME_SEARCHABLE_PATH = Path(TWITTER_OUTPUT, 'ticker_names_searchable.csv')

TWITTER_TRASH_OUTPUT = Path(TWITTER_OUTPUT, "trash")
ensure_dir(TWITTER_TRASH_OUTPUT)

YAHOO_OUTPUT_PATH = Path(FIN_DATA, 'yahoo')
YAHOO_COMPANY_INFO = Path(YAHOO_OUTPUT_PATH, 'company_info')
ensure_dir(str(YAHOO_COMPANY_INFO))

QUANDL_TBLS_DIR = Path(FIN_DATA, 'quandl', 'tables')
SHAR_TICKERS = os.path.join(QUANDL_TBLS_DIR, "shar_tickers.csv")

TOP_100K_WORDS_PATH = Path(DATA_PATH, 'english', 'top100KWords.txt')

QUANDL_DIR = os.path.join(FIN_DATA, "quandl")
QUANDL_TBLS_DIR = os.path.join(QUANDL_DIR, "tables")
SHAR_CORE_FUNDY_FILE_PATH = Path(QUANDL_TBLS_DIR, "shar_core_fundamentals.csv")

SHARADAR_ACTIONS_DIR = os.path.join(QUANDL_TBLS_DIR, "shar_actions")
ensure_dir(SHARADAR_ACTIONS_DIR)
SHARADAR_ACTIONS_FILEPATH = Path(SHARADAR_ACTIONS_DIR, "actions.csv")

SHAR_SPLIT_EQUITY_EOD_DIR = Path(QUANDL_TBLS_DIR, "splits_eod")
SHAR_TICKER_DETAIL_INFO_PATH = Path(QUANDL_TBLS_DIR, "shar_tickers.csv")

KAFKA_URL = "localhost:9092"

GOOGLE_NEWS_OUTPUT_DIR_PATH = Path(OVERFLOW_DATA_PATH, "news\\google\\alpha_media_signal")

TWITTER_INFERENCE_MODEL_PATH = Path(TWITTER_OUTPUT, "inference_model_drop")
TRAIN_READY_TWEETS = Path(TWITTER_INFERENCE_MODEL_PATH, "train_ready_tweets.mod")

BACKUP_ROOT_PATH = Path("I:/sp_backups")
BACKUP_VOLUME_LABEL = "Flesche"
ALPHA_MEDIA_SIGNAL_PROJ = Path(WORKSPACE_PATH, "alpha_media_signal")
NEWS_GESTALT_PROJ = Path(WORKSPACE_PATH, "news_gestalt")

BERT_PATH = Path(TWITTER_OUTPUT, "bert")
BERT_APPS_DATA_PATH = Path(BERT_PATH, "apps.csv")
BERT_REVIEWS_DATA_PATH = Path(BERT_PATH, "reviews.csv")

COLA_DIR_PATH = Path(OVERFLOW_DATA_PATH, "cola", "cola_public_1.1", "cola_public", "raw")
COLA_IN_DOMAIN_DEV_PATH = Path(COLA_DIR_PATH, "in_domain_dev.tsv")
COLA_IN_DOMAIN_TRAIN_PATH = Path(COLA_DIR_PATH, "in_domain_train.tsv")
COLA_OUT_OF_DOMAIN_DEV_PATH = Path(COLA_DIR_PATH, "out_of_domain_dev.tsv")

TWITTER_TEXT_LABEL_TRAIN_PATH = Path(TWITTER_INFERENCE_MODEL_PATH, "twitter_text_with_proper_labels.parquet")

TWITTER_MODEL_PATH = Path(TWITTER_INFERENCE_MODEL_PATH, "models")
ensure_dir(TWITTER_MODEL_PATH)

TIP_RANKS_DATA_DIR = Path(FIN_DATA, 'tip_ranks')
ensure_dir(TIP_RANKS_DATA_DIR)
TIP_RANKS_STOCK_DATA_PATH = os.path.join(TIP_RANKS_DATA_DIR, "tip_ranks_stock.parquet")

DAILY_ROI_NASDAQ_PATH = os.path.join(QUANDL_DIR, "daily_roi_nasdaq.parquet")

TICK_ON_DAY_PATH = Path(FIN_DATA, "tickers_on_day")
ensure_dir(TICK_ON_DAY_PATH)

TOD_PICKLE_PATH = Path(TICK_ON_DAY_PATH, "tod.pickle")

THREE_BELOW_HISTORY_PATH = Path(TICK_ON_DAY_PATH, "three_below_history.csv")

TWITTER_TRADE_HISTORY_PATH = Path(TWITTER_OUTPUT, "trade_history")
ensure_dir(TWITTER_TRADE_HISTORY_PATH)

TWITTER_TRADE_HISTORY_FILE_PATH = Path(TWITTER_TRADE_HISTORY_PATH, "twitter_trade_history.csv")

