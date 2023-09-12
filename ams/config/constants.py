import json
import os
from pathlib import Path
from types import SimpleNamespace

import yaml

from ams.config.Credentials import Credentials

PROJECT_ROOT = Path(__file__).parent.parent.parent
STOCK_PREDICTOR_ROOT = Path(PROJECT_ROOT.parent, 'stock-predictor')


def ensure_dir(dir):
    os.makedirs(dir, exist_ok=True)


def get_api_key(file_path: Path):
    with open(str(file_path), "r") as f:
        return f.readline()


if os.name == 'nt':
    WORKSPACE_PATH = Path('C:\\Users\\Chris\\workspaces\\')
    OVERFLOW_WORKSPACE_PATH = Path('F:\\Users\\chris\\')
    WORKSPACE_E_PATH = Path("E:\\workspaces\\")
else:
    # /home/jupyter/alpha_media_signal/
    WORKSPACE_PATH = Path('/home', 'jupyter')
    OVERFLOW_WORKSPACE_PATH = Path('/home', 'jupyter', 'overflow_workspace')

F_WORKSPACE_PATH = Path(r'F:\Users\Chris')
DATA_PATH = Path(F_WORKSPACE_PATH, 'data')
ensure_dir(str(OVERFLOW_WORKSPACE_PATH))

LOGGING_PATH = Path(DATA_PATH, 'logs', 'alpha_media_signal')
ensure_dir(LOGGING_PATH)

RESOURCES_PATH = Path(DATA_PATH, 'credentials')
TWITTER_CREDS_PATH = Path(RESOURCES_PATH, "search_tweets_creds.yaml")

fletch22_key = 'search_tweets_fullarchive_development'
standard_search_key = 'standard_search_tweets'
rogd_key = "standard_search_tweets_rogd"

TWITTER_OUTPUT_RAW_PATH = Path(DATA_PATH, 'twitter')
TWITTER_RAW_TWEETS_PREFIX = 'tweets_raw'

OVERFLOW_DATA_PATH = Path(OVERFLOW_WORKSPACE_PATH, 'data')

WEATHER_DATA_DIR = Path(OVERFLOW_DATA_PATH, "weather")
WEATHER_DATA_PATH = Path(WEATHER_DATA_DIR, "station_1.txt")

FIN_DATA = Path(OVERFLOW_DATA_PATH, 'financial')

TWITTER_OVERFLOW_OUTPUT = Path(OVERFLOW_DATA_PATH, 'twitter')
TWITTER_OVERFLOW_OUTPUT.mkdir(exist_ok=True)
TICKER_NAME_SEARCHABLE_PATH = Path(TWITTER_OVERFLOW_OUTPUT, 'ticker_names_searchable.csv')

SP500_PATH = Path(FIN_DATA, 'sp500')
SP500_LIST_HIST_PATH = Path(SP500_PATH, "sp500_list.csv")
SP500_LIST_HIST_REF_PATH = Path(SP500_PATH, "sp500_list_refined.csv")
SP_TOP_N_PATH = Path(SP500_PATH, "sp500_top_n_path.csv")
SP500_HTML_PATH = Path(SP500_PATH, "html")
SP500_WIKI_TICKERS = Path(SP500_PATH, "wiki")

TWITTER_TRASH_OUTPUT = Path(TWITTER_OVERFLOW_OUTPUT, "trash")
ensure_dir(TWITTER_TRASH_OUTPUT)

YAHOO_OUTPUT_PATH = Path(FIN_DATA, 'yahoo')
YAHOO_COMPANY_INFO = Path(YAHOO_OUTPUT_PATH, 'company_info')
ensure_dir(str(YAHOO_COMPANY_INFO))

QUANDL_DIR = os.path.join(FIN_DATA, "nasdaq")
SHAR_TICKERS = os.path.join(QUANDL_DIR, "SHARADAR_TICKERS.csv")

DAILY_STOCK_DIR = Path(QUANDL_DIR, 'daily')
FUNDY_DIR = Path(QUANDL_DIR, 'fundamentals')
SHAR_CORE_FUNDY_FILE_PATH = Path(QUANDL_DIR, "shar_core_fundamentals.csv")

SHARADAR_ACTIONS_DIR = os.path.join(QUANDL_DIR, "shar_actions")
ensure_dir(SHARADAR_ACTIONS_DIR)
SHARADAR_ACTIONS_FILEPATH = Path(SHARADAR_ACTIONS_DIR, "actions.csv")

SHAR_SPLIT_EQUITY_EOD_DIR = Path(QUANDL_DIR, "splits_eod")
SHAR_TICKER_DETAIL_INFO_PATH = Path(QUANDL_DIR, "SHARADAR_TICKERS.csv")

SHAR_INDICATORS_CSV = Path(QUANDL_DIR, "SHARADAR_INDICATORS.csv")

KAFKA_URL = "localhost:9092"

GOOGLE_NEWS_OUTPUT_DIR_PATH = Path(OVERFLOW_DATA_PATH, "news\\google\\alpha_media_signal")

TWITTER_INFERENCE_MODEL_PATH = Path(TWITTER_OVERFLOW_OUTPUT, "inference_model_drop")
TRAIN_READY_TWEETS = Path(TWITTER_INFERENCE_MODEL_PATH, "train_ready_tweets.mod")

BACKUP_ROOT_PATH = Path("I:/sp_backups")
BACKUP_VOLUME_LABEL = "Flesche"
ALPHA_MEDIA_SIGNAL_PROJ = Path(WORKSPACE_PATH, "alpha_media_signal")
NEWS_GESTALT_PROJ = Path(WORKSPACE_PATH, "news_gestalt")
BLOGGER_HIGH_SCORE_PROJ = Path(WORKSPACE_PATH, "blogger_high_score")

BERT_PATH = Path(TWITTER_OVERFLOW_OUTPUT, "bert")
BERT_APPS_DATA_PATH = Path(BERT_PATH, "apps.csv")
BERT_REVIEWS_DATA_PATH = Path(BERT_PATH, "reviews.csv")

COLA_DIR_PATH = Path(OVERFLOW_DATA_PATH, "cola", "cola_public_1.1", "cola_public", "raw")
COLA_IN_DOMAIN_DEV_PATH = Path(COLA_DIR_PATH, "in_domain_dev.tsv")
COLA_IN_DOMAIN_TRAIN_PATH = Path(COLA_DIR_PATH, "in_domain_train.tsv")
COLA_OUT_OF_DOMAIN_DEV_PATH = Path(COLA_DIR_PATH, "out_of_domain_dev.tsv")

TWITTER_TEXT_LABEL_TRAIN_PATH = Path(TWITTER_INFERENCE_MODEL_PATH,
                                     "twitter_text_with_proper_labels.parquet")

TWITTER_MODEL_PATH = Path(TWITTER_INFERENCE_MODEL_PATH, "models")
ensure_dir(TWITTER_MODEL_PATH)

TWITTER_XGB_MODEL_PATH = Path(TWITTER_MODEL_PATH, "xgb_model.pkl")

TIP_RANKS_DATA_DIR = Path(FIN_DATA, 'tip_ranks')
ensure_dir(TIP_RANKS_DATA_DIR)
TIP_RANKS_STOCK_DATA_PATH = os.path.join(TIP_RANKS_DATA_DIR, "tip_ranks_stock.parquet")

TIP_RANKED_DATA_PATH = Path(TWITTER_OUTPUT_RAW_PATH, "tip_ranked", "main", "tip_rank_2020-12-14_22-48-27-354.17.parquet")

DAILY_ROI_NASDAQ_PATH = Path(QUANDL_DIR, "daily_roi_nasdaq.parquet")

TICK_ON_DAY_PATH = Path(FIN_DATA, "tickers_on_day")
ensure_dir(TICK_ON_DAY_PATH)

TOD_PICKLE_PATH = Path(TICK_ON_DAY_PATH, "tod.pickle")

THREE_BELOW_HISTORY_PATH = Path(TICK_ON_DAY_PATH, "three_below_history.csv")

TWITTER_TRADE_HISTORY_PATH = Path(TWITTER_OVERFLOW_OUTPUT, "trade_history")
ensure_dir(TWITTER_TRADE_HISTORY_PATH)

TWITTER_TRADE_HISTORY_FILE_PATH = Path(TWITTER_TRADE_HISTORY_PATH, "twitter_trade_history.csv")

CREDENTIALS_ROOT = Path(DATA_PATH, "credentials")
if not CREDENTIALS_ROOT.exists():
    raise Exception("The credentials root was not found.")

slack_cred_path = os.path.join(CREDENTIALS_ROOT, 'slack.json')
with open(slack_cred_path, "r") as f:
    SLACK_CREDENTIALS = json.loads(f.read())

finnhub_path = Path(CREDENTIALS_ROOT, "finnhub.io.api_key")
FINNHUB_API_KEY = get_api_key(file_path=finnhub_path)

alpaca_cred_path = CREDENTIALS_ROOT / 'alpaca.json'
with open(alpaca_cred_path, "r") as f:
    ALPACA_CREDENTIALS = json.loads(f.read())['PAPER-API']

US_MARKET_HOLIDAYS_PATH = Path(FIN_DATA, "us_market_holidays.csv")

TWITTER_GREAT_REDUCTION_DIR = Path(TWITTER_OUTPUT_RAW_PATH, "great_reduction")

TWITTER_MODEL_PREDICTION_DIR_PATH = Path(TWITTER_MODEL_PATH, "predictions")

TWITTER_TRAINING_PREDICTIONS_FILE_PATH = Path(TWITTER_MODEL_PREDICTION_DIR_PATH, "predictions.csv")
ensure_dir(TWITTER_TRAINING_PREDICTIONS_FILE_PATH.parent)

TWITTER_REAL_MONEY_PREDICTIONS_FILE_PATH = Path(TWITTER_MODEL_PREDICTION_DIR_PATH, "real_money_predictions.csv")
ensure_dir(TWITTER_REAL_MONEY_PREDICTIONS_FILE_PATH.parent)

TWEET_RAW_DROP_ARCHIVE = Path(TWITTER_OVERFLOW_OUTPUT, "raw_drop", "archive")

TESTS_ROOT = Path(PROJECT_ROOT, "tests")
TESTS_RESOURCES = Path(TESTS_ROOT, "resources")

TEMP_PATH = Path("e:\\tmp")

TRANSIENT_DIR_PATH = Path(TEMP_PATH, "transient")
ensure_dir(TRANSIENT_DIR_PATH)

REFINED_TWEETS_BUCKET_PATH = Path(TWITTER_OUTPUT_RAW_PATH, "refined_tweets_bucket")

xgb_defaults = SimpleNamespace(defaults=SimpleNamespace(max_depth=4))

AUTO_ML_PATH = Path(TRANSIENT_DIR_PATH, "auto_ml_train_test_val.csv")

TWITTER_STACKING_MODEL_PATH = Path(TWITTER_MODEL_PATH, "twitter_stacking_model.pkl")

TWIT_STOCK_MERGE_DROP_PATH = Path(TWITTER_OUTPUT_RAW_PATH, "stock_merge_drop", "main")

PERF_METRICS_ROIS = Path(TRANSIENT_DIR_PATH, "perf_metrics_rois.csv")

CATBOOST_TRAIN_DIR = Path(TWITTER_INFERENCE_MODEL_PATH, "train")

STOCK_AGG_DATAFILE = Path(OVERFLOW_DATA_PATH, "basic_stock_agg.parquet")

E_DATA_PATH = Path(WORKSPACE_E_PATH, "data")
WIKI_PAGES_PATH = Path(E_DATA_PATH, "wiki_pages")
WIKI_ARTICLES_PATH = Path(WIKI_PAGES_PATH, "enwiki-latest-pages-articles.xml.bz2")
WIKI_CORPUS_PATH = Path(WIKI_PAGES_PATH, "corpus")

TOP_100K_WORDS_PATH = Path(DATA_PATH, 'english', 'top100KWords.txt')

SURVIVOR_RESULTS_DIR = DATA_PATH / 'survivor'
SURVIVOR_RESULTS_DIR.mkdir(exist_ok=True, parents=True)
SURVIVOR_RESULTS_PATH = SURVIVOR_RESULTS_DIR / 'base_results.csv'

STOCKS_US_LOW_PRICE_PATH = TESTS_RESOURCES / 'stocks_us_low_price.json'
STOCKS_US_TECH_STOCKS_PATH = TESTS_RESOURCES / 'stocks_us_tech.json'

# INSIDER_BUYS_DATA_PATH = Path(r'F:\Users\Chris\data\financial\insider_deals\misc\insider_buys.csv')
INSIDER_DIR = FIN_DATA / 'insider_deals'
INSIDER_BUYS_DATA_PATH = INSIDER_DIR / 'misc' / 'insider_buys.csv'
INS_BUYS_SUBMISSIONS_DATA_PATH = INSIDER_DIR / 'combined' / 'all_SUBMISSIONS.csv'
