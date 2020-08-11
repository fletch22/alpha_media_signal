from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

WORKSPACE_PATH = Path('C:\\Users\\Chris\\workspaces\\')
LOGGING_PATH = Path(WORKSPACE_PATH, 'data\\logs\\alpha_media_signal')

RESOURCES_PATH = Path(WORKSPACE_PATH, 'data\\credentials')
TWITTER_CREDS_PATH = Path(RESOURCES_PATH, "./search_tweets_creds.yaml")

TWITTER_OUTPUT_RAW_PATH = Path(WORKSPACE_PATH, 'data', 'twitter')
TWITTER_RAW_TWEETS_PREFIX = 'tweets_raw'