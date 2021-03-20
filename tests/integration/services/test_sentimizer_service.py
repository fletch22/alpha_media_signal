import csv
from pathlib import Path

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from ams.config import constants
from ams.services import sentimizer_service, file_services, bert_sentiment_service
from ams.utils.Stopwatch import Stopwatch

from ams.config import logger_factory

logger = logger_factory.create(__name__)


def test_get_sentiment():
    # Arrange
    text = "Sorrow breaks me!"
    text_training = "o immediately use a model on a given text, we provide the pipeline API. Pipelines group together a pretrained model with the preprocessing that was used during that model training. Here is how to quickly use a pipeline to classify positive versus negative texts"
    # Act
    stopwatch = Stopwatch(start_now=True)
    sentiment = sentimizer_service.get_sentiment(text=text)
    sentiment = sentimizer_service.get_sentiment(text="foo manchu")
    sentiment = sentimizer_service.get_sentiment(text=text_training)
    stopwatch.end(msg="3 sentiments")

    # Assert
    logger.info(sentiment)

    analyzer = SentimentIntensityAnalyzer()

    logger.info(analyzer.polarity_scores(text_training))


def test_bert():
    # Arrange
    twitter_folder = 'twitter'
    flat_tweet_path = Path(constants.DATA_PATH, twitter_folder, r'flattened_drop', "tweets_flat_2020-10-17_10-09-09-14.92")
    files = file_services.list_files(parent_path=flat_tweet_path, ends_with=".csv")

    df_red = None
    for f in files:
        df_red = pd.read_csv(str(f), dialect=csv.unix_dialect(), error_bad_lines=False, index_col=False, dtype='unicode')
        break

    # Act
    preds = bert_sentiment_service.get_bert_preds(df=df_red)

    # Assert
    logger.info(preds)
