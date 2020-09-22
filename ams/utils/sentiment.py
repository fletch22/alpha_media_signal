from typing import Dict

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()


def get_sentiment_intensity_score(text: str) -> Dict:
    return analyzer.polarity_scores(text)
