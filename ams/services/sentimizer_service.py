from transformers import pipeline

# Allocate a pipeline for sentiment-analysis
classifier = pipeline('sentiment-analysis')

LBL_NEGATIVE = "NEGATIVE"
LBL_POSITIVE = "POSITIVE"


def get_sentiment(text: str):
    sentiment_list = classifier(text)

    sentiment = sentiment_list[0]
    label = sentiment["label"]
    score = sentiment["score"]
    if label == LBL_NEGATIVE:
        score *= -1

    return score


