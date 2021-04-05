from ams.config import constants
from ams.marios_workbench.twitter.import_and_predict import valve as gmap_valve
from ams.twitter.TrainAndPredictionParams import PredictionMode


def test_get_message_and_predict():
    gmap_valve.get_and_message_predictions(twitter_root_path=constants.TWITTER_OUTPUT_RAW_PATH,
                                           prediction_mode=PredictionMode.DevelopmentAndTraining)