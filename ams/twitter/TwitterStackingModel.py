from pathlib import Path

from ams.config import constants, logger_factory
from ams.services import pickle_service
from ams.twitter.TrainAndPredictionParams import TrainAndPredictionParams
import pandas as pd

logger = logger_factory.create(__name__)


class TwitterStackingModel:
    models = None

    def __init__(self):
        self.models = list()

    def add_trained_model(self, model: any):
        self.models.append(model)

    @classmethod
    def persist(cls, twitter_stacking_model: any):
        pickle_service.save(twitter_stacking_model, file_path=constants.TWITTER_STACKING_MODEL_PATH)

    @classmethod
    def load(cls):
        return pickle_service.load(file_path=constants.TWITTER_STACKING_MODEL_PATH)