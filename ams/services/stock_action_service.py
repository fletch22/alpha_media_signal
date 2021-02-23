import pandas as pd

from ams.config import constants


def get_splits():
    df = pd.read_csv(constants.SHARADAR_ACTIONS_FILEPATH)
    return df[df["action"] == "split"]