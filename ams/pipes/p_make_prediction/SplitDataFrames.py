from typing import List

import pandas as pd

from ams.config import logger_factory

logger = logger_factory.create(__name__)


class SplitDataFrames:
    dfs = []

    def __init__(self, split_dfs: List[pd.DataFrame]):
        self.dfs = split_dfs

    def get_dates(self):
        dates = set()

        for sd in self.dfs:
            dates |= set(sd["date"].unique())

        return sorted(dates)

    def get_dataframe(self, date_str: str):
        logger.info(f"finding: {date_str}")
        for sd in self.dfs:
            if date_str in sd["date"].unique():
                return sd
        return None