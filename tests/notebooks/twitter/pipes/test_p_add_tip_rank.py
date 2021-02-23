from pathlib import Path

from ams.config import constants
import pandas as pd


def test_get():
    file_path = Path(constants.TWITTER_OUTPUT_RAW_PATH, "tip_ranked", "main", "tip_rank_2020-12-14_22-48-27-354.17.parquet")

    df = pd.read_parquet(str(file_path))

    logger.info(df.head())