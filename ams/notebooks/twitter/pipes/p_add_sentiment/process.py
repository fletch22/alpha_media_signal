import os
from pathlib import Path
from typing import List

import dask
import numpy as np
import pandas as pd
from dask.dataframe import from_pandas
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from ams.config import constants
from ams.config import logger_factory
from ams.notebooks.twitter.pipes import batchy_bae
from ams.services import file_services

logger = logger_factory.create(__name__)


def process(source_dir_path: Path, output_dir_path: Path):
    files = file_services.list_files(source_dir_path, ends_with=".parquet.in_transition", use_dir_recursion=True)

    analyzer = SentimentIntensityAnalyzer()

    def add_senti(text) -> List[str]:
        result = analyzer.polarity_scores(text)
        return [result["neg"], result["neu"], result["pos"], result["compound"]]

    dask.config.set(scheduler='processes')

    for f in files:
        pdf = pd.read_parquet(f)

        split_dfs = np.array_split(pdf, 12)
        del pdf

        for sdf in split_dfs:
            print("Converting Pandas dataframe to Dask DF ...")
            ddf = from_pandas(sdf, npartitions=22)

            ddf = ddf.assign(sent_list=ddf.nlp_text.map(lambda x: add_senti(x)))
            ddf = ddf.assign(f22_sentiment_neg=ddf.sent_list.map(lambda x: x[0]))
            ddf = ddf.assign(f22_sentiment_neu=ddf.sent_list.map(lambda x: x[1]))
            ddf = ddf.assign(f22_sentiment_pos=ddf.sent_list.map(lambda x: x[2]))
            ddf = ddf.assign(f22_sentiment_compound=ddf.sent_list.map(lambda x: x[-1]))
            ddf.drop("sent_list", axis=1)

            ddf.compute()

            sent_drop_path = file_services.create_unique_folder_name(str(output_dir_path), prefix="sd")
            ddf.to_parquet(path=str(sent_drop_path), engine="pyarrow", compression="snappy")


def start():
    source_dir_path = Path(constants.TWITTER_OUTPUT_RAW_PATH, "deduped", "main")
    file_services.unnest_files(parent=source_dir_path, target_path=source_dir_path, filename_ends_with=".parquet")

    output_dir_path = Path(constants.TWITTER_OUTPUT_RAW_PATH, 'sent_drop', "main")
    os.makedirs(output_dir_path, exist_ok=True)

    if not file_services.is_empty(output_dir_path):
        raise Exception(f"Output folder '{output_dir_path}' is not empty.")

    batchy_bae.start(source_path=source_dir_path, output_dir_path=output_dir_path, process_callback=process, should_archive=False)


if __name__ == '__main__':
    start()