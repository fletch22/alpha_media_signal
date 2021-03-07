from pathlib import Path
from typing import List

import dask
import pandas as pd
from dask.dataframe import from_pandas
from distributed import Client
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from ams.config import logger_factory
from ams.config.constants import ensure_dir
from ams.pipes import batchy_bae
from ams.services import file_services

logger = logger_factory.create(__name__)


def process(source_dir_path: Path, output_dir_path: Path):
    files = file_services.list_files(source_dir_path, ends_with=".parquet.in_transition", use_dir_recursion=True)

    analyzer = SentimentIntensityAnalyzer()

    def add_senti(text) -> List[str]:
        result = analyzer.polarity_scores(text)
        return [result["neg"], result["neu"], result["pos"], result["compound"]]

    dask.config.set(scheduler='processes')

    num_files = len(files)

    logger.info(f"Num files: {num_files}")
    client = Client()

    total_count = 0
    for ndx, f in enumerate(files):
        logger.info(f"Reading {ndx + 1} of {num_files}: '{f}'")
        pdf = pd.read_parquet(f)

        total_count += pdf.shape[0]

        logger.info(f"Converting Pandas dataframe ({pdf.shape[0]} rows) to Dask DF ...")
        ddf = from_pandas(pdf, npartitions=10)
        ddf.persist()

        ddf = ddf.assign(sent_list=ddf.nlp_text.map(lambda x: add_senti(x)))
        ddf = ddf.assign(f22_sentiment_neg=ddf.sent_list.map(lambda x: x[0]))
        ddf = ddf.assign(f22_sentiment_neu=ddf.sent_list.map(lambda x: x[1]))
        ddf = ddf.assign(f22_sentiment_pos=ddf.sent_list.map(lambda x: x[2]))
        ddf = ddf.assign(f22_sentiment_compound=ddf.sent_list.map(lambda x: x[-1]))
        ddf.drop("sent_list", axis=1)

        sent_drop_path = file_services.create_unique_folder_name(str(output_dir_path), prefix="sd")

        ddf.to_parquet(path=str(sent_drop_path), schema='infer', engine="pyarrow", compression="snappy")

        client.compute(ddf)

    logger.info(f"Total records processed: {total_count}")

    client.close()


def start(source_dir_path: Path, twitter_root_path: Path, snow_plow_stage: bool, should_delete_leftovers: bool):
    file_services.unnest_files(parent=source_dir_path, target_path=source_dir_path, filename_ends_with=".parquet")

    output_dir_path = Path(twitter_root_path, 'sent_drop', "main")
    ensure_dir(output_dir_path)

    batchy_bae.ensure_clean_output_path(output_dir_path, should_delete_remaining=should_delete_leftovers)

    batchy_bae.start(source_path=source_dir_path, out_dir_path=output_dir_path,
                     process_callback=process, should_archive=False,
                     snow_plow_stage=snow_plow_stage, should_delete_leftovers=should_delete_leftovers)

    return output_dir_path
