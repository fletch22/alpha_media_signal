from pathlib import Path

import pandas as pd

from ams.config import constants, logger_factory
from ams.pipes.p_add_id import process as process_id
from ams.services import file_services, ticker_service

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

logger = logger_factory.create(__name__)

def test_add_id():
    process_id.start()


def test_get_inf_drop():
    file_path_str = str(Path(constants.OVERFLOW_DATA_PATH, "twitter", "inference_model_drop", "twitter_id_with_label.csv"))

    df = pd.read_csv(file_path_str)

    logger.info(f"Cols: {list(df.columns)}")

    df_short = df[["f22_id", "text"]]
    logger.info(df_short.head(20))


def test_all_twitter_dates():
    # folder_path = Path(constants.TWITTER_OUTPUT, "flattened_drop")
    # folder_path = Path(constants.TWITTER_OUTPUT_RAW_PATH, "id_fixed")
    folder_path = Path(constants.TWITTER_OUTPUT_RAW_PATH, "test")
    # folder_path = Path(constants.TWITTER_OUTPUT_RAW_PATH, "sent_drop", "staging")
    # folder_path = Path(constants.TWITTER_OUTPUT_RAW_PATH, "sent_drop", "test")
    files = file_services.list_files(folder_path, ends_with=".csv", use_dir_recursion=True)
    output_path = folder_path = Path(constants.TWITTER_OUTPUT_RAW_PATH, "test_output", "test.csv")

    df_all = []
    rows = 0
    count_limit = 400000
    replace_chars = "foo"

    logger.info(f"Found {len(files)} files.")

    for f in files:
        df = pd.read_csv(f)

        df_text = df
        df_text = df[["text"]]

        df_text["text"] = df_text["text"].apply(replace_chars)

        logger.info(df_text.iloc[0, 0])

        df_text.to_csv(output_path)


def convert_date():
    date_str = "Wed Aug 12 01:50:19 +0000 2020"


def test_load_and_test():
    folder_path = Path(constants.TWITTER_OUTPUT_RAW_PATH, "id_fixed", "id_fixed_2020-11-15_23-38-23-675.74.csv")

    df = pd.read_csv(str(folder_path), sep="|")

    logger.info(df[["text"]].head())


def test_random_choice():
    import numpy as np

    for i in range(10):
        val = np.random.choice(2, 1)
        logger.info(val)


def test_find_funny_roi():
    # Arrange
    # Act
    df = pd.read_csv(constants.TWITTER_TRAINING_PREDICTIONS_FILE_PATH)

    # print(df.columns)

    df = df[(df["purchase_date"] == "2020-10-20") & (df["num_hold_days"] == 10)]

    print(df.head())

    df = ticker_service.get_ticker_eod_data("NTEC")

    row = df[df["date"] == "2020-10-20"].iloc[0]
    purchase_price = row["close"]

    row = df[df["date"] == "2020-11-03"].iloc[0]
    sell_price = row["close"]

    print(f"Pp: {purchase_price}; Sp: {sell_price}")

    # Assert