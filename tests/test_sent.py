from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from ams.config import constants
from ams.notebooks.twitter.pipes.p_add_id import process as process_id
from ams.services import file_services

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

def test_add_id():
    process_id.start()



def test_get_inf_drop():
    file_path_str = str(Path(constants.OVERFLOW_DATA_PATH, "twitter", "inference_model_drop", "twitter_id_with_label.csv"))

    df = pd.read_csv(file_path_str)

    print(f"Cols: {list(df.columns)}")

    df_short = df[["f22_id", "text"]]
    print(df_short.head(20))





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

    print(f"Found {len(files)} files.")

    for f in files:
        # df = pd.read_csv(f, header=0, sep='\n', quoting=csv.QUOTE_ALL, quotechar='"', engine="python")
        df = pd.read_csv(f)

        df_text = df
        df_text = df[["text"]]

        # print(f"Num cols: {len(list(df_text.columns))}")
        # print(f"Cols: {str(list(df_text.columns))}")

        # df["text"] = df["text"].replace(["\\"], "\"\"")
        df_text["text"] = df_text["text"].apply(replace_chars)

        # df.drop("text", axis=1)

        # df_text["create_at_timestamp"]
        print(df_text.iloc[0, 0])

        df_text.to_csv(output_path)

        # print(df_text.head(10))
        # break


def convert_date():
    date_str = "Wed Aug 12 01:50:19 +0000 2020"


def test_load_and_test():
    folder_path = Path(constants.TWITTER_OUTPUT_RAW_PATH, "id_fixed", "id_fixed_2020-11-15_23-38-23-675.74.csv")

    df = pd.read_csv(str(folder_path), sep="|")

    print(df[["text"]].head())
