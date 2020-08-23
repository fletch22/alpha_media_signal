import csv
import json
import re
from pathlib import Path

import pandas as pd

from alpha_media_signal.config import constants
from alpha_media_signal.services import file_services
from pyspark.sql.types import StructType


def test_load():
    # csv_path_str = r"C:\Users\Chris\workspaces\data\twitter\flattened_drop\tweets_flat_2020-08-21_22-27-23-826.42\part-00000-94e14f84-2c63-4aec-8c07-b865fdaf12a9-c000.csv"

    output_folder_path = Path(f'{constants.DATA_PATH}\\twitter\\flattened_drop\\tweets_flat_2020-08-22_18-04-19-516.66')
    csv_list = list(file_services.list_files(output_folder_path, ends_with=".csv"))

    csv_path_str = str(csv_list[0])
    df = pd.read_csv(csv_path_str, dialect=csv.unix_dialect(), error_bad_lines=False, index_col=False, dtype='unicode')

    print(df.shape[0])
    print(df.columns)


def test_replace_newlines():
    sample = "Healthcare news covering:\n\n- FDA Approvals & Recalls\n- Stage 1/2/3 Updates\n- New Drug Applications\n- FDA Press Releases\n- Company Announcements\n- SEC Filings"
    pattern = re.compile('\n')
    result = re.sub(pattern, '', sample)

    print(result)

    import csv

    unix_dialect = csv.unix_dialect()

def test_schema():
    test = {'ticker': 'foo', 'has_cashtag': True, 'ticker_in_text': False}
    test_schema = StructType.fromJson(test)

    print(test_schema)

    # schema = StructType(ArrayType(StructType(StructField("ticker", StringType()),
    #                          StructField("has_cashtag", BooleanType()),
    #                          StructField("ticker_in_text", BooleanType()))))
