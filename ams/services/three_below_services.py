import csv
from typing import Dict, List

from ams.config import constants
from ams.services.csv_service import write_dicts_as_csv


def write_3_below_history(rows: List[Dict], overwrite: bool = False):
    if len(rows) == 0:
        raise Exception("Nothing to write. Rows are empty.")

    write_dicts_as_csv(output_file_path=constants.THREE_BELOW_HISTORY_PATH, overwrite=overwrite, rows=rows)



