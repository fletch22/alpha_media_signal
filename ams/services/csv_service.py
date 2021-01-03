import csv
from pathlib import Path
from typing import List, Dict


def write_dicts_as_csv(output_file_path: Path, overwrite: bool, rows: List[Dict]):
    field_names = rows[0].keys()

    if overwrite or not output_file_path.exists():
        with open(str(output_file_path), 'w+', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=field_names)
            writer.writeheader()
    with open(str(output_file_path), 'a+', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=field_names)

        for r in rows:
            writer.writerow(r)