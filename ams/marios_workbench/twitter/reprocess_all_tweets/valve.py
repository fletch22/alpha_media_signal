import sys
import traceback

from ams.config import logger_factory
from ams.marios_workbench.twitter.import_and_predict import valve as import_and_pred_valve

logger = logger_factory.create(__name__)

import shutil
from datetime import datetime
from pathlib import Path
from typing import Set

from ams.config import constants
from ams.config.constants import ensure_dir
from ams.services import file_services, temp_dir_service, zip_service
from ams.utils.date_utils import get_next_market_date
from ams.utils import date_utils

# NOTE: 2021-02-20: chris.flesche: 3GB
CHUNK_SIZE = 1E9


class ReProcessZip:
    accum = []
    src_zips = None
    trp = None
    raw_root_acc = None
    raw_root_main = None
    end_drop_path = None
    iap = None
    already_proc_filenames = set()
    total_acc_size = 0
    last_proc_dt = None

    def __init__(self, source_zips: Path, twitter_root_path: Path, input_archive_path: Path):
        self.source_zips = source_zips
        self.twitter_root_path = twitter_root_path
        self.input_archive_path = input_archive_path

        self.raw_root_main = Path(twitter_root_path, "raw_drop", "main")
        self.raw_root_acc = Path(twitter_root_path, "raw_drop", "accumulator")
        ensure_dir(self.raw_root_acc)
        file_services.clean_dir(parent_path=self.raw_root_acc)

        self.end_drop_path = Path(twitter_root_path, "end_drop")
        self.processed_path = Path(self.input_archive_path, "leftover_raw_main")

        for f in file_services.list_files(self.processed_path):
            self.already_proc_filenames.add(f.name)

    def process_all(self):
        with temp_dir_service.TempDir() as temp_dir_path:
            for tmp_zip_path in zip_service.raw_from_zip_generator(temp_dir_path=temp_dir_path, source_path=self.source_zips,
                                                                   already_proc_filenames=self.already_proc_filenames):
                self.process_zip(tmp_zip_path=tmp_zip_path)

            self.move_and_proc_accumulated()

    def process_zip(self, tmp_zip_path: Path):
        logger.info(f"Extracted {tmp_zip_path.name}")

        raw_acc_path = move_file_to_folder(tmp_zip_path, self.raw_root_acc)
        self.accum.append(raw_acc_path)

        self.total_acc_size += raw_acc_path.stat().st_size

        logger.info(f"Chunk size: {CHUNK_SIZE}; Data size: {self.total_acc_size}")

        has_enough_data = True if self.total_acc_size > CHUNK_SIZE else False

        if has_enough_data:
            self.move_and_proc_accumulated()

            self.accum = []
            self.total_acc_size = 0

    def move_and_proc_accumulated(self):
        for a_path in self.accum:
            move_file_to_folder(a_path, self.raw_root_main)

        skip_external_data_dl = True

        files = file_services.list_files(self.raw_root_main)

        logger.info(f"About to process files ... {len(files)}")

        import_and_pred_valve.process(twitter_root_path=trp,
                                      input_archive_path=self.input_archive_path,
                                      skip_external_data_dl=skip_external_data_dl,
                                      archive_raw=False,
                                      should_delete_leftovers=True)

        self.last_proc_dt = datetime.now()

        proc_files = create_dummy_files(source_dir_path=self.raw_root_main, output_dir_path=self.processed_path)

        self.already_proc_filenames |= proc_files

        print(self.already_proc_filenames)

    @classmethod
    def was_less_than_market_day_ago(cls, dt: datetime):
        dt_str = date_utils.get_standard_ymd_format(dt)
        dt_next_dt = date_utils.parse_std_datestring(get_next_market_date(dt_str))

        return dt_next_dt < datetime.now()


def create_dummy_files(source_dir_path, output_dir_path: Path) -> Set[str]:
    ensure_dir(output_dir_path)
    files = file_services.list_files(parent_path=source_dir_path)
    proc_files = set()
    for f in files:
        f.unlink()
        target_path = Path(output_dir_path, f.name)
        with open(target_path, "w+") as fw:
            fw.write("dummy file")
        proc_files.add(f.name)

    return proc_files


def move_file_to_folder(a_path, target_dir_path):
    raw_main_path = Path(target_dir_path, a_path.name)
    shutil.move(a_path, raw_main_path)
    return raw_main_path


def reprocess(source_zips: Path, twitter_root_path: Path, input_archive_path: Path):
    reprocess_zip = ReProcessZip(source_zips=source_zips,
                                 twitter_root_path=twitter_root_path,
                                 input_archive_path=input_archive_path)
    reprocess_zip.process_all()


if __name__ == '__main__':
    src_zips = Path(constants.TEMP_PATH, "raw_zip_archive")
    trp = Path(constants.TEMP_PATH, "twitter")
    iap = Path(constants.TEMP_PATH, "reprocess")

    file_services.remove_folder_read_only(dir_path=trp, recursive=True)

    CHUNK_SIZE = 1E5

    count = 0
    max_retry = 2
    while count < max_retry:
        # try:
        reprocess(source_zips=src_zips,
                  twitter_root_path=trp,
                  input_archive_path=iap)
        # except BaseException as be:
        #     traceback.print_stack(file=sys.stdout)
        #     logger.error(be)
        count += 1
        break