import shutil
from datetime import datetime
from pathlib import Path

from ams.config import constants
from ams.config import logger_factory
from ams.config.constants import ensure_dir
from ams.pipes import pipe_fitter
from ams.services import file_services
from ams.twitter.twitter_ml_utils import get_next_market_date
from ams.utils import date_utils
from tests.services import zip_service

logger = logger_factory.create(__name__)

# NOTE: 2021-02-20: chris.flesche: 3GB
CHUNK_SIZE = 3E9


class ReProcessZip:
    accum = []
    source_zips = None
    twitter_root_path = None
    raw_root_acc = None
    raw_root_main = None
    end_drop_path = None
    input_archive_path = None
    already_proc_filenames = []
    total_acc_size = 0
    last_proc_dt = None

    def __init__(self, source_zips: Path, twitter_root_path: Path, input_archive_path: Path):
        self.source_zips = source_zips
        self.twitter_root_path = twitter_root_path
        self.input_archive_path = input_archive_path

        self.raw_root_main = Path(twitter_root_path, "raw_drop", "main")
        self.raw_root_acc = Path(twitter_root_path, "raw_drop", "accumulator")
        ensure_dir(self.raw_root_acc)
        self.end_drop_path = Path(twitter_root_path, "end_drop")

        output_dir_path = Path(input_archive_path, "leftover_raw_main")
        move_files_in_flder(source_dir_path=self.raw_root_main, output_dir_path=output_dir_path)

    def process_all(self):
        for tmp_zip_path in zip_service.raw_from_zip_generator(source_path=self.source_zips, already_proc_filenames=self.already_proc_filenames):
            self.process_zip(tmp_zip_path=tmp_zip_path)

        self.move_and_proc_accumulated()

    def process_zip(self, tmp_zip_path: Path):
        logger.info(f"Extracted {tmp_zip_path.name}")

        raw_main_path = move_file_to_folder(tmp_zip_path, self.raw_root_acc)

        self.total_acc_size += raw_main_path.stat().st_size

        has_enough_data = True if self.total_acc_size > CHUNK_SIZE else False
        if has_enough_data:
            self.move_and_proc_accumulated()

            self.accum = []
            self.total_acc_size = 0
        else:
            self.accum.append(raw_main_path)

    def move_and_proc_accumulated(self):
        for a_path in self.accum:
            move_file_to_folder(a_path, self.raw_root_main)

        skip_external_data_dl = False
        if self.last_proc_dt is not None:
            skip_external_data_dl = ReProcessZip.was_less_than_market_day_ago(self.last_proc_dt)

        if skip_external_data_dl:
            logger.info(f"Will skip external data download.")

        # FIXME: 2021-02-20: chris.flesche: Temporary
        skip_external_data_dl = True
        pipe_fitter.process(twitter_root_path=twitter_root_path,
                            end_drop_path=self.end_drop_path,
                            input_archive_path=self.input_archive_path,
                            skip_external_data_dl=skip_external_data_dl)

        self.last_proc_dt = datetime.now()

    @classmethod
    def was_less_than_market_day_ago(cls, dt: datetime):
        dt_str = date_utils.get_standard_ymd_format(dt)
        dt_next_dt = date_utils.parse_std_datestring(get_next_market_date(dt_str, 1))

        return dt_next_dt < datetime.now()


def move_files_in_flder(source_dir_path, output_dir_path: Path):
    ensure_dir(output_dir_path)
    files = file_services.list_files(parent_path=source_dir_path)
    for f in files:
        move_file_to_folder(a_path=f, target_dir_path=output_dir_path)


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
    source_zips = Path(constants.TEST_TEMP_PATH, "raw_zip_archive")
    twitter_root_path = Path(constants.TEST_TEMP_PATH, "twitter")
    input_archive_path = Path(constants.TEST_TEMP_PATH, "reprocess")

    CHUNK_SIZE = 100000

    reprocess(source_zips=source_zips,
              twitter_root_path=twitter_root_path,
              input_archive_path=input_archive_path)