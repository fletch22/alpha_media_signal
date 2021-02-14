import shutil
from pathlib import Path
from unittest.mock import patch

from ams.config import constants
from ams.config.constants import ensure_dir
from ams.pipes import pipe_fitter
from ams.services import file_services, ticker_service

sample_raw_drop_path = Path(constants.TESTS_RESOURCES, "sample_raw_drop.txt")


def test_create_test_records():
    create_sample_raw_drop(num_records=10)


def create_sample_raw_drop(num_records: int = 100):
    source_path = Path(constants.TWITTER_OVERFLOW_OUTPUT, "raw_drop", "archive", "multithreaded_drop_2021-02-04_23-58-08-285.9.txt")
    with open(str(source_path), "r") as r:
        with open(str(sample_raw_drop_path), "w") as w:
            for i in range(num_records):
                line = r.readline()
                w.write(line)


def test_old_starts():
    raw_drop_main = Path(constants.TWITTER_OUTPUT_RAW_PATH, "raw_drop", "main", sample_raw_drop_path.name)
    shutil.copy(sample_raw_drop_path, raw_drop_main)

    pipe_fitter.process_old()


def test_process():
    # Arrange
    twitter_root_path = Path(constants.TEST_TEMP_PATH, "twitter")
    source_file_test_path = Path(constants.TESTS_RESOURCES, "sample_raw_drop.txt")
    test_source_path = Path(twitter_root_path, "raw_drop", "main")
    end_drop_path = Path(twitter_root_path, "end_drop")

    ensure_dir(test_source_path)
    test_path_new = Path(test_source_path, source_file_test_path.name)
    shutil.copy(source_file_test_path, test_path_new)

    output_dir_path = Path(file_services.create_unique_folder_name(constants.TEST_TEMP_PATH, prefix="archive_input"))

    with patch("ams.pipes.pipe_fitter.command_service.get_equity_daily_data", return_value=None) as mock_get_equity, \
        patch("ams.pipes.pipe_fitter.command_service.get_equity_fundamentals_data", return_value=None) as mock_get_eq_funda, \
        patch("ams.pipes.pipe_fitter.equity_performance.start") as mock_eq_perf_start:
        pipe_fitter.process(twitter_root_path=twitter_root_path, end_drop_path=end_drop_path, input_archive_path=output_dir_path)

        mock_get_equity.assert_called_once()
        mock_get_eq_funda.assert_called_once()
        mock_eq_perf_start.assert_called_once()

    # Act

    # Assert

def test_foo():
    # Arrange
    # Act
    print(ticker_service.get_ticker_eod_data("AAPL")["date"].max())
    # Assert