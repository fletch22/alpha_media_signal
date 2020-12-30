import os
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

from ams import config
from ams.config import constants, logger_factory
from ams.services.backup_service import backup_folder, backup_file
from ams.utils.Stopwatch import Stopwatch

logger = logger_factory.create(__name__)


def backup_project_to_gcs(include_data: bool = True):
    stopWatch = Stopwatch(start_now=True)

    with tempfile.TemporaryDirectory() as t:
        # t = config.constants.TWITTER_OUTPUT
        print(t)

        source_dir_project = config.constants.ALPHA_MEDIA_SIGNAL_PROJ
        output_path = Path(t, f"project.zip")
        backup_folder(source_dir_project, output_path=output_path)

        if include_data:
            source_dir_project = constants.SHAR_SPLIT_EQUITY_EOD_DIR
            output_path = Path(t, f"eods.zip")
            backup_folder(source_dir_project, output_path=output_path)

            backup_file(constants.SHAR_CORE_FUNDY_FILE_PATH, output_path=Path(t, "shar_core_fundamentals.zip"))

            backup_file(constants.SHAR_TICKER_DETAIL_INFO_PATH, output_path=Path(t, "shar_tickers.zip"))

            shell_script = constants.SHARADAR_ACTIONS_FILEPATH
            output_path = Path(t, shell_script.name)
            shutil.copy(shell_script, output_path)

            shell_script = constants.TICKER_NAME_SEARCHABLE_PATH
            output_path = Path(t, shell_script.name)
            shutil.copy(shell_script, output_path)

            shell_script = constants.BERT_REVIEWS_DATA_PATH
            output_path = Path(t, shell_script.name)
            shutil.copy(shell_script, output_path)

            shell_script = constants.COLA_IN_DOMAIN_TRAIN_PATH
            output_path = Path(t, shell_script.name)
            shutil.copy(shell_script, output_path)

            shell_script = constants.COLA_IN_DOMAIN_DEV_PATH
            output_path = Path(t, shell_script.name)
            shutil.copy(shell_script, output_path)

            shell_script = constants.COLA_OUT_OF_DOMAIN_DEV_PATH
            output_path = Path(t, shell_script.name)
            shutil.copy(shell_script, output_path)

            backup_file(constants.TWITTER_TEXT_LABEL_TRAIN_PATH, output_path=Path(t, "twitter_text_with_proper_labels.zip"))

            source_data_dir = Path(constants.DATA_PATH, "twitter", "learning_prep_drop", "main")
            output_path = Path(t, f"lpd.zip")
            backup_folder(source_data_dir, output_path=output_path)

            backup_file(constants.DAILY_ROI_NASDAQ_PATH, output_path=Path(t, "daily_roi_nasdaq.parquet.zip"))

        shell_script = Path(constants.PROJECT_ROOT, "scripts", "gc_install.sh")
        output_path = Path(t, shell_script.name)
        shutil.copy(shell_script, output_path)

        shell_script = Path(constants.PROJECT_ROOT, "scripts", "gc_init.sh")
        output_path = Path(t, shell_script.name)
        shutil.copy(shell_script, output_path)

        command = ["gsutil", "rsync", str(t), "gs://api_uploads/twitter"]
        completed_process = subprocess.run(command, shell=True, check=True)

        print(completed_process.returncode)

    stopWatch.end(msg="Backup")


if __name__ == '__main__':
    backup_project_to_gcs(include_data=False)