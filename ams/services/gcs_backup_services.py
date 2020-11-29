import os
import shutil
import subprocess
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path

from ams import config
from ams.config import constants, logger_factory
from ams.services import file_services
from ams.utils import date_utils
from ams.utils.Stopwatch import Stopwatch

logger = logger_factory.create(__name__)


def zipdir(path, ziph, omit_folders: list = None):
    root_len = len(path)
    for root, dirs, files in os.walk(path):
        for f in files:
            source_path = os.path.join(root, f)
            arcname = f"{root[root_len:]}/{f}"
            skip_file = False
            for o in omit_folders:
                if arcname.startswith(o):
                    skip_file = True

            if not skip_file:
                print(f"{arcname}")
                ziph.write(source_path, arcname)


def backup_folder(backup_source_dir, output_path: Path):
    zipf = zipfile.ZipFile(str(output_path), 'w', zipfile.ZIP_DEFLATED)

    zipdir(backup_source_dir, zipf, omit_folders=["\\venv"])

    zipf.close()


def backup_file(backup_source_file: Path, output_path: Path):
    with zipfile.ZipFile(str(output_path), 'w') as zipf:
        zipf.write(str(backup_source_file), arcname=backup_source_file.name, compress_type=zipfile.ZIP_DEFLATED)


def backup_project():
    backup_root = str(config.constants.BACKUP_ROOT_PATH)
    backup_dest_dirname = os.path.join(backup_root, date_utils.format_file_system_friendly_date(datetime.now()))
    os.makedirs(backup_dest_dirname, exist_ok=True)

    print(f"Will back up to: {backup_dest_dirname}")

    volume = file_services.get_windows_drive_volume_label(backup_root[0])
    logger.info(f"Volume Name: '{volume}'.")

    if volume != config.constants.BACKUP_VOLUME_LABEL:
        raise Exception("Error. Backup failed! Volume label does not match expected label.")

    stopWatch = Stopwatch(start_now=True)

    source_dir_project = os.path.join(constants.PROJECT_ROOT)

    output_path = Path(backup_dest_dirname, f"stock-predictor.zip")
    backup_folder(source_dir_project, output_path=output_path)

    source_dir_project = str(config.constants.ALPHA_MEDIA_SIGNAL_PROJ)
    output_path = Path(backup_dest_dirname, f"alpha_media_signal.zip")
    backup_folder(source_dir_project, output_path=output_path)

    source_dir_project = str(config.constants.NEWS_GESTALT_PROJ)
    output_path = Path(backup_dest_dirname, f"news_gestalt.zip")
    backup_folder(source_dir_project, output_path=output_path)

    stopWatch.end(msg="Backup")


def backup_project_to_gcs(include_data: bool = True):
    stopWatch = Stopwatch(start_now=True)

    with tempfile.TemporaryDirectory() as t:
        # t = config.constants.TWITTER_OUTPUT
        print(t)

        source_dir_project = str(config.constants.ALPHA_MEDIA_SIGNAL_PROJ)
        output_path = Path(t, f"project.zip")
        backup_folder(source_dir_project, output_path=output_path)

        if include_data:
            source_dir_project = str(constants.SHAR_SPLIT_EQUITY_EOD_DIR)
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

        shell_script = Path(constants.PROJECT_ROOT, "scripts", "gc_install.sh")
        output_path = Path(t, shell_script.name)
        shutil.copy(shell_script, output_path)

        shell_script = Path(constants.PROJECT_ROOT, "scripts", "gc_init.sh")
        output_path = Path(t, shell_script.name)
        shutil.copy(shell_script, output_path)

        backup_file(constants.TWITTER_TEXT_LABEL_TRAIN_PATH, output_path=Path(t, "twitter_text_with_proper_labels.zip"))

        command = ["gsutil", "rsync", str(t), "gs://api_uploads/twitter"]
        completed_process = subprocess.run(command, shell=True, check=True)

        print(completed_process.returncode)

    stopWatch.end(msg="Backup")


if __name__ == '__main__':
    backup_project_to_gcs(include_data=False)
