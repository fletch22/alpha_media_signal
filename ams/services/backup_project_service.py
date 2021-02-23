import os
from datetime import datetime
from pathlib import Path

from ams.config import constants, logger_factory
from ams.services import file_services
from ams.services.backup_service import backup_folder
from ams.utils import date_utils
from ams.utils.Stopwatch import Stopwatch

logger = logger_factory.create(__name__)


def backup_project():
    backup_root = str(constants.BACKUP_ROOT_PATH)
    backup_dest_dirname = os.path.join(backup_root, date_utils.format_file_system_friendly_date(datetime.now()))
    os.makedirs(backup_dest_dirname, exist_ok=True)

    logger.info(f"Will back up to: {backup_dest_dirname}")

    volume = file_services.get_windows_drive_volume_label(backup_root[0])
    logger.info(f"Volume Name: '{volume}'.")

    if volume != constants.BACKUP_VOLUME_LABEL:
        raise Exception("Error. Backup failed! Volume label does not match expected label.")

    stopWatch = Stopwatch(start_now=True)

    source_dir_project = constants.PROJECT_ROOT

    output_path = Path(backup_dest_dirname, f"stock-predictor.zip")
    backup_folder(source_dir_project, output_path=output_path)

    source_dir_project = constants.ALPHA_MEDIA_SIGNAL_PROJ
    output_path = Path(backup_dest_dirname, f"alpha_media_signal.zip")
    backup_folder(source_dir_project, output_path=output_path)

    source_dir_project = constants.NEWS_GESTALT_PROJ
    output_path = Path(backup_dest_dirname, f"news_gestalt.zip")
    backup_folder(source_dir_project, output_path=output_path)

    source_dir_project = constants.BLOGGER_HIGH_SCORE_PROJ
    output_path = Path(backup_dest_dirname, f"blogger_high_score.zip")
    backup_folder(source_dir_project, output_path=output_path)

    stopWatch.end(msg="Backup")


if __name__ == '__main__':
    backup_project()