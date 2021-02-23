from pathlib import Path

from ams.config import constants, logger_factory
from ams.services import file_services

logger = logger_factory.create(__name__)

def test_file_walk():
    # Arrange
    parent = Path(__file__).parent.parent
    files = file_services.walk(parent)

    # Act
    logger.info(files)

    # Assert
    assert (len(files) > 0)


def test_list_folders():
    # Arrange
    source = constants.TWITTER_OUTPUT_RAW_PATH

    # Act
    dirs = file_services.list_child_folders(parent_path=source)

    logger.info(dirs)
    # Assert
