from pathlib import Path

from ams.config import constants
from ams.services import file_services


def test_file_walk():
    # Arrange
    parent = Path(__file__).parent.parent
    files = file_services.walk(parent)

    # Act
    print(files)

    # Assert
    assert (len(files) > 0)


def test_list_folders():
    # Arrange
    source = constants.TWITTER_OUTPUT_RAW_PATH

    # Act
    dirs = file_services.list_child_folders(parent_path=source)

    print(dirs)
    # Assert
