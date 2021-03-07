from pathlib import Path

from ams.services import temp_dir_service
from tests.testing_utils import create_random_text_file


def test_temp_dir():
    # Arrange
    # Act
    with temp_dir_service.TempDir() as td:
        print(td)
        create_random_text_file(parent_dir_path=td)


    # Assert
    assert(not Path(td).exists())