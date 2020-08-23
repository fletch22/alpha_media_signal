from pathlib import Path

from alpha_media_signal.services import file_services


def test_file_walk():
    # Arrange
    parent = Path(__file__).parent.parent
    files = file_services.walk(parent)

    # Act
    print(files)

    # Assert
    assert(len(files) > 0)