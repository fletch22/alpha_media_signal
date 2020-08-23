from alpha_media_signal import config
from alpha_media_signal.config import constants


def test_config():
    # Arrange
    # Act
    creds = constants.FLETCH22_CREDS

    print(creds.api_key)

    # Assert
    assert(len(creds.api_key) > 10)
