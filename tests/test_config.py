from ams import config
from ams.config import constants


def test_config():
    # Arrange
    # Act
    creds = constants.FLETCH22_CREDS

    print(creds.api_key)

    # Assert
    assert(len(creds.api_key) > 10)
