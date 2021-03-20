from ams.config import constants
from ams.services import three_below_services


def test_write_lines():
    # Arrange
    rows = [{"foo": "bar"}]

    three_below_services.write_3_below_history(rows, overwrite=True)
    # Act

    # Assert
    assert(constants.THREE_BELOW_HISTORY_PATH.exists())