from ams.services import slack_service


def test_messaging():
    # Arrange
    # Act
    slack_service.send_direct_message_to_chris("This is a test.")
    # Assert