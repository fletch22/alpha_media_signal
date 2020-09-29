from ams.services import spark_service


def test_connection():
    # Arrange

    # Act
    spark_service.get_or_create("test")
    # Assert