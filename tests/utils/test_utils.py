from alpha_media_signal import utils


def test_load_common_words():
    # Arrange
    # Act
    common_words = utils.load_common_words()

    # Assert
    assert(len(common_words) > 10000)

