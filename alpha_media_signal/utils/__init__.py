from alpha_media_signal.config import constants
import codecs


def load_common_words():
    with codecs.open(constants.TOP_100K_WORDS_PATH, encoding='utf-8') as f:
        lines = f.readlines()
        return [l.strip() for l in lines]
