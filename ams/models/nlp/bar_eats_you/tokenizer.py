from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.normalizers import NFKC, Sequence
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer
from pathlib import Path
from tokenizers.models import BPE
import os

from ams.config import constants


class BPE_token(object):
    def __init__(self):
        self.tokenizer = Tokenizer(BPE())
        self.tokenizer.normalizer = Sequence([
            NFKC()
        ])
        self.tokenizer.pre_tokenizer = ByteLevel()
        self.tokenizer.decoder = ByteLevelDecoder()
        # self.tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        # self.tokenizer.pre_tokenizer = Whitespace()

    def bpe_train(self, paths):
        trainer = BpeTrainer(vocab_size=50000, show_progress=True, initial_alphabet=ByteLevel.alphabet(), special_tokens=[
            "<s>",
            "<pad>",
            "</s>",
            "<unk>",
            "<mask>"
        ])
        self.tokenizer.train(paths, trainer)

        # self.tokenizer.train(paths, trainer)

    def save_tokenizer(self, location: str, prefix=None):
        if not os.path.exists(location):
            os.makedirs(location)
        self.tokenizer.model.save(location, prefix)


def get_corpus_path(lang: str = "en"):
    return Path(constants.WIKI_CORPUS_PATH, f"{lang}_corpus")


def get_tokenized_files_path(lang: str = "en"):
    corpus_path = get_corpus_path(lang=lang)
    return Path(corpus_path, "tokenized_data")


def get_text_input_files(lang: str, max_files: int = None):
    corpus_path = get_corpus_path(lang=lang)
    # the folder 'text' contains all the files
    file_glob_lst = corpus_path.glob("**/*.txt")
    file_paths = []
    for ndx, file in enumerate(file_glob_lst):
        if max_files is not None:
            if ndx >= max_files:
                break
        file_paths.append(str(file))
    return file_paths


if __name__ == '__main__':
    lang = "en"

    paths = get_text_input_files(lang=lang)

    print(f"Found {len(paths):,} files.")
    tokenizer = BPE_token()
    # train the tokenizer model
    tokenizer.bpe_train(paths)
    # saving the tokenized data in our specified folder
    save_path = get_tokenized_files_path(lang=lang)
    if save_path.exists():
        raise Exception("Save path already exists.")
    tokenizer.save_tokenizer(location=str(save_path))
