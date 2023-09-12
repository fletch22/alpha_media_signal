import json
import shutil
from pathlib import Path

from pyspark.sql import functions as F
from transformers import GPT2Config, TFGPT2LMHeadModel, GPT2Tokenizer

from ams.config import constants
from ams.models.nlp.bar_eats_you.tokenizer import get_text_input_files, get_corpus_path
from ams.services import spark_service
from ams.utils.Stopwatch import Stopwatch


def write_tokenized_output(output_par_path: Path, tokenizer, single_string: str, file_output_ndx: int):
    string_tokenized = tokenizer.encode(single_string)
    output_path = Path(output_par_path, f"{file_output_ndx}.txt")
    with open(str(output_path), "w") as fw:
        st_json = json.dumps(string_tokenized)
        fw.write(st_json)


def init(save_path: Path, lang: str):
    # loading tokenizer from the saved model path
    print("About to create GPT2Tokenizer from pretrained files.")
    tokenizer = GPT2Tokenizer.from_pretrained(save_path)
    tokenizer.add_special_tokens({
        "eos_token": "</s>",
        "bos_token": "<s>",
        "unk_token": "<unk>",
        "pad_token": "<pad>",
        "mask_token": "<mask>"
    })
    # creating the configurations from which the model can be made
    config = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    # creating the model
    print("About to construct initial model.")
    model = TFGPT2LMHeadModel(config)

    single_string = ''
    print("About to get text input files.")
    paths = get_text_input_files(lang=lang, max_files=None)
    print(f"Got {len(paths)} paths.")

    output_par_path = Path(constants.WIKI_CORPUS_PATH, "tokenized", lang)
    if output_par_path.exists():
        shutil.rmtree(output_par_path)
    output_par_path.mkdir()

    single_string = ""
    file_output_ndx = 0
    for ndx, filename in enumerate(paths):
        with open(filename, "r", encoding='utf-8') as f:
            x = f.read()
        single_string = single_string.join([x, tokenizer.eos_token])
        if ndx != 0 and ndx % 25000 == 0:
            write_tokenized_output(output_par_path=output_par_path,
                                   tokenizer=tokenizer,
                                   single_string=single_string,
                                   file_output_ndx=file_output_ndx)
            file_output_ndx += 1
            single_string = ""

    if len(single_string) > 0:
        write_tokenized_output(output_par_path=output_par_path,
                               tokenizer=tokenizer,
                               single_string=single_string,
                               file_output_ndx=file_output_ndx)


# @udf(returnType=StringType())
# def convert_timestamp_to_nyc_date_str_udf(utc_timestamp):
#     return convert_timestamp_to_nyc_date_str(utc_timestamp=utc_timestamp)

def get_tokenized_files_path(lang: str = "en"):
    corpus_path = get_corpus_path(lang=lang)
    return Path(corpus_path, "tokenized_data")


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def process():
    stop_watch = Stopwatch()

    lang = "en"
    print("About to create GPT2Tokenizer from pretrained files.")
    save_path = str(get_tokenized_files_path(lang=lang))
    tokenizer = GPT2Tokenizer.from_pretrained(save_path)
    tokenizer.add_special_tokens({
        "eos_token": "</s>",
        "bos_token": "<s>",
        "unk_token": "<unk>",
        "pad_token": "<pad>",
        "mask_token": "<mask>"
    })
    # creating the configurations from which the model can be made
    config = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    # creating the model
    print("About to construct initial model.")
    model = TFGPT2LMHeadModel(config)
    max_files = None

    output_par_path = Path(constants.WIKI_CORPUS_PATH, "tokenized", lang)
    if output_par_path.exists():
        shutil.rmtree(output_par_path)
    output_par_path.mkdir()

    # spark = SparkSession.builder.appName("bar_eats_you").getOrCreate()
    spark_session = spark_service.get_or_create("bar_eats_you")

    paths = get_text_input_files(lang=lang, max_files=max_files)
    paths = [str(p) for p in paths]

    chunked = chunks(paths, 100000)
    # paths = ['E:\\workspaces\\data\\wiki_pages\\corpus\\en_corpus\\article_1.txt', 'E:\\workspaces\\data\\wiki_pages\\corpus\\en_corpus\\article_10.txt',
    #          'E:\\workspaces\\data\\wiki_pages\\corpus\\en_corpus\\article_100.txt', 'E:\\workspaces\\data\\wiki_pages\\corpus\\en_corpus\\article_1000.txt',
    #          'E:\\workspaces\\data\\wiki_pages\\corpus\\en_corpus\\article_10000.txt', 'E:\\workspaces\\data\\wiki_pages\\corpus\\en_corpus\\article_100000.txt',
    #          'E:\\workspaces\\data\\wiki_pages\\corpus\\en_corpus\\article_1000000.txt', 'E:\\workspaces\\data\\wiki_pages\\corpus\\en_corpus\\article_1000001.txt',
    #          'E:\\workspaces\\data\\wiki_pages\\corpus\\en_corpus\\article_1000002.txt', 'E:\\workspaces\\data\\wiki_pages\\corpus\\en_corpus\\article_1000003.txt']

    for file_output_ndx, chunk_paths in enumerate(chunked):
        df = spark_session.read.text(chunk_paths)
        df = df.withColumn("test", F.lit("t"))
        df_grouped = df.groupBy(F.col("test")).agg(F.collect_list(F.col("value")).alias("concat_value"))
        sep = " "
        df = df_grouped.select("concat_value") \
            .withColumn("concat_value", F.concat_ws(sep, F.col("concat_value")))
        df_grouped = None
        print(df.columns)
        single_string = df.select(F.col("concat_value")).first()[0]
        df = None

        write_tokenized_output(output_par_path=output_par_path, tokenizer=tokenizer, single_string=single_string, file_output_ndx=file_output_ndx)

        print(f"Length: {len(single_string):,}")
        # print(value)

    stop_watch.end("Finished processing")


if __name__ == '__main__':
    process()
