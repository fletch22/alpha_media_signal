import re
from pathlib import Path
from typing import Dict

import pandas as pd
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql.functions import explode
from pyspark.sql.functions import udf, struct
from pyspark.sql.types import IntegerType, StringType
from pyspark.sql.types import StructType, StructField, BooleanType, ArrayType, Row
from retry import retry

from ams.config import logger_factory, constants
from ams.config.constants import ensure_dir
from ams.pipes import batchy_bae
from ams.services import dataframe_services
from ams.services import spark_service
from ams.services import twitter_service, file_services
from ams.services.dataframe_services import PersistedDataFrameTypes

logger = logger_factory.create(__name__)

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

entity_comma = '&#44;'
line_ending_pattern = re.compile("[\r\n]")
CHUNK_SIZE = 16

search_tuples = None
schema = StructType(fields=[StructField('place_country', StringType()),
                            StructField('place_full_name', StringType()),
                            StructField('place_name', StringType()),
                            ])


def get_search_tuples():
    global search_tuples
    if search_tuples is None:
        search_tuples = twitter_service.get_ticker_searchable_tuples()
    return search_tuples


def clean_text(text: str):
    global line_ending_pattern
    global entity_comma
    result = text
    if text is not None and len(text) > 0:
        result = re.sub(line_ending_pattern, '', text)
        result = re.sub(",", entity_comma, result)
    return result


clean_text_udf = udf(clean_text, T.StringType())


def get_cashtag_info(ticker: str, has_cashtag: bool, ticker_in_text: bool) -> Dict:
    return {"ticker": ticker, "has_cashtag": has_cashtag, "ticker_in_text": ticker_in_text}


def fix_columns(df: DataFrame):
    sel_columns = ['created_at',
                   'id',
                   'text',
                   'truncated',
                   'source',
                   'in_reply_to_status_id',
                   'in_reply_to_user_id',
                   'in_reply_to_screen_name',
                   'contributors',
                   'is_quote_status',
                   'retweet_count',
                   'favorite_count',
                   'retweeted',
                   'possibly_sensitive',
                   'lang',
                   F.col('entities.user_mentions')[0].alias('entities_user_mentions_0').cast(StringType()),
                   F.col('entities.user_mentions')[1].alias('entities_user_mentions_1').cast(StringType()),
                   F.col('entities.user_mentions')[2].alias('entities_user_mentions_2').cast(StringType()),
                   F.col('entities.user_mentions')[3].alias('entities_user_mentions_3').cast(StringType()),
                   F.col('entities.urls')[0].alias('entities_urls_0').cast(StringType()),
                   F.col('entities.urls')[1].alias('entities_urls_1').cast(StringType()),
                   F.col('entities.urls')[2].alias('entities_urls_2').cast(StringType()),
                   F.col('entities.urls')[3].alias('entities_urls_3').cast(StringType()),
                   F.col('metadata.iso_language_code').alias('metadata_iso_language_code'),
                   F.col('metadata.result_type').alias('metadata_result_type'),
                   F.col('user.id').alias('user_id'),
                   F.col('user.name').alias('user_name'),
                   F.col('user.screen_name').alias('user_screen_name'),
                   F.col('user.location').alias('user_location'),
                   F.col('user.description').alias('user_description'),
                   F.col('user.url').alias('user_url'),
                   F.col('user.protected').alias('user_protected'),
                   F.col('user.followers_count').alias('user_followers_count').cast(IntegerType()),
                   F.col('user.friends_count').alias('user_friends_count').cast(IntegerType()),
                   F.col('user.listed_count').alias('user_listed_count'),
                   F.col('user.created_at').alias('user_created_at'),
                   F.col('user.favourites_count').alias('user_favourites_count').cast(IntegerType()),
                   F.col('user.utc_offset').alias('user_utc_offset'),
                   F.col('user.time_zone').alias('user_time_zone'),
                   F.col('user.geo_enabled').alias('user_geo_enabled'),
                   F.col('user.verified').alias('user_verified'),
                   F.col('user.statuses_count').alias('user_statuses_count').cast(IntegerType()),
                   F.col('user.lang').alias('user_lang'),
                   F.col('user.contributors_enabled').alias('user_contributors_enabled'),
                   F.col('user.is_translator').alias('user_is_translator'),
                   F.col('user.is_translation_enabled').alias('user_is_translation_enabled'),
                   F.col('user.profile_background_color').alias('user_profile_background_color'),
                   F.col('user.profile_background_image_url').alias('user_profile_background_image_url'),
                   F.col('user.profile_background_image_url_https').alias('user_profile_background_image_url_https'),
                   F.col('user.profile_background_tile').alias('user_profile_background_tile'),
                   F.col('user.profile_image_url').alias('user_profile_image_url'),
                   F.col('user.profile_image_url_https').alias('user_profile_image_url_https'),
                   F.col('user.profile_banner_url').alias('user_profile_banner_url'),
                   F.col('user.profile_link_color').alias('user_profile_link_color'),
                   F.col('user.profile_sidebar_border_color').alias('user_profile_sidebar_border_color'),
                   F.col('user.profile_sidebar_fill_color').alias('user_profile_sidebar_fill_color'),
                   F.col('user.profile_text_color').alias('user_profile_text_color'),
                   F.col('user.profile_use_background_image').alias('user_profile_use_background_image'),
                   F.col('user.has_extended_profile').alias('user_has_extended_profile'),
                   F.col('user.default_profile').alias('user_default_profile'),
                   F.col('user.default_profile_image').alias('user_default_profile_image'),
                   F.col('user.following').alias('user_following'),
                   F.col('user.follow_request_sent').alias('user_follow_request_sent'),
                   F.col('user.notifications').alias('user_notifications'),
                   F.col('user.translator_type').alias('user_translator_type'),
                   F.col('f22_place.place_country').alias('place_country').cast(StringType()),
                   F.col('f22_place.place_full_name').alias('place_full_name').cast(StringType()),
                   F.col('f22_place.place_name').alias('place_name').cast(StringType())
                   ]

    df = df.select(*sel_columns)
    return df.drop(*['user', 'metadata', 'entities', 'f22_place'])


def clean_columns(df: DataFrame):
    return df.withColumn("text", clean_text_udf(F.col("text"))) \
        .withColumn("user_name", clean_text_udf(F.col("user_name"))) \
        .withColumn("user_screen_name", clean_text_udf(F.col("user_screen_name"))) \
        .withColumn("user_location", clean_text_udf(F.col("user_location"))) \
        .withColumn("user_description", clean_text_udf(F.col("user_description"))) \
        .withColumn("entities_user_mentions_0", clean_text_udf(F.col("entities_user_mentions_0"))) \
        .withColumn("entities_user_mentions_1", clean_text_udf(F.col("entities_user_mentions_1"))) \
        .withColumn("entities_user_mentions_2", clean_text_udf(F.col("entities_user_mentions_2"))) \
        .withColumn("entities_user_mentions_3", clean_text_udf(F.col("entities_user_mentions_3"))) \
        .withColumn("entities_urls_0", clean_text_udf(F.col("entities_urls_0"))) \
        .withColumn("entities_urls_1", clean_text_udf(F.col("entities_urls_1"))) \
        .withColumn("entities_urls_2", clean_text_udf(F.col("entities_urls_2"))) \
        .withColumn("entities_urls_3", clean_text_udf(F.col("entities_urls_3"))) \
        .withColumn("place_name", clean_text_udf(F.col("place_name"))) \
        .withColumn("user_url", clean_text_udf(F.col("user_url"))) \
        .withColumn("user_profile_background_image_url", clean_text_udf(F.col("user_profile_background_image_url"))) \
        .withColumn("source", clean_text_udf(F.col("source"))) \
        .withColumn("in_reply_to_screen_name", clean_text_udf(F.col("in_reply_to_screen_name"))) \
        .withColumn("place_country", clean_text_udf(F.col("place_country"))) \
        .dropDuplicates(['id'])


def get_cashtags_row_wise(row: Row):
    cashtags_stock = []

    row_dict = row.asDict()
    all_thing = ''

    text = ''
    text_len = 0
    for k in row_dict.keys():
        if k.endswith('_lc'):
            if k == 'text_lc':
                text = row_dict[k]
                if text is None:
                    text = ''
                text_len = len(str(text))
            else:
                cell = row_dict[k]
                cell = '' if cell is None else cell

                if type(cell) != 'str':
                    cell = str(cell)

                if cell is None:
                    cell = ''
                all_thing += cell
    all_thing = text + all_thing

    for s in get_search_tuples():
        ticker = s[0]
        ticker_lc = ticker.lower()
        name_lc = s[1].lower()

        index = all_thing.find(f'${ticker_lc}')
        if index > -1:
            ticker_in_text = True if index < text_len else False
            cashtags_stock.append(get_cashtag_info(ticker=ticker, has_cashtag=True, ticker_in_text=ticker_in_text))
        else:
            index_ticker = all_thing.find(ticker_lc)
            index_name = all_thing.find(name_lc)

            if index_ticker > -1 and index_name > -1:
                ticker_in_text = True if index_ticker < text_len else False
                cashtags_stock.append(get_cashtag_info(ticker=ticker, has_cashtag=False, ticker_in_text=ticker_in_text))

        num_other_tickers = len(cashtags_stock) - 1
        for tag in cashtags_stock:
            tag['num_other_tickers_in_tweet'] = num_other_tickers

    return cashtags_stock


def find_tickers_and_explode(df: DataFrame):
    columns_to_search = ['text', 'source', 'entities_user_mentions_0', 'entities_user_mentions_1', 'entities_user_mentions_2', 'entities_user_mentions_3',
                         'entities_urls_0', 'entities_urls_1', 'entities_urls_2', 'entities_urls_3', 'user_description', 'user_url']

    lc_cols = []
    for c in columns_to_search:
        lc_cols.append(f'{c}_lc')
        df = df.withColumn(f'{c}_lc', F.lower(F.col(c)))

    cashtags_schema = ArrayType(StructType(fields=[StructField('ticker', StringType()),
                                                   StructField('has_cashtag', BooleanType()),
                                                   StructField('ticker_in_text', BooleanType()),
                                                   StructField('num_other_tickers_in_tweet', IntegerType())
                                                   ]))

    get_cashtags_row_wise_udf = udf(get_cashtags_row_wise, cashtags_schema)

    df = df.withColumn("f22", get_cashtags_row_wise_udf((struct([df[x] for x in df.columns]))))
    df = df.withColumn('f22', explode(F.col('f22')))

    se_columns = list(set(df.columns) - set(lc_cols)) + [F.col('f22.ticker').alias('f22_ticker'),
                                                         F.col('f22.has_cashtag').alias('f22_has_cashtag'),
                                                         F.col('f22.ticker_in_text').alias('f22_ticker_in_text'),
                                                         F.col('f22.num_other_tickers_in_tweet').alias('f22_num_other_tickers_in_tweet')
                                                         ]
    return df.select(*se_columns).drop('f22')


@retry(tries=3)
def persist(df: DataFrame, output_drop_folder_path: Path, prefix: str = "tweets_flat", file_type: PersistedDataFrameTypes = PersistedDataFrameTypes.PARQUET,
            num_output_files: int = -1):
    dataframe_services.persist_dataframe(df=df,
                                         output_drop_folder_path=output_drop_folder_path,
                                         prefix=prefix,
                                         num_output_files=num_output_files,
                                         file_type=file_type)


def clean_unexpected_null_column(place: dict):
    result = {"place_country": None, "place_full_name": None, "place_name": None}
    if place is not None:
        result["place_country"] = place["country"]
        result["place_full_name"] = place["full_name"]
        result["place_name"] = place["name"]
    return result


clean_unexpected_null_column_udf = udf(clean_unexpected_null_column, schema)


def chunk_it(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def clean_place(df: DataFrame):
    df = df.withColumn("f22_place", clean_unexpected_null_column_udf(F.col("place")))
    df = df.drop("place")

    return df


def process(source_dir_path: Path, output_dir_path: Path):
    spark = spark_service.get_or_create(app_name='twitter')
    sc = spark.sparkContext
    log4jLogger = sc._jvm.org.apache.log4j
    LOGGER = log4jLogger.LogManager.getLogger(__name__)
    LOGGER.info("pyspark script logger initialized")

    files = file_services.list_files(parent_path=source_dir_path, ends_with=".txt.in_transition")
    files = [f for f in files if f.stat().st_size > 0]
    files = [str(f) for f in files]

    logger.info(f"Number of files: {len(files)}")

    df = spark.read.option("charset", "UTF-8").json(files)

    # df_init = df_init.sample(fraction=.01)

    df = df.dropDuplicates(['id'])

    df = clean_place(df=df)

    df = df.persist()

    df = fix_columns(df=df)

    df = df.persist()

    df = clean_columns(df=df)

    df = df.persist()

    df = find_tickers_and_explode(df=df)

    df = df.persist()

    logger.info(f"Will attempt to write {CHUNK_SIZE} files to {output_dir_path}")
    persist(df=df, output_drop_folder_path=output_dir_path)

    df.unpersist()



def start(source_dir_path: Path, twitter_root_path: Path, snow_plow_stage: bool, should_delete_leftovers: bool):
    file_services.unnest_files(parent=source_dir_path, target_path=source_dir_path, filename_ends_with=".txt")

    output_dir_path = Path(twitter_root_path, 'flattened_drop', "main")
    ensure_dir(output_dir_path)

    batchy_bae.ensure_clean_output_path(output_dir_path, should_delete_remaining=should_delete_leftovers)

    batchy_bae.start(source_path=source_dir_path, out_dir_path=output_dir_path,
                     process_callback=process, should_archive=False,
                     snow_plow_stage=snow_plow_stage, should_delete_leftovers=should_delete_leftovers)

    return output_dir_path


if __name__ == '__main__':
    twit_root_dir = Path(constants.TEMP_PATH, "twitter")
    src_dir_path = Path(twit_root_dir, "smallified_raw_drop", "main")

    start(source_dir_path=src_dir_path,
          twitter_root_path=twit_root_dir,
          snow_plow_stage=False,
          should_delete_leftovers=False)