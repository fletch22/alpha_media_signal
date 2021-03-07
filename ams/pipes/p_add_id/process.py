import os
import shutil
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from ams.config import constants, logger_factory
from ams.config.constants import ensure_dir
from ams.pipes import batchy_bae
from ams.services import file_services

logger = logger_factory.create(__name__)


def move_nested(parent: Path, target_path: Path):
    files = file_services.list_files(parent_path=parent, ends_with=".parquet", use_dir_recursion=True)

    for f in files:
        file_dest = str(Path(target_path, f.name))
        shutil.move(str(f), dst=file_dest)


def process(source_dir_path: Path, output_dir_path: Path):
    files = file_services.list_files(source_dir_path, ends_with=".parquet.in_transition", use_dir_recursion=True)

    num_to_proc = len(files)
    logger.info(f"{num_to_proc} files found to process.")

    df_all = []
    rows = 0
    count_limit = 400000
    total_count = 0
    for f in files:
        logger.info(f"Processing {f}")
        df = pd.read_parquet(f)
        total_count += df.shape[0]

        cols_str = ['entities_user_mentions_2', 'user_profile_background_image_url_https', 'text', 'user_created_at', 'user_default_profile_image', 'user_name',
                    'user_profile_background_tile', 'metadata_result_type', 'entities_urls_0', 'user_location', 'user_notifications', 'user_lang',
                    'entities_urls_3', 'place_country', 'user_description', 'lang',
                    'source', 'user_url', 'id', 'user_profile_banner_url', 'entities_user_mentions_0', 'user_id',
                    'in_reply_to_screen_name', 'entities_urls_1', 'user_geo_enabled', 'user_profile_link_color', 'place_name', 'user_translator_type', 'is_quote_status',
                    'user_statuses_count', 'user_profile_background_image_url', 'entities_user_mentions_1', 'in_reply_to_user_id', 'contributors',
                    'user_profile_use_background_image',
                    'user_profile_image_url_https', 'user_profile_sidebar_border_color', 'user_following', 'user_profile_sidebar_fill_color', 'user_screen_name',
                    'in_reply_to_status_id',
                    'user_profile_text_color', 'user_profile_background_color', 'entities_user_mentions_3', 'created_at', 'entities_urls_2', 'metadata_iso_language_code',
                    'user_profile_image_url', 'user_default_profile', 'f22_ticker']

        for c in cols_str:
            df[c] = df[c].astype(str).apply(replace_chars)

        cols_numeric = ['user_followers_count', 'user_time_zone', 'user_friends_count',
                        'user_utc_offset', 'favorite_count', 'f22_num_other_tickers_in_tweet',
                        'user_favourites_count', 'retweet_count', 'user_listed_count']

        for c in cols_numeric:
            df[c] = df[c].apply(convert_to_numeric).astype(np.float64)

        cols_bool = ['user_follow_request_sent', 'user_has_extended_profile', 'retweeted', 'user_is_translator',
                     'user_is_translation_enabled', 'truncated', 'possibly_sensitive', 'f22_has_cashtag', 'f22_ticker_in_text',
                     'user_verified', 'user_protected', 'user_contributors_enabled']

        for c in cols_bool:
            df[c] = df[c].apply(convert_to_boolean).astype(bool)

        df_all.append(df)

        rows += df.shape[0]
        if rows > count_limit:
            persist_df(df_all=df_all, output_dir_path=output_dir_path)
            df_all = []
            rows = 0

    if len(df_all) > 0:
        persist_df(df_all=df_all, output_dir_path=output_dir_path)

    logger.info(f"Total records processed: {total_count}")


def add_synth_id(df: pd.DataFrame):
    df["f22_id"] = df.apply(lambda row: hash(str(row["user_name"]) + str(row["text"]) + str(row["created_at"])), axis=1)


def persist_df(df_all: List[pd.DataFrame], output_dir_path: Path):
    df_large = pd.concat(df_all, axis=0)

    add_synth_id(df_large)

    folder_path_str = file_services.create_unique_filename(parent_dir=str(output_dir_path), prefix="id_fixed", extension="parquet")
    df_large.to_parquet(folder_path_str)


def convert_to_boolean(value):
    if value is not None:
        if type(value) == str:
            value = value.lower()
            if value == 'true':
                value = True
            elif value == 'false':
                value = False
            else:
                value = None
    return value


def convert_to_numeric(value):
    val_num = value
    try:
        val_num = float(value)
    except Exception as e:
        val_num = None
    return val_num


def replace_chars(text):
    return text.replace("\\\"", "").replace("\"", "").replace("\"", "")


def start(source_dir_path: Path, twitter_root_path: Path, snow_plow_stage: bool, should_delete_leftovers: bool):
    move_nested(parent=source_dir_path, target_path=source_dir_path)

    output_dir_path = Path(twitter_root_path, "id_fixed", "main")
    ensure_dir(output_dir_path)

    batchy_bae.ensure_clean_output_path(output_dir_path, should_delete_remaining=should_delete_leftovers)

    batchy_bae.start(source_path=source_dir_path, out_dir_path=output_dir_path, process_callback=process,
                     should_delete_leftovers=should_delete_leftovers,
                     should_archive=False, snow_plow_stage=snow_plow_stage)

    return output_dir_path
