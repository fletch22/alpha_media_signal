import shutil
from datetime import datetime
from pathlib import Path
from typing import List

import pandas as pd

from ams.config import constants, logger_factory
from ams.config.constants import ensure_dir
from ams.pipes.p_add_id import process as add_id_process
from ams.pipes.p_add_learning_prep import process as add_learn_prep_process
from ams.pipes.p_add_sentiment import process as add_sent_process
from ams.pipes.p_coalesce import process as coalesce_process
from ams.pipes.p_flatten import process as flatten_process
from ams.pipes.p_remove_dupes import process as rem_dupes_process
from ams.pipes.p_smallify_files import process as smallify_process
from ams.pipes.p_twitter_reduction import process as twit_reduce_process
from ams.services import command_service, file_services, slack_service
from ams.services.equities import equity_performance
from ams.twitter import twitter_ml_utils, twitter_ml
from ams.utils import date_utils

logger = logger_factory.create(__name__)


def validate_final_output(output_dir_path: Path, num_orig_files: int):
    files = file_services.walk(output_dir_path)
    num_output_files_new = len(files)
    assert (num_orig_files == num_output_files_new - 1)

    for f in files:
        df = pd.read_parquet(str(f))
        assert (df.shape[0] > 0)

    logger.info("Output data seems valid.")


def archive_input(source_dir_path: Path, output_dir_path: Path):
    logger.info("Archiving input ...")
    files = file_services.walk(the_path=source_dir_path)
    for f in files:
        new_path = Path(output_dir_path, f.name)
        shutil.move(str(f), str(new_path))


def process(twitter_root_path: Path, end_drop_path: Path, input_archive_path: Path):
    num_orig_files = len(file_services.walk(end_drop_path))

    command_service.get_equity_daily_data()
    command_service.get_equity_fundamentals_data()
    equity_performance.start()

    source_dir_path = Path(twitter_root_path, "raw_drop", "main")
    ensure_dir(source_dir_path)

    if file_services.has_no_files(source_dir_path):
        return False

    small_source_path, small_output_path = smallify_process.start(source_dir_path=source_dir_path, twitter_root_path=twitter_root_path, snow_plow_stage=False)

    flat_output_dir = flatten_process.start(source_dir_path=small_output_path, twitter_root_path=twitter_root_path, snow_plow_stage=True)

    # # flat_output_dir = Path("e:\\tmp\\twitter\\flattened_drop\\main")
    add_output_dir = add_id_process.start(source_dir_path=flat_output_dir, twitter_root_path=twitter_root_path, snow_plow_stage=True)

    # add_output_dir = Path("e:\\tmp\\twitter\\id_fixed\\main")
    rem_output_dir = rem_dupes_process.start(source_dir_path=add_output_dir, twitter_root_path=twitter_root_path, snow_plow_stage=True)

    # rem_output_dir = Path("C:\\Users\\Chris\\workspaces\\data\\twitter\\deduped\\main")
    coal_output_dir = coalesce_process.start(source_dir_path=rem_output_dir, twitter_root_path=twitter_root_path, snow_plow_stage=True)

    # coal_output_dir = Path("e:\\tmp\\twitter\\coalesced\\main")
    sent_output_dir = add_sent_process.start(source_dir_path=coal_output_dir, twitter_root_path=twitter_root_path, snow_plow_stage=True)

    # sent_output_dir = Path("e:\\tmp\\twitter\\sent_drop\\main")
    learn_output_dir = add_learn_prep_process.start(source_dir_path=sent_output_dir, twitter_root_path=twitter_root_path, snow_plow_stage=True)
    twit_output_dir = twit_reduce_process.start(source_dir_path=learn_output_dir, twitter_root_path=twitter_root_path, snow_plow_stage=True)

    transfer_to_end_drop(end_drop_path=end_drop_path, twit_output_dir=twit_output_dir)

    validate_final_output(output_dir_path=end_drop_path, num_orig_files=num_orig_files)

    # small_source_path = Path("C:\\Users\\Chris\\workspaces\\data\\twitter\\raw_drop\\main")
    archive_input(source_dir_path=small_source_path, output_dir_path=input_archive_path)

    return True


def transfer_to_end_drop(twit_output_dir: Path, end_drop_path: Path):
    files_final = file_services.list_files(twit_output_dir, ends_with=".parquet")
    ensure_dir(end_drop_path)
    for f in files_final:
        dest_path = Path(end_drop_path, f.name)
        shutil.copy(str(f), str(dest_path))


def process_old():
    # command_service.get_equity_daily_data()
    # command_service.get_equity_fundamentals_data()
    # equity_performance.start()

    smallify_process.start_old()
    flatten_process.start_old()
    add_id_process.start_old()
    rem_dupes_process.start_old()
    coalesce_process.start_old()
    add_sent_process.start_old()
    add_learn_prep_process.start_old()
    # twit_reduce_process.start_old()


def predict(predict_date_str: str, num_hold_days: int):
    twitter_ml.make_a_real_prediction(predict_date_str=predict_date_str, num_hold_days=num_hold_days)

    return twitter_ml_utils.get_real_predictions(sample_size=8,
                                                 num_hold_days=num_hold_days,
                                                 min_price=5.,
                                                 purchase_date_str=predict_date_str)


def predict_multiple_holds(num_hold_list: List[int]) -> List[str]:
    msgs = list()
    for num_hold_days in num_hold_list:
        tod_dt = datetime.now()
        today_str = date_utils.get_standard_ymd_format(tod_dt)
        prev_mark_dt_str = twitter_ml_utils.get_next_market_date(date_str=today_str, num_days=-1)
        sample_tickers = predict(predict_date_str=prev_mark_dt_str, num_hold_days=num_hold_days)

        message = f"{today_str} process complete: {num_hold_days} day prediction: {sample_tickers}"
        msgs.append(message)
    return msgs


def get_todays_prediction(twitter_root_path: Path, input_archive_path: Path):
    end_drop_path = constants.TWITTER_END_DROP
    try:
        was_successful = process(twitter_root_path=twitter_root_path, end_drop_path=end_drop_path, input_archive_path=input_archive_path)

        if was_successful:
            messages = predict_multiple_holds(num_hold_list=[5, 2])
        else:
            messages = ["Evidently no data today."]
    except Exception as e:
        messages = [f"Encountered problem."]
        logger.info(e)

    for m in messages:
        try:
            logger.info(m)
            slack_service.send_direct_message_to_chris(message=m)
        except BaseException as be:
            logger.info(be)

def spec_1():
    nh_list = [5, 2, 4, 3, 1]
    for i in nh_list:
        m = predict_multiple_holds(num_hold_list=[i])
        try:
            logger.info(m)
            slack_service.send_direct_message_to_chris(message=m)
        except BaseException as be:
            logger.info(be)


if __name__ == '__main__':
    # end_drop_path = constants.TWITTER_END_DROP
    # get_todays_prediction(twitter_root_path=constants.TWITTER_OUTPUT_RAW_PATH, input_archive_path=constants.TWEET_RAW_DROP_ARCHIVE)
    # process(twitter_root_path=constants.TWITTER_OUTPUT_RAW_PATH, end_drop_path=end_drop_path, input_archive_path=constants.TWEET_RAW_DROP_ARCHIVE)

    # today_str = "2021-02-11"
    # sample_tickers = predict(predict_date_str=today_str, num_hold_days=2)
    # print(sample_tickers)
    spec_1()