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
from ams.services.equities.equity_performance import calc_and_persist_nasdaq_roi
from ams.twitter import twitter_ml_utils, twitter_ml
from ams.utils import date_utils
from tests.services import zip_service

logger = logger_factory.create(__name__)


def validate_final_output(output_dir_path: Path):
    files = file_services.walk(output_dir_path)

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

        output_path = Path(new_path.parent, f"{new_path.stem}.zip")
        zip_service.zip_file(file_path=new_path, output_path=output_path)

        # NOTE: 2021-02-17: chris.flesche: delete file if zip was successful?
        # new_path.unlink()


def process(twitter_root_path: Path, end_drop_path: Path, input_archive_path: Path, skip_external_data_dl: bool = False):

    if not skip_external_data_dl:
        command_service.get_equity_daily_data()
        command_service.get_equity_fundamentals_data()
        equity_performance.start()

        earliest_dt = date_utils.parse_std_datestring("2020-08-10")
        calc_and_persist_nasdaq_roi(date_from=earliest_dt, date_to=datetime.now(), days_hold_stock=1)
    else:
        logger.info("Skipping external data download.")

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

    # rem_output_dir = Path(constants.TWITTER_OUTPUT_RAW_PATH, "deduped", "main")
    coal_output_dir = coalesce_process.start(source_dir_path=rem_output_dir, twitter_root_path=twitter_root_path, snow_plow_stage=True)

    # coal_output_dir = Path(constants.TWITTER_OUTPUT_RAW_PATH, "coalesced\\main")
    sent_output_dir = add_sent_process.start(source_dir_path=coal_output_dir, twitter_root_path=twitter_root_path, snow_plow_stage=True)

    # sent_output_dir = Path("e:\\tmp\\twitter\\sent_drop\\main")
    learn_output_dir = add_learn_prep_process.start(source_dir_path=sent_output_dir, twitter_root_path=twitter_root_path, snow_plow_stage=True)
    twit_output_dir = twit_reduce_process.start(source_dir_path=learn_output_dir, twitter_root_path=twitter_root_path, snow_plow_stage=True)

    transfer_to_end_drop(end_drop_path=end_drop_path, twit_output_dir=twit_output_dir)

    validate_final_output(output_dir_path=end_drop_path)

    # small_source_path = Path(constants.TWITTER_OUTPUT_RAW_PATH, "raw_drop\\main")
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


def predict(tweet_date_str: str, num_hold_days: int):
    purchase_date_str = twitter_ml.make_a_real_prediction(tweet_date_str=tweet_date_str, num_hold_days=num_hold_days)

    return twitter_ml_utils.get_real_predictions(sample_size=16,
                                                 num_hold_days=num_hold_days,
                                                 min_price=5.,
                                                 purchase_date_str=purchase_date_str)


def predict_multiple_holds(num_hold_list: List[int]) -> List[str]:
    msgs = list()
    for num_hold_days in num_hold_list:
        tod_dt = datetime.now()
        today_str = date_utils.get_standard_ymd_format(tod_dt)
        tweet_date_str = twitter_ml_utils.get_next_market_date(date_str=today_str, num_days=-1)

        logger.info(f"TDS: {tweet_date_str}")

        sample_tickers = predict(tweet_date_str=tweet_date_str, num_hold_days=num_hold_days)

        message = f"{today_str} process complete: {num_hold_days} day prediction: {sample_tickers}"
        msgs.append(message)
    return msgs


def get_todays_prediction(twitter_root_path: Path, input_archive_path: Path):
    end_drop_path = constants.TWITTER_END_DROP
    messages = []
    try:
        was_successful = process(twitter_root_path=twitter_root_path, end_drop_path=end_drop_path, input_archive_path=input_archive_path)

        if was_successful:
            get_and_message_predictions(num_days_hold_list=[10, 5, 4, 3, 2, 1])
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
    get_and_message_predictions(nh_list)


def get_and_message_predictions(num_days_hold_list):
    for i in num_days_hold_list:
        m = predict_multiple_holds(num_hold_list=[i])
        try:
            logger.info(m)
            slack_service.send_direct_message_to_chris(message=m)
        except BaseException as be:
            logger.info(be)


if __name__ == '__main__':
    end_drop_path = constants.TWITTER_END_DROP
    get_todays_prediction(twitter_root_path=constants.TWITTER_OUTPUT_RAW_PATH, input_archive_path=constants.TWEET_RAW_DROP_ARCHIVE)
    # process(twitter_root_path=constants.TWITTER_OUTPUT_RAW_PATH, end_drop_path=end_drop_path, input_archive_path=constants.TWEET_RAW_DROP_ARCHIVE)

    # constants.xgb.defaults.max_depth = 4
    # # tweet_date_str = "2021-01-13"
    # tweet_date_str = "2021-02-18"
    #
    # n_hold_list = [5, 2, 1, 10, 3, 4]
    # # n_hold_list = [9, 8, 7, 6]
    # for i in n_hold_list:
    #     num_hold_days = i
    #     message = predict(tweet_date_str, num_hold_days)
    #     message = f"Tweet date: {tweet_date_str}: {i} days: {message}"
    #     try:
    #         logger.info(message)
    #         slack_service.send_direct_message_to_chris(message=message)
    #     except BaseException as be:
    #         logger.info(be)

    # purchase_date_str = "2021-01-14"
    # tickers = twitter_ml_utils.get_real_predictions(sample_size=16,
    #                                       num_hold_days=5,
    #                                       min_price=5.,
    #                                       purchase_date_str=purchase_date_str)
    #
    # print(tickers)