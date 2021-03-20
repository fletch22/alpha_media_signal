import shutil
from datetime import datetime
from pathlib import Path
from typing import List

import pandas as pd

import ams.utils.date_utils
from ams.config import constants, logger_factory
from ams.config.constants import ensure_dir
from ams.pipes.p_add_id import process as add_id_process
from ams.pipes.p_add_learning_prep import process as add_learn_prep_process
from ams.pipes.p_add_sentiment import process as add_sent_process
from ams.pipes.p_coalesce import process as coalesce_process
from ams.pipes.p_flatten import process as flatten_process
from ams.pipes.p_make_prediction import mp_process
from ams.pipes.p_remove_dupes import process as rem_dupes_process
from ams.pipes.p_smallify_files import process as smallify_process
from ams.pipes.p_stock_merge import sm_process
from ams.pipes.p_twitter_reduction import process as twit_reduce_process
from ams.services import command_service, file_services, slack_service, zip_service
from ams.services.equities import equity_performance
from ams.services.equities.equity_performance import calc_and_persist_nasdaq_roi
from ams.twitter import twitter_ml_utils, twitter_ml
from ams.twitter.TrainAndPredictionParams import PredictionMode
from ams.utils import date_utils

logger = logger_factory.create(__name__)


def process(twitter_root_path: Path,
            input_archive_path: Path, should_delete_leftovers: bool,
            skip_external_data_dl: bool = False, archive_raw: bool = True):

    if not skip_external_data_dl:
        command_service.get_equity_daily_data()
        command_service.get_equity_fundamentals_data()
        equity_performance.start()

        earliest_dt = date_utils.parse_std_datestring("2020-08-10")
        calc_and_persist_nasdaq_roi(date_from=earliest_dt, date_to=datetime.now(), days_hold_stock=1)
    else:
        logger.info("Skipping external data download.")

    start_source_path = Path(twitter_root_path, "raw_drop", "main")
    ensure_dir(start_source_path)

    if file_services.has_no_files(start_source_path):
        return False

    files = file_services.list_files(start_source_path)
    logger.info(f"Processing {files}")

    srd_dir_path = Path(twitter_root_path, "smallified_raw_drop", "main")
    smallify_process.start(src_dir_path=start_source_path, dest_dir_path=srd_dir_path,
                                                                  snow_plow_stage=False, should_delete_leftovers=should_delete_leftovers)

    fd_dir_path = Path(twitter_root_path, 'flattened_drop', "main")
    flatten_process.start(source_dir_path=srd_dir_path, dest_dir_path=fd_dir_path,
                                            snow_plow_stage=True, should_delete_leftovers=should_delete_leftovers)

    # # flat_output_dir = Path("e:\\tmp\\twitter\\flattened_drop\\main")
    if_dir_path = Path(twitter_root_path, "id_fixed", "main")
    add_id_process.start(source_dir_path=fd_dir_path, dest_dir_path=if_dir_path,
                                          snow_plow_stage=True, should_delete_leftovers=should_delete_leftovers)

    # add_output_dir = Path("e:\\tmp\\twitter\\id_fixed\\main")
    dd_dir_path = Path(twitter_root_path, "deduped", "main")
    rem_dupes_process.start(source_dir_path=if_dir_path, dest_dir_path=dd_dir_path,
                                             snow_plow_stage=True, should_delete_leftovers=should_delete_leftovers)

    # rem_output_dir = Path(constants.TWITTER_OUTPUT_RAW_PATH, "deduped", "main")
    c_dir_path = Path(twitter_root_path, 'coalesced', "main")
    coalesce_process.start(source_dir_path=dd_dir_path, dest_dir_path=c_dir_path,
                                             snow_plow_stage=True, should_delete_leftovers=should_delete_leftovers)

    # coal_output_dir = Path(constants.TWITTER_OUTPUT_RAW_PATH, "coalesced\\main")
    sd_dir_path = Path(twitter_root_path, 'sent_drop', "main")
    add_sent_process.start(source_dir_path=c_dir_path, dest_dir_path=sd_dir_path,
                                             snow_plow_stage=True, should_delete_leftovers=should_delete_leftovers)

    # sent_output_dir = Path("e:\\tmp\\twitter\\sent_drop\\main")
    lpd_dir_path = Path(twitter_root_path, 'learning_prep_drop', "main")
    add_learn_prep_process.start(source_dir_path=sd_dir_path, dest_dir_path=lpd_dir_path,
                                                    snow_plow_stage=True, should_delete_leftovers=should_delete_leftovers)

    tr_dir_path = twit_reduce_process.get_output_dir(twitter_root_path=twitter_root_path)
    twit_reduce_process.start(source_dir_path=lpd_dir_path, dest_dir_path=tr_dir_path,
                                                snow_plow_stage=True, should_delete_leftovers=should_delete_leftovers)

    ref_tweet_bucket = Path(twitter_root_path, "refined_tweets_bucket")
    transfer_to_refined_tweets_bucket(src_dir_path=tr_dir_path, refined_tweets_path=ref_tweet_bucket)

    validate_ref_tweet_bucket(output_dir_path=ref_tweet_bucket)

    if archive_raw:
        # small_source_path = Path(constants.TWITTER_OUTPUT_RAW_PATH, "raw_drop\\main")
        archive_input(source_dir_path=start_source_path, output_dir_path=input_archive_path)

    return True


def validate_ref_tweet_bucket(output_dir_path: Path):
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


def transfer_to_refined_tweets_bucket(src_dir_path: Path, refined_tweets_path: Path):
    files_final = file_services.list_files(src_dir_path, ends_with=".parquet")
    ensure_dir(refined_tweets_path)
    for f in files_final:
        dest_path = Path(refined_tweets_path, f.name)
        shutil.copy(str(f), str(dest_path))


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
        tweet_date_str = ams.utils.date_utils.get_next_market_date(date_str=today_str, is_reverse=True)

        # NOTE: 2021-03-11: chris.flesche: Temporarily disabled.
        # sample_tickers = predict(tweet_date_str=tweet_date_str, num_hold_days=num_hold_days)
        #
        # message = f"{today_str} process complete: {num_hold_days} day prediction: {sample_tickers}"
        # msgs.append(message)
        msgs.append("Predictions temporarily disabled.")
    return msgs


def get_todays_prediction(twitter_root_path: Path, input_archive_path: Path):
    messages = []
    try:
        was_successful = process(twitter_root_path=twitter_root_path,
                                 input_archive_path=input_archive_path,
                                 should_delete_leftovers=False,
                                 skip_external_data_dl=False)

        # NOTE: 2021-03-17: chris.flesche: Replace with pipe.
        if was_successful:
            get_and_message_predictions(twitter_root_path=twitter_root_path)
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


def get_and_message_predictions(twitter_root_path: Path):
    ref_tweet_bucket = Path(twitter_root_path, "refined_tweets_bucket")
    sm_dir_path = Path(twitter_root_path, "stock_merge_drop", "main")

    sm_process.start(src_dir_path=ref_tweet_bucket,
                     dest_dir_path=sm_dir_path,
                     should_delete_leftovers=True)

    pb_dir_path = Path(twitter_root_path, "prediction_bucket")
    mp_process.start(src_path=sm_dir_path,
                     dest_path=pb_dir_path,
                     prediction_mode=PredictionMode.RealMoneyStockRecommender)


if __name__ == '__main__':
    get_todays_prediction(twitter_root_path=constants.TWITTER_OUTPUT_RAW_PATH, input_archive_path=constants.TWEET_RAW_DROP_ARCHIVE)
    # end_bucket_path = constants.REFINED_TWEETS_BUCKET_PATH
    # process(twitter_root_path=constants.TWITTER_OUTPUT_RAW_PATH,
    #         end_bucket_path=end_bucket_path,
    #         input_archive_path=constants.TWEET_RAW_DROP_ARCHIVE,
    #         should_delete_leftovers=False)

    # get_and_message_predictions(twitter_root_path=constants.TWITTER_OUTPUT_RAW_PATH)

