import time

import schedule

from ams.config import logger_factory
from ams.marios_workbench.twitter.import_and_predict import valve as valve_import_and_predict
from ams.services import twitter_service

logger = logger_factory.create(__name__)


def process_twitter_signal():
    twitter_service.fetch_up_to_date_tweets()
    valve_import_and_predict.open()


def process_daily_twitter_signal():
    jobs_start = "00:00"
    schedule.every().day.at(jobs_start).do(process_twitter_signal)

    while True:
        logger.info(f"Waiting to start job at {jobs_start}pm ...")
        schedule.run_pending()
        time.sleep(120)


if __name__ == '__main__':
    process_daily_twitter_signal()