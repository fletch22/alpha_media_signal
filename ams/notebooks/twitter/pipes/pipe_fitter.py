import gc

from ams.notebooks.twitter.pipes.p_smallify_files import process as smallify_process
from ams.notebooks.twitter.pipes.p_fix_tweet_multi_key import process as fix_tweet_process
from ams.notebooks.twitter.pipes.p_flatten import process as flatten_process
from ams.notebooks.twitter.pipes.p_add_id import process as add_id_process
from ams.notebooks.twitter.pipes.p_remove_dupes import process as rem_dupes_process
from ams.notebooks.twitter.pipes.p_add_sentiment import process as add_sent_process
from ams.notebooks.twitter.pipes.p_add_learning_prep import process as add_learn_prep_process
from ams.notebooks.twitter.pipes.p_twitter_reduction import process as twit_reduce_process
from ams.services import command_service
from ams.services.equities import equity_performance


def start():
    command_service.get_equity_daily_data()
    command_service.get_equity_fundamentals_data()
    equity_performance.start()

    smallify_process.start()
    fix_tweet_process.start()
    flatten_process.start()
    add_id_process.start()
    rem_dupes_process.start()
    add_sent_process.start()
    add_learn_prep_process.start()
    twit_reduce_process.start()


if __name__ == '__main__':
    start()