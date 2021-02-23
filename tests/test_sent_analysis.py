import csv
import json
import re
import time
from pathlib import Path
from typing import Dict, List

import pandas as pd
from pyspark.sql.types import StructType

from ams.config import constants, logger_factory
from ams.services import file_services
from ams.utils import sentiment
from tests.test_tweets import test_multithreads

logger = logger_factory.create(__name__)


def test_load():
    # csv_path_str = r"C:\Users\Chris\workspaces\data\twitter\flattened_drop\tweets_flat_2020-08-21_22-27-23-826.42\part-00000-94e14f84-2c63-4aec-8c07-b865fdaf12a9-c000.csv"

    output_folder_path = Path(f'{constants.DATA_PATH}\\twitter\\flattened_drop\\tweets_flat_2020-08-22_18-04-19-516.66')
    csv_list = list(file_services.list_files(output_folder_path, ends_with=".csv"))

    csv_path_str = str(csv_list[0])
    df = pd.read_csv(csv_path_str, dialect=csv.unix_dialect(), error_bad_lines=False, index_col=False, dtype='unicode')

    logger.info(df.shape[0])
    logger.info(df.columns)


def test_replace_newlines():
    sample = "Healthcare news covering:\n\n- FDA Approvals & Recalls\n- Stage 1/2/3 Updates\n- New Drug Applications\n- FDA Press Releases\n- Company Announcements\n- SEC Filings"
    pattern = re.compile('\n')
    result = re.sub(pattern, '', sample)

    logger.info(result)

    import csv

    unix_dialect = csv.unix_dialect()


def test_schema():
    test = {'ticker': 'foo', 'has_cashtag': True, 'ticker_in_text': False}
    test_schema = StructType.fromJson(test)

    logger.info(test_schema)


def get_cashtag_info(ticker: str, has_cashtag: bool, ticker_in_text: bool) -> Dict:
    return {"ticker": ticker, "has_cashtag": has_cashtag, "ticker_in_text": ticker_in_text}


def get_cashtags_row_wise(raw_line: str, search_tuples: List):
    tweet = json.loads(raw_line)
    text = tweet['text']

    cashtags_stock = []
    for s in search_tuples:
        ticker = s[0].strip()
        name = s[1].strip()

        if re.search(f'${ticker}', text) and re.search(name, text, re.IGNORECASE):
            cashtags_stock.append(get_cashtag_info(ticker=ticker, has_cashtag=True, ticker_in_text=True))

    if len(cashtags_stock) == 0:
        for s in search_tuples:
            ticker = s[0].strip()
            name = s[1].strip()

            if re.search(ticker, text) and re.search(name, text, re.IGNORECASE):
                cashtags_stock.append(get_cashtag_info(ticker=ticker, has_cashtag=False, ticker_in_text=True))

    if len(cashtags_stock) == 0:

        for s in search_tuples:
            ticker = s[0]
            name = s[1]
            if re.search(ticker, raw_line) and re.search(name, raw_line, re.IGNORECASE):
                cashtags_stock.append(get_cashtag_info(ticker=ticker, has_cashtag=True, ticker_in_text=True))

    logger.info(f'Tweet associate with: {cashtags_stock}')

    return cashtags_stock


def test_fast_search():
    # Arrange

    def load_magazine(searches: List[Dict]):
        retained = {}
        for s in searches:
            search_dict = retained
            ticker = s["ticker"]
            for char in ticker:
                logger.info(char)
                if char not in list(search_dict.keys()):
                    current = {}
                    search_dict[char] = current
                else:
                    current = search_dict[char]
                search_dict = current

        return retained

    companies = [{"ticker": "ma", "company": "Man on the Moon!"}, {"ticker": "mum", "company": "Z Thunder!"}, {"ticker": "mun", "company": "Magic Mike"}]
    magazine = load_magazine(searches=companies)

    def search_and_pop(text: str, current_node: Dict, ticker: str, start: int = 0, ticker_index: int = 0):
        current_char = ticker[ticker_index]
        index = text.find(current_char, start)
        if index == -1:
            if current_char in current_node.keys():
                current_node.pop(current_char)
            return
        else:
            if len(ticker) == ticker_index + 1:
                return
            ticker_index += 1
            next_node = current_node[current_char]
            search_and_pop(text=text, current_node=next_node, ticker=ticker, start=index, ticker_index=ticker_index)

    ticker = 'mu'.lower()
    text = 'The magic man toffed his cap.'.lower()

    for c in companies:
        search_and_pop(text=text, current_node=magazine, ticker=c['ticker'])

    logger.info(f'{magazine}')

    # Act+

    # Assert


def test_timer():
    search_str = 'Graphs'

    text = """Sat Aug 15 16:01:29 +0000 20201294665524555722753Earnings growth rate (%) among largest #stocks $SPX $SPY1. �� APPLE INC. $AAPL: 8.42. �� MICROSOFT CORPORATIO.… https://t.co/twfQJiMcG0True<a href    ="https://11graphs.com
" rel="nofollow">11GTweetBot</a>NoneNoneNoneNoneFalse21FalseFalseenNoneNoneNoneNone[twitter.com/i/web/status/1…&#44; https://twitter.com/i/web/status/1294665524555722753&#44; [114&#44; 137]&#44; https://t.co/twfQJiMcG0]NoneNoneNonee
nrecent113052520361903718511Graphs11GraphsParis&#44; France• � Algorithm that ranks stocks and ETFs • � Outperform the S&P 500 • � Worldwide companies screened • � Ready to beat the market? •https://t.co/2aMU8Jk48MFalse191810934
Mon May 20 17:26:27 +0000 2019529NoneNoneFalseFalse3354NoneFalseFalseFalse000000http://abs.twimg.com/images/themes/theme1/bg.pnghttps://abs.twimg.com/images/themes/theme1/bg.pngFalsehttp://pbs.twimg.com/profile_images/12840645120318
25927/YtJp1zO6_normal.jpghttps://pbs.twimg.com/profile_images/1284064512031825927/YtJp1zO6_normal.jpghttps://pbs.twimg.com/profile_banners/1130525203619037185/15949796071B95E0000000000000000000FalseFalseFalseFalseNoneNoneNonenoneNon
eNone"""

    start = time.time()
    for i in range(100000):
        # re.search(search_str, text)
        text = text.lower()
        search_str = search_str.lower()
        text.find(search_str)
    end = time.time()

    elapsed_time = end - start
    logger.info(f'Elapsed time: {elapsed_time} seconds')


def test_sent_analysis():
    text_1 = """@Codieisfree @JRSP_1978 @ValeTudoBro I understand. What I don't like is captain cop and jive turkey Biden😂 and like… https://t.co/fnOwHpLqm1"""
    text_2 = r'RT @iprathmeshs: We belong to Atif Aslam&#44; KK &#44; Himesh reshammiya &#44; Shreya Ghoshal&#44; A.R.Rahman  music era do not doubt our taste in music. P…'
    text_3 = ""

    sent_1 = sentiment.get_sentiment_intensity_score(text_3)
    logger.info(sent_1)

    num_analysis = 10000
    start = time.time()
    for i in range(num_analysis):
        sent_1 = sentiment.get_sentiment_intensity_score(text_1)
    end = time.time()

    sent_per_sec = num_analysis / (end - start)
    logger.info(f'{sent_per_sec} SA per second')


if __name__ == '__main__':
    # test_queue()
    test_multithreads()