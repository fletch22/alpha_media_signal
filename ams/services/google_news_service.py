import csv
import time
from concurrent.futures.thread import ThreadPoolExecutor
from datetime import timedelta, datetime
from pathlib import Path
from typing import Tuple, List

import requests

from ams.DateRange import DateRange
from ams.config import constants
from ams.services import file_services, twitter_service
from ams.utils import date_utils
from ams.utils.PrinterThread import PrinterThread

EMPTY_STRING = ""


def multithreaded_query(ticker_tuples: List[Tuple[str, str]], date_range: DateRange, output_path: Path):
    pt = PrinterThread()
    try:
        pt.start()

        ticker_tuples = [(t[0], t[1], date_range) for t in ticker_tuples]

        f_date_str = date_utils.get_standard_ymd_format(date_range.from_date)
        t_date_str = date_utils.get_standard_ymd_format(date_range.to_date)

        pt.print(f"Query will be {f_date_str}-{t_date_str}.")

        total_pages = 0
        for ticker, name, date_range in ticker_tuples:

            if ticker <= "AACB":
                continue
            news_topic = f"{ticker} {name}"
            total_pages += query(news_topic=news_topic, output_path=output_path, date_range=date_range)

        # def custom_request(search_terms: Tuple[str, DateRange]):
        #     ticker = search_terms[0]
        #     name = search_terms[1]
        #     date_range = search_terms[2]
        #
        #     if ticker <= "AAXN":
        #         return 0
        #
        #     pt.print(f'{ticker}  {name} | from {f_date_str} thru {t_date_str}')
        #
        #     news_topic = f"{ticker} {name}"
        #
        #     return query(news_topic=news_topic, output_path=output_path, date_range=date_range)

        # with ThreadPoolExecutor(1) as executor:
        #     results = executor.map(custom_request, ticker_tuples, timeout=None)
        #
        # total_pages = 0
        # for page_count in results:
        #     total_pages += page_count
        # pt.print(results)
    finally:
        pt.end()

    return total_pages


def query(news_topic: str, output_path: Path, date_range: DateRange, max_results: int = 12):
    pt = PrinterThread()
    end_of_pages_token = "In order to show you the most relevant results"
    captcha_token = "captcha"
    has_content_token = "kCrYT"

    query_pause = 2

    try:
        pt.start()

        count = 0
        start = 0

        from_date_str = date_utils.get_standard_ymd_format(date_range.from_date)
        to_date_str = date_utils.get_standard_ymd_format(date_range.to_date)

        min_date = date_utils.get_us_mdy_format(date_range.from_date)
        max_date = date_utils.get_us_mdy_format(date_range.to_date)

        sleep_time = 600
        while True:
            url = f"https://www.google.com/search?q={news_topic}&tbs=sbd:1,cdr:1,cd_min:{min_date},cd_max:{max_date}&tbm=nws&sxsrf=ALeKk01AMAuhZOHeQnFTht-WbV4I63ILcw:1601601578458&ei=KoB2X5q0G4bJtQbm4LT4AQ&start={start}&sa=N&ved=0ahUKEwia8Z7p3pTsAhWGZM0KHWYwDR84ZBDw0wMIyAY&biw=1903&bih=919&dpr=1"
            response = requests.get(url)
            news_html = response.text

            if captcha_token in news_html:
                pt.print("Exceeded request allowance.")
                time.sleep(sleep_time)
                sleep_time += 40 * 60
                continue

            if len(news_html) < 40000 and (end_of_pages_token in news_html):
                pt.print("Got last page.")
                break

            if 'errors' in news_html:
                pt.print(news_html['errors'])
                pt.print('Pausing...')
                break

            found_content = False
            if has_content_token in news_html:
                found_content = True
                write_news(from_date_str=from_date_str, news_html=news_html, to_date_str=to_date_str, news_topic=news_topic, output_path=output_path)

            pt.print(f"query: {news_topic}; {from_date_str}-{to_date_str} returned results: {found_content}")

            start += 12
            count += 1

            if max_results <= start or not found_content:
                break

            time.sleep(query_pause)


    finally:
        pt.end()

    return count


def write_news(from_date_str: str, to_date_str: str, news_html: str, news_topic: str, output_path: Path):
    news_clean = news_html.replace("\n", EMPTY_STRING).replace("\r", EMPTY_STRING).replace("\t", EMPTY_STRING)
    row = [from_date_str, to_date_str, news_topic, news_clean]

    is_new_file = False
    if not output_path.exists() or output_path.stat().st_size == 0:
        is_new_file = True

    with open(str(output_path), "a+") as output:
        wr = csv.writer(output, quoting=csv.QUOTE_ALL, lineterminator="\n")
        if is_new_file:
            wr.writerow(["from_date", "to_date", "news_topic", "new_html"])
        wr.writerow(row)


def roll_through_the_days(date_range: DateRange, output_path: Path):
    ticker_tuples = twitter_service.get_ticker_searchable_tuples()

    chunk_size = 2
    ticker_tuple_chunks = [ticker_tuples[i:i + chunk_size] for i in range(0, len(ticker_tuples), chunk_size)]

    da_date = date_range.from_date

    acc_pages = 0
    for ttc in ticker_tuple_chunks:
        while da_date <= date_range.to_date:
            da_dr = DateRange(from_date=da_date, to_date=da_date)
            total_pages = multithreaded_query(ticker_tuples=ttc, date_range=da_dr, output_path=output_path)
            acc_pages += total_pages
            da_date = da_date + timedelta(days=1)
        da_date = date_range.from_date

    return acc_pages


def get_news():
    output_path = file_services.create_unique_filename(constants.GOOGLE_NEWS_OUTPUT_DIR_PATH, prefix="goog_news", extension="txt")
    today_dt_str = date_utils.get_standard_ymd_format(datetime.now())
    date_range = DateRange.from_date_strings("2020-08-01", today_dt_str)

    start = time.time()
    total_pages = roll_through_the_days(date_range=date_range, output_path=output_path)
    end = time.time()

    print(f"Elapsed: {end - start}; total_pages; {total_pages}")

    return total_pages, end - start


if __name__ == '__main__':
    get_news()
