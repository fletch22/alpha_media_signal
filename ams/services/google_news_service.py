import csv
import time
from concurrent.futures.thread import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path
from typing import Tuple, List

import requests

from ams.DateRange import DateRange
from ams.config import constants
from ams.services import file_services, ticker_service
from ams.utils import date_utils
from ams.utils.PrinterThread import PrinterThread

EMPTY_STRING = ""


def multithreaded_query(tickers: List[str], date_range: DateRange, output_path: Path):
    pt = PrinterThread()
    try:
        pt.start()

        ticker_tuples = [(t, date_range) for t in tickers]

        f_date_str = date_utils.get_standard_ymd_format(date_range.from_date)
        t_date_str = date_utils.get_standard_ymd_format(date_range.to_date)

        print(f"Query will be {f_date_str}-{t_date_str}.")

        def custom_request(search_terms: Tuple[str, DateRange]):
            ticker = search_terms[0]
            date_range = search_terms[1]

            pt.print(f'{ticker}: from {f_date_str} thru {t_date_str}')

            return query(news_topic=ticker, output_path=output_path, date_range=date_range)

        results = 0
        with ThreadPoolExecutor(4) as executor:
            results = executor.map(custom_request, ticker_tuples, timeout=None)

        total_pages = 0
        for page_count in results:
            total_pages += page_count

        # pt.print(results)
    finally:
        pt.end()

    return total_pages


def query(news_topic: str, output_path: Path, date_range: DateRange, max_results: int = 12):
    pt = PrinterThread()
    end_of_pages_token = "In order to show you the most relevant results"
    captcha_token = "captcha"

    query_pause = 1

    try:
        pt.start()

        count = 0
        start = 0

        from_date_str = date_utils.get_standard_ymd_format(date_range.from_date)
        to_date_str = date_utils.get_standard_ymd_format(date_range.to_date)

        min_date = date_utils.get_us_mdy_format(date_range.from_date)
        max_date = date_utils.get_us_mdy_format(date_range.to_date)

        while True:
            url = f"https://www.google.com/search?q={news_topic}&tbs=sbd:1,cdr:1,cd_min:{min_date},cd_max:{max_date}&tbm=nws&sxsrf=ALeKk01AMAuhZOHeQnFTht-WbV4I63ILcw:1601601578458&ei=KoB2X5q0G4bJtQbm4LT4AQ&start={start}&sa=N&ved=0ahUKEwia8Z7p3pTsAhWGZM0KHWYwDR84ZBDw0wMIyAY&biw=1903&bih=919&dpr=1"
            time.sleep(query_pause)
            response = requests.get(url)
            news_html = response.text

            write_news(from_date_str=from_date_str, news_html=news_html, to_date_str=to_date_str, news_topic=news_topic, output_path=output_path)
            pt.print(f"query: {news_topic}; cumulative results: {start}")

            if len(news_html) < 40000 and (end_of_pages_token in news_html or captcha_token in news_html):
                pt.print("Got last page.")
                break

            if 'errors' in news_html:
                pt.print(news_html['errors'])
                pt.print('Pausing...')
                break

            if max_results >= start:
                break

            start += 12
            count += 1


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
    tickers = ticker_service.get_all_tickers()
    tickers = tickers[3:5]

    chunk_size = 2
    ticker_chunks = [tickers[i:i + chunk_size] for i in range(0, len(tickers), chunk_size)]

    da_date = date_range.from_date

    acc_pages = 0
    for tc in ticker_chunks:
        while da_date <= date_range.to_date:
            da_dr = DateRange(from_date=da_date, to_date=da_date)
            total_pages = multithreaded_query(tickers=tc, date_range=da_dr, output_path=output_path)
            acc_pages += total_pages
            da_date = da_date + timedelta(days=1)

    return acc_pages


if __name__ == '__main__':
    output_path = file_services.create_unique_filename(constants.GOOGLE_NEWS_OUTPUT_DIR_PATH, prefix="goog_news", extension="txt")
    date_range = DateRange.from_date_strings("2019-09-01", "2020-09-02")

    start = time.time()
    total_pages = roll_through_the_days(date_range=date_range, output_path=output_path)
    end = time.time()

    print(f"Elapsed: {end - start}; total_pages; {total_pages}")
