import pickle
import shutil
from pathlib import Path

import bs4 as bs
import requests

from ams.config import constants


def get_sp500_table():
    html = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(html.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})
    return table


def fetch_sp500_tickers():
    tickers = []
    table = get_sp500_table()
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text
        ticker = ticker[:-1]
        tickers.append(ticker)

    if constants.SP500_WIKI_TICKERS.exists():
        shutil.rmtree(str(constants.SP500_WIKI_TICKERS))
    constants.SP500_WIKI_TICKERS.mkdir()

    sp500_lst_path = Path(constants.SP500_WIKI_TICKERS, "sp500tickers.pickle")
    with open(sp500_lst_path, "wb") as f:
        pickle.dump(tickers, f)


def write_to_tickers():
    import datetime as dt
    import pandas_datareader.data as pdr

    sp500_lst_path = Path(constants.SP500_WIKI_TICKERS, "sp500tickers.pickle")
    with open(sp500_lst_path, "rb") as f:
        tickers = pickle.load(f)

    sd_path = Path(constants.SP500_WIKI_TICKERS, "stock_dfs")
    if sd_path.exists():
        shutil.rmtree(str(sd_path))
    sd_path.mkdir()

    start = dt.datetime(2007, 1, 1)
    end = dt.datetime.now()

    for ticker in tickers:
        output_path = Path(sd_path, f"{ticker}.csv")
        if not output_path.exists():
            df = pdr.DataReader(ticker.replace('.', '-'), 'yahoo', start, end)
            df.reset_index(inplace=True)
            df.set_index("Date", inplace=True)
            df.to_csv(str(output_path))
        else:
            print('Already have {}'.format(ticker))


if __name__ == '__main__':
    # fetch_sp500_tickers()
    write_to_tickers()