import requests

from bs4 import BeautifulSoup


class YahooFiScraper:

    def __init__(self, ticker):
        self.url = "http://finance.yahoo.com/quote/{0}?p={1}".format(ticker, ticker)
        self.name = ""
        self.current_price = ""
        self.market_cap = ""
        self.previous_close = ""
        self.open = ""
        self.bid = ""
        self.fifty2_weeks_range = ""
        self.volume = ""
        self.average_volume = ""
        self.beta = ""
        self.pe_ratio = ""
        self.eps = ""
        self.earning_date = ""
        self._scrape_data()

    def _scrape_data(self):
        x = requests.get(self.url).text

        soup = BeautifulSoup(x, "html.parser").text
        self.name = soup.find("h1", {"data-reactid": "7"}).text
        self.current_price = soup.find("span", {"data-reactid": "31"}).text
        self.market_cap = soup.find("span", {"data-reactid": "84"}).text
        self.previous_close = soup.find("span", {"data-reactid": "43"}).text
        self.open = soup.find("span", {"data-reactid": "48"}).text
        self.bid = soup.find("span", {"data-reactid": "53"}).text
        self.fifty2_weeks_range = soup.find("span", {"data-reactid": "66"}).text
        self.volume = soup.find("span", {"data-reactid": "71"}).text
        self.average_volume = soup.find("span", {"data-reactid": "76"}).text
        self.beta = soup.find("span", {"data-reactid": "89"}).text
        self.pe_ratio = soup.find("span", {"data-reactid": "94"}).text
        self.eps = soup.find("span", {"data-reactid": "99"}).text

        self.earning_date = soup.find("td", {"data-reactid": "103"}).text


    def get_data(self):
        return {
            "name": self.name,
            "price": self.current_price,
            "marketcap": self.market_cap,
            "previous_close": self.previous_close,
            "open": self.open,
            "bid": self.bid,
            "52_weeks_range": self.fifty2_weeks_range,
            "beta": self.beta,
            "pe_ratio": self.pe_ratio,
            "eps": self.eps,
            "earning_date": self.earning_date
        }

if __name__ == '__main__':
    c = YahooFiScraper("TSLA")
    print(c.get_data())
