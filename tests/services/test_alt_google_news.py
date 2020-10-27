import pandas as pd
from GoogleNews import GoogleNews

def test_google_news_alt():
    googlenews = GoogleNews(start='05/01/2020', end='05/31/2020')
    googlenews.search('Coronavirus')
    result = googlenews.result()
    df = pd.DataFrame(result)
    print(df.head())


import requests


def test_send_request():
    # url = "http://httpbin.org/headers?json"
    url = "https://www.google.com/search?q=AAPL"
    response = requests.get(
        url="https://app.scrapingbee.com/api/v1/",
        params={
            "api_key": "5ZD9C416DHWKXNEJ47R0186TJM5CZX1BPDW9MBT64P7D8QGCXPZSNKRY7X4E9EO06EWDK6DBLSYVFGKM",
            "url": url,
            "premium_proho_split_by_daysxy": "true",
            "country_code": "us",
            "custom_google": "true"
    }
    )
    print('Response HTTP Status Code: ', response.status_code)
    print('Response HTTP Response Body: ', response.content)


#  Install the Python Requests library:
# `pip install requests`
# import requests
# def send_request():
#     response = requests.get(
#         url="https://app.scrapingbee.com/api/v1/",
#         params={
#             "api_key": "5ZD9C416DHWKXNEJ47R0186TJM5CZX1BPDW9MBT64P7D8QGCXPZSNKRY7X4E9EO06EWDK6DBLSYVFGKM",
#             "url": "http://httpbin.org/headers?json",
#         },
#
#     )
#     print('Response HTTP Status Code: ', response.status_code)
#     print('Response HTTP Response Body: ', response.content)


# send_request()