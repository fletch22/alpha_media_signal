from ams.monte_carlo import ma_bottom_feeder as mba
from ams.monte_carlo.ma_bottom_feed_data import stock_dict_high_roe, stock_dict_low_evebitda


def test_get_maos():
    stocks = stock_dict_low_evebitda
    mba.get_roi(stocks=stocks, min_days_under_sought=39)
    mba.get_roi(stocks=stocks, min_days_under_sought=29)


def test_get_bottom_feeder_recommendations():
    mba.get_recommendations()
