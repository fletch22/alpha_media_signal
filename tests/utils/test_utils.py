from datetime import datetime

# from ams import utils
#
#
# def test_load_common_words():
#     # Arrange
#     # Act
#     common_words = utils.load_common_words()
#
#     # Assert
#     assert (len(common_words) > 10000)
from ams.utils import date_utils


def test_convert_twitter_date():
    date_string = "Wed Aug 26 12:41:33 +0000 2020"
    # format = "%a %b %d %H:%M:%S %z %Y"

    ts = date_utils.parse_twitter_date_string_as_timestamp(date_string=date_string)

    print(ts)
    ts = int(ts)
    print(datetime.fromtimestamp(ts))
    # print(dt.timestamp())

def test_binning():
    import pandas as pd
    import numpy as np

    result = pd.cut(np.array([1, 7, 5, 4, 6, 3]), 3, labels=range(1,4))

    # print(result.codes)

    df_foo = pd.DataFrame([{"foo": 1},
                  {"foo": 2},
                  {"foo": 3},
                  {"foo": 3}
                  ])

    result = pd.cut(df_foo["foo"].values, 3, labels=range(1,4))
    df_foo['bin'] = result.codes

    # result = df_foo.groupby('bin').agg(['count'])
    df_foo['count_it'] = df_foo.groupby('bin').count()

    print(df_foo.head())




