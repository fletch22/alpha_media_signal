import json
import math
import operator
import re
import time
import urllib
from concurrent.futures.thread import ThreadPoolExecutor
from datetime import timedelta, datetime
from pathlib import Path
from random import shuffle
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import requests
import schedule
from searchtweets import load_credentials, gen_rule_payload, ResultStream
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from ams import utils
from ams.DateRange import DateRange
from ams.config import logger_factory, constants
from ams.services import file_services, ticker_service
from ams.services import stock_action_service as sas
from ams.services.equities import equity_fundy_service
from ams.services.equities.ExchangeType import ExchangeType
from ams.services.equities.TickerService import TickerService
from ams.services.ticker_service import fillna_column
from ams.utils import date_utils, equity_utils, twitter_utils
from ams.utils.PrinterThread import PrinterThread

logger = logger_factory.create(__name__)

COL_AFTER_HOURS = "f22_is_tweet_after_hours"

EARLIEST_TWEET_DATE_STR = "2020-08-10"


def _get_credentials():
    yaml_key = 'search_tweets_fullarchive_development'

    return load_credentials(filename=constants.TWITTER_CREDS_PATH,
                            yaml_key=yaml_key,
                            env_overwrite=False)


def search(query: str, date_range: DateRange = None):
    language = 'lang:en'
    query_esc = f'{query} {language}'

    print(query_esc)

    kwargs = {}
    if date_range is not None:
        from_date = date_utils.get_standard_ymd_format(date_range.from_date)
        to_date = date_utils.get_standard_ymd_format(date_range.to_date)
        kwargs = dict(from_date=from_date, to_date=to_date)

    rule = gen_rule_payload(pt_rule=query_esc, results_per_call=500, **kwargs)

    dev_search_args = _get_credentials()

    rs = ResultStream(rule_payload=rule,
                      max_results=270000,
                      max_pages=1,
                      **dev_search_args)

    return process_tweets_stream(rs)


def process_tweets_stream(rs):
    tweet_generator = rs.stream()
    cache_length = 9
    count = 0
    tweets = []
    tweet_raw_output_path = file_services.create_unique_filename(constants.TWITTER_OUTPUT_RAW_PATH,
                                                                 prefix=constants.TWITTER_RAW_TWEETS_PREFIX,
                                                                 extension='txt')
    for tweet in tweet_generator:
        tweets.append(tweet)
        if len(tweets) >= cache_length:
            append_tweets_to_output_file(output_path=tweet_raw_output_path, tweets=tweets)
            count += len(tweets)
            tweets = []
    if len(tweets) > 0:
        append_tweets_to_output_file(output_path=tweet_raw_output_path, tweets=tweets)
        count += len(tweets)
    return tweet_raw_output_path if count > 0 else None, count


def append_tweets_to_output_file(output_path: Path, tweets: List[Dict], ticker: str):
    json_lines = [json.dumps(t) for t in tweets]
    if len(json_lines):
        with open(str(output_path), 'a+') as f:
            json_lines_nl = [f'{{"version": "0.9.2", "f22_ticker": "{ticker}", "tweet": {j}}}\n' for j in json_lines]
            f.writelines(json_lines_nl)


def search_standard(query: str, tweet_raw_output_path: Path, date_range: DateRange, ticker: str, max_count: int = 5000):
    original_query = query
    pt = PrinterThread()
    sprint = pt.print
    count = 0
    try:
        pt.start()
        language = 'lang:en'
        query_esc = urllib.parse.quote(f'{query} {language}')
        creds = constants.CURRENT_CREDS
        endpoint = creds.endpoint

        start_date_str = date_utils.get_standard_ymd_format(date_range.from_date)
        end_date_str = date_utils.get_standard_ymd_format(date_range.to_date)

        # "https://api.twitter.com/1.1/search/tweets.json?q=Vodafone VOD lang:en&result_type=recent&since=2020-08-10&until=2020-08-11"
        url = f'{endpoint}?q={query_esc}&result_type=mixed&since={start_date_str}&until={end_date_str}&count=100'

        next_results_token = 'next_results'
        status_key = 'statuses'

        while True:
            try:
                response = requests.get(url, headers={
                    "Authorization": f"Bearer {creds.default_bearer_token}"})

                search_results = response.json()
            except Exception as e:
                sprint(e)
                time.sleep(120)
                continue

            if 'errors' in search_results:
                # '{'errors': [{'message': 'Rate limit exceeded', 'code': 88}]}
                sprint(search_results['errors'])
                sprint('Pausing...')
                time.sleep(240)
                continue
            if status_key in search_results:
                tweets = search_results[status_key]
                num_tweets = len(tweets)
                count += num_tweets
                if num_tweets > 0:
                    sprint(f'Fetched {len(tweets)} {original_query} tweets')
                append_tweets_to_output_file(output_path=tweet_raw_output_path, tweets=tweets, ticker=ticker)
            else:
                break

            if count > 5000:
                break

            search_metadata = search_results["search_metadata"]
            if next_results_token in search_metadata:
                query = search_metadata[next_results_token]
                url = f'{endpoint}{query}'
            else:
                break
    finally:
        pt.end()

    return count


def fix_common(name: str, ticker: str, common_words: List[str]):
    if name in common_words and ticker in common_words:
        name = f'{name} stock'

    return name


def create_colloquial_twitter_stock_search_tokens():
    ticker_tuples = TickerService.get_list_of_tickers_by_exchange(cols=['ticker', 'name'],
                                                                  exchange_types=[
                                                                      ExchangeType.NASDAQ])

    ticker_tuples.sort(key=operator.itemgetter(0))

    common_words = utils.load_common_words()

    ticker_data = []
    for t in ticker_tuples:
        ticker = t[0]
        name = t[1]

        name = equity_utils.convert_equity_name_to_common(name)
        name = fix_common(name, ticker, common_words=common_words)
        ticker_data.append({'name': name, 'ticker': ticker})

    df = pd.DataFrame(ticker_data)
    df.to_csv(constants.TICKER_NAME_SEARCHABLE_PATH, index=False)


def remove_items(ticker_tuples, ticker_to_flag: str, delete_before: bool):
    num = 0
    for num, tuple in enumerate(ticker_tuples):
        ticker = tuple[0]
        if ticker == ticker_to_flag:
            break

    return ticker_tuples[num + 1:] if delete_before is True else ticker_tuples[:num + 1]


def get_ticker_searchable_tuples() -> List:
    df = pd.read_csv(constants.TICKER_NAME_SEARCHABLE_PATH)
    ticker_tuples = list(map(tuple, df[['ticker', 'name']].drop_duplicates(subset=["ticker"]).to_numpy()))
    return ticker_tuples


def compose_search_and_query_double(date_range: DateRange, name_1: str, ticker_1: str, name_2: str,
                                    ticker_2: str, tweet_raw_output_path: Path):
    # continue
    query = f'\"{ticker_1}\" {name_1} OR \"{ticker_2}\" {name_2}'

    return search_standard(query=query, tweet_raw_output_path=tweet_raw_output_path,
                           date_range=date_range)


def compose_search_and_query(date_range: DateRange, name: str, ticker: str,
                             tweet_raw_output_path: Path):
    # continue
    query = f'\"{ticker}\" {name}'

    return search_standard(query=query, tweet_raw_output_path=tweet_raw_output_path,
                           date_range=date_range, ticker=ticker)


def get_cashtag_info(ticker: str, has_cashtag: bool) -> Dict:
    return {"ticker": ticker, 'has_cashtag': has_cashtag}


def find_cashtag(raw_line: str, search_tuples: List) -> List[str]:
    tweet = json.loads(raw_line)
    text = tweet['text']
    cashtags_stock = []
    for s in search_tuples:
        ticker = s[0].strip()
        name = s[1].strip()

        if re.search(f'\${ticker}', text) and re.search(name, text, re.IGNORECASE):
            cashtags_stock.append(get_cashtag_info(ticker=ticker, has_cashtag=True))

    if len(cashtags_stock) == 0:
        for s in search_tuples:
            ticker = s[0].strip()
            name = s[1].strip()

            if re.search(ticker, text) and re.search(name, text, re.IGNORECASE):
                cashtags_stock.append(ticker)
                get_cashtag_info(ticker=ticker, has_cashtag=False)

    if len(cashtags_stock) == 0:
        for s in search_tuples:
            ticker = s[0]
            name = s[1]
            if re.search(ticker, raw_line) and re.search(name, raw_line, re.IGNORECASE):
                cashtags_stock.append(ticker)

    print(cashtags_stock)

    tweet['flagged_stocks'] = cashtags_stock
    return tweet


def search_one_day_at_a_time(date_range: DateRange):
    from_date = date_range.from_date
    to_date = date_range.to_date

    num_days = (to_date - from_date).days
    to_date = from_date + timedelta(days=1)
    for i in range(num_days):
        day_range = DateRange(from_date=from_date, to_date=to_date)

        search_with_multi_thread(day_range)

        from_date = from_date + timedelta(days=1)
        to_date = to_date + timedelta(days=1)


def search_with_multi_thread(date_range: DateRange):
    ticker_tuples = get_ticker_searchable_tuples()

    from_date_str = date_utils.get_standard_ymd_format(date_range.from_date)
    if from_date_str == "2021-02-05":
        ticker_tuples = remove_items(ticker_tuples=ticker_tuples, ticker_to_flag='RAND', delete_before=True)

    parent = Path(constants.TWITTER_OUTPUT_RAW_PATH, 'raw_drop', "main")
    tweet_raw_output_path = file_services.create_unique_filename(str(parent),
                                                                 prefix="multithreaded_drop",
                                                                 extension='txt')
    print(f'Output path: {str(tweet_raw_output_path)}')

    pt = PrinterThread()
    sprint = pt.print
    ticker_tweet_count = 0
    try:
        pt.start()

        def custom_request(ticker_info: Tuple[str, str]):
            ticker = ticker_info[0]
            name = ticker_info[1]

            f_date_str = date_utils.get_standard_ymd_format(date_range.from_date)
            t_date_str = date_utils.get_standard_ymd_format(date_range.from_date)
            sprint(f'{ticker}: {name} from {f_date_str} thru {t_date_str}')

            return compose_search_and_query(date_range=date_range, name=name, ticker=ticker,
                                            tweet_raw_output_path=tweet_raw_output_path)

        results = 0
        with ThreadPoolExecutor(6) as executor:
            executor.map(custom_request, ticker_tuples, timeout=None)

        # for tt in ticker_tuples:
        #     ticker_tweet_count += custom_request(ticker_info=tt)

    finally:
        pt.end()

    print(f"Total tweets: {ticker_tweet_count}")


def get_stock_data_for_twitter_companies(df_tweets: pd.DataFrame, num_days_in_future: int = 1):
    ttd = ticker_service.extract_ticker_tweet_dates(df_tweets)
    return ticker_service.get_ticker_on_dates(ttd, num_days_in_future=num_days_in_future)


def get_stock_data_for_twitter_companies_2(df_tweets: pd.DataFrame, num_hold_days: int, num_days_until_purchase: int):
    ttd = ticker_service.extract_ticker_tweet_dates(df_tweets)
    return ticker_service.get_ticker_on_dates_2(ttd, num_hold_days=num_hold_days, num_days_until_purchase=num_days_until_purchase)


def get_rec_quarter_for_twitter():
    df_rec_quart = equity_fundy_service.get_most_recent_quarter_data()
    return df_rec_quart.drop(
        columns=["lastupdated", "dimension", "calendardate", "datekey", "reportperiod"])


def get_all_quarterly_data_for_twitter():
    df_rec_quart = equity_fundy_service.get_all_quarterly_data()
    return df_rec_quart.drop(columns=["lastupdated", "dimension", "datekey", "reportperiod"])


def exagerrate_stock_val_change(value):
    is_neg = -1 if value < 0 else 1
    exag_val = (abs(value + 1)) ** 2

    return is_neg * exag_val


def std_col(df: pd.DataFrame, col_name: str):
    standard_scaler = StandardScaler()

    df = fillna_column(df=df, col=col_name)

    with pd.option_context('mode.chained_assignment', None):
        df.loc[:, col_name] = standard_scaler.fit_transform(df[[col_name]])


def add_buy_sell(df: pd.DataFrame):
    roi_threshold_pct = 0
    df.loc[:, 'stock_val_change'] = ((df['future_close'] - df['close']) / df['close']) - df["nasdaq_day_roi"]

    df.loc[:, 'buy_sell'] = df['stock_val_change'].apply(lambda x: 1 if x >= roi_threshold_pct else -1)
    df.loc[:, 'stock_val_change_ex'] = df["stock_val_change"].apply(exagerrate_stock_val_change)

    std_col(df=df, col_name="stock_val_change_ex")

    return df


def fill_null_numeric(df: pd.DataFrame, cols_fundy_numeric: List[str]):
    zero_fill_cols = ['created_at_timestamp',
                      'favorite_count',
                      'user_listed_count',
                      'user_statuses_count',
                      'user_friends_count',
                      'retweet_count', 'user_followers_count',
                      'f22_num_other_tickers_in_tweet', 'f22_sentiment_pos',
                      'f22_sentiment_neu', 'f22_sentiment_neg', 'f22_sentiment_compound',
                      'f22_compound_score']

    cols_zero_combined = zero_fill_cols + cols_fundy_numeric

    zero_fills = {z: df[z][df[z].notnull()].median() for z in cols_zero_combined}

    for k, v in zero_fills.items():
        v = -1 if str(v) == 'nan' else v
        df[k] = df[k].fillna(v)

    return df


def fill_empty_str_cols(df: pd.DataFrame):
    df = df.dropna(subset=["f22_ticker"])

    empty_fill_cols = ['user_screen_name',
                       'in_reply_to_screen_name',
                       'metadata_result_type',
                       'user_location', 'place_country',
                       'place_name']

    df[empty_fill_cols] = df[empty_fill_cols].fillna("")
    df["lang"] = df["lang"].fillna("en")
    df_filtered = df.dropna(subset=["date"])

    return df_filtered


def is_after_nasdaq_closed(created_at_timestamp: int):
    return date_utils.is_after_nasdaq_closed(utc_timestamp=created_at_timestamp)


def add_is_tweet_after_hours(df: pd.DataFrame):
    df[COL_AFTER_HOURS] = None
    df[COL_AFTER_HOURS] = df.apply(lambda x: is_after_nasdaq_closed(x['created_at_timestamp']), axis=1)
    return df


def add_tweet_count(df: pd.DataFrame):
    df_g_counting = df.groupby(['f22_ticker', 'purchase_date'])
    s_sized = df_g_counting.size()

    df_renamed = s_sized.to_frame().rename(columns={0: 'f22_day_tweet_count'})

    return df.join(df_renamed, on=["f22_ticker", "purchase_date"], how="inner")


def convert_col_to_bool(df: pd.DataFrame, cols: List[str]):
    for c in cols:
        df = df.astype({c: str})
        df[c] = df[c].apply(lambda x: x.lower().strip())
        df = df.replace({c: {'true': True, 'false': False, '': False, '0': False, '1': True}})
        df[c] = df[c].fillna(False).astype('bool')

    return df.copy()


def convert_to_bool(df: pd.DataFrame):
    return convert_col_to_bool(df, ['possibly_sensitive', 'f22_ticker_in_text', 'user_verified',
                                    'f22_has_cashtag',
                                    'user_has_extended_profile', 'user_is_translation_enabled',
                                    'f22_ticker_in_text',
                                    'user_protected', 'user_geo_enabled'])


# def std_numeric_cols(df: pd.DataFrame, cols_additional: List[str]):
#     df["original_close_price"] = df["close"]
#     std_cols = [
#         'favorite_count',
#         'user_listed_count',
#         'user_statuses_count',
#         'user_friends_count',
#         'retweet_count', 'user_follow_request_sent', 'user_followers_count',
#         'f22_num_other_tickers_in_tweet', 'f22_sentiment_pos',
#         'f22_sentiment_neu', 'f22_sentiment_neg', 'f22_sentiment_compound',
#         'f22_compound_score', 'open', 'high', 'low', 'close', 'closeunadj', 'volume', "f22_day_tweet_count"]
#
#     std_cols += cols_additional
#
#     for i, c in enumerate(std_cols):
#         df[c] = scaler.fit_transform(df[[c]])
#
#     return df


def refine_pool(df: pd.DataFrame, min_volume: int = None, min_price: float = None, max_price: float = None):
    if min_volume is not None:
        df = df[df["prev_volume"] > min_volume].copy()
    if min_price is not None:
        df = df[df["prev_close"] >= min_price].copy()
    if max_price is not None:
        df = df[df["prev_close"] <= max_price].copy()
    return df


def join_with_stock_splits(df: pd.DataFrame):
    df_stock_splits = sas.get_splits()[["ticker", "date", "value"]]

    df_holdouts_clean = df.drop(columns=["date"])

    df_split_aware = pd.merge(df_holdouts_clean, df_stock_splits, how='left',
                              left_on=["f22_ticker", "purchase_date"], right_on=["ticker", "date"])
    df_splitted = df_split_aware.rename(columns={"value": "split_share_multiplier"}).drop(
        columns=["ticker", "date"])
    df_splitted["split_share_multiplier"] = df_splitted["split_share_multiplier"].fillna(1.0)

    return df_splitted


def dec_tree(X_train: np.array, y_train: np.array, X_test: np.array, y_test: np.array, max_depth: int):
    decisiontree = DecisionTreeClassifier(max_depth=max_depth)
    model = decisiontree.fit(X_train, y_train)
    y_test_pred = model.predict(X_test)

    from sklearn.metrics import accuracy_score

    print(accuracy_score(y_test, y_test_pred))

    return model


def dec_tree_regressor(X_train: np.array, y_train: np.array, X_test: np.array, y_test: np.array, max_depth: int):
    regressor = DecisionTreeRegressor(random_state=0, max_depth=max_depth)
    model = regressor.fit(X_train, y_train)
    y_test_pred = model.predict(X_test)

    from sklearn.metrics import r2_score

    print(r2_score(y_test, y_test_pred))

    return model


def rnd_forest_clf(X_train: np.array, y_train: np.array, X_test: np.array, y_test: np.array,
                   max_depth: int):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score

    rf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                                max_depth=None, max_features='auto', max_leaf_nodes=None,
                                min_impurity_split=1e-07, min_samples_leaf=1,
                                min_samples_split=2, min_weight_fraction_leaf=0.0,
                                n_estimators=16, n_jobs=1, oob_score=False, random_state=None,
                                verbose=0, warm_start=False)
    rf.fit(X=X_train, y=y_train)

    y_test_pred = rf.predict(X_test)

    print(accuracy_score(y_test, y_test_pred))

    return rf


def train_mlp(X_train: np.array, y_train: np.array, X_test: np.array, y_test: np.array):
    start = time.time()
    num_input_features = X_train.shape[1]
    classes = 2  # Buy/Sell
    num_hidden_neurons = int(num_input_features / classes)
    clf = MLPClassifier(hidden_layer_sizes=(num_hidden_neurons), max_iter=800, tol=1e-19, activation='relu',
                        solver='adam')
    clf.fit(X_train, y_train)  # Fit data
    y_pred = clf.predict(X_test)  # Predict results for x_test
    accs = accuracy_score(y_test, y_pred)  # Accuracy Score
    end = time.time()

    print(f"{accs}: Elapsed time: {end - start} seconds.")

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.set_title("MLP Classifier Loss Curve")
    ax.plot(clf.loss_curve_)

    fig.canvas.draw()

    return clf


def omit_columns(df: pd.DataFrame):
    omit_cols = ['created_at_timestamp', 'in_reply_to_status_id', 'place_country', 'user_time_zone',
                 'place_name', "famasector", "f22_id",
                 'user_location', 'metadata_result_type', 'place_name', 'place_country',
                 'lang', 'in_reply_to_screen_name', 'lastupdated', 'created_at', "prev_date", "future_open", "future_close",
                 "future_low", "future_high", "calendardate", "reportperiod", "dimension", "datekey"]

    narrow_cols = list(set(df.columns) - set(omit_cols))

    return df[narrow_cols]


def balance_df(df: pd.DataFrame):
    df_samp_buy = df[df["buy_sell"] == 1].copy()
    df_samp_sell = df[df["buy_sell"] != 1].copy()

    num_buy = df_samp_buy.shape[0]
    num_sell = df_samp_sell.shape[0]
    if num_buy <= num_sell:
        frac = df_samp_buy.shape[0] / df_samp_sell.shape[0]
        df_samp_sell = df_samp_sell.sample(frac=frac)
    else:
        frac = df_samp_sell.shape[0] / df_samp_buy.shape[0]
        df_samp_buy = df_samp_buy.sample(frac=frac)

    return pd.concat([df_samp_buy, df_samp_sell]).sample(frac=1.0)


def split_train_test(train_set: pd.DataFrame, test_set: pd.DataFrame, train_cols: List[str], label_col: str = "buy_sell"):
    train_set_bal = balance_df(train_set)
    test_set_bal = balance_df(test_set)

    X_train = np.array(train_set_bal[train_cols])
    X_test = np.array(test_set_bal[train_cols])

    y_train = np.array(train_set_bal[label_col])
    y_test = np.array(test_set_bal[label_col])

    return X_train, y_train, X_test, y_test, train_cols


def split_df_for_learning(df: pd.DataFrame, train_cols: List[str], label_col: str = "buy_sell", require_balance: bool = True):
    if require_balance:
        df = balance_df(df)
        logger.info(f"balanced data: {df.shape[0]}")

    if df.shape[0] == 0:
        return None, None

    X = np.array(df[train_cols])
    y = np.array(df[label_col])

    return X, y


def get_feature_columns(narrow_cols):
    omit_cols = {'buy_sell', 'date', 'purchase_date', "future_open", 'future_low', "future_high",
                 "future_close", "stock_val_change_ex",
                 "stock_val_change_scaled", "stock_val_change", "roi", "user_screen_name",
                 "future_date", "user_follow_request_sent", "f22_ticker"}
    omit_cols |= {"nasdaq_day_roi"}
    # End FIXME
    return list(set(narrow_cols) - omit_cols)


def split_by_date(df):
    df.sort_values(by=['purchase_date'], inplace=True)

    num_rows = df.shape[0]

    df_train = df.iloc[:math.ceil(num_rows * .8)]
    df_test = df.iloc[math.ceil(-(num_rows * .2)):]

    f22_dates = df_test["purchase_date"].unique().tolist()
    df_train = df_train[~df_train["purchase_date"].isin(f22_dates)]

    return df_train, df_test


def ho_split_by_days(df, small_data_days_to_pull: int = None, small_data_frac: float = None,
                     use_only_recent_for_holdout: bool = False):
    if small_data_days_to_pull is not None and small_data_frac is not None:
        raise Exception(
            "Cannot pass data in both 'small_data_days_to_pull' and 'small_data_frac' function arguments.")

    date_max = df["purchase_date"].max()
    date_min = df["purchase_date"].min()

    logger.info(f"Split | min: {date_min} | max: {date_max}")

    dt_max = date_utils.parse_std_datestring(date_max)
    dt_min = date_utils.parse_std_datestring(date_min)

    total_days = (dt_max - dt_min).days

    if total_days < 1:
        return None, None

    if small_data_days_to_pull is not None:
        num_days_to_pull = small_data_days_to_pull
    else:
        num_days_to_pull = math.ceil(small_data_frac * total_days)

    pull_days = list(range(total_days + 1))
    if not use_only_recent_for_holdout:
        shuffle(pull_days)
    pull_days = pull_days[-num_days_to_pull:]

    days_to_pull = []
    for i in pull_days:
        rand_dt = dt_min + timedelta(days=i)
        rand_date_string = date_utils.get_standard_ymd_format(rand_dt)
        days_to_pull.append(rand_date_string)

    logger.info(f"Split dates for small dataset: {days_to_pull}")

    df_samp = df[~df["purchase_date"].isin(days_to_pull)]
    df_holdouts = df[df["purchase_date"].isin(days_to_pull)]

    return df_samp, df_holdouts


def remove_last_days(df: pd.DataFrame, num_days: int):
    result = df
    has_remaining_days = False
    if num_days > 0:
        min_date_str = df["purchase_date"].min()
        max_date_str = df["purchase_date"].max()

        dt_max = date_utils.parse_std_datestring(max_date_str)

        dt_youngest = dt_max - timedelta(days=num_days)
        dt_youngest_str = date_utils.get_standard_ymd_format(dt_youngest)

        has_remaining_days = False if min_date_str >= dt_youngest_str else True

        result = df[df["purchase_date"] <= dt_youngest_str]

    return result, has_remaining_days


def fetch_up_to_date_tweets():
    date_range = get_search_date_range()

    search_one_day_at_a_time(date_range=date_range)

    return date_range


def get_search_date_range():
    youngest_tweet_date_str = twitter_utils.get_youngest_tweet_date_in_system()
    from_date = date_utils.parse_std_datestring(youngest_tweet_date_str) + timedelta(days=1)
    to_date = datetime.now()

    to_date_str = date_utils.get_standard_ymd_format(to_date)
    from_date_str = date_utils.get_standard_ymd_format(to_date)
    if to_date_str <= from_date_str:
        to_date = from_date + timedelta(days=1)
    date_range = DateRange(from_date=from_date, to_date=to_date)
    return date_range


def get_daily_prediction():
    jobs_start = "01:30"
    schedule.every().day.at(jobs_start).do(fetch_up_to_date_tweets)

    while True:
        logger.info(f"Waiting to start job at {jobs_start} ...")
        schedule.run_pending()
        time.sleep(120)


if __name__ == '__main__':
    # get_daily_prediction()

    fetch_up_to_date_tweets()
    # date_range = DateRange.from_date_strings(from_date_str="2021-02-12", to_date_str="2021-02-13")
    # search_one_day_at_a_time(date_range=date_range)
    # youngest_tweet_date_str = twitter_utils.get_youngest_tweet_date_in_system()
    # print(youngest_tweet_date_str)