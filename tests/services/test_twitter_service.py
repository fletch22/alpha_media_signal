import json
import statistics

from ams.DateRange import DateRange
from ams.config import constants
from ams.services import twitter_service, ticker_service


def test_tuples():
    tickers_tuples = twitter_service.get_ticker_searchable_tuples()

    print(len(list(set(tickers_tuples))))

    print(f"Num ticker tuples: {len(tickers_tuples)}")
    sample = tickers_tuples[:4]

    print(sample)


def test_get_cashtags():
    # Arrange
    ticker = "AAPL"
    text = "Buy buy buy $AAPL stock!"

    # Act
    # result = re.search(f'\${ticker}', text)
    index = text.find(f'${ticker}')

    # Assert
    # assert(result is not None)
    print(index)


def test_twitter_service():
    from ams.services import file_services
    # Arrange
    query = "AAPL"
    date_range = DateRange.from_date_strings("2020-10-04", "2020-10-06")
    output_path = file_services.create_unique_filename(constants.TWITTER_TRASH_OUTPUT, prefix="search_twitter_test")

    # Act
    twitter_service.search_standard(query=query, tweet_raw_output_path=output_path, date_range=date_range)
    # Assert


def test_dict():
    ticker = "AAA"
    group_preds = {}
    info = {}
    group_preds[ticker] = info
    info["foo"] = 1

    print(group_preds)


def test_variance():
    xgd_roi = [0.01016489518739397, 0.005431473886264778, 0.0041906347964202504, 0.0032364696599932424,
               0.003079144070843302, -0.0034775125215308366, 0.004949152172345926, 0.014665985945089976,
               0.042262781499111154, -0.007055702278903755, -0.006475380892203079, 0.009944160932423602,
               0.01225172693921719, 0.01430208122052239, -0.003576006694340998, -0.006829779210202892,
               -0.014290912524036524, -0.018709054576318992, -0.014105749861243692, 0.0034707887891016697,
               0.006285752420679376, 0.00044407548071677703, 0.005369998448713367, -0.0014652196571048994,
               -0.0035726428180204847, 0.0009415602860621213, -0.0041319402768939095, -0.0019397130848269582,
               0.003585340367653943, 0.0029561792410786256, 0.007912720973937495, 0.0109789888246501,
               0.012236541590707079, 0.009642140823852494, 0.015333620906635995, 0.005950767015591503,
               0.005029145838466293, 0.007595850557538718, 0.013286442151762648, 0.013841033590808701,
               -0.0010103689160377185, -0.0224412848409366, -0.018221099905199416, -0.027506634989750804,
               0.002573627175276247, -0.0005823738580257419, 0.0006519800838628998, 0.003531365866339583,
               0.023835092714383617, 0.011603688373690156, -0.006927938715145362, 0.00013382903106703474,
               0.00642892470124381, -0.0035299866859325657, -0.0006997822341065042, 0.0007178811979486065,
               0.0028709947129961035, -0.009030206963279044, -0.007146772386615384, -0.002315261432719855,
               -0.006468927275210507, 0.009784060072983436, -0.0005660773319218205, -0.002449978774734192]

    nas_roi = [0.0018062180653949, 0.0014940770395308256, -0.001569225190245889, 0.005286247583284451, -0.007211932514484991, -0.00028312293261191116, -0.004218958508642669,
               -0.010672034628326846, -0.0013324513498838254, 0.008282044034146842, -0.003931259700315608, -0.004908462694112828, 0.010492061731315498, 0.009977077738488284,
               0.00028033798161251265, 0.005256138544715342, -0.02624905121931827, -0.007085697822482278, -0.0090545627472927, 0.01403293752248785, -0.006525418299935892,
               -0.005271756902691373, 0.02392764404619495, 0.003922026336346108, 0.007843207944603008, -0.00015456972273578896, 0.005661770508390591, -0.027187342114291375,
               -0.0003304226421702049, -0.024765272466523366, -0.004115911980550246, 0.017886053251231562, 0.01946375402539589, 0.0055727456461636135, 0.005352977230668436,
               0.009374115278017634, -0.0008450207499160761, 0.02213346743388073, -0.0015494405551833782, 0.021224536006606183, 0.008945994551017673, 0.006318168203757112,
               0.004914718496879112, -0.0026342191850376607, -0.004781657058121968, 0.0036894701530060152, 0.001078446288846616, 0.04458474905741843, 0.0002504398843868148,
               -0.0057486908449396355, 0.012561564265838446, 0.005257190159840517, 0.013670166695223139, -0.005707159970003562, -0.026696780127356302, 0.010221578133953317,
               -0.016697137007659162, 0.009675116835638917, 0.02361919568536519, 0.0035142930805712605, 0.023418870512890128, -0.007696058582134161, 0.026734660815077003,
               0.012252307861201045, 0.007856155148335797, -0.008080259692148959, 0.01569338648447066, 0.017584965488348116, 0.003936927566187403, -0.0029643278617197473,
               0.012615811885924429, 0.005346573424422124, 0.019675474616984788]

    xgd_std = statistics.stdev(xgd_roi)
    nas_std = statistics.stdev(nas_roi)

    print(f"xgd std: {xgd_std}; nas std: {nas_std}")

    print(f"{xgd_std}/{nas_std}")

    import statistics as s
    print(f"Mean: xgd: {s.mean(xgd_roi)}; nas roi: {s.mean(nas_roi)}")


def test_nas_roi():
    import pandas as pd
    df_roi_nasdaq = pd.read_parquet(str(constants.DAILY_ROI_NASDAQ_PATH))

    print(df_roi_nasdaq.head(100))


def test_bad_file():
    # findspark.init()
    # spark = spark_service.get_or_create(app_name='twitter_flatten_test')
    # sc = spark.sparkContext

    file_path_str = """C:\\Users\\Chris\\workspaces\\data\\twitter\\fixed_drop\\main\\smallified_2020-12-26_16-06-28-263.01.parquet.txt_fixed.txt"""

    with open(file_path_str, "r+") as rf:
        all_lines = rf.readlines()
        for line in all_lines:
            thing = json.loads(line)
            print(thing["user"])


def test_twitter_trade_history():
    import pandas as pd

    df = pd.read_csv(constants.TWITTER_TRADE_HISTORY_FILE_PATH, )

    df = df.sample(frac=1.0)

    df = df[df["purchase_price"] > 5.]

    df["roi"] = (df["sell_price"] - df["purchase_price"]) / df["purchase_price"]

    df_g = df.groupby(by=["purchase_dt"])

    all_days = []
    max_stock_buy = 8
    for ndx, (group_name, df_group) in enumerate(df_g):
        num_samples = df_group.shape[0]
        num_samples = max_stock_buy if num_samples >= max_stock_buy else num_samples
        df_g_samp = df_group.iloc[:num_samples]
        day_mean = df_g_samp["roi"].mean()
        tickers = df_g_samp["ticker"].to_list()
        print(tickers)
        all_days.append(day_mean)

    print(f'roi with max stock buy {max_stock_buy}: {statistics.mean(all_days)} ')
    print(df['roi'].mean())
    print(f"Total trades: {len(all_days)}: {all_days}")

    initial_inv = 1000
    total = initial_inv
    for roi in all_days:
       ret = total * roi
       total += ret
    print(f"Total roi: {(total - initial_inv) / initial_inv}")

    # validate_roi_data(df)




def validate_roi_data(df):
    df_samp = df.iloc[:1000]
    num_days_to_wait = 5
    cols = ["future_close", "future_date"]
    for index, row in df_samp.iterrows():
        ticker = row["ticker"]
        purchase_date = row["purchase_dt"]
        purchase_price = row["purchase_price"]
        sell_price = row["sell_price"]

        df_tick = ticker_service.get_ticker_eod_data(ticker=ticker)
        df_tick["future_close"] = df_tick["close"]
        df_tick["future_date"] = df_tick["date"]
        df_tick[cols] = df_tick[cols].shift(-num_days_to_wait)
        df_tick = df_tick[df_tick["date"] == purchase_date]
        tick_row = df_tick.iloc[0]
        close = tick_row["close"]
        future_close = tick_row["future_close"]

        print(f"PP: {purchase_price}; SP: {sell_price}")
        print(f"close: {close}; fp: {future_close}")
        assert (round(purchase_price, 3) == round(close, 3))
        assert (round(sell_price, 3) == round(future_close, 3))
