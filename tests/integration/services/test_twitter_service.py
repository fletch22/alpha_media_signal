from ams.config import logger_factory
from ams.services import twitter_service as ts

import json
import statistics
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from sklearn.neural_network import MLPClassifier

from ams.DateRange import DateRange
from ams.config import constants, logger_factory
from ams.services import twitter_service, ticker_service, stock_action_service
from ams.services.spark_service import get_or_create
from ams.services.twitter_service import get_split_prepped_twitter_data
from ams.utils import date_utils
import pandas as pd

logger = logger_factory.create(__name__)

pd.set_option('display.max_rows', 5000)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

tweet_raw_output_path = Path(constants.TWITTER_OUTPUT_RAW_PATH, "test", "test_output.json")


def test_get_search_date_range():
    # Arrange
    # Act
    date_range = ts.get_search_date_range()

    # Assert
    logger.info(f"from {date_range.from_date}")
    logger.info(f"to {date_range.to_date}")


def test_tuples():
    tickers_tuples = twitter_service.get_ticker_searchable_tuples()

    logger.info(len(list(set(tickers_tuples))))

    logger.info(f"Num ticker tuples: {len(tickers_tuples)}")
    sample = tickers_tuples[:4]

    logger.info(sample)


def test_get_cashtags():
    # Arrange
    ticker = "AAPL"
    text = "Buy buy buy $AAPL stock!"

    # Act
    # result = re.search(f'\${ticker}', text)
    index = text.find(f'${ticker}')

    # Assert
    # assert(result is not None)
    logger.info(index)


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

    logger.info(group_preds)


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

    logger.info(f"xgd std: {xgd_std}; nas std: {nas_std}")

    logger.info(f"{xgd_std}/{nas_std}")

    import statistics as s
    logger.info(f"Mean: xgd: {s.mean(xgd_roi)}; nas roi: {s.mean(nas_roi)}")


def test_nas_roi():
    import pandas as pd
    df_roi_nasdaq = pd.read_parquet(str(constants.DAILY_ROI_NASDAQ_PATH))

    logger.info(df_roi_nasdaq.head(100))


def test_bad_file():
    # findspark.init()
    # spark = spark_service.get_or_create(app_name='twitter_flatten_test')
    # sc = spark.sparkContext

    file_path_str = """C:\\Users\\Chris\\workspaces\\data\\twitter\\fixed_drop\\main\\smallified_2020-12-26_16-06-28-263.01.parquet.txt_fixed.txt"""

    with open(file_path_str, "r+") as rf:
        all_lines = rf.readlines()
        for line in all_lines:
            thing = json.loads(line)
            logger.info(thing["user"])


def test_twitter_trade_history():
    import pandas as pd

    df = pd.read_csv(constants.TWITTER_TRADE_HISTORY_FILE_PATH)
    logger.info(f"Num rows: {df.shape[0]}")

    df = df.sample(frac=1.0)

    max_price = 5.
    df = df[df["purchase_price"] > max_price]

    df["roi"] = (df["sell_price"] - df["purchase_price"]) / df["purchase_price"]

    df_g = df.groupby(by=["purchase_dt"])

    all_days = []
    max_stock_buy = 8
    for group_name, df_group in df_g:
        num_samples = df_group.shape[0]
        num_samples = max_stock_buy if num_samples >= max_stock_buy else num_samples
        df_g_samp = df_group.iloc[:num_samples]
        day_mean = df_g_samp["roi"].mean()
        tickers = df_g_samp["ticker"].to_list()
        logger.info(tickers)
        all_days.append(day_mean)

    logger.info(f'roi with max stock buy {max_stock_buy}: {statistics.mean(all_days)} ')
    logger.info(df['roi'].mean())
    logger.info(f"Total trades: {len(all_days)}: {all_days}")

    initial_inv = 1000
    total = initial_inv
    for roi in all_days:
        total = (total * roi) + total
    logger.info(f"Total roi: {(total - initial_inv) / initial_inv}")

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

        logger.info(f"PP: {purchase_price}; SP: {sell_price}")
        logger.info(f"close: {close}; fp: {future_close}")
        assert (round(purchase_price, 3) == round(close, 3))
        assert (round(sell_price, 3) == round(future_close, 3))


def test_compare():
    from_xgb = {'table': ['SEP', 'SF1', 'SF3B', '<unknown>', 'SFP'],
                'category': ['Domestic Preferred Stock', 'ADR Preferred Stock', 'ETF', 'ETN', 'Canadian Preferred Stock', 'Domestic Stock Warrant', 'Canadian Common Stock',
                             'ADR Common Stock Primary Class', 'Canadian Common Stock Primary Class', 'ETD', 'CEF', 'ADR Common Stock', 'IDX',
                             'Domestic Common Stock Primary Class',
                             '<unknown>', 'Canadian Stock Warrant', 'ADR Stock Warrant', 'Domestic Common Stock Secondary Class', 'Domestic Common Stock',
                             'ADR Common Stock Secondary Class'],
                'sicsector': ['Agriculture Forestry And Fishing', 'Transportation Communications Electric Gas And Sanitary Service', 'Finance Insurance And Real Estate',
                              '<unknown>',
                              'Public Administration', 'Services', 'Nonclassifiable', 'Construction', 'Wholesale Trade', 'Retail Trade', 'Manufacturing', 'Mining'],
                'famaindustry': ['Textiles', 'Healthcare', 'Restaraunts Hotels Motels', 'Candy & Soda', 'Pharmaceutical Products', 'Medical Equipment', 'Consumer Goods',
                                 'Machinery',
                                 'Shipping Containers', 'Petroleum and Natural Gas', 'Measuring and Control Equipment', 'Apparel', 'Coal', 'Construction Materials',
                                 'Electronic Equipment', 'Tobacco Products', 'Printing and Publishing', 'Personal Services', 'Chemicals', 'Insurance', 'Retail',
                                 'Electrical Equipment',
                                 'Automobiles and Trucks', 'Wholesale', 'Almost Nothing', 'Banking', 'Food Products', 'Transportation', '<unknown>', 'Trading', 'Construction',
                                 'Business Supplies', 'Steel Works Etc', 'Defense', 'Utilities', 'Recreation', 'Rubber and Plastic Products',
                                 'Non-Metallic and Industrial Metal Mining',
                                 'Shipbuilding Railroad Equipment', 'Real Estate', 'Communication', 'Agriculture', 'Beer & Liquor', 'Precious Metals', 'Entertainment',
                                 'Fabricated Products', 'Computers', 'Business Services', 'Aircraft'],
                'sector': ['Basic Materials', 'Technology', 'Healthcare', 'Energy', 'Real Estate', 'Communication Services', '<unknown>', 'Utilities', 'Consumer Cyclical',
                           'Consumer Defensive', 'Financial Services', 'Industrials'],
                'industry': ['Publishing', 'Industrial Metals & Minerals', 'Medical Care Facilities', 'Shipping & Ports', 'Utilities - Renewable', 'Packaged Foods',
                             'Integrated Freight & Logistics', 'Other Industrial Metals & Mining', 'Metal Fabrication', 'Luxury Goods', 'Aluminum', 'Trucking',
                             'Medical Distribution',
                             'Medical Instruments & Supplies', 'Semiconductor Equipment & Materials', 'Pollution & Treatment Controls', 'Long-Term Care Facilities',
                             'Electrical Equipment & Parts', 'Furnishings', 'Media - Diversified', 'Footwear & Accessories', 'Textile Manufacturing', 'Recreational Vehicles',
                             'Insurance - Reinsurance', 'Silver', 'Computer Hardware', 'Aerospace & Defense', 'Telecom Services', 'Grocery Stores',
                             'Security & Protection Services',
                             'Insurance - Life', 'Lumber & Wood Production', 'Semiconductors', 'Specialty Chemicals', 'Software - Application', 'Healthcare Plans',
                             'Beverages - Brewers',
                             'Specialty Retail', 'Farm Products', 'Internet Content & Information', 'Farm & Construction Equipment', 'Business Equipment',
                             'Specialty Industrial Machinery', 'Oil & Gas Refining & Marketing', 'Banks - Regional - US', 'Diversified Industrials',
                             'REIT - Healthcare Facilities',
                             'Financial Conglomerates', 'Other Precious Metals & Mining', 'Banks - Regional', 'Utilities - Regulated Gas',
                             'Utilities - Independent Power Producers',
                             'Banks - Global', 'Financial Data & Stock Exchanges', 'Specialty Finance', 'Tobacco', 'Pharmaceutical Retailers', 'Building Products & Equipment',
                             'Real Estate - Development', 'Utilities - Regulated Water', 'Software - Infrastructure', 'Shell Companies', 'Tools & Accessories',
                             'Apparel Manufacturing',
                             'Insurance - Property & Casualty', 'Diagnostics & Research', 'Apparel Retail', '<unknown>', 'Capital Markets', 'Auto & Truck Dealerships',
                             'Discount Stores',
                             'Solar', 'REIT - Mortgage', 'REIT - Diversified', 'Staffing & Outsourcing Services', 'REIT - Industrial', 'Asset Management', 'Travel Services',
                             'Semiconductor Memory', 'Advertising Agencies', 'Airports & Air Services', 'Savings & Cooperative Banks', 'Medical Devices', 'Internet Retail',
                             'Leisure',
                             'Oil & Gas Integrated', 'Agricultural Inputs', 'Information Technology Services', 'Broadcasting - Radio', 'Auto Manufacturers', 'Biotechnology',
                             'Apparel Stores', 'Personal Services', 'Chemicals', 'Electronic Gaming & Multimedia', 'Beverages - Soft Drinks', 'Business Equipment & Supplies',
                             'Packaging & Containers', 'Lodging', 'REIT - Retail', 'Real Estate Services', 'Insurance Brokers', 'Broadcasting - TV', 'Oil & Gas Midstream',
                             'Scientific & Technical Instruments', 'Drug Manufacturers - Specialty & Generic', 'Insurance - Diversified', 'Medical Care',
                             'Drug Manufacturers - Major',
                             'Confectioners', 'Marine Shipping', 'Farm & Heavy Construction Machinery', 'Residential Construction', 'Staffing & Employment Services', 'Steel',
                             'Electronics & Computer Distribution', 'Railroads', 'Electronic Components', 'Paper & Paper Products', 'Entertainment', 'Conglomerates',
                             'Home Furnishings & Fixtures', 'Furnishings Fixtures & Appliances', 'Business Services', 'Department Stores', 'Engineering & Construction',
                             'Infrastructure Operations', 'Airlines', 'Food Distribution', 'Oil & Gas E&P', 'Mortgage Finance', 'Resorts & Casinos',
                             'Real Estate - Diversified',
                             'Copper', 'REIT - Hotel & Motel', 'Coking Coal', 'Thermal Coal', 'Rental & Leasing Services', 'Consumer Electronics', 'REIT - Specialty',
                             'Credit Services',
                             'Coal', 'Consulting Services', 'Waste Management', 'Beverages - Non-Alcoholic', 'REIT - Residential', 'REIT - Office', 'Building Materials',
                             'Utilities - Regulated Electric', 'Beverages - Wineries & Distilleries', 'Computer Systems', 'Home Improvement Stores', 'Oil & Gas Drilling',
                             'Home Improvement Retail', 'Uranium', 'Household & Personal Products', 'Gold', 'Health Care Plans', 'Data Storage', 'Auto Parts',
                             'Health Information Services', 'Financial Exchanges', 'Utilities - Diversified', 'Real Estate - General', 'Education & Training Services',
                             'Specialty Business Services', 'Oil & Gas Equipment & Services', 'Drug Manufacturers - General', 'Insurance - Specialty', 'Broadcasting',
                             'Industrial Distribution', 'Communication Equipment', 'Restaurants', 'Gambling', 'Banks - Diversified'],
                'scalemarketcap': ['2 - Micro', '1 - Nano', '<unknown>', '5 - Large', '6 - Mega', '4 - Mid', '3 - Small'],
                'scalerevenue': ['2 - Micro', '1 - Nano', '<unknown>', '5 - Large', '6 - Mega', '4 - Mid', '3 - Small'],
                'currency': ['EUR', 'COP', 'MXN', 'SEK', 'PEN', 'JPY', 'AUD', 'ILS', 'HKD', 'BRL', 'TRY', 'CHF', 'KRW', 'INR', 'CLP', '<unknown>', 'DKK', 'PLN', 'IDR', 'NZD',
                             'CNY',
                             'ARS', 'VEF', 'NOK', 'USD', 'GBP', 'ZAR', 'MYR', 'RUB', 'PHP', 'TWD', 'CAD'],
                'location': ['Colombia', 'North Carolina; U.S.A', "Democratic People'S Republic Of Korea", 'Ohio; U.S.A', 'United Kingdom', 'South Dakota; U.S.A',
                             'Marshall Islands',
                             'Singapore', 'Rhode Island; U.S.A', 'Brazil', 'Maldives', 'Arizona; U.S.A', 'Belgium', 'Turkey', 'Republic Of Korea', 'Oklahoma; U.S.A',
                             'United Republic Of Tanzania', 'Michigan; U.S.A', 'Jersey', 'California; U.S.A', 'South Carolina; U.S.A', 'Ghana', 'Alaska; U.S.A',
                             'West Virginia; U.S.A',
                             'Ireland', 'Nova Scotia; Canada', 'Colorado; U.S.A', 'Peru', 'Manitoba; Canada', 'Maryland; U.S.A', 'Illinois; U.S.A', 'Hong Kong', 'Venezuela',
                             'Guernsey',
                             'Chile', 'Philippines', 'India', 'Italy', 'Israel', 'Switzerland', 'Maine; U.S.A', 'Greece', 'Mississippi; U.S.A', 'Puerto Rico',
                             'Kentucky; U.S.A',
                             'Finland', 'Panama', 'China', 'District Of Columbia; U.S.A', 'United States; U.S.A', 'Alabama; U.S.A', 'Georgia U.S.A.', 'Czech Republic',
                             'New Jersey; U.S.A', 'Japan', 'Guam', 'United Arab Emirates', 'Sweden', 'Indiana; U.S.A', 'Bermuda', '<unknown>', 'Macau',
                             'British Columbia; Canada',
                             'Delaware; U.S.A', 'Louisiana; U.S.A', 'Argentina', 'Norway', 'Hawaii; U.S.A', 'New Hampshire; U.S.A', 'Massachusetts; U.S.A',
                             'North Dakota; U.S.A',
                             'Oregon; U.S.A', 'Virgin Islands; U.S.', 'Montana; U.S.A', 'Monaco', 'Netherlands Antilles', 'Hungary', 'Iowa; U.S.A', 'Denmark', 'Jordan',
                             'Texas; U.S.A',
                             'Idaho; U.S.A', 'Cayman Islands', 'Netherlands', 'Missouri; U.S.A', 'Cyprus', 'Vermont; U.S.A', 'Gibraltar', 'South Africa',
                             'Canada (Federal Level)',
                             'Pennsylvania; U.S.A', 'Thailand', 'Florida; U.S.A', 'New York; U.S.A', 'Nebraska; U.S.A', 'Tennessee; U.S.A', 'Luxembourg', 'Poland',
                             'Saint Vincent And The Grenadines', 'Washington; U.S.A', 'Russian Federation', 'British Virgin Islands', 'Taiwan', 'Costa Rica', 'Australia',
                             'Alberta; Canada', 'Germany', 'Israel-Syria', 'Arkansas; U.S.A', 'Quebec; Canada', 'Bahamas', 'France', 'Israel-Jordan', 'Austria', 'Malaysia',
                             'Iceland',
                             'Virginia; U.S.A', 'Saskatchewan; Canada', 'Newfoundland; Canada', 'Georgia; U.S.A', 'Wyoming; U.S.A', 'Connecticut; U.S.A', 'Indonesia',
                             'Utah; U.S.A',
                             'Uruguay', 'Canada', 'New Zealand', 'Unknown', 'Wisconsin; U.S.A', 'New Brunswick; Canada', 'Oman', 'Ontario; Canada', 'Saudi Arabia',
                             'Minnesota; U.S.A',
                             'Kansas; U.S.A', 'Nevada; U.S.A', 'Isle Of Man', 'Malta', 'Mauritius', 'Mexico', 'New Mexico; U.S.A', 'Spain'],
                'f22_ticker': ['MIRM', 'SEDG', 'LQDT', 'ALGN', 'ARVN', 'HJLI', 'CELH', 'LIND', 'SPSC', 'BLMN', 'ALGT', 'UTSI', 'EVER', 'EMMA', 'HQY', 'CHCO', 'GLAD', 'RGLD',
                               'MIDD',
                               'EQBK', 'PRGS', 'PCAR', 'OESX', 'JJSF', 'GOOD', 'TIPT', 'AREC', 'AIMT', 'ADI', 'FLIR', 'TGTX', 'TXN', 'BCOR', 'STRT', 'MBIN', 'SANM', 'CDAK',
                               'CLNE',
                               'AUB', 'INWK', 'MGEE', 'PCB', 'CRTX', 'VRTS', 'SRCL', 'AMBA', 'CYCC', 'CAPR', 'NODK', 'CVET', 'ELSE', 'PSNL', 'CARV', 'DOCU', 'FREQ', 'NOVT',
                               'FSRV',
                               'LFVN', 'OPI', 'INSM', 'SONA', 'APDN', 'WMGI', 'QRVO', 'SGRY', 'GRBK', 'BOWX', 'FNCB', 'FXNC', 'FFWM', 'CRDF', 'MMAC', 'QLYS', 'HEC', 'ADTN',
                               'AVNW',
                               'BRKS', 'FIVN', 'GIII', 'PAYX', 'BDTX', 'OBLN', 'SPRT', 'SRRA', 'CEMI', 'MLHR', 'LIVN', 'ALIM', 'CLCT', 'VCEL', 'ISEE', 'PULM', 'AEMD', 'GTLS',
                               'ETSY',
                               'DXYN', 'PANL', 'ZIXI', 'FPRX', 'PRPH', 'CORE', 'OSUR', 'CALA', 'MBCN', 'PRSC', 'KRNY', 'NAKD', 'AAWW', 'BLPH', 'ALKS', 'FOUR', 'RGNX', 'PCTY',
                               'GENC',
                               'MOTS', 'HAYN', 'TGEN', 'ADP', 'ZS', 'NTLA', 'AESE', 'FISI', 'RUBY', 'AXTI', 'LUNG', 'BBBY', 'FOCS', 'OMER', 'DBX', 'FBMS', 'JCOM', 'SCWX',
                               'ADVM', 'ABMD',
                               'HEAR', 'TROW', 'CD', 'WTRE', 'CVGI', 'SEIC', 'IHRT', 'CBAT', 'WSFS', 'BIIB', 'FREE', 'ATHX', 'NIU', 'PRPL', 'ARNA', 'CIGI', 'GBIO', 'MSFT',
                               'SLAB',
                               'FEIM', 'EZPW', 'CLVS', 'DNLI', 'KE', 'DLHC', 'TNXP', 'SUNW', 'CNXN', 'STOK', 'OPRA', 'AFIN', 'WLDN', 'USAT', 'BWB', 'FCCY', 'RRR', 'LCA',
                               'WIX', 'OMCL',
                               'NEPT', 'EPAY', 'RMR', 'AGYS', 'IMGN', 'COWN', 'ESSA', 'CARG', 'CVAC', 'SRTS', 'LEDS', 'NNBR', 'SMIT', 'CMCSA', 'CNCE', 'QFIN', 'XBIT', 'ATLC',
                               'STSA',
                               'ASML', 'PRTS', 'TXRH', 'ZYNE', 'PPSI', 'ACHV', 'ASND', 'IDYA', 'MPWR', 'VFF', 'MTSI', 'VERO', 'SABR', 'PEIX', 'STKL', 'SPFI', 'OTEX', 'AVT',
                               'VREX',
                               'PROV', 'FDUS', 'PAHC', 'IIN', 'SONO', 'CPST', 'CVV', 'FFIC', 'GRPN', 'MIND', 'RELL', 'EVOP', 'VRNT', 'LSTR', 'EIGI', 'KEQU', 'QRHC', 'CLLS',
                               'EGOV',
                               'TTMI', 'NTGR', 'RBBN', 'ITRM', 'AYRO', 'TLGT', 'AVCO', 'NVAX', 'VICR', 'FAT', 'NMCI', 'CMLF', 'XNCR', 'CVCY', 'PFPT', 'SND', 'AMEH', 'TESS',
                               'SG', 'TAST',
                               'ETTX', 'RRBI', 'CODX', 'CLFD', 'VERU', 'TSCO', 'MSTR', 'CNOB', 'REKR', 'PINC', 'BIGC', 'ZBRA', 'DMTK', 'FTEK', 'SFT', 'IVAC', 'DHIL', 'KURA',
                               'QK',
                               'FIVE', 'ICUI', 'KTCC', 'PPBI', 'TELL', 'BANF', 'IRBT', 'CPSH', 'APLT', 'NRC', 'OSPN', 'FGBI', 'CSCO', 'OKTA', 'QNTA', 'NTEC', 'STXS', 'MEET',
                               'TAYD',
                               'CATM', 'JYNT', 'FBIO', 'SMTC', 'PETQ', 'VTSI', 'XLNX', 'IMVT', 'FUSE', 'XRAY', 'BLIN', 'IOVA', 'PENN', 'DYNT', 'KLIC', 'BBGI', 'NXST', 'SLRC',
                               'CTSH',
                               'ARAY', 'VKTX', 'AVO', 'NSIT', 'SRPT', 'LMAT', 'ATRI', 'LOOP', 'HA', 'BEAT', 'SSNC', 'HGSH', 'MGTA', 'YMAB', 'CG', 'ISBC', 'CDK', 'ABEO',
                               'HEES', 'EXTR',
                               'ARWR', 'LKQ', 'CDNS', 'HTLD', 'NKLA', 'ONEM', 'ESTA', 'FRPT', 'BNGO', 'STFC', 'MDXG', 'SGMA', 'CALB', 'CONN', 'ZAGG', 'PFSW', 'PEP', 'DSKE',
                               'FFIV',
                               'GLYC', 'MCRI', 'TUSK', 'BKCC', 'CASS', 'SIBN', 'DXPE', 'DCTH', 'VRSN', 'TYME', 'HALO', 'EBIX', 'AMRK', 'NMRK', 'DOYU', 'DXCM', 'AMRH', 'IRTC',
                               'VIR',
                               'AROW', 'EAR', 'ATOS', 'GLDD', 'IOSP', 'PLXS', 'IROQ', 'SAL', 'HOFV', 'WDC', 'NGM', 'TSC', 'ALGS', 'APVO', 'GSHD', 'RIVE', 'BPMC', 'CTMX',
                               'VRNS', 'NICK',
                               'CWBC', 'AXDX', 'QCOM', 'WWD', 'PATK', 'KYMR', 'PRVB', 'NAII', 'FEYE', 'CPRX', 'CUTR', 'VYNE', 'SBUX', 'PEBO', 'RELV', 'TXG', 'OPGN', 'KTRA',
                               'XEL',
                               'SONM', 'SVC', 'RVSB', 'SREV', 'KFRC', 'OPTT', 'TER', 'FBNC', 'BCEL', 'CLGN', 'ARCC', 'MBIO', 'NEXT', 'PLUG', 'ACMR', 'GRTX', 'KROS', 'PGNY',
                               'EMKR',
                               'CFBI', 'JCS', 'STTK', 'UIHC', 'FOXA', 'EQ', 'SEER', 'QRTEA', 'MARK', 'SVVC', 'RNET', 'QUIK', 'GNTX', 'ILPT', 'VC', 'KRBP', 'RESN', 'EVOL',
                               'NNDM', 'FIXX',
                               'NTRS', 'ETFC', 'IVA', 'AMD', 'PPC', 'SRCE', 'PECK', 'HIBB', 'VIVE', 'AVGR', 'XSPA', 'CREX', 'IMMU', 'LHCG', 'MXIM', 'AUTO', 'PCVX', 'SNFCA',
                               'BLKB',
                               'BOXL', 'VBTX', 'CETX', 'BDGE', 'LJPC', 'MRSN', 'DXLG', 'BHF', 'JVA', 'SFM', 'CLRO', 'RAVN', 'NEPH', 'HUIZ', 'BSRR', 'XFOR', 'IDXG', 'TRMB',
                               'LKFN',
                               'CSCW', 'ABIO', 'SP', 'TLRY', 'ADTX', 'NTRP', 'ADUS', 'AWRE', 'DENN', 'MRTN', 'HFWA', 'EGRX', 'VCNX', 'WSBC', 'PSTL', 'PYPL', 'ENTG', 'GERN',
                               'ALRM',
                               'RNWK', 'BCOV', 'LYTS', 'GSKY', 'STCN', 'OPRX', 'FRGI', 'WYNN', 'EBAY', 'MNRO', 'FLMN', 'ACGL', 'ASPS', 'OPBK', 'FRG', 'CSOD', 'MYOK', 'HTLF',
                               'MRIN',
                               'CME', 'GWGH', 'VRSK', 'GXGX', 'ASRT', 'HOL', 'STAY', 'DORM', 'MCFT', 'VSTM', 'CTRE', 'APPN', 'SILK', 'VIVO', 'JD', 'YRCW', 'NTAP', 'TVTY',
                               'UMPQ', 'GLUU',
                               'CHTR', 'GOVX', 'WMG', 'CBIO', 'CONE', 'FULT', 'IPDN', 'DMLP', 'VIRT', 'SIVB', 'CBLI', 'MICT', 'PRTK', 'WLTW', 'LBRDA', 'CATB', 'BFC', 'DISH',
                               'FARO',
                               'ISRG', 'IDEX', 'MNCL', 'MNOV', 'AQST', 'SSP', 'PTE', 'BLU', 'BOTJ', 'SAMG', 'OPK', 'ASFI', 'ECPG', 'ABCB', 'LTBR', 'ICCC', 'EXLS', 'PBPB',
                               'VIAV', 'ATRS',
                               'PIH', 'NBIX', 'IPWR', 'INBK', 'CEVA', 'IDCC', 'LTRPA', 'HRZN', 'EWBC', 'UMBF', 'MCHP', 'BPFH', 'CCXI', 'GT', 'MESO', 'VERB', 'DJCO', 'NARI',
                               'ANIX',
                               'SNSS', 'FATE', 'SPRO', 'JKHY', 'PICO', 'ARAV', 'APTX', 'IMMR', 'RETA', 'AMTB', 'SBSI', 'HOPE', 'GRIL', 'HBIO', 'SHBI', 'ASRV', 'ANCN', 'IEP',
                               'INFI',
                               'DAKT', 'DARE', 'LPLA', 'AMSF', 'LBC', 'ALLK', 'IIVI', 'CFBK', 'UPWK', 'TRS', 'COOP', 'HDSN', 'CODA', 'CWBR', 'KRON', 'MDRX', 'NVCR', 'RSSS',
                               'QADB',
                               'AGRX', 'ASYS', 'WHLM', 'SAFM', 'TTGT', 'IMAC', 'NVUS', 'MOFG', 'DRRX', 'TTOO', 'OTRK', 'IMRA', 'SMMT', 'VXRT', 'SYNA', 'APEX', 'ATNX', 'EYES',
                               'HVBC',
                               'NCSM', 'PIXY', 'ATCX', 'FELE', 'DRNA', 'FIBK', 'PDD', 'FLDM', 'OLED', 'CBFV', 'TWST', 'BRY', 'TILE', 'ALSK', 'CTRN', 'CIIC', 'CCOI', 'BKYI',
                               'DTIL',
                               'ROLL', 'BECN', 'AAPL', 'ODT', 'VTVT', 'PNRG', 'BIMI', 'FMNB', 'DKNG', 'WLFC', 'BMRN', 'PBHC', 'STAA', 'LUNA', 'AIMC', 'ULBI', 'BLBD', 'AQB',
                               'BBQ',
                               'CMCT', 'FLXS', 'OFLX', 'SBGI', 'EOLS', 'AXAS', 'DVAX', 'NUVA', 'CBAY', 'ARRY', 'LEVI', 'SOLY', 'ACAM', 'NCNA', 'BYFC', 'LOAC', 'UNAM', 'AGTC',
                               'PNFP',
                               'AKAM', 'CAKE', 'SCOR', 'VCTR', 'MGI', 'AMCX', 'FOSL', 'BVXV', 'SRAX', 'LI', 'CSSE', 'SURF', 'SFIX', 'SVMK', 'ECOR', 'FFNW', 'RP', 'ALEC',
                               'TLND', 'IIIV',
                               'ACET', 'HLIT', 'VG', 'FAST', 'SWAV', 'ENG', 'BCBP', 'BSY', 'DRIO', 'HLIO', 'CWST', 'PLMR', 'TENX', 'TPIC', 'AGNC', 'CCRN', 'IBKR', 'TWIN',
                               'INDB', 'URGN',
                               'TCDA', 'CDMO', 'SSYS', 'WKHS', 'HURC', 'AAON', 'LASR', 'MITK', 'ESXB', 'LOVE', 'IRMD', 'JAKK', 'OBCI', 'NUAN', 'NATH', 'SCKT', 'HZNP', 'PS',
                               'GLSI',
                               'ATRO', 'HONE', 'NDSN', 'ADMA', 'CFMS', 'EYE', 'PCYO', 'ADMP', 'PRAX', 'CAC', 'SMED', 'ONB', 'MORN', 'WTFC', 'AUPH', 'RUN', 'GLG', 'KOD',
                               'CERN', 'SGA',
                               'TRUP', 'TTNP', 'WEN', 'ICAD', 'EKSO', 'MTSC', 'CMBM', 'OMEX', 'SESN', 'GBDC', 'FWONA', 'MSVB', 'CHMA', 'NTWK', 'IRWD', 'VRM', 'MTRX', 'OCGN',
                               'MRVL',
                               'EBTC', 'VIAC', 'ORGS', 'MCEP', 'VTNR', 'LGIH', 'XGN', 'CJJD', 'LPSN', 'LMPX', 'PBCT', 'AMNB', 'STKS', 'KTOS', 'ATRA', 'AEY', 'RMBL', 'OFS',
                               'INSE',
                               'XENT', 'AMGN', 'TH', 'IBTX', 'MGEN', 'GRMN', 'CROX', 'THRM', 'EPZM', 'SIGI', 'APPS', 'CNBKA', 'BIOL', 'TTCF', 'CASH', 'SPLK', 'ALXO', 'GPRE',
                               'SLM',
                               'NWFL', 'NBAC', 'ONVO', 'DTEA', 'TECH', 'ROCK', 'WVVI', 'CLSK', 'CZNC', 'PEGA', 'SNDE', 'WNEB', 'MELI', 'UXIN', 'RPD', 'AZRX', 'NOVN', 'STRS',
                               'CFFI',
                               'NKTX', 'BAND', 'KVHI', 'PLYA', 'REGI', 'SYKE', 'ESGR', 'EQOS', 'OPTN', 'MRCC', 'SPTN', 'REYN', 'OTEL', 'APLS', 'NERV', 'KELYA', 'KNDI', 'CLSN',
                               'UCTT',
                               'COUP', 'IGAC', 'LNDC', 'ACER', 'CYAN', 'ALPN', 'CPRT', 'LACQ', 'ATVI', 'ACRS', 'ICHR', 'CRVS', 'TREE', 'JAGX', 'RDUS', 'ALRN', 'PSMT', 'AOUT',
                               'FRPH',
                               'MU', 'TPTX', 'KINS', 'IBOC', 'AINV', 'PTGX', 'COLB', 'GEVO', 'LILA', 'NBSE', 'CTHR', 'WINT', 'LOCO', 'AUTL', 'FORR', 'FOLD', 'ACTG', 'GECC',
                               'PMD',
                               'TITN', 'CRON', 'EXPE', 'GABC', 'GNPX', 'PTC', 'GHSI', 'LYRA', 'INTG', 'MBRX', 'HSTM', 'DMRC', 'SNOA', 'MNSB', 'CYBE', 'IBEX', 'SDC', 'RIOT',
                               'HURN',
                               'KIN', 'CARA', 'TNDM', 'HDS', 'CNSL', 'ETON', 'MTEM', 'USCR', 'RILY', 'CSX', 'SUNS', 'CYTK', 'CBTX', 'DGLY', 'XONE', 'CYRX', 'FNKO', 'GSBC',
                               'SMSI',
                               'BOOM', 'SNES', 'ARDX', 'BSET', 'CDLX', 'DNKN', 'CMTL', 'CGNX', 'AIHS', 'CRMT', 'LCNB', 'OPRT', 'IMUX', 'PAYS', 'ASUR', 'SYRS', 'WHLR', 'CASY',
                               'ESPR',
                               'OTIC', 'HWC', 'ANGI', 'BIOC', 'TIG', 'NNOX', 'MTBC', 'RDNT', 'IPGP', 'LQDA', 'FMAO', 'CDZI', 'HROW', 'BGCP', 'LEVL', 'VTGN', 'DWSN', 'TC',
                               'OFED', 'NVEC',
                               'INTU', 'FOXF', 'OTTR', 'XENE', 'HSIC', 'MRUS', 'RGEN', 'LSBK', 'ITRI', 'NDRA', 'SPT', 'PRPO', 'TZOO', 'CVBF', 'PCRX', 'FSLR', 'GILD', 'ACNB',
                               'CDEV',
                               'BREW', 'KERN', 'RXT', 'HWCC', 'AMED', 'SMPL', 'LANC', 'KRYS', 'NATI', 'COHU', 'NH', 'NTES', 'SELB', 'AWH', 'CZWI', 'LIVX', 'PACB', 'AFIB',
                               'SIEB', 'CPAH',
                               'IIIN', 'ESSC', 'AVDL', 'AMST', 'BYND', 'HBT', 'ADES', 'IMXI', 'ICFI', 'ORBC', 'INVE', 'SNEX', 'AMSWA', 'PAVM', 'VLDR', 'CLAR', 'PLSE', 'USIO',
                               'SANW',
                               'WIRE', 'AMZN', 'TRMK', 'ATAX', 'FCCO', 'GVP', 'THFF', 'BCML', 'NDAQ', 'GNLN', 'ARCT', 'LITE', 'RGLS', 'INFN', 'GLPI', 'LFUS', 'RADA', 'IGMS',
                               'HFFG',
                               'FLUX', 'GNCA', 'CTSO', 'ZIOP', 'IONS', 'ON', 'CSTL', 'ACHC', 'CPIX', 'NTIC', 'CLXT', 'SCHL', 'TBK', 'ANIP', 'FUTU', 'XPEL', 'CREG', 'NEOS',
                               'AVAV',
                               'XERS', 'SIEN', 'NTNX', 'STX', 'UFPT', 'RRGB', 'TACT', 'SYBT', 'STRA', 'EXPD', 'RCKT', 'MKTX', 'AXGN', 'TOTA', 'WORX', 'XPER', 'BPTH', 'ANSS',
                               'VNDA',
                               'PVAC', 'NBTB', 'BASI', 'ULTA', 'MGRC', 'CACC', 'MBUU', 'ONEW', 'TTWO', 'VNOM', 'KPTI', 'MNKD', 'OLLI', 'DLTH', 'CMPS', 'AMTX', 'AMRN', 'CMLS',
                               'AHCO',
                               'CENT', 'SEAC', 'ECOL', 'CRWD', 'EGBN', 'COCP', 'INAQ', 'BILI', 'SRRK', 'SAFT', 'SGBX', 'MRNA', 'TARA', 'BMCH', 'POAI', 'AAXN', 'ASO', 'NDLS',
                               'THCA',
                               'JAMF', 'AMRS', 'ADAP', 'CPTA', 'HWKN', 'DLTR', 'MAT', 'CLMT', 'FFBW', 'FBIZ', 'MBOT', 'LMRK', 'UEIC', 'AVID', 'ADMS', 'GNUS', 'OPES', 'TSRI',
                               'SPWH',
                               'RDI', 'KBNT', 'AKUS', 'SAVA', 'GFN', 'QTNT', 'ENDP', 'MSON', 'KOSS', 'BLDR', 'STEP', 'PEAK', 'HCAT', 'CHFS', 'URBN', 'ARTL', 'INO', 'CRUS',
                               'HAIN',
                               'SBFG', 'APOG', 'PRNB', 'SYNH', 'INPX', 'UTHR', 'TCON', 'BXRX', 'LCUT', 'USEG', 'AXNX', 'SCYX', 'BCRX', 'NYMT', 'TSLA', 'SUPN', 'ATRC', 'LMST',
                               'RAPT',
                               'RDFN', 'SASR', 'HSII', 'TRMD', 'SLNO', 'BBCP', 'LGND', 'PDCO', 'AIRT', 'CBAN', 'UG', 'UVSP', 'PRAA', 'GIFI', 'RVMD', 'PCSA', 'POWL', 'APEN',
                               'CATY',
                               'BANR', 'ORRF', 'ADOM', 'VRAY', 'ANDE', 'METX', 'MOXC', 'COLL', 'SMBC', 'FLGT', 'BFST', 'FTNT', 'PACW', 'SMCI', 'YNDX', 'CMPI', 'COGT', 'PDSB',
                               'HAS',
                               'CBRL', 'CLDX', 'ODP', 'TTEK', 'ACST', 'WASH', 'MOR', 'CDXS', 'MLVF', 'KLDO', 'DFFN', 'ACOR', 'CCBG', 'OYST', 'CMRX', 'EVFM', 'NFLX', 'SGMO',
                               'WEYS',
                               'EGAN', 'ZG', 'CRSA', 'ZM', 'CGIX', 'SAMA', 'SHYF', 'VSEC', 'ITCI', 'ONCT', 'DSPG', 'ATXI', 'PRLD', 'PZZA', 'APTO', 'HCSG', 'RGCO', 'LOGI',
                               'FITB', 'KRUS',
                               'BBIO', 'ARLP', 'ELOX', 'SLGN', 'ABTX', 'TRCH', 'HOMB', 'APXT', 'NURO', 'ORGO', 'MOMO', 'CHRW', 'BGFV', 'CNET', 'SWTX', 'VUZI', 'NKTR', 'SKYW',
                               'EDIT',
                               'FWRD', 'IBCP', 'INVA', 'AMPH', 'ABUS', 'LNSR', 'BEEM', 'PSEC', 'SNBR', 'UNIT', 'STIM', 'CCMP', 'DOMO', 'MTCH', 'IDXX', 'NTRA', 'UAL', 'MEDP',
                               'RNA',
                               'CRNX', 'MGPI', 'IESC', 'KXIN', 'ALTM', 'LECO', 'XOMA', 'FROG', 'FLL', 'BKSC', 'CECE', 'CRNC', 'UROV', 'PXLW', 'HCKT', 'PLAB', 'FULC', 'HGBL',
                               'CAR',
                               'OBNK', 'TRNS', 'YORW', 'AEGN', 'GEC', 'VRRM', 'FB', 'OTLK', 'GBT', 'ICPT', 'PRGX', 'PRCP', 'IDRA', 'RYTM', 'LOPE', 'PLAY', 'SLDB', 'GRIF',
                               'MANT', 'OSW',
                               'BLUE', 'CWCO', 'ALOT', 'KZR', 'HMNF', 'POOL', 'SUMO', 'FLNT', 'MORF', 'INSG', 'BBI', 'XELB', 'UONE', 'KLAC', 'BEAM', 'PLXP', 'POLA', 'FUV',
                               'MVBF',
                               'QUMU', 'AKTS', 'TWNK', 'MCRB', 'GFED', 'VBLT', 'AVCT', 'NHLD', 'TCBK', 'BGNE', 'PWOD', 'TENB', 'NSYS', 'EHTH', 'GHIV', 'AGLE', 'PBYI', 'CENX',
                               'DADA',
                               'NEOG', 'SMMC', 'SWKH', 'MSEX', 'TACO', 'RLMD', 'LORL', 'CRTD', 'TCPC', 'HSDT', 'VCYT', 'MIST', 'ISIG', 'AVXL', 'AAME', 'RICK', 'CORT', 'EQIX',
                               'ANY',
                               'KNSL', 'ADXS', 'SIRI', 'JAN', 'IPAR', 'NEO', 'UI', 'CZR', 'INMB', 'GEOS', 'HTBK', 'FLIC', 'GSIT', 'GPRO', 'INCY', 'MACK', 'CLSD', 'JBLU',
                               'ATEX', 'GDEN',
                               'SINT', 'PRAH', 'LULU', 'TSBK', 'VERI', 'OCFC', 'ASTE', 'EML', 'PSTX', 'CTBI', 'NTCT', 'SCPH', 'JAZZ', 'SQBG', 'RAIL', 'ALLO', 'MBWM', 'MRTX',
                               'YGYI',
                               'NVFY', 'DCPH', 'HMST', 'SSNT', 'GALT', 'IMBI', 'ATOM', 'LAMR', 'ACAD', 'HOLX', 'REGN', 'FCEL', 'ALTR', 'ZI', 'RAVE', 'SPPI', 'ISNS', 'TTEC',
                               'BL', 'CASI',
                               'TDAC', 'WBA', 'CUE', 'EGLE', 'SMBK', 'REPL', 'ALT', 'CALM', 'HALL', 'PSTI', 'CTIB', 'DMAC', 'HTBX', 'PASG', 'MGTX', 'PRIM', 'PTON', 'HELE',
                               'EFSC',
                               'FANG', 'OSIS', 'KBAL', 'DGICA', 'ACIW', 'AGBA', 'ATEC', 'UNTY', 'SFBS', 'CATC', 'CIDM', 'LPRO', 'NXPI', 'CRAI', 'ESCA', 'LOB', 'BLI', 'WINA',
                               'ZSAN',
                               'RKDA', 'SVRA', 'AMOT', 'RIDE', 'TMDX', 'MYRG', 'XAIR', 'HWBK', 'PFMT', 'WING', 'CHUY', 'ARDS', 'GCBC', 'MLAB', 'CPSS', 'NBEV', 'LPCN', 'ALAC',
                               'HSON',
                               'SSB', 'ASTC', 'PIRS', 'AACQ', 'AGIO', 'PHUN', 'VYGR', 'SNGX', '<unknown>', 'CRSR', 'MWK', 'PGEN', 'LAZY', 'UNFI', 'RDVT', 'RAMP', 'KOPN',
                               'LPTX', 'RIGL',
                               'PRVL', 'ISTR', 'IZEA', 'AMTD', 'FVE', 'LSCC', 'NFE', 'LIVE', 'LMB', 'MRBK', 'SBRA', 'TRHC', 'BRKL', 'MEIP', 'MASI', 'SYNC', 'EIDX', 'LNTH',
                               'AYTU',
                               'STRO', 'ADIL', 'XBIO', 'BBSI', 'MOHO', 'BNTC', 'TXMD', 'FIZZ', 'PME', 'CMPR', 'TPCO', 'CNST', 'BHTG', 'WW', 'ROIC', 'PRFT', 'NXTC', 'CERS',
                               'WRLD',
                               'UEPS', 'AMAG', 'AEIS', 'MDLZ', 'AMBC', 'CETV', 'WSBF', 'CSTR', 'EDUC', 'NETE', 'KRTX', 'EDSA', 'WETF', 'CTAS', 'NAVI', 'ACBI', 'CREE', 'OEG',
                               'ALNA',
                               'PHIO', 'ZION', 'FSBW', 'EYPT', 'UFPI', 'GWRS', 'ODFL', 'CNDT', 'STLD', 'REFR', 'CRSP', 'QURE', 'VRA', 'JNCE', 'LBAI', 'SLGG', 'TW', 'FRME',
                               'BRPA',
                               'OXFD', 'QNST', 'SHEN', 'ATNI', 'ARCB', 'JOUT', 'BNFT', 'RCM', 'KIDS', 'LINK', 'BZUN', 'PRTH', 'RMCF', 'PFBI', 'DZSI', 'SSKN', 'SALM', 'STBA',
                               'TCX',
                               'HYAC', 'PEBK', 'NWSA', 'CTXS', 'CYCN', 'SCON', 'PTI', 'UHAL', 'NSEC', 'ALXN', 'ONCS', 'TURN', 'MMSI', 'CRIS', 'ICMB', 'LMNR', 'SAGE', 'WTER',
                               'CHRS',
                               'LVGO', 'TA', 'HBCP', 'REG', 'OPT', 'STRM', 'TWOU', 'MVIS', 'AKBA', 'REAL', 'SSTI', 'AMSC', 'MTLS', 'MANH', 'REPH', 'COLM', 'FONR', 'ANAB',
                               'FRBK', 'SLP',
                               'GWPH', 'IAC', 'BSGM', 'HRTX', 'ENPH', 'OCUL', 'BCPC', 'DGII', 'MPAA', 'XLRN', 'VALU', 'PCH', 'PBIP', 'WSTL', 'KIRK', 'PFIE', 'VERY', 'AVRO',
                               'AEYE',
                               'BLFS', 'SLCT', 'BKNG', 'FENC', 'DLPN', 'CSPI', 'SPWR', 'TCRR', 'TMUS', 'AKRO', 'IRDM', 'MIME', 'GLRE', 'NMRD', 'MERC', 'PRMW', 'CRWS', 'PPD',
                               'ALBO',
                               'LOAN', 'CDNA', 'AIRG', 'LOGC', 'GTIM', 'OSS', 'BOKF', 'CSII', 'SFNC', 'MRNS', 'STRL', 'BSQR', 'PDCE', 'RUTH', 'DIOD', 'QTRX', 'GOGO', 'HYRE',
                               'ICBK',
                               'FVAM', 'SYBX', 'PNBK', 'CNMD', 'ALNY', 'AMKR', 'LAUR', 'OPHC', 'WDFC', 'MOBL', 'SNCA', 'CGBD', 'ADPT', 'CBSH', 'KALU', 'UBX', 'EVOK', 'OSTK',
                               'NMTR',
                               'CBMG', 'ALTA', 'RBB', 'SSSS', 'TFSL', 'LMFA', 'USAU', 'KALA', 'MPB', 'DAIO', 'LRCX', 'CINF', 'MESA', 'PI', 'EVK', 'MKSI', 'SDGR', 'NEON',
                               'ALDX', 'MDCA',
                               'CCB', 'LYFT', 'MRAM', 'MIK', 'HEPA', 'BRQS', 'AOSL', 'IEA', 'MNST', 'CPHC', 'CRBP', 'QTT', 'GBCI', 'ENSG', 'AMAT', 'ADBE', 'CLIR', 'HARP',
                               'QCRH', 'SLS',
                               'EVLO', 'RBKB', 'YTEN', 'FISV', 'USAK', 'KALV', 'VRTX', 'COHR', 'LFAC', 'CBMB', 'FTFT', 'MNTA', 'RUSHA', 'WIFI', 'CPSI', 'PSTV', 'MFIN', 'ICON',
                               'SGLB',
                               'JACK', 'TRVN', 'GAIA', 'CDW', 'RPRX', 'ONTX', 'FSTR', 'NCBS', 'WISA', 'RBCAA', 'NWBI', 'AVGO', 'PDLI', 'SWKS', 'STAF', 'RARE', 'BCLI', 'VRTU',
                               'RMTI',
                               'LBTYA', 'BLNK', 'AMCI', 'NXGN', 'MRCY', 'MARA', 'RMBS', 'APPF', 'ENTA', 'HUBG', 'FLWS', 'ACRX', 'JRVR', 'AMWD', 'INOV', 'MATW', 'MGNX', 'HOOK',
                               'FHB',
                               'RNST', 'BCDA', 'ACIA', 'SYNL', 'TCBI', 'VOXX', 'ASMB', 'VRME', 'NESR', 'CELC', 'PHAT', 'CTXR', 'TRMT', 'INTC', 'HOTH', 'WATT', 'CHNG', 'RAND',
                               'RFIL',
                               'WDAY', 'CAMP', 'CLRB', 'HMSY', 'PTMN', 'LLNW', 'PLUS', 'NK', 'SRNE', 'EEFT', 'FLEX', 'TCF', 'EFOI', 'KNSA', 'FFIN', 'MBII', 'TBLT', 'OM',
                               'TYHT', 'LE',
                               'ICCH', 'KRMD', 'COMM', 'GP', 'SPOK', 'TCFC', 'TNAV', 'GLIBA', 'GOOGL', 'TBBK', 'ILMN', 'THMO', 'ANAT', 'ALJJ', 'EXC', 'JBHT', 'MNPR', 'FGEN',
                               'CRVL',
                               'NEWT', 'FORM', 'CASA', 'LUMO', 'METC', 'CDXC', 'PCSB', 'FUNC', 'BTAI', 'LIVK', 'HLNE', 'AZPN', 'BSVN', 'GO', 'CHCI', 'TTD', 'MCHX', 'TAIT',
                               'PTCT',
                               'PKBK', 'API', 'ECHO', 'ZGYH', 'BWEN', 'KRNT', 'RVNC', 'GROW', 'CGRO', 'GOSS', 'DCOM', 'SIGA', 'PFLT', 'AGEN', 'GH', 'GRNQ', 'GNMK', 'VSAT',
                               'TEAM',
                               'SLRX', 'SGMS', 'HTGM', 'HCCH', 'ADSK', 'FRTA', 'ZUMZ', 'SBBP', 'BIDU', 'LWAY', 'AGFS', 'ZGNX', 'EXEL', 'EVBG', 'LAKE', 'SUMR', 'CRTO', 'SINO',
                               'INGN',
                               'POWI', 'HAFC', 'SMTX', 'CCNE', 'HBAN', 'SEEL', 'GTHX', 'GTEC', 'ERII', 'CHDN', 'ROST', 'CFB', 'ACCD', 'SOHO', 'VBIV', 'PLCE', 'OCC', 'ROKU',
                               'FPAY',
                               'CTIC', 'CKPT', 'GDS', 'HCAC', 'SGH', 'FARM', 'BMRA', 'BRP', 'HBP', 'SAIA', 'DISCA', 'QELL', 'TRIP', 'NMIH', 'CSWC', 'TCMD', 'APEI', 'XELA',
                               'PAAS',
                               'SHSP', 'RPAY', 'ACLS', 'RMG', 'SELF', 'GOCO', 'MHLD', 'CVLT', 'TBIO', 'SPKE', 'NSTG', 'RMNI', 'ZYXI', 'UPLD', 'MGLN', 'WWR', 'AKU', 'CCLP',
                               'FCNCA',
                               'CLDB', 'LMNX', 'NVIV', 'MTEX', 'WTRH', 'NTUS', 'EA', 'PFG', 'CHEF', 'MRKR', 'MDB', 'HNNA', 'LAND', 'FBRX', 'ANIK', 'DRAD', 'PNNT', 'VIRC',
                               'SBAC', 'LEGH',
                               'GRWG', 'CLBS', 'PODD', 'PMBC', 'CYBR', 'PETS', 'QDEL', 'MYGN', 'SGEN', 'MAR', 'WVE', 'XP', 'LPTH', 'CVGW', 'EXPO', 'NVDA', 'BRKR', 'COST',
                               'RCMT', 'GDYN',
                               'OMP', 'STNE', 'EXAS', 'EXPI', 'SCPL', 'KHC', 'OSBC', 'ZNGA', 'MMLP', 'SNDX', 'VISL', 'SYPR', 'BCTF', 'FTDR', 'FSDC', 'BMRC', 'SRDX', 'UTMD',
                               'MCBC',
                               'BPOP', 'CLBK', 'AAL', 'AKER', 'SHOO', 'HCCI', 'ASPU', 'FNWB', 'UFCS', 'SNPS', 'IMKTA', 'OVID', 'CSGS', 'NBRV', 'AERI', 'WERN', 'CFFN']}

    from_saved = {'table': ['SEP', 'SF1', 'SF3B', '<unknown>', 'SFP'],
                  'category': ['Domestic Preferred Stock', 'ADR Preferred Stock', 'ETF', 'ETN', 'Canadian Preferred Stock', 'Domestic Stock Warrant', 'Canadian Common Stock',
                               'ADR Common Stock Primary Class', 'Canadian Common Stock Primary Class', 'ETD', 'CEF', 'ADR Common Stock', 'IDX',
                               'Domestic Common Stock Primary Class', '<unknown>', 'Canadian Stock Warrant', 'ADR Stock Warrant', 'Domestic Common Stock Secondary Class',
                               'Domestic Common Stock', 'ADR Common Stock Secondary Class'],
                  'sicsector': ['Agriculture Forestry And Fishing', 'Transportation Communications Electric Gas And Sanitary Service', 'Finance Insurance And Real Estate',
                                '<unknown>', 'Public Administration', 'Services', 'Nonclassifiable', 'Construction', 'Wholesale Trade', 'Retail Trade', 'Manufacturing',
                                'Mining'],
                  'famaindustry': ['Textiles', 'Healthcare', 'Restaraunts Hotels Motels', 'Candy & Soda', 'Pharmaceutical Products', 'Medical Equipment', 'Consumer Goods',
                                   'Machinery', 'Shipping Containers', 'Petroleum and Natural Gas', 'Measuring and Control Equipment', 'Apparel', 'Coal',
                                   'Construction Materials', 'Electronic Equipment', 'Tobacco Products', 'Printing and Publishing', 'Personal Services', 'Chemicals',
                                   'Insurance', 'Retail', 'Electrical Equipment', 'Automobiles and Trucks', 'Wholesale', 'Almost Nothing', 'Banking', 'Food Products',
                                   'Transportation', '<unknown>', 'Trading', 'Construction', 'Business Supplies', 'Steel Works Etc', 'Defense', 'Utilities', 'Recreation',
                                   'Rubber and Plastic Products', 'Non-Metallic and Industrial Metal Mining', 'Shipbuilding Railroad Equipment', 'Real Estate',
                                   'Communication', 'Agriculture', 'Beer & Liquor', 'Precious Metals', 'Entertainment', 'Fabricated Products', 'Computers',
                                   'Business Services', 'Aircraft'],
                  'sector': ['Basic Materials', 'Technology', 'Healthcare', 'Energy', 'Real Estate', 'Communication Services', '<unknown>', 'Utilities', 'Consumer Cyclical',
                             'Consumer Defensive', 'Financial Services', 'Industrials'],
                  'industry': ['Publishing', 'Industrial Metals & Minerals', 'Medical Care Facilities', 'Shipping & Ports', 'Utilities - Renewable', 'Packaged Foods',
                               'Integrated Freight & Logistics', 'Other Industrial Metals & Mining', 'Metal Fabrication', 'Luxury Goods', 'Aluminum', 'Trucking',
                               'Medical Distribution', 'Medical Instruments & Supplies', 'Semiconductor Equipment & Materials', 'Pollution & Treatment Controls',
                               'Long-Term Care Facilities', 'Electrical Equipment & Parts', 'Furnishings', 'Media - Diversified', 'Footwear & Accessories',
                               'Textile Manufacturing', 'Recreational Vehicles', 'Insurance - Reinsurance', 'Silver', 'Computer Hardware', 'Aerospace & Defense',
                               'Telecom Services', 'Grocery Stores', 'Security & Protection Services', 'Insurance - Life', 'Lumber & Wood Production', 'Semiconductors',
                               'Specialty Chemicals', 'Software - Application', 'Healthcare Plans', 'Beverages - Brewers', 'Specialty Retail', 'Farm Products',
                               'Internet Content & Information', 'Farm & Construction Equipment', 'Business Equipment', 'Specialty Industrial Machinery',
                               'Oil & Gas Refining & Marketing', 'Banks - Regional - US', 'Diversified Industrials', 'REIT - Healthcare Facilities', 'Financial Conglomerates',
                               'Other Precious Metals & Mining', 'Banks - Regional', 'Utilities - Regulated Gas', 'Utilities - Independent Power Producers', 'Banks - Global',
                               'Financial Data & Stock Exchanges', 'Specialty Finance', 'Tobacco', 'Pharmaceutical Retailers', 'Building Products & Equipment',
                               'Real Estate - Development', 'Utilities - Regulated Water', 'Software - Infrastructure', 'Shell Companies', 'Tools & Accessories',
                               'Apparel Manufacturing', 'Insurance - Property & Casualty', 'Diagnostics & Research', 'Apparel Retail', '<unknown>', 'Capital Markets',
                               'Auto & Truck Dealerships', 'Discount Stores', 'Solar', 'REIT - Mortgage', 'REIT - Diversified', 'Staffing & Outsourcing Services',
                               'REIT - Industrial', 'Asset Management', 'Travel Services', 'Semiconductor Memory', 'Advertising Agencies', 'Airports & Air Services',
                               'Savings & Cooperative Banks', 'Medical Devices', 'Internet Retail', 'Leisure', 'Oil & Gas Integrated', 'Agricultural Inputs',
                               'Information Technology Services', 'Broadcasting - Radio', 'Auto Manufacturers', 'Biotechnology', 'Apparel Stores', 'Personal Services',
                               'Chemicals', 'Electronic Gaming & Multimedia', 'Beverages - Soft Drinks', 'Business Equipment & Supplies', 'Packaging & Containers', 'Lodging',
                               'REIT - Retail', 'Real Estate Services', 'Insurance Brokers', 'Broadcasting - TV', 'Oil & Gas Midstream', 'Scientific & Technical Instruments',
                               'Drug Manufacturers - Specialty & Generic', 'Insurance - Diversified', 'Medical Care', 'Drug Manufacturers - Major', 'Confectioners',
                               'Marine Shipping', 'Farm & Heavy Construction Machinery', 'Residential Construction', 'Staffing & Employment Services', 'Steel',
                               'Electronics & Computer Distribution', 'Railroads', 'Electronic Components', 'Paper & Paper Products', 'Entertainment', 'Conglomerates',
                               'Home Furnishings & Fixtures', 'Furnishings Fixtures & Appliances', 'Business Services', 'Department Stores', 'Engineering & Construction',
                               'Infrastructure Operations', 'Airlines', 'Food Distribution', 'Oil & Gas E&P', 'Mortgage Finance', 'Resorts & Casinos',
                               'Real Estate - Diversified', 'Copper', 'REIT - Hotel & Motel', 'Coking Coal', 'Thermal Coal', 'Rental & Leasing Services',
                               'Consumer Electronics', 'REIT - Specialty', 'Credit Services', 'Coal', 'Consulting Services', 'Waste Management', 'Beverages - Non-Alcoholic',
                               'REIT - Residential', 'REIT - Office', 'Building Materials', 'Utilities - Regulated Electric', 'Beverages - Wineries & Distilleries',
                               'Computer Systems', 'Home Improvement Stores', 'Oil & Gas Drilling', 'Home Improvement Retail', 'Uranium', 'Household & Personal Products',
                               'Gold', 'Health Care Plans', 'Data Storage', 'Auto Parts', 'Health Information Services', 'Financial Exchanges', 'Utilities - Diversified',
                               'Real Estate - General', 'Education & Training Services', 'Specialty Business Services', 'Oil & Gas Equipment & Services',
                               'Drug Manufacturers - General', 'Insurance - Specialty', 'Broadcasting', 'Industrial Distribution', 'Communication Equipment', 'Restaurants',
                               'Gambling', 'Banks - Diversified'], 'scalemarketcap': ['2 - Micro', '1 - Nano', '<unknown>', '5 - Large', '6 - Mega', '4 - Mid', '3 - Small'],
                  'scalerevenue': ['2 - Micro', '1 - Nano', '<unknown>', '5 - Large', '6 - Mega', '4 - Mid', '3 - Small'],
                  'currency': ['EUR', 'COP', 'MXN', 'SEK', 'PEN', 'JPY', 'AUD', 'ILS', 'HKD', 'BRL', 'TRY', 'CHF', 'KRW', 'INR', 'CLP', '<unknown>', 'DKK', 'PLN', 'IDR',
                               'NZD', 'CNY', 'ARS', 'VEF', 'NOK', 'USD', 'GBP', 'ZAR', 'MYR', 'RUB', 'PHP', 'TWD', 'CAD'],
                  'location': ['Colombia', 'North Carolina; U.S.A', "Democratic People'S Republic Of Korea", 'Ohio; U.S.A', 'United Kingdom', 'South Dakota; U.S.A',
                               'Marshall Islands', 'Singapore', 'Rhode Island; U.S.A', 'Brazil', 'Maldives', 'Arizona; U.S.A', 'Belgium', 'Turkey', 'Republic Of Korea',
                               'Oklahoma; U.S.A', 'United Republic Of Tanzania', 'Michigan; U.S.A', 'Jersey', 'California; U.S.A', 'South Carolina; U.S.A', 'Ghana',
                               'Alaska; U.S.A', 'West Virginia; U.S.A', 'Ireland', 'Nova Scotia; Canada', 'Colorado; U.S.A', 'Peru', 'Manitoba; Canada', 'Maryland; U.S.A',
                               'Illinois; U.S.A', 'Hong Kong', 'Venezuela', 'Guernsey', 'Chile', 'Philippines', 'India', 'Italy', 'Israel', 'Switzerland', 'Maine; U.S.A',
                               'Greece', 'Mississippi; U.S.A', 'Puerto Rico', 'Kentucky; U.S.A', 'Finland', 'Panama', 'China', 'District Of Columbia; U.S.A',
                               'United States; U.S.A', 'Alabama; U.S.A', 'Georgia U.S.A.', 'Czech Republic', 'New Jersey; U.S.A', 'Japan', 'Guam', 'United Arab Emirates',
                               'Sweden', 'Indiana; U.S.A', 'Bermuda', '<unknown>', 'Macau', 'British Columbia; Canada', 'Delaware; U.S.A', 'Louisiana; U.S.A', 'Argentina',
                               'Norway', 'Hawaii; U.S.A', 'New Hampshire; U.S.A', 'Massachusetts; U.S.A', 'North Dakota; U.S.A', 'Oregon; U.S.A', 'Virgin Islands; U.S.',
                               'Montana; U.S.A', 'Monaco', 'Netherlands Antilles', 'Hungary', 'Iowa; U.S.A', 'Denmark', 'Jordan', 'Texas; U.S.A', 'Idaho; U.S.A',
                               'Cayman Islands', 'Netherlands', 'Missouri; U.S.A', 'Cyprus', 'Vermont; U.S.A', 'Gibraltar', 'South Africa', 'Canada (Federal Level)',
                               'Pennsylvania; U.S.A', 'Thailand', 'Florida; U.S.A', 'New York; U.S.A', 'Nebraska; U.S.A', 'Tennessee; U.S.A', 'Luxembourg', 'Poland',
                               'Saint Vincent And The Grenadines', 'Washington; U.S.A', 'Russian Federation', 'British Virgin Islands', 'Taiwan', 'Costa Rica', 'Australia',
                               'Alberta; Canada', 'Germany', 'Israel-Syria', 'Arkansas; U.S.A', 'Quebec; Canada', 'Bahamas', 'France', 'Israel-Jordan', 'Austria', 'Malaysia',
                               'Iceland', 'Virginia; U.S.A', 'Saskatchewan; Canada', 'Newfoundland; Canada', 'Georgia; U.S.A', 'Wyoming; U.S.A', 'Connecticut; U.S.A',
                               'Indonesia', 'Utah; U.S.A', 'Uruguay', 'Canada', 'New Zealand', 'Unknown', 'Wisconsin; U.S.A', 'New Brunswick; Canada', 'Oman',
                               'Ontario; Canada', 'Saudi Arabia', 'Minnesota; U.S.A', 'Kansas; U.S.A', 'Nevada; U.S.A', 'Isle Of Man', 'Malta', 'Mauritius', 'Mexico',
                               'New Mexico; U.S.A', 'Spain'],
                  'f22_ticker': ['MIRM', 'SEDG', 'LQDT', 'ALGN', 'ARVN', 'HJLI', 'CELH', 'LIND', 'SPSC', 'BLMN', 'ALGT', 'UTSI', 'EVER', 'EMMA', 'HQY', 'CHCO', 'GLAD', 'RGLD',
                                 'MIDD', 'EQBK', 'PRGS', 'PCAR', 'OESX', 'JJSF', 'GOOD', 'TIPT', 'AREC', 'AIMT', 'ADI', 'FLIR', 'TGTX', 'TXN', 'BCOR', 'STRT', 'MBIN', 'SANM',
                                 'CDAK', 'CLNE', 'AUB', 'INWK', 'MGEE', 'PCB', 'CRTX', 'VRTS', 'SRCL', 'AMBA', 'CYCC', 'CAPR', 'NODK', 'CVET', 'ELSE', 'PSNL', 'CARV', 'DOCU',
                                 'FREQ', 'NOVT', 'FSRV', 'LFVN', 'OPI', 'INSM', 'SONA', 'APDN', 'WMGI', 'QRVO', 'SGRY', 'GRBK', 'BOWX', 'FNCB', 'FXNC', 'FFWM', 'CRDF', 'MMAC',
                                 'QLYS', 'HEC', 'ADTN', 'AVNW', 'BRKS', 'FIVN', 'GIII', 'PAYX', 'BDTX', 'OBLN', 'SPRT', 'SRRA', 'CEMI', 'MLHR', 'LIVN', 'ALIM', 'CLCT', 'VCEL',
                                 'ISEE', 'PULM', 'AEMD', 'GTLS', 'ETSY', 'DXYN', 'PANL', 'ZIXI', 'FPRX', 'PRPH', 'CORE', 'OSUR', 'CALA', 'MBCN', 'PRSC', 'KRNY', 'NAKD',
                                 'AAWW', 'BLPH', 'ALKS', 'FOUR', 'RGNX', 'PCTY', 'GENC', 'MOTS', 'HAYN', 'TGEN', 'ADP', 'ZS', 'NTLA', 'AESE', 'FISI', 'RUBY', 'AXTI', 'LUNG',
                                 'BBBY', 'FOCS', 'OMER', 'DBX', 'FBMS', 'JCOM', 'SCWX', 'ADVM', 'ABMD', 'HEAR', 'TROW', 'CD', 'WTRE', 'CVGI', 'SEIC', 'IHRT', 'CBAT', 'WSFS',
                                 'BIIB', 'FREE', 'ATHX', 'NIU', 'PRPL', 'ARNA', 'CIGI', 'GBIO', 'MSFT', 'SLAB', 'FEIM', 'EZPW', 'CLVS', 'DNLI', 'KE', 'DLHC', 'TNXP', 'SUNW',
                                 'CNXN', 'STOK', 'OPRA', 'AFIN', 'WLDN', 'USAT', 'BWB', 'FCCY', 'RRR', 'LCA', 'WIX', 'OMCL', 'NEPT', 'EPAY', 'RMR', 'AGYS', 'IMGN', 'COWN',
                                 'ESSA', 'CARG', 'CVAC', 'SRTS', 'LEDS', 'NNBR', 'SMIT', 'CMCSA', 'CNCE', 'QFIN', 'XBIT', 'ATLC', 'STSA', 'ASML', 'PRTS', 'TXRH', 'ZYNE',
                                 'PPSI', 'ACHV', 'ASND', 'IDYA', 'MPWR', 'VFF', 'MTSI', 'VERO', 'SABR', 'PEIX', 'STKL', 'SPFI', 'OTEX', 'AVT', 'VREX', 'PROV', 'FDUS', 'PAHC',
                                 'IIN', 'SONO', 'CPST', 'CVV', 'FFIC', 'GRPN', 'MIND', 'RELL', 'EVOP', 'VRNT', 'LSTR', 'EIGI', 'KEQU', 'QRHC', 'CLLS', 'EGOV', 'TTMI', 'NTGR',
                                 'RBBN', 'ITRM', 'AYRO', 'TLGT', 'AVCO', 'NVAX', 'VICR', 'FAT', 'NMCI', 'CMLF', 'XNCR', 'CVCY', 'PFPT', 'SND', 'AMEH', 'TESS', 'SG', 'TAST',
                                 'ETTX', 'RRBI', 'CODX', 'CLFD', 'VERU', 'TSCO', 'MSTR', 'CNOB', 'REKR', 'PINC', 'BIGC', 'ZBRA', 'DMTK', 'FTEK', 'SFT', 'IVAC', 'DHIL', 'KURA',
                                 'QK', 'FIVE', 'ICUI', 'KTCC', 'PPBI', 'TELL', 'BANF', 'IRBT', 'CPSH', 'APLT', 'NRC', 'OSPN', 'FGBI', 'CSCO', 'OKTA', 'QNTA', 'NTEC', 'STXS',
                                 'MEET', 'TAYD', 'CATM', 'JYNT', 'FBIO', 'SMTC', 'PETQ', 'VTSI', 'XLNX', 'IMVT', 'FUSE', 'XRAY', 'BLIN', 'IOVA', 'PENN', 'DYNT', 'KLIC',
                                 'BBGI', 'NXST', 'SLRC', 'CTSH', 'ARAY', 'VKTX', 'AVO', 'NSIT', 'SRPT', 'LMAT', 'ATRI', 'LOOP', 'HA', 'BEAT', 'SSNC', 'HGSH', 'MGTA', 'YMAB',
                                 'CG', 'ISBC', 'CDK', 'ABEO', 'HEES', 'EXTR', 'ARWR', 'LKQ', 'CDNS', 'HTLD', 'NKLA', 'ONEM', 'ESTA', 'FRPT', 'BNGO', 'STFC', 'MDXG', 'SGMA',
                                 'CALB', 'CONN', 'ZAGG', 'PFSW', 'PEP', 'DSKE', 'FFIV', 'GLYC', 'MCRI', 'TUSK', 'BKCC', 'CASS', 'SIBN', 'DXPE', 'DCTH', 'VRSN', 'TYME', 'HALO',
                                 'EBIX', 'AMRK', 'NMRK', 'DOYU', 'DXCM', 'AMRH', 'IRTC', 'VIR', 'AROW', 'EAR', 'ATOS', 'GLDD', 'IOSP', 'PLXS', 'IROQ', 'SAL', 'HOFV', 'WDC',
                                 'NGM', 'TSC', 'ALGS', 'APVO', 'GSHD', 'RIVE', 'BPMC', 'CTMX', 'VRNS', 'NICK', 'CWBC', 'AXDX', 'QCOM', 'WWD', 'PATK', 'KYMR', 'PRVB', 'NAII',
                                 'FEYE', 'CPRX', 'CUTR', 'VYNE', 'SBUX', 'PEBO', 'RELV', 'TXG', 'OPGN', 'KTRA', 'XEL', 'SONM', 'SVC', 'RVSB', 'SREV', 'KFRC', 'OPTT', 'TER',
                                 'FBNC', 'BCEL', 'CLGN', 'ARCC', 'MBIO', 'NEXT', 'PLUG', 'ACMR', 'GRTX', 'KROS', 'PGNY', 'EMKR', 'CFBI', 'JCS', 'STTK', 'UIHC', 'FOXA', 'EQ',
                                 'SEER', 'QRTEA', 'MARK', 'SVVC', 'RNET', 'QUIK', 'GNTX', 'ILPT', 'VC', 'KRBP', 'RESN', 'EVOL', 'NNDM', 'FIXX', 'NTRS', 'ETFC', 'IVA', 'AMD',
                                 'PPC', 'SRCE', 'PECK', 'HIBB', 'VIVE', 'AVGR', 'XSPA', 'CREX', 'IMMU', 'LHCG', 'MXIM', 'AUTO', 'PCVX', 'SNFCA', 'BLKB', 'BOXL', 'VBTX',
                                 'CETX', 'BDGE', 'LJPC', 'MRSN', 'DXLG', 'BHF', 'JVA', 'SFM', 'CLRO', 'RAVN', 'NEPH', 'HUIZ', 'BSRR', 'XFOR', 'IDXG', 'TRMB', 'LKFN', 'CSCW',
                                 'ABIO', 'SP', 'TLRY', 'ADTX', 'NTRP', 'ADUS', 'AWRE', 'DENN', 'MRTN', 'HFWA', 'EGRX', 'VCNX', 'WSBC', 'PSTL', 'PYPL', 'ENTG', 'GERN', 'ALRM',
                                 'RNWK', 'BCOV', 'LYTS', 'GSKY', 'STCN', 'OPRX', 'FRGI', 'WYNN', 'EBAY', 'MNRO', 'FLMN', 'ACGL', 'ASPS', 'OPBK', 'FRG', 'CSOD', 'MYOK', 'HTLF',
                                 'MRIN', 'CME', 'GWGH', 'VRSK', 'GXGX', 'ASRT', 'HOL', 'STAY', 'DORM', 'MCFT', 'VSTM', 'CTRE', 'APPN', 'SILK', 'VIVO', 'JD', 'YRCW', 'NTAP',
                                 'TVTY', 'UMPQ', 'GLUU', 'CHTR', 'GOVX', 'WMG', 'CBIO', 'CONE', 'FULT', 'IPDN', 'DMLP', 'VIRT', 'SIVB', 'CBLI', 'MICT', 'PRTK', 'WLTW',
                                 'LBRDA', 'CATB', 'BFC', 'DISH', 'FARO', 'ISRG', 'IDEX', 'MNCL', 'MNOV', 'AQST', 'SSP', 'PTE', 'BLU', 'BOTJ', 'SAMG', 'OPK', 'ASFI', 'ECPG',
                                 'ABCB', 'LTBR', 'ICCC', 'EXLS', 'PBPB', 'VIAV', 'ATRS', 'PIH', 'NBIX', 'IPWR', 'INBK', 'CEVA', 'IDCC', 'LTRPA', 'HRZN', 'EWBC', 'UMBF',
                                 'MCHP', 'BPFH', 'CCXI', 'GT', 'MESO', 'VERB', 'DJCO', 'NARI', 'ANIX', 'SNSS', 'FATE', 'SPRO', 'JKHY', 'PICO', 'ARAV', 'APTX', 'IMMR', 'RETA',
                                 'AMTB', 'SBSI', 'HOPE', 'GRIL', 'HBIO', 'SHBI', 'ASRV', 'ANCN', 'IEP', 'INFI', 'DAKT', 'DARE', 'LPLA', 'AMSF', 'LBC', 'ALLK', 'IIVI', 'CFBK',
                                 'UPWK', 'TRS', 'COOP', 'HDSN', 'CODA', 'CWBR', 'KRON', 'MDRX', 'NVCR', 'RSSS', 'QADB', 'AGRX', 'ASYS', 'WHLM', 'SAFM', 'TTGT', 'IMAC', 'NVUS',
                                 'MOFG', 'DRRX', 'TTOO', 'OTRK', 'IMRA', 'SMMT', 'VXRT', 'SYNA', 'APEX', 'ATNX', 'EYES', 'HVBC', 'NCSM', 'PIXY', 'ATCX', 'FELE', 'DRNA',
                                 'FIBK', 'PDD', 'FLDM', 'OLED', 'CBFV', 'TWST', 'BRY', 'TILE', 'ALSK', 'CTRN', 'CIIC', 'CCOI', 'BKYI', 'DTIL', 'ROLL', 'BECN', 'AAPL', 'ODT',
                                 'VTVT', 'PNRG', 'BIMI', 'FMNB', 'DKNG', 'WLFC', 'BMRN', 'PBHC', 'STAA', 'LUNA', 'AIMC', 'ULBI', 'BLBD', 'AQB', 'BBQ', 'CMCT', 'FLXS', 'OFLX',
                                 'SBGI', 'EOLS', 'AXAS', 'DVAX', 'NUVA', 'CBAY', 'ARRY', 'LEVI', 'SOLY', 'ACAM', 'NCNA', 'BYFC', 'LOAC', 'UNAM', 'AGTC', 'PNFP', 'AKAM',
                                 'CAKE', 'SCOR', 'VCTR', 'MGI', 'AMCX', 'FOSL', 'BVXV', 'SRAX', 'LI', 'CSSE', 'SURF', 'SFIX', 'SVMK', 'ECOR', 'FFNW', 'RP', 'ALEC', 'TLND',
                                 'IIIV', 'ACET', 'HLIT', 'VG', 'FAST', 'SWAV', 'ENG', 'BCBP', 'BSY', 'DRIO', 'HLIO', 'CWST', 'PLMR', 'TENX', 'TPIC', 'AGNC', 'CCRN', 'IBKR',
                                 'TWIN', 'INDB', 'URGN', 'TCDA', 'CDMO', 'SSYS', 'WKHS', 'HURC', 'AAON', 'LASR', 'MITK', 'ESXB', 'LOVE', 'IRMD', 'JAKK', 'OBCI', 'NUAN',
                                 'NATH', 'SCKT', 'HZNP', 'PS', 'GLSI', 'ATRO', 'HONE', 'NDSN', 'ADMA', 'CFMS', 'EYE', 'PCYO', 'ADMP', 'PRAX', 'CAC', 'SMED', 'ONB', 'MORN',
                                 'WTFC', 'AUPH', 'RUN', 'GLG', 'KOD', 'CERN', 'SGA', 'TRUP', 'TTNP', 'WEN', 'ICAD', 'EKSO', 'MTSC', 'CMBM', 'OMEX', 'SESN', 'GBDC', 'FWONA',
                                 'MSVB', 'CHMA', 'NTWK', 'IRWD', 'VRM', 'MTRX', 'OCGN', 'MRVL', 'EBTC', 'VIAC', 'ORGS', 'MCEP', 'VTNR', 'LGIH', 'XGN', 'CJJD', 'LPSN', 'LMPX',
                                 'PBCT', 'AMNB', 'STKS', 'KTOS', 'ATRA', 'AEY', 'RMBL', 'OFS', 'INSE', 'XENT', 'AMGN', 'TH', 'IBTX', 'MGEN', 'GRMN', 'CROX', 'THRM', 'EPZM',
                                 'SIGI', 'APPS', 'CNBKA', 'BIOL', 'TTCF', 'CASH', 'SPLK', 'ALXO', 'GPRE', 'SLM', 'NWFL', 'NBAC', 'ONVO', 'DTEA', 'TECH', 'ROCK', 'WVVI',
                                 'CLSK', 'CZNC', 'PEGA', 'SNDE', 'WNEB', 'MELI', 'UXIN', 'RPD', 'AZRX', 'NOVN', 'STRS', 'CFFI', 'NKTX', 'BAND', 'KVHI', 'PLYA', 'REGI', 'SYKE',
                                 'ESGR', 'EQOS', 'OPTN', 'MRCC', 'SPTN', 'REYN', 'OTEL', 'APLS', 'NERV', 'KELYA', 'KNDI', 'CLSN', 'UCTT', 'COUP', 'IGAC', 'LNDC', 'ACER',
                                 'CYAN', 'ALPN', 'CPRT', 'LACQ', 'ATVI', 'ACRS', 'ICHR', 'CRVS', 'TREE', 'JAGX', 'RDUS', 'ALRN', 'PSMT', 'AOUT', 'FRPH', 'MU', 'TPTX', 'KINS',
                                 'IBOC', 'AINV', 'PTGX', 'COLB', 'GEVO', 'LILA', 'NBSE', 'CTHR', 'WINT', 'LOCO', 'AUTL', 'FORR', 'FOLD', 'ACTG', 'GECC', 'PMD', 'TITN', 'CRON',
                                 'EXPE', 'GABC', 'GNPX', 'PTC', 'GHSI', 'LYRA', 'INTG', 'MBRX', 'HSTM', 'DMRC', 'SNOA', 'MNSB', 'CYBE', 'IBEX', 'SDC', 'RIOT', 'HURN', 'KIN',
                                 'CARA', 'TNDM', 'HDS', 'CNSL', 'ETON', 'MTEM', 'USCR', 'RILY', 'CSX', 'SUNS', 'CYTK', 'CBTX', 'DGLY', 'XONE', 'CYRX', 'FNKO', 'GSBC', 'SMSI',
                                 'BOOM', 'SNES', 'ARDX', 'BSET', 'CDLX', 'DNKN', 'CMTL', 'CGNX', 'AIHS', 'CRMT', 'LCNB', 'OPRT', 'IMUX', 'PAYS', 'ASUR', 'SYRS', 'WHLR',
                                 'CASY', 'ESPR', 'OTIC', 'HWC', 'ANGI', 'BIOC', 'TIG', 'NNOX', 'MTBC', 'RDNT', 'IPGP', 'LQDA', 'FMAO', 'CDZI', 'HROW', 'BGCP', 'LEVL', 'VTGN',
                                 'DWSN', 'TC', 'OFED', 'NVEC', 'INTU', 'FOXF', 'OTTR', 'XENE', 'HSIC', 'MRUS', 'RGEN', 'LSBK', 'ITRI', 'NDRA', 'SPT', 'PRPO', 'TZOO', 'CVBF',
                                 'PCRX', 'FSLR', 'GILD', 'ACNB', 'CDEV', 'BREW', 'KERN', 'RXT', 'HWCC', 'AMED', 'SMPL', 'LANC', 'KRYS', 'NATI', 'COHU', 'NH', 'NTES', 'SELB',
                                 'AWH', 'CZWI', 'LIVX', 'PACB', 'AFIB', 'SIEB', 'CPAH', 'IIIN', 'ESSC', 'AVDL', 'AMST', 'BYND', 'HBT', 'ADES', 'IMXI', 'ICFI', 'ORBC', 'INVE',
                                 'SNEX', 'AMSWA', 'PAVM', 'VLDR', 'CLAR', 'PLSE', 'USIO', 'SANW', 'WIRE', 'AMZN', 'TRMK', 'ATAX', 'FCCO', 'GVP', 'THFF', 'BCML', 'NDAQ',
                                 'GNLN', 'ARCT', 'LITE', 'RGLS', 'INFN', 'GLPI', 'LFUS', 'RADA', 'IGMS', 'HFFG', 'FLUX', 'GNCA', 'CTSO', 'ZIOP', 'IONS', 'ON', 'CSTL', 'ACHC',
                                 'CPIX', 'NTIC', 'CLXT', 'SCHL', 'TBK', 'ANIP', 'FUTU', 'XPEL', 'CREG', 'NEOS', 'AVAV', 'XERS', 'SIEN', 'NTNX', 'STX', 'UFPT', 'RRGB', 'TACT',
                                 'SYBT', 'STRA', 'EXPD', 'RCKT', 'MKTX', 'AXGN', 'TOTA', 'WORX', 'XPER', 'BPTH', 'ANSS', 'VNDA', 'PVAC', 'NBTB', 'BASI', 'ULTA', 'MGRC',
                                 'CACC', 'MBUU', 'ONEW', 'TTWO', 'VNOM', 'KPTI', 'MNKD', 'OLLI', 'DLTH', 'CMPS', 'AMTX', 'AMRN', 'CMLS', 'AHCO', 'CENT', 'SEAC', 'ECOL',
                                 'CRWD', 'EGBN', 'COCP', 'INAQ', 'BILI', 'SRRK', 'SAFT', 'SGBX', 'MRNA', 'TARA', 'BMCH', 'POAI', 'AAXN', 'ASO', 'NDLS', 'THCA', 'JAMF', 'AMRS',
                                 'ADAP', 'CPTA', 'HWKN', 'DLTR', 'MAT', 'CLMT', 'FFBW', 'FBIZ', 'MBOT', 'LMRK', 'UEIC', 'AVID', 'ADMS', 'GNUS', 'OPES', 'TSRI', 'SPWH', 'RDI',
                                 'KBNT', 'AKUS', 'SAVA', 'GFN', 'QTNT', 'ENDP', 'MSON', 'KOSS', 'BLDR', 'STEP', 'PEAK', 'HCAT', 'CHFS', 'URBN', 'ARTL', 'INO', 'CRUS', 'HAIN',
                                 'SBFG', 'APOG', 'PRNB', 'SYNH', 'INPX', 'UTHR', 'TCON', 'BXRX', 'LCUT', 'USEG', 'AXNX', 'SCYX', 'BCRX', 'NYMT', 'TSLA', 'SUPN', 'ATRC',
                                 'LMST', 'RAPT', 'RDFN', 'SASR', 'HSII', 'TRMD', 'SLNO', 'BBCP', 'LGND', 'PDCO', 'AIRT', 'CBAN', 'UG', 'UVSP', 'PRAA', 'GIFI', 'RVMD', 'PCSA',
                                 'POWL', 'APEN', 'CATY', 'BANR', 'ORRF', 'ADOM', 'VRAY', 'ANDE', 'METX', 'MOXC', 'COLL', 'SMBC', 'FLGT', 'BFST', 'FTNT', 'PACW', 'SMCI',
                                 'YNDX', 'CMPI', 'COGT', 'PDSB', 'HAS', 'CBRL', 'CLDX', 'ODP', 'TTEK', 'ACST', 'WASH', 'MOR', 'CDXS', 'MLVF', 'KLDO', 'DFFN', 'ACOR', 'CCBG',
                                 'OYST', 'CMRX', 'EVFM', 'NFLX', 'SGMO', 'WEYS', 'EGAN', 'ZG', 'CRSA', 'ZM', 'CGIX', 'SAMA', 'SHYF', 'VSEC', 'ITCI', 'ONCT', 'DSPG', 'ATXI',
                                 'PRLD', 'PZZA', 'APTO', 'HCSG', 'RGCO', 'LOGI', 'FITB', 'KRUS', 'BBIO', 'ARLP', 'ELOX', 'SLGN', 'ABTX', 'TRCH', 'HOMB', 'APXT', 'NURO',
                                 'ORGO', 'MOMO', 'CHRW', 'BGFV', 'CNET', 'SWTX', 'VUZI', 'NKTR', 'SKYW', 'EDIT', 'FWRD', 'IBCP', 'INVA', 'AMPH', 'ABUS', 'LNSR', 'BEEM',
                                 'PSEC', 'SNBR', 'UNIT', 'STIM', 'CCMP', 'DOMO', 'MTCH', 'IDXX', 'NTRA', 'UAL', 'MEDP', 'RNA', 'CRNX', 'MGPI', 'IESC', 'KXIN', 'ALTM', 'LECO',
                                 'XOMA', 'FROG', 'FLL', 'BKSC', 'CECE', 'CRNC', 'UROV', 'PXLW', 'HCKT', 'PLAB', 'FULC', 'HGBL', 'CAR', 'OBNK', 'TRNS', 'YORW', 'AEGN', 'GEC',
                                 'VRRM', 'FB', 'OTLK', 'GBT', 'ICPT', 'PRGX', 'PRCP', 'IDRA', 'RYTM', 'LOPE', 'PLAY', 'SLDB', 'GRIF', 'MANT', 'OSW', 'BLUE', 'CWCO', 'ALOT',
                                 'KZR', 'HMNF', 'POOL', 'SUMO', 'FLNT', 'MORF', 'INSG', 'BBI', 'XELB', 'UONE', 'KLAC', 'BEAM', 'PLXP', 'POLA', 'FUV', 'MVBF', 'QUMU', 'AKTS',
                                 'TWNK', 'MCRB', 'GFED', 'VBLT', 'AVCT', 'NHLD', 'TCBK', 'BGNE', 'PWOD', 'TENB', 'NSYS', 'EHTH', 'GHIV', 'AGLE', 'PBYI', 'CENX', 'DADA',
                                 'NEOG', 'SMMC', 'SWKH', 'MSEX', 'TACO', 'RLMD', 'LORL', 'CRTD', 'TCPC', 'HSDT', 'VCYT', 'MIST', 'ISIG', 'AVXL', 'AAME', 'RICK', 'CORT',
                                 'EQIX', 'ANY', 'KNSL', 'ADXS', 'SIRI', 'JAN', 'IPAR', 'NEO', 'UI', 'CZR', 'INMB', 'GEOS', 'HTBK', 'FLIC', 'GSIT', 'GPRO', 'INCY', 'MACK',
                                 'CLSD', 'JBLU', 'ATEX', 'GDEN', 'SINT', 'PRAH', 'LULU', 'TSBK', 'VERI', 'OCFC', 'ASTE', 'EML', 'PSTX', 'CTBI', 'NTCT', 'SCPH', 'JAZZ', 'SQBG',
                                 'RAIL', 'ALLO', 'MBWM', 'MRTX', 'YGYI', 'NVFY', 'DCPH', 'HMST', 'SSNT', 'GALT', 'IMBI', 'ATOM', 'LAMR', 'ACAD', 'HOLX', 'REGN', 'FCEL',
                                 'ALTR', 'ZI', 'RAVE', 'SPPI', 'ISNS', 'TTEC', 'BL', 'CASI', 'TDAC', 'WBA', 'CUE', 'EGLE', 'SMBK', 'REPL', 'ALT', 'CALM', 'HALL', 'PSTI',
                                 'CTIB', 'DMAC', 'HTBX', 'PASG', 'MGTX', 'PRIM', 'PTON', 'HELE', 'EFSC', 'FANG', 'OSIS', 'KBAL', 'DGICA', 'ACIW', 'AGBA', 'ATEC', 'UNTY',
                                 'SFBS', 'CATC', 'CIDM', 'LPRO', 'NXPI', 'CRAI', 'ESCA', 'LOB', 'BLI', 'WINA', 'ZSAN', 'RKDA', 'SVRA', 'AMOT', 'RIDE', 'TMDX', 'MYRG', 'XAIR',
                                 'HWBK', 'PFMT', 'WING', 'CHUY', 'ARDS', 'GCBC', 'MLAB', 'CPSS', 'NBEV', 'LPCN', 'ALAC', 'HSON', 'SSB', 'ASTC', 'PIRS', 'AACQ', 'AGIO', 'PHUN',
                                 'VYGR', 'SNGX', '<unknown>', 'CRSR', 'MWK', 'PGEN', 'LAZY', 'UNFI', 'RDVT', 'RAMP', 'KOPN', 'LPTX', 'RIGL', 'PRVL', 'ISTR', 'IZEA', 'AMTD',
                                 'FVE', 'LSCC', 'NFE', 'LIVE', 'LMB', 'MRBK', 'SBRA', 'TRHC', 'BRKL', 'MEIP', 'MASI', 'SYNC', 'EIDX', 'LNTH', 'AYTU', 'STRO', 'ADIL', 'XBIO',
                                 'BBSI', 'MOHO', 'BNTC', 'TXMD', 'FIZZ', 'PME', 'CMPR', 'TPCO', 'CNST', 'BHTG', 'WW', 'ROIC', 'PRFT', 'NXTC', 'CERS', 'WRLD', 'UEPS', 'AMAG',
                                 'AEIS', 'MDLZ', 'AMBC', 'CETV', 'WSBF', 'CSTR', 'EDUC', 'NETE', 'KRTX', 'EDSA', 'WETF', 'CTAS', 'NAVI', 'ACBI', 'CREE', 'OEG', 'ALNA', 'PHIO',
                                 'ZION', 'FSBW', 'EYPT', 'UFPI', 'GWRS', 'ODFL', 'CNDT', 'STLD', 'REFR', 'CRSP', 'QURE', 'VRA', 'JNCE', 'LBAI', 'SLGG', 'TW', 'FRME', 'BRPA',
                                 'OXFD', 'QNST', 'SHEN', 'ATNI', 'ARCB', 'JOUT', 'BNFT', 'RCM', 'KIDS', 'LINK', 'BZUN', 'PRTH', 'RMCF', 'PFBI', 'DZSI', 'SSKN', 'SALM', 'STBA',
                                 'TCX', 'HYAC', 'PEBK', 'NWSA', 'CTXS', 'CYCN', 'SCON', 'PTI', 'UHAL', 'NSEC', 'ALXN', 'ONCS', 'TURN', 'MMSI', 'CRIS', 'ICMB', 'LMNR', 'SAGE',
                                 'WTER', 'CHRS', 'LVGO', 'TA', 'HBCP', 'REG', 'OPT', 'STRM', 'TWOU', 'MVIS', 'AKBA', 'REAL', 'SSTI', 'AMSC', 'MTLS', 'MANH', 'REPH', 'COLM',
                                 'FONR', 'ANAB', 'FRBK', 'SLP', 'GWPH', 'IAC', 'BSGM', 'HRTX', 'ENPH', 'OCUL', 'BCPC', 'DGII', 'MPAA', 'XLRN', 'VALU', 'PCH', 'PBIP', 'WSTL',
                                 'KIRK', 'PFIE', 'VERY', 'AVRO', 'AEYE', 'BLFS', 'SLCT', 'BKNG', 'FENC', 'DLPN', 'CSPI', 'SPWR', 'TCRR', 'TMUS', 'AKRO', 'IRDM', 'MIME',
                                 'GLRE', 'NMRD', 'MERC', 'PRMW', 'CRWS', 'PPD', 'ALBO', 'LOAN', 'CDNA', 'AIRG', 'LOGC', 'GTIM', 'OSS', 'BOKF', 'CSII', 'SFNC', 'MRNS', 'STRL',
                                 'BSQR', 'PDCE', 'RUTH', 'DIOD', 'QTRX', 'GOGO', 'HYRE', 'ICBK', 'FVAM', 'SYBX', 'PNBK', 'CNMD', 'ALNY', 'AMKR', 'LAUR', 'OPHC', 'WDFC',
                                 'MOBL', 'SNCA', 'CGBD', 'ADPT', 'CBSH', 'KALU', 'UBX', 'EVOK', 'OSTK', 'NMTR', 'CBMG', 'ALTA', 'RBB', 'SSSS', 'TFSL', 'LMFA', 'USAU', 'KALA',
                                 'MPB', 'DAIO', 'LRCX', 'CINF', 'MESA', 'PI', 'EVK', 'MKSI', 'SDGR', 'NEON', 'ALDX', 'MDCA', 'CCB', 'LYFT', 'MRAM', 'MIK', 'HEPA', 'BRQS',
                                 'AOSL', 'IEA', 'MNST', 'CPHC', 'CRBP', 'QTT', 'GBCI', 'ENSG', 'AMAT', 'ADBE', 'CLIR', 'HARP', 'QCRH', 'SLS', 'EVLO', 'RBKB', 'YTEN', 'FISV',
                                 'USAK', 'KALV', 'VRTX', 'COHR', 'LFAC', 'CBMB', 'FTFT', 'MNTA', 'RUSHA', 'WIFI', 'CPSI', 'PSTV', 'MFIN', 'ICON', 'SGLB', 'JACK', 'TRVN',
                                 'GAIA', 'CDW', 'RPRX', 'ONTX', 'FSTR', 'NCBS', 'WISA', 'RBCAA', 'NWBI', 'AVGO', 'PDLI', 'SWKS', 'STAF', 'RARE', 'BCLI', 'VRTU', 'RMTI',
                                 'LBTYA', 'BLNK', 'AMCI', 'NXGN', 'MRCY', 'MARA', 'RMBS', 'APPF', 'ENTA', 'HUBG', 'FLWS', 'ACRX', 'JRVR', 'AMWD', 'INOV', 'MATW', 'MGNX',
                                 'HOOK', 'FHB', 'RNST', 'BCDA', 'ACIA', 'SYNL', 'TCBI', 'VOXX', 'ASMB', 'VRME', 'NESR', 'CELC', 'PHAT', 'CTXR', 'TRMT', 'INTC', 'HOTH', 'WATT',
                                 'CHNG', 'RAND', 'RFIL', 'WDAY', 'CAMP', 'CLRB', 'HMSY', 'PTMN', 'LLNW', 'PLUS', 'NK', 'SRNE', 'EEFT', 'FLEX', 'TCF', 'EFOI', 'KNSA', 'FFIN',
                                 'MBII', 'TBLT', 'OM', 'TYHT', 'LE', 'ICCH', 'KRMD', 'COMM', 'GP', 'SPOK', 'TCFC', 'TNAV', 'GLIBA', 'GOOGL', 'TBBK', 'ILMN', 'THMO', 'ANAT',
                                 'ALJJ', 'EXC', 'JBHT', 'MNPR', 'FGEN', 'CRVL', 'NEWT', 'FORM', 'CASA', 'LUMO', 'METC', 'CDXC', 'PCSB', 'FUNC', 'BTAI', 'LIVK', 'HLNE', 'AZPN',
                                 'BSVN', 'GO', 'CHCI', 'TTD', 'MCHX', 'TAIT', 'PTCT', 'PKBK', 'API', 'ECHO', 'ZGYH', 'BWEN', 'KRNT', 'RVNC', 'GROW', 'CGRO', 'GOSS', 'DCOM',
                                 'SIGA', 'PFLT', 'AGEN', 'GH', 'GRNQ', 'GNMK', 'VSAT', 'TEAM', 'SLRX', 'SGMS', 'HTGM', 'HCCH', 'ADSK', 'FRTA', 'ZUMZ', 'SBBP', 'BIDU', 'LWAY',
                                 'AGFS', 'ZGNX', 'EXEL', 'EVBG', 'LAKE', 'SUMR', 'CRTO', 'SINO', 'INGN', 'POWI', 'HAFC', 'SMTX', 'CCNE', 'HBAN', 'SEEL', 'GTHX', 'GTEC',
                                 'ERII', 'CHDN', 'ROST', 'CFB', 'ACCD', 'SOHO', 'VBIV', 'PLCE', 'OCC', 'ROKU', 'FPAY', 'CTIC', 'CKPT', 'GDS', 'HCAC', 'SGH', 'FARM', 'BMRA',
                                 'BRP', 'HBP', 'SAIA', 'DISCA', 'QELL', 'TRIP', 'NMIH', 'CSWC', 'TCMD', 'APEI', 'XELA', 'PAAS', 'SHSP', 'RPAY', 'ACLS', 'RMG', 'SELF', 'GOCO',
                                 'MHLD', 'CVLT', 'TBIO', 'SPKE', 'NSTG', 'RMNI', 'ZYXI', 'UPLD', 'MGLN', 'WWR', 'AKU', 'CCLP', 'FCNCA', 'CLDB', 'LMNX', 'NVIV', 'MTEX', 'WTRH',
                                 'NTUS', 'EA', 'PFG', 'CHEF', 'MRKR', 'MDB', 'HNNA', 'LAND', 'FBRX', 'ANIK', 'DRAD', 'PNNT', 'VIRC', 'SBAC', 'LEGH', 'GRWG', 'CLBS', 'PODD',
                                 'PMBC', 'CYBR', 'PETS', 'QDEL', 'MYGN', 'SGEN', 'MAR', 'WVE', 'XP', 'LPTH', 'CVGW', 'EXPO', 'NVDA', 'BRKR', 'COST', 'RCMT', 'GDYN', 'OMP',
                                 'STNE', 'EXAS', 'EXPI', 'SCPL', 'KHC', 'OSBC', 'ZNGA', 'MMLP', 'SNDX', 'VISL', 'SYPR', 'BCTF', 'FTDR', 'FSDC', 'BMRC', 'SRDX', 'UTMD', 'MCBC',
                                 'BPOP', 'CLBK', 'AAL', 'AKER', 'SHOO', 'HCCI', 'ASPU', 'FNWB', 'UFCS', 'SNPS', 'IMKTA', 'OVID', 'CSGS', 'NBRV', 'AERI', 'WERN', 'CFFN']}

    from_xgb_keys = list(from_xgb.keys())
    from_sav_keys = list(from_saved.keys())

    sorted(from_xgb_keys)
    sorted(from_sav_keys)

    assert (from_xgb_keys == from_sav_keys)

    for k in from_xgb_keys:
        logger.info(k)
        list_xgb = list(from_xgb[k])
        sorted(list_xgb)
        list_saved = list(from_saved[k])
        sorted(list_saved)

        assert (list_xgb == list_saved)


def test_compare_columns():
    from_train = ['industry_Beverages - Non-Alcoholic', 'category_CEF', 'location_Alberta; Canada', 'f22_sentiment_neg', 'deferredrev', 'ncf', 'scalerevenue_6 - Mega',
                  'location_Michigan; U.S.A', 'buy_sell', 'location_British Columbia; Canada', 'location_Virginia; U.S.A', 'currency_KRW', 'user_followers_count',
                  'currency_HKD', 'location_Ohio; U.S.A', 'dividends', 'table_SF1', 'nasdaq_day_roi', 'industry_Rental & Leasing Services', 'location_Oregon; U.S.A', 'close',
                  'famaindustry_Candy & Soda', 'ebitda', 'famaindustry_Defense', 'industry_Consumer Electronics', 'industry_<unknown>', 'location_Taiwan', 'industry_Silver',
                  'location_Denmark', 'scalerevenue_<unknown>', 'user_is_translation_enabled', 'category_ADR Common Stock Primary Class', 'currency_BRL',
                  'location_Puerto Rico', 'industry_Diversified Industrials', 'location_Japan', 'location_Ontario; Canada', 'close_SMA_50', 'location_Maine; U.S.A',
                  'location_Finland', 'location_Oklahoma; U.S.A', 'industry_Specialty Finance', 'famaindustry_Entertainment', 'location_Kansas; U.S.A',
                  'category_Canadian Common Stock Primary Class', 'famaindustry_Fabricated Products', 'taxliabilities', 'industry_Uranium', 'location_Quebec; Canada', 'debtc',
                  'industry_Drug Manufacturers - General', 'location_Poland', 'industry_Recreational Vehicles', 'netincdis', 'future_open', 'location_Mexico',
                  'category_Canadian Preferred Stock', 'location_New Brunswick; Canada', 'location_Marshall Islands', 'sector_Consumer Defensive', 'location_Nebraska; U.S.A',
                  'ebitdausd', 'table_<unknown>', 'location_Hong Kong', 'close_SMA_200', 'location_Georgia U.S.A.', 'location_Thailand', 'industry_Healthcare Plans',
                  'user_statuses_count', 'industry_Insurance - Life', 'industry_Diagnostics & Research', 'ppnenet', 'famaindustry_Shipping Containers', 'ebitdamargin',
                  'f22_ticker_in_text', 'industry_Furnishings', 'future_close', 'industry_Restaurants', 'industry_REIT - Healthcare Facilities', 'location_Illinois; U.S.A',
                  'currency_USD', 'industry_Insurance - Reinsurance', 'intexp', 'industry_Computer Hardware', 'industry_Broadcasting - TV', 'taxexp',
                  'sicsector_Nonclassifiable', 'user_listed_count', 'industry_Biotechnology', 'location_Utah; U.S.A', 'location_South Carolina; U.S.A', 'days_since',
                  'industry_Business Equipment', 'industry_Advertising Agencies', 'industry_Insurance - Specialty', 'close_SMA_100', 'industry_Software - Infrastructure',
                  'opinc', 'famaindustry_Recreation', 'category_ADR Stock Warrant', 'industry_Medical Devices', 'location_District Of Columbia; U.S.A',
                  'location_Republic Of Korea', 'industry_Auto & Truck Dealerships', 'industry_Credit Services', 'location_New York; U.S.A',
                  'industry_Financial Conglomerates', 'industry_Utilities - Independent Power Producers', 'location_Manitoba; Canada', 'industry_Software - Application',
                  'industry_Steel', 'industry_Medical Distribution', 'location_Nova Scotia; Canada', 'industry_REIT - Retail', 'famaindustry_Beer & Liquor',
                  'industry_Farm & Heavy Construction Machinery', 'industry_Tools & Accessories', 'category_IDX', 'industry_Semiconductors', 'currency_PLN', 'currency_VEF',
                  'industry_REIT - Industrial', 'industry_Beverages - Soft Drinks', 'industry_Apparel Stores', 'siccode', 'ncfbus', 'industry_Infrastructure Operations',
                  'industry_Airlines', 'location_Saudi Arabia', 'retearn', 'ros', 'currency_CNY', 'date', 'industry_Packaged Foods', 'currency_CLP', 'industry_Broadcasting',
                  'price', 'f22_has_cashtag', 'industry_Oil & Gas Equipment & Services', 'sicsector_Construction', 'currentratio', 'currency_PEN', 'currency_<unknown>', 'pe',
                  'famaindustry_Restaraunts Hotels Motels', 'industry_Auto Manufacturers', 'taxassets', 'currency_NZD', 'scalemarketcap_3 - Small', 'scalerevenue_5 - Large',
                  'location_New Mexico; U.S.A', 'fd_day_of_week', 'industry_Apparel Retail', 'industry_Industrial Metals & Minerals', 'sicsector_Retail Trade',
                  'industry_Luxury Goods', 'user_screen_name', 'user_has_extended_profile', 'famaindustry_Medical Equipment', 'industry_Consulting Services',
                  'industry_Health Care Plans', 'location_Argentina', 'location_Saint Vincent And The Grenadines', 'future_high', 'location_New Hampshire; U.S.A',
                  'category_Domestic Preferred Stock', 'user_verified', 'location_Nevada; U.S.A', 'industry_Trucking', 'industry_Staffing & Outsourcing Services', 'debtusd',
                  'industry_Leisure', 'famaindustry_Transportation', 'famaindustry_Construction Materials', 'location_Bahamas', 'industry_Media - Diversified',
                  'location_Hungary', 'industry_Travel Services', 'roe', 'location_South Dakota; U.S.A', 'cor', 'location_Delaware; U.S.A', 'location_Indonesia',
                  'location_Malta', 'sgna', 'industry_Medical Care', 'famaindustry_Insurance', 'industry_Oil & Gas Midstream', 'industry_Lumber & Wood Production',
                  'famaindustry_Real Estate', 'liabilitiesc', 'location_United Republic Of Tanzania', 'fd_day_of_year', 'investments', 'famaindustry_Tobacco Products', 'open',
                  'industry_Banks - Regional', 'sicsector_Transportation Communications Electric Gas And Sanitary Service', 'sicsector_Agriculture Forestry And Fishing',
                  'location_Vermont; U.S.A', 'industry_Farm Products', 'industry_Utilities - Regulated Water', 'industry_Specialty Retail', 'location_Isle Of Man',
                  'sharesbas', 'ebit', 'original_close_price', 'industry_REIT - Hotel & Motel', 'location_<unknown>', 'industry_Home Improvement Retail', 'location_Canada',
                  'famaindustry_Computers', 'location_Costa Rica', 'future_low', 'future_date', 'user_friends_count', 'industry_Coking Coal', 'scalerevenue_3 - Small',
                  "location_Democratic People'S Republic Of Korea", 'location_Texas; U.S.A', 'currency_INR', 'evebitda', 'roic', 'netinc', 'famaindustry_Wholesale',
                  'industry_Packaging & Containers', 'currency_JPY', 'industry_Gold', 'days_util_sale', 'close_SMA_20_days_since_under', 'location_Uruguay', 'low',
                  'famaindustry_Personal Services', 'location_Massachusetts; U.S.A', 'industry_Education & Training Services', 'industry_Electronics & Computer Distribution',
                  'assetsc', 'industry_Oil & Gas E&P', 'ebt', 'tbvps', 'stock_val_change', 'industry_Waste Management', 'industry_Banks - Global',
                  'category_ADR Preferred Stock', 'location_Germany', 'location_Ireland', 'location_Colombia', 'sector_Utilities', 'industry_Electronic Gaming & Multimedia',
                  'location_Malaysia', 'famaindustry_Textiles', 'location_Norway', 'location_Louisiana; U.S.A', 'netmargin', 'revenueusd', 'sicsector_<unknown>',
                  'industry_Utilities - Regulated Electric', 'consolinc', 'famaindustry_Measuring and Control Equipment', 'payoutratio', 'location_Missouri; U.S.A',
                  'location_Cyprus', 'industry_Textile Manufacturing', 'famaindustry_Petroleum and Natural Gas', 'revenue', 'f22_compound_score', 'scalemarketcap_2 - Micro',
                  'famaindustry_Rubber and Plastic Products', 'industry_Entertainment', 'industry_Asset Management', 'sector_Healthcare', 'location_Georgia; U.S.A',
                  'location_Singapore', 'prefdivis', 'industry_REIT - Diversified', 'industry_Real Estate - General', 'industry_Marine Shipping', 'scalemarketcap_1 - Nano',
                  'famaindustry_Construction', 'epsdil', 'cashneq', 'netinccmnusd', 'location_Alaska; U.S.A', 'assetsnc', 'sicsector_Finance Insurance And Real Estate',
                  'industry_REIT - Mortgage', 'location_California; U.S.A', 'location_Austria', 'famaindustry_Communication', 'location_Russian Federation', 'debtnc',
                  'location_North Dakota; U.S.A', 'category_ETD', 'industry_Integrated Freight & Logistics', 'famaindustry_Aircraft', 'location_Idaho; U.S.A',
                  'user_protected', 'intangibles', 'industry_Beverages - Wineries & Distilleries', 'location_Maryland; U.S.A', 'location_Hawaii; U.S.A',
                  'industry_Medical Care Facilities', 'location_Venezuela', 'divyield', 'invcap', 'location_Guam', 'industry_Data Storage', 'gp',
                  'famaindustry_Printing and Publishing', 'cashnequsd', 'location_Wyoming; U.S.A', 'famasector', 'sicsector_Manufacturing', 'industry_Resorts & Casinos',
                  'f22_ticker', 'industry_Other Industrial Metals & Mining', 'industry_Other Precious Metals & Mining', 'location_British Virgin Islands', 'eps',
                  'scalemarketcap_<unknown>', 'category_<unknown>', 'location_Iowa; U.S.A', 'payables', 'location_Peru', 'famaindustry_<unknown>', 'location_Jordan',
                  'industry_Drug Manufacturers - Specialty & Generic', 'location_New Zealand', 'purchase_date', 'location_Israel-Jordan', 'accoci', 'table_SEP',
                  'famaindustry_Chemicals', 'fd_day_of_month', 'sector_Real Estate', 'category_Domestic Common Stock Secondary Class',
                  'famaindustry_Non-Metallic and Industrial Metal Mining', 'location_Greece', 'industry_Furnishings Fixtures & Appliances',
                  'industry_Medical Instruments & Supplies', 'industry_Communication Equipment', 'currency_COP', 'industry_Savings & Cooperative Banks',
                  'industry_Oil & Gas Drilling', 'category_Domestic Common Stock Primary Class', 'sicsector_Public Administration', 'close_SMA_200_days_since_under',
                  'user_geo_enabled', 'currency_CHF', 'grossmargin', 'location_Newfoundland; Canada', 'location_Chile', 'industry_Household & Personal Products',
                  'industry_Business Services', 'industry_Mortgage Finance', 'industry_Specialty Business Services', 'location_Cayman Islands',
                  'location_United States; U.S.A', 'industry_Specialty Industrial Machinery', 'industry_Semiconductor Equipment & Materials', 'famaindustry_Business Supplies',
                  'industry_Shell Companies', 'currency_PHP', 'industry_Semiconductor Memory', 'location_Canada (Federal Level)', 'ebitusd', 'f22_id', 'epsusd',
                  'sicsector_Mining', 'industry_Utilities - Renewable', 'netincnci', 'pb', 'shareswadil', 'evebit', 'location_Maldives', 'fcf',
                  'close_SMA_15_days_since_under', 'sector_Basic Materials', 'famaindustry_Utilities', 'industry_Publishing', 'industry_Industrial Distribution',
                  'famaindustry_Precious Metals', 'industry_REIT - Office', 'sector_Consumer Cyclical', 'industry_Home Improvement Stores', 'industry_Internet Retail',
                  'location_Macau', 'investmentsc', 'location_Netherlands', 'industry_Tobacco', 'industry_REIT - Residential', 'location_Panama',
                  'category_Canadian Stock Warrant', 'industry_Health Information Services', 'liabilities', 'rnd', 'currency_IDR', 'famaindustry_Apparel',
                  'industry_Insurance Brokers', 'location_Turkey', 'sector_Energy', 'location_Brazil', 'category_ETF', 'category_Canadian Common Stock',
                  'industry_Financial Exchanges', 'location_Sweden', 'industry_Conglomerates', 'location_Mississippi; U.S.A', 'industry_Oil & Gas Refining & Marketing',
                  'currency_CAD', 'assets', 'ncfdiv', 'ncff', 'industry_Department Stores', 'location_Pennsylvania; U.S.A', 'industry_Utilities - Regulated Gas',
                  'location_North Carolina; U.S.A', 'industry_Long-Term Care Facilities', 'industry_Chemicals', 'industry_Real Estate Services',
                  'close_SMA_100_days_since_under', 'location_Bermuda', 'industry_Aluminum', 'industry_Solar', 'famaindustry_Shipbuilding Railroad Equipment', 'currency_SEK',
                  'industry_Real Estate - Diversified', 'category_Domestic Stock Warrant', 'industry_Shipping & Ports', 'location_Minnesota; U.S.A',
                  'category_ADR Common Stock', 'industry_Insurance - Property & Casualty', 'industry_Metal Fabrication', 'industry_Aerospace & Defense',
                  'famaindustry_Steel Works Etc', 'high', 'inventory', 'assetturnover', 'deposits', 'equityavg', 'ncfdebt', 'industry_Oil & Gas Integrated', 'equityusd',
                  'ncfcommon', 'ncfx', 'famaindustry_Business Services', 'location_Australia', 'bvps', 'location_Tennessee; U.S.A', 'industry_Internet Content & Information',
                  'dps', 'close_SMA_20', 'famaindustry_Machinery', 'favorite_count', 'famaindustry_Trading', 'industry_Business Equipment & Supplies',
                  'location_United Kingdom', 'ev', 'industry_Gambling', 'currency_NOK', 'location_France', 'sector_Industrials', 'industry_Grocery Stores', 'location_Ghana',
                  'famaindustry_Consumer Goods', 'sps', 'possibly_sensitive', 'industry_Engineering & Construction', 'industry_Copper', 'industry_Building Materials',
                  'sicsector_Services', 'industry_Financial Data & Stock Exchanges', 'location_West Virginia; U.S.A', 'f22_sentiment_pos', 'f22_is_tweet_after_hours',
                  'sector_Communication Services', 'currency_AUD', 'location_Italy', 'f22_num_other_tickers_in_tweet', 'famaindustry_Electronic Equipment',
                  'famaindustry_Coal', 'currency_ARS', 'category_ETN', 'location_Connecticut; U.S.A', 'invcapavg', 'scalemarketcap_5 - Large', 'location_Philippines',
                  'scalerevenue_2 - Micro', 'marketcap', 'tangibles', 'location_Washington; U.S.A', 'location_United Arab Emirates', 'industry_Home Furnishings & Fixtures',
                  'ps1', 'sicsector_Wholesale Trade', 'industry_Computer Systems', 'de', 'industry_Paper & Paper Products', 'scalerevenue_1 - Nano', 'industry_Thermal Coal',
                  'location_New Jersey; U.S.A', 'ncfi', 'pe1', 'industry_Confectioners', 'industry_Discount Stores', 'location_Colorado; U.S.A', 'currency_EUR',
                  'industry_Telecom Services', 'location_China', 'location_Jersey', 'location_Israel-Syria', 'location_Wisconsin; U.S.A', 'location_Israel',
                  'location_Virgin Islands; U.S.', 'industry_Insurance - Diversified', 'location_Spain', 'opex', 'location_Alabama; U.S.A', 'industry_Apparel Manufacturing',
                  'industry_Security & Protection Services', 'stock_val_change_ex', 'location_South Africa', 'industry_Auto Parts', 'currency_GBP', 'f22_sentiment_compound',
                  'industry_Staffing & Employment Services', 'currency_DKK', 'assetsavg', 'location_Monaco', 'sector_Technology', 'currency_ZAR',
                  'industry_Banks - Diversified', 'industry_Banks - Regional - US', 'location_Czech Republic', 'industry_Pharmaceutical Retailers',
                  'location_Saskatchewan; Canada', 'roa', 'scalerevenue_4 - Mid', 'close_SMA_15', 'shareswa', 'equity', 'ncfinv', 'location_Switzerland',
                  'industry_Footwear & Accessories', 'location_Kentucky; U.S.A', 'famaindustry_Almost Nothing', 'industry_Broadcasting - Radio', 'location_Gibraltar', 'ncfo',
                  'receivables', 'workingcapital', 'currency_MYR', 'depamor', 'location_Montana; U.S.A', 'sbcomp', 'currency_TWD', 'f22_sentiment_neu',
                  'sector_Financial Services', 'fxusd', 'user_follow_request_sent', 'retweet_count', 'currency_ILS', 'industry_Electrical Equipment & Parts',
                  'location_Arkansas; U.S.A', 'industry_Lodging', 'industry_Building Products & Equipment', 'industry_Food Distribution', 'industry_REIT - Specialty',
                  'location_Iceland', 'volume', 'category_Domestic Common Stock', 'famaindustry_Food Products', 'netinccmn', 'industry_Information Technology Services',
                  'location_Belgium', 'famaindustry_Healthcare', 'industry_Farm & Construction Equipment', 'sharefactor', 'industry_Real Estate - Development',
                  'close_SMA_50_days_since_under', 'currency_RUB', 'liabilitiesnc', 'location_India', 'location_Arizona; U.S.A', 'currency_MXN',
                  'industry_Airports & Air Services', 'industry_Utilities - Diversified', 'ps', 'industry_Residential Construction', 'famaindustry_Pharmaceutical Products',
                  'industry_Electronic Components', 'closeunadj', 'industry_Specialty Chemicals', 'table_SFP', 'debt', 'location_Luxembourg', 'table_SF3B',
                  'location_Netherlands Antilles', 'location_Rhode Island; U.S.A', 'famaindustry_Automobiles and Trucks', 'industry_Railroads', 'industry_Beverages - Brewers',
                  'fcfps', 'location_Florida; U.S.A', 'famaindustry_Banking', 'capex', 'location_Mauritius', 'famaindustry_Electrical Equipment', 'scalemarketcap_4 - Mid',
                  'industry_Agricultural Inputs', 'sector_<unknown>', 'location_Unknown', 'location_Guernsey', 'industry_Coal', 'industry_Personal Services',
                  'industry_Drug Manufacturers - Major', 'famaindustry_Retail', 'location_Oman', 'location_Indiana; U.S.A', 'industry_Pollution & Treatment Controls',
                  'scalemarketcap_6 - Mega', 'famaindustry_Agriculture', 'category_ADR Common Stock Secondary Class', 'industry_Capital Markets', 'currency_TRY',
                  'investmentsnc', 'industry_Scientific & Technical Instruments', 'f22_day_tweet_count', 'rating', 'rank', 'rating_age_days', 'target_price', 'rank_roi',
                  'f22_ticker_MIRM', 'f22_ticker_SEDG', 'f22_ticker_LQDT', 'f22_ticker_ALGN', 'f22_ticker_ARVN', 'f22_ticker_HJLI', 'f22_ticker_CELH', 'f22_ticker_LIND',
                  'f22_ticker_SPSC', 'f22_ticker_BLMN', 'f22_ticker_ALGT', 'f22_ticker_UTSI', 'f22_ticker_EVER', 'f22_ticker_EMMA', 'f22_ticker_HQY', 'f22_ticker_CHCO',
                  'f22_ticker_GLAD', 'f22_ticker_RGLD', 'f22_ticker_MIDD', 'f22_ticker_EQBK', 'f22_ticker_PRGS', 'f22_ticker_PCAR', 'f22_ticker_OESX', 'f22_ticker_JJSF',
                  'f22_ticker_GOOD', 'f22_ticker_TIPT', 'f22_ticker_AREC', 'f22_ticker_AIMT', 'f22_ticker_ADI', 'f22_ticker_FLIR', 'f22_ticker_TGTX', 'f22_ticker_TXN',
                  'f22_ticker_BCOR', 'f22_ticker_MBIN', 'f22_ticker_SANM', 'f22_ticker_CDAK', 'f22_ticker_CLNE', 'f22_ticker_AUB', 'f22_ticker_MGEE', 'f22_ticker_PCB',
                  'f22_ticker_CRTX', 'f22_ticker_VRTS', 'f22_ticker_SRCL', 'f22_ticker_AMBA', 'f22_ticker_CYCC', 'f22_ticker_CAPR', 'f22_ticker_NODK', 'f22_ticker_CVET',
                  'f22_ticker_FIII', 'f22_ticker_ELSE', 'f22_ticker_PSNL', 'f22_ticker_CARV', 'f22_ticker_DOCU', 'f22_ticker_FREQ', 'f22_ticker_NOVT', 'f22_ticker_LFVN',
                  'f22_ticker_UBSI', 'f22_ticker_OPI', 'f22_ticker_INSM', 'f22_ticker_SONA', 'f22_ticker_APDN', 'f22_ticker_WMGI', 'f22_ticker_QRVO', 'f22_ticker_SGRY',
                  'f22_ticker_GRBK', 'f22_ticker_FNCB', 'f22_ticker_FFWM', 'f22_ticker_CRDF', 'f22_ticker_MMAC', 'f22_ticker_QLYS', 'f22_ticker_VIOT', 'f22_ticker_ADTN',
                  'f22_ticker_BPYU', 'f22_ticker_AVNW', 'f22_ticker_ARTW', 'f22_ticker_BRKS', 'f22_ticker_HMTV', 'f22_ticker_FIVN', 'f22_ticker_NAOV', 'f22_ticker_GIII',
                  'f22_ticker_PAYX', 'f22_ticker_OBLN', 'f22_ticker_SRRA', 'f22_ticker_CEMI', 'f22_ticker_MLHR', 'f22_ticker_LIVN', 'f22_ticker_ALIM', 'f22_ticker_CLCT',
                  'f22_ticker_VCEL', 'f22_ticker_ISEE', 'f22_ticker_PULM', 'f22_ticker_AEMD', 'f22_ticker_GTLS', 'f22_ticker_ETSY', 'f22_ticker_DXYN', 'f22_ticker_PANL',
                  'f22_ticker_ZIXI', 'f22_ticker_FPRX', 'f22_ticker_PRPH', 'f22_ticker_CORE', 'f22_ticker_JRSH', 'f22_ticker_OSUR', 'f22_ticker_CALA', 'f22_ticker_MBCN',
                  'f22_ticker_PRSC', 'f22_ticker_KRNY', 'f22_ticker_NAKD', 'f22_ticker_AAWW', 'f22_ticker_BLPH', 'f22_ticker_ALKS', 'f22_ticker_CFRX', 'f22_ticker_RGNX',
                  'f22_ticker_PCTY', 'f22_ticker_GENC', 'f22_ticker_MOTS', 'f22_ticker_HAYN', 'f22_ticker_TGEN', 'f22_ticker_ADP', 'f22_ticker_ZS', 'f22_ticker_NTLA',
                  'f22_ticker_AESE', 'f22_ticker_FISI', 'f22_ticker_RUBY', 'f22_ticker_AXTI', 'f22_ticker_LUNG', 'f22_ticker_BBBY', 'f22_ticker_FOCS', 'f22_ticker_OMER',
                  'f22_ticker_DBX', 'f22_ticker_EBC', 'f22_ticker_FBMS', 'f22_ticker_JCOM', 'f22_ticker_SCWX', 'f22_ticker_ADVM', 'f22_ticker_ABMD', 'f22_ticker_HEAR',
                  'f22_ticker_TROW', 'f22_ticker_CD', 'f22_ticker_WTRE', 'f22_ticker_CVGI', 'f22_ticker_SEIC', 'f22_ticker_MNTX', 'f22_ticker_IHRT', 'f22_ticker_CBAT',
                  'f22_ticker_GTYH', 'f22_ticker_WSFS', 'f22_ticker_BIIB', 'f22_ticker_FREE', 'f22_ticker_ATHX', 'f22_ticker_AUVI', 'f22_ticker_NIU', 'f22_ticker_PRPL',
                  'f22_ticker_ARNA', 'f22_ticker_CIGI', 'f22_ticker_GBIO', 'f22_ticker_MSFT', 'f22_ticker_FEIM', 'f22_ticker_EZPW', 'f22_ticker_CLVS', 'f22_ticker_DNLI',
                  'f22_ticker_KE', 'f22_ticker_DLHC', 'f22_ticker_TNXP', 'f22_ticker_SUNW', 'f22_ticker_CNXN', 'f22_ticker_STOK', 'f22_ticker_OPRA', 'f22_ticker_AFIN',
                  'f22_ticker_USAT', 'f22_ticker_FCCY', 'f22_ticker_RRR', 'f22_ticker_OVLY', 'f22_ticker_LCA', 'f22_ticker_WIX', 'f22_ticker_OMCL', 'f22_ticker_NEPT',
                  'f22_ticker_EPAY', 'f22_ticker_PAE', 'f22_ticker_RMR', 'f22_ticker_AGYS', 'f22_ticker_IMGN', 'f22_ticker_RADI', 'f22_ticker_COWN', 'f22_ticker_ESSA',
                  'f22_ticker_CARG', 'f22_ticker_CVAC', 'f22_ticker_LARK', 'f22_ticker_SRTS', 'f22_ticker_NNBR', 'f22_ticker_SMIT', 'f22_ticker_CMCSA', 'f22_ticker_CNCE',
                  'f22_ticker_XBIT', 'f22_ticker_ATLC', 'f22_ticker_STSA', 'f22_ticker_ASML', 'f22_ticker_PRTS', 'f22_ticker_TXRH', 'f22_ticker_ZYNE', 'f22_ticker_PPSI',
                  'f22_ticker_ACHV', 'f22_ticker_ASND', 'f22_ticker_IDYA', 'f22_ticker_MPWR', 'f22_ticker_VFF', 'f22_ticker_MTSI', 'f22_ticker_VERO', 'f22_ticker_SABR',
                  'f22_ticker_PEIX', 'f22_ticker_RMBI', 'f22_ticker_STKL', 'f22_ticker_OTEX', 'f22_ticker_AVT', 'f22_ticker_VREX', 'f22_ticker_PROV', 'f22_ticker_FDUS',
                  'f22_ticker_SGRP', 'f22_ticker_IIN', 'f22_ticker_SONO', 'f22_ticker_CPST', 'f22_ticker_CVV', 'f22_ticker_FFIC', 'f22_ticker_GRPN', 'f22_ticker_MIND',
                  'f22_ticker_RELL', 'f22_ticker_EVOP', 'f22_ticker_VRNT', 'f22_ticker_LSTR', 'f22_ticker_EIGI', 'f22_ticker_QRHC', 'f22_ticker_CLLS', 'f22_ticker_EGOV',
                  'f22_ticker_TTMI', 'f22_ticker_NTGR', 'f22_ticker_RBBN', 'f22_ticker_ITRM', 'f22_ticker_AYRO', 'f22_ticker_AVCO', 'f22_ticker_NVAX', 'f22_ticker_AEHR',
                  'f22_ticker_VICR', 'f22_ticker_AMRB', 'f22_ticker_FAT', 'f22_ticker_AZN', 'f22_ticker_CMLF', 'f22_ticker_SNCR', 'f22_ticker_CVCY', 'f22_ticker_XNCR',
                  'f22_ticker_CVLY', 'f22_ticker_PFPT', 'f22_ticker_SND', 'f22_ticker_TESS', 'f22_ticker_SG', 'f22_ticker_ETTX', 'f22_ticker_CODX', 'f22_ticker_CLFD',
                  'f22_ticker_VERU', 'f22_ticker_TSCO', 'f22_ticker_MSTR', 'f22_ticker_FRAF', 'f22_ticker_SBT', 'f22_ticker_CNOB', 'f22_ticker_REKR', 'f22_ticker_PINC',
                  'f22_ticker_BIGC', 'f22_ticker_ZBRA', 'f22_ticker_DMTK', 'f22_ticker_FTEK', 'f22_ticker_SFT', 'f22_ticker_QLGN', 'f22_ticker_IVAC', 'f22_ticker_FLXN',
                  'f22_ticker_DHIL', 'f22_ticker_KURA', 'f22_ticker_QK', 'f22_ticker_FIVE', 'f22_ticker_ICUI', 'f22_ticker_KTCC', 'f22_ticker_PPBI', 'f22_ticker_TELL',
                  'f22_ticker_BANF', 'f22_ticker_IRBT', 'f22_ticker_CPSH', 'f22_ticker_APLT', 'f22_ticker_NRC', 'f22_ticker_OSPN', 'f22_ticker_FGBI', 'f22_ticker_CSCO',
                  'f22_ticker_OKTA', 'f22_ticker_HOFT', 'f22_ticker_NTEC', 'f22_ticker_STXS', 'f22_ticker_CATM', 'f22_ticker_JYNT', 'f22_ticker_FBIO', 'f22_ticker_SMTC',
                  'f22_ticker_PETQ', 'f22_ticker_VTSI', 'f22_ticker_XLNX', 'f22_ticker_IMVT', 'f22_ticker_XRAY', 'f22_ticker_BLIN', 'f22_ticker_VMD', 'f22_ticker_IOVA',
                  'f22_ticker_PENN', 'f22_ticker_DYNT', 'f22_ticker_KLIC', 'f22_ticker_BBGI', 'f22_ticker_NXST', 'f22_ticker_SLRC', 'f22_ticker_CTSH', 'f22_ticker_ARAY',
                  'f22_ticker_VKTX', 'f22_ticker_AVO', 'f22_ticker_NSIT', 'f22_ticker_SRPT', 'f22_ticker_LMAT', 'f22_ticker_ATRI', 'f22_ticker_LOGM', 'f22_ticker_LOOP',
                  'f22_ticker_HA', 'f22_ticker_BEAT', 'f22_ticker_SSNC', 'f22_ticker_HGSH', 'f22_ticker_MGTA', 'f22_ticker_YMAB', 'f22_ticker_CG', 'f22_ticker_ISBC',
                  'f22_ticker_CDK', 'f22_ticker_ABEO', 'f22_ticker_ARPO', 'f22_ticker_ENOB', 'f22_ticker_HEES', 'f22_ticker_EXTR', 'f22_ticker_ARWR', 'f22_ticker_LKQ',
                  'f22_ticker_CDNS', 'f22_ticker_HTLD', 'f22_ticker_NKLA', 'f22_ticker_ONEM', 'f22_ticker_ESTA', 'f22_ticker_FRPT', 'f22_ticker_BNGO', 'f22_ticker_STFC',
                  'f22_ticker_MDXG', 'f22_ticker_SGMA', 'f22_ticker_CALB', 'f22_ticker_CONN', 'f22_ticker_ZAGG', 'f22_ticker_PFSW', 'f22_ticker_PEP', 'f22_ticker_DSKE',
                  'f22_ticker_FFIV', 'f22_ticker_GLYC', 'f22_ticker_MCRI', 'f22_ticker_TUSK', 'f22_ticker_CASS', 'f22_ticker_SIBN', 'f22_ticker_DCTH', 'f22_ticker_RBCN',
                  'f22_ticker_EMCF', 'f22_ticker_VRSN', 'f22_ticker_TYME', 'f22_ticker_HALO', 'f22_ticker_EBIX', 'f22_ticker_AMRK', 'f22_ticker_NMRK', 'f22_ticker_DOYU',
                  'f22_ticker_DXCM', 'f22_ticker_RPLA', 'f22_ticker_AMRH', 'f22_ticker_IRTC', 'f22_ticker_VIR', 'f22_ticker_EAR', 'f22_ticker_ATOS', 'f22_ticker_IOSP',
                  'f22_ticker_PLXS', 'f22_ticker_IROQ', 'f22_ticker_SAL', 'f22_ticker_HOFV', 'f22_ticker_WDC', 'f22_ticker_NGM', 'f22_ticker_TSC', 'f22_ticker_ALGS',
                  'f22_ticker_ULH', 'f22_ticker_APVO', 'f22_ticker_GSHD', 'f22_ticker_RIVE', 'f22_ticker_BPMC', 'f22_ticker_CTMX', 'f22_ticker_VRNS', 'f22_ticker_NICK',
                  'f22_ticker_AXDX', 'f22_ticker_QCOM', 'f22_ticker_WWD', 'f22_ticker_PATK', 'f22_ticker_KYMR', 'f22_ticker_PRVB', 'f22_ticker_FEYE', 'f22_ticker_CPRX',
                  'f22_ticker_CUTR', 'f22_ticker_VYNE', 'f22_ticker_SBUX', 'f22_ticker_SFST', 'f22_ticker_PEBO', 'f22_ticker_TXG', 'f22_ticker_OPGN', 'f22_ticker_KTRA',
                  'f22_ticker_XEL', 'f22_ticker_SONM', 'f22_ticker_SREV', 'f22_ticker_KFRC', 'f22_ticker_OPTT', 'f22_ticker_TER', 'f22_ticker_SPNE', 'f22_ticker_FBNC',
                  'f22_ticker_BCEL', 'f22_ticker_CLGN', 'f22_ticker_ARCC', 'f22_ticker_MBIO', 'f22_ticker_NEXT', 'f22_ticker_LAWS', 'f22_ticker_PLUG', 'f22_ticker_ACMR',
                  'f22_ticker_KROS', 'f22_ticker_PGNY', 'f22_ticker_EMKR', 'f22_ticker_JCS', 'f22_ticker_STTK', 'f22_ticker_UIHC', 'f22_ticker_FOXA', 'f22_ticker_EQ',
                  'f22_ticker_SEER', 'f22_ticker_ATSG', 'f22_ticker_MARK', 'f22_ticker_QRTEA', 'f22_ticker_QUIK', 'f22_ticker_GNTX', 'f22_ticker_ILPT', 'f22_ticker_FFBC',
                  'f22_ticker_VC', 'f22_ticker_KRBP', 'f22_ticker_RESN', 'f22_ticker_EVOL', 'f22_ticker_NNDM', 'f22_ticker_FIXX', 'f22_ticker_IRIX', 'f22_ticker_NTRS',
                  'f22_ticker_ETFC', 'f22_ticker_IVA', 'f22_ticker_AMD', 'f22_ticker_PPC', 'f22_ticker_SRCE', 'f22_ticker_PECK', 'f22_ticker_HIBB', 'f22_ticker_VIVE',
                  'f22_ticker_AVGR', 'f22_ticker_XSPA', 'f22_ticker_CREX', 'f22_ticker_IMMU', 'f22_ticker_LHCG', 'f22_ticker_MXIM', 'f22_ticker_NCMI', 'f22_ticker_AUTO',
                  'f22_ticker_PCVX', 'f22_ticker_BLKB', 'f22_ticker_BOXL', 'f22_ticker_VBTX', 'f22_ticker_CETX', 'f22_ticker_LJPC', 'f22_ticker_BDGE', 'f22_ticker_MRSN',
                  'f22_ticker_DXLG', 'f22_ticker_BHF', 'f22_ticker_JVA', 'f22_ticker_SFM', 'f22_ticker_CLRO', 'f22_ticker_GRTS', 'f22_ticker_RAVN', 'f22_ticker_NEPH',
                  'f22_ticker_HUIZ', 'f22_ticker_SECO', 'f22_ticker_XFOR', 'f22_ticker_TRMB', 'f22_ticker_NKSH', 'f22_ticker_LKFN', 'f22_ticker_ABIO', 'f22_ticker_SP',
                  'f22_ticker_TLRY', 'f22_ticker_NTRP', 'f22_ticker_ADUS', 'f22_ticker_AWRE', 'f22_ticker_ORPH', 'f22_ticker_DENN', 'f22_ticker_MRTN', 'f22_ticker_HFWA',
                  'f22_ticker_EGRX', 'f22_ticker_VCNX', 'f22_ticker_WSBC', 'f22_ticker_JBSS', 'f22_ticker_PYPL', 'f22_ticker_ENTG', 'f22_ticker_GERN', 'f22_ticker_ALRM',
                  'f22_ticker_RNWK', 'f22_ticker_HBMD', 'f22_ticker_BCOV', 'f22_ticker_VECO', 'f22_ticker_GSKY', 'f22_ticker_STCN', 'f22_ticker_OPRX', 'f22_ticker_FRGI',
                  'f22_ticker_WYNN', 'f22_ticker_EBAY', 'f22_ticker_MNRO', 'f22_ticker_FLMN', 'f22_ticker_ACGL', 'f22_ticker_ASPS', 'f22_ticker_OPBK', 'f22_ticker_CTG',
                  'f22_ticker_FRG', 'f22_ticker_CSOD', 'f22_ticker_TZAC', 'f22_ticker_ITI', 'f22_ticker_MYOK', 'f22_ticker_VRCA', 'f22_ticker_CNTY', 'f22_ticker_OPNT',
                  'f22_ticker_HTLF', 'f22_ticker_MRIN', 'f22_ticker_CME', 'f22_ticker_GWGH', 'f22_ticker_KOR', 'f22_ticker_VRSK', 'f22_ticker_GXGX', 'f22_ticker_ASRT',
                  'f22_ticker_HOL', 'f22_ticker_STAY', 'f22_ticker_DORM', 'f22_ticker_VSTM', 'f22_ticker_CTRE', 'f22_ticker_APPN', 'f22_ticker_SILK', 'f22_ticker_VIVO',
                  'f22_ticker_TBPH', 'f22_ticker_JD', 'f22_ticker_YRCW', 'f22_ticker_NTAP', 'f22_ticker_TVTY', 'f22_ticker_UMPQ', 'f22_ticker_GLUU', 'f22_ticker_CHTR',
                  'f22_ticker_GOVX', 'f22_ticker_WMG', 'f22_ticker_ARTNA', 'f22_ticker_CBIO', 'f22_ticker_CONE', 'f22_ticker_FULT', 'f22_ticker_IPDN', 'f22_ticker_VIRT',
                  'f22_ticker_SIVB', 'f22_ticker_AUBN', 'f22_ticker_CBLI', 'f22_ticker_BELFA', 'f22_ticker_MICT', 'f22_ticker_PRTK', 'f22_ticker_BUSE', 'f22_ticker_WLTW',
                  'f22_ticker_LBRDA', 'f22_ticker_CATB', 'f22_ticker_UCBI', 'f22_ticker_BFC', 'f22_ticker_DISH', 'f22_ticker_GNTY', 'f22_ticker_FARO', 'f22_ticker_ISRG',
                  'f22_ticker_IDEX', 'f22_ticker_MNCL', 'f22_ticker_MNOV', 'f22_ticker_AQST', 'f22_ticker_SSP', 'f22_ticker_PTE', 'f22_ticker_BLU', 'f22_ticker_OPK',
                  'f22_ticker_ASFI', 'f22_ticker_ECPG', 'f22_ticker_ABCB', 'f22_ticker_LTBR', 'f22_ticker_EXLS', 'f22_ticker_CDTX', 'f22_ticker_PBPB', 'f22_ticker_VIAV',
                  'f22_ticker_ATRS', 'f22_ticker_NBIX', 'f22_ticker_IPWR', 'f22_ticker_INBK', 'f22_ticker_CEVA', 'f22_ticker_IDCC', 'f22_ticker_LTRPA', 'f22_ticker_HRZN',
                  'f22_ticker_EWBC', 'f22_ticker_UMBF', 'f22_ticker_MCHP', 'f22_ticker_BPFH', 'f22_ticker_CCXI', 'f22_ticker_GT', 'f22_ticker_MESO', 'f22_ticker_VERB',
                  'f22_ticker_DJCO', 'f22_ticker_ANIX', 'f22_ticker_SPRO', 'f22_ticker_FATE', 'f22_ticker_JKHY', 'f22_ticker_ARYA', 'f22_ticker_PICO', 'f22_ticker_ARAV',
                  'f22_ticker_APTX', 'f22_ticker_CAAS', 'f22_ticker_IMMR', 'f22_ticker_RETA', 'f22_ticker_AMTB', 'f22_ticker_SBSI', 'f22_ticker_SRAC', 'f22_ticker_HOPE',
                  'f22_ticker_SHBI', 'f22_ticker_ANCN', 'f22_ticker_ORMP', 'f22_ticker_INFI', 'f22_ticker_DAKT', 'f22_ticker_DARE', 'f22_ticker_LPLA', 'f22_ticker_AMSF',
                  'f22_ticker_LBC', 'f22_ticker_ALLK', 'f22_ticker_IIVI', 'f22_ticker_CFBK', 'f22_ticker_UPWK', 'f22_ticker_TRS', 'f22_ticker_COOP', 'f22_ticker_ANNX',
                  'f22_ticker_CWBR', 'f22_ticker_MDRX', 'f22_ticker_SBCF', 'f22_ticker_ORIC', 'f22_ticker_NVCR', 'f22_ticker_QADB', 'f22_ticker_AGRX', 'f22_ticker_SAFM',
                  'f22_ticker_TTGT', 'f22_ticker_IMAC', 'f22_ticker_NVUS', 'f22_ticker_TTOO', 'f22_ticker_DRRX', 'f22_ticker_USLM', 'f22_ticker_IMRA', 'f22_ticker_SMMT',
                  'f22_ticker_VXRT', 'f22_ticker_SYNA', 'f22_ticker_APEX', 'f22_ticker_ATNX', 'f22_ticker_EYES', 'f22_ticker_HVBC', 'f22_ticker_NVEE', 'f22_ticker_NCSM',
                  'f22_ticker_PIXY', 'f22_ticker_ATCX', 'f22_ticker_FELE', 'f22_ticker_PCYG', 'f22_ticker_DRNA', 'f22_ticker_FIBK', 'f22_ticker_PDD', 'f22_ticker_FLDM',
                  'f22_ticker_OLED', 'f22_ticker_CBFV', 'f22_ticker_TWST', 'f22_ticker_BRY', 'f22_ticker_TILE', 'f22_ticker_CTRN', 'f22_ticker_CIIC', 'f22_ticker_CCOI',
                  'f22_ticker_BKYI', 'f22_ticker_FRAN', 'f22_ticker_DTIL', 'f22_ticker_ROLL', 'f22_ticker_BECN', 'f22_ticker_AAPL', 'f22_ticker_ODT', 'f22_ticker_VTVT',
                  'f22_ticker_WVFC', 'f22_ticker_PNRG', 'f22_ticker_BIMI', 'f22_ticker_FMNB', 'f22_ticker_DKNG', 'f22_ticker_WLFC', 'f22_ticker_BMRN', 'f22_ticker_STAA',
                  'f22_ticker_LUNA', 'f22_ticker_AIMC', 'f22_ticker_AMTI', 'f22_ticker_LINC', 'f22_ticker_BLBD', 'f22_ticker_AQB', 'f22_ticker_BBQ', 'f22_ticker_OFLX',
                  'f22_ticker_FLXS', 'f22_ticker_SBGI', 'f22_ticker_EOLS', 'f22_ticker_AXAS', 'f22_ticker_DVAX', 'f22_ticker_NUVA', 'f22_ticker_CBAY', 'f22_ticker_ARRY',
                  'f22_ticker_LEVI', 'f22_ticker_SOLY', 'f22_ticker_ACAM', 'f22_ticker_NCNA', 'f22_ticker_BYFC', 'f22_ticker_LOAC', 'f22_ticker_UNAM', 'f22_ticker_AGTC',
                  'f22_ticker_PNFP', 'f22_ticker_AKAM', 'f22_ticker_CAKE', 'f22_ticker_SCOR', 'f22_ticker_VCTR', 'f22_ticker_MGI', 'f22_ticker_AMCX', 'f22_ticker_FOSL',
                  'f22_ticker_BVXV', 'f22_ticker_SRAX', 'f22_ticker_LI', 'f22_ticker_CSSE', 'f22_ticker_SURF', 'f22_ticker_SFIX', 'f22_ticker_SVMK', 'f22_ticker_ECOR',
                  'f22_ticker_FFNW', 'f22_ticker_PESI', 'f22_ticker_RP', 'f22_ticker_ALEC', 'f22_ticker_TLND', 'f22_ticker_IIIV', 'f22_ticker_ACET', 'f22_ticker_HLIT',
                  'f22_ticker_FUSN', 'f22_ticker_VG', 'f22_ticker_FAST', 'f22_ticker_TEUM', 'f22_ticker_SWAV', 'f22_ticker_ENG', 'f22_ticker_BCBP', 'f22_ticker_DRIO',
                  'f22_ticker_CWST', 'f22_ticker_PLMR', 'f22_ticker_TENX', 'f22_ticker_TPIC', 'f22_ticker_AGNC', 'f22_ticker_CCRN', 'f22_ticker_IBKR', 'f22_ticker_TWIN',
                  'f22_ticker_INDB', 'f22_ticker_URGN', 'f22_ticker_TCDA', 'f22_ticker_ORTX', 'f22_ticker_CDMO', 'f22_ticker_SSYS', 'f22_ticker_WKHS', 'f22_ticker_AAON',
                  'f22_ticker_LASR', 'f22_ticker_MITK', 'f22_ticker_ESXB', 'f22_ticker_LOVE', 'f22_ticker_IRMD', 'f22_ticker_JAKK', 'f22_ticker_OBCI', 'f22_ticker_NUAN',
                  'f22_ticker_NATH', 'f22_ticker_SCKT', 'f22_ticker_STMP', 'f22_ticker_HZNP', 'f22_ticker_PS', 'f22_ticker_GLSI', 'f22_ticker_ATRO', 'f22_ticker_HONE',
                  'f22_ticker_NDSN', 'f22_ticker_ADMA', 'f22_ticker_CFMS', 'f22_ticker_EYE', 'f22_ticker_PCYO', 'f22_ticker_ADMP', 'f22_ticker_CAC', 'f22_ticker_SMED',
                  'f22_ticker_ONB', 'f22_ticker_MORN', 'f22_ticker_WTFC', 'f22_ticker_AUPH', 'f22_ticker_RUN', 'f22_ticker_GLG', 'f22_ticker_KOD', 'f22_ticker_CERN',
                  'f22_ticker_SGA', 'f22_ticker_TRUP', 'f22_ticker_TTNP', 'f22_ticker_WEN', 'f22_ticker_ICAD', 'f22_ticker_THBR', 'f22_ticker_EKSO', 'f22_ticker_MTSC',
                  'f22_ticker_CMBM', 'f22_ticker_SESN', 'f22_ticker_GBDC', 'f22_ticker_FWONA', 'f22_ticker_MSVB', 'f22_ticker_CHMA', 'f22_ticker_IRWD', 'f22_ticker_MTRX',
                  'f22_ticker_MFNC', 'f22_ticker_MRVL', 'f22_ticker_EBTC', 'f22_ticker_VIAC', 'f22_ticker_ORGS', 'f22_ticker_MCEP', 'f22_ticker_VTNR', 'f22_ticker_LGIH',
                  'f22_ticker_CJJD', 'f22_ticker_FCBP', 'f22_ticker_LPSN', 'f22_ticker_LMPX', 'f22_ticker_PBCT', 'f22_ticker_STKS', 'f22_ticker_ATRA', 'f22_ticker_AEY',
                  'f22_ticker_RMBL', 'f22_ticker_OFS', 'f22_ticker_INSE', 'f22_ticker_AMGN', 'f22_ticker_TH', 'f22_ticker_IBTX', 'f22_ticker_MGEN', 'f22_ticker_GRMN',
                  'f22_ticker_CROX', 'f22_ticker_THRM', 'f22_ticker_DCT', 'f22_ticker_EPZM', 'f22_ticker_SIGI', 'f22_ticker_APPS', 'f22_ticker_CNBKA', 'f22_ticker_BIOL',
                  'f22_ticker_GAIN', 'f22_ticker_TTCF', 'f22_ticker_CASH', 'f22_ticker_CCCC', 'f22_ticker_SPLK', 'f22_ticker_GPRE', 'f22_ticker_SLM', 'f22_ticker_NWFL',
                  'f22_ticker_NBAC', 'f22_ticker_ONVO', 'f22_ticker_DTEA', 'f22_ticker_TECH', 'f22_ticker_ROCK', 'f22_ticker_CLSK', 'f22_ticker_PTEN', 'f22_ticker_PEGA',
                  'f22_ticker_SNDE', 'f22_ticker_MELI', 'f22_ticker_SCSC', 'f22_ticker_UXIN', 'f22_ticker_RPD', 'f22_ticker_AZRX', 'f22_ticker_NOVN', 'f22_ticker_STRS',
                  'f22_ticker_ALCO', 'f22_ticker_REGI', 'f22_ticker_SYKE', 'f22_ticker_BAND', 'f22_ticker_KVHI', 'f22_ticker_PLYA', 'f22_ticker_ESGR', 'f22_ticker_EQOS',
                  'f22_ticker_OPTN', 'f22_ticker_MRCC', 'f22_ticker_SPTN', 'f22_ticker_OTEL', 'f22_ticker_APLS', 'f22_ticker_NERV', 'f22_ticker_KELYA', 'f22_ticker_KNDI',
                  'f22_ticker_CLSN', 'f22_ticker_INOD', 'f22_ticker_UCTT', 'f22_ticker_COUP', 'f22_ticker_IGAC', 'f22_ticker_LNDC', 'f22_ticker_EXPC', 'f22_ticker_ACER',
                  'f22_ticker_CYAN', 'f22_ticker_MAXN', 'f22_ticker_ALPN', 'f22_ticker_CPRT', 'f22_ticker_ATVI', 'f22_ticker_ACRS', 'f22_ticker_ICHR', 'f22_ticker_CRVS',
                  'f22_ticker_TREE', 'f22_ticker_LNT', 'f22_ticker_PATI', 'f22_ticker_JAGX', 'f22_ticker_RDUS', 'f22_ticker_OCSL', 'f22_ticker_ALRN', 'f22_ticker_PSMT',
                  'f22_ticker_AOUT', 'f22_ticker_FRPH', 'f22_ticker_MU', 'f22_ticker_TPTX', 'f22_ticker_OLB', 'f22_ticker_KINS', 'f22_ticker_IBOC', 'f22_ticker_AINV',
                  'f22_ticker_PTGX', 'f22_ticker_COLB', 'f22_ticker_GEVO', 'f22_ticker_LILA', 'f22_ticker_NBSE', 'f22_ticker_CTHR', 'f22_ticker_WINT', 'f22_ticker_LOCO',
                  'f22_ticker_AUTL', 'f22_ticker_FORR', 'f22_ticker_FOLD', 'f22_ticker_ACTG', 'f22_ticker_GECC', 'f22_ticker_TITN', 'f22_ticker_CRON', 'f22_ticker_EXPE',
                  'f22_ticker_NFBK', 'f22_ticker_GABC', 'f22_ticker_GNPX', 'f22_ticker_PTC', 'f22_ticker_LYRA', 'f22_ticker_HFBL', 'f22_ticker_MBRX', 'f22_ticker_HSTM',
                  'f22_ticker_DMRC', 'f22_ticker_SNOA', 'f22_ticker_MNSB', 'f22_ticker_CYBE', 'f22_ticker_IBEX', 'f22_ticker_SDC', 'f22_ticker_FNLC', 'f22_ticker_RIOT',
                  'f22_ticker_KIN', 'f22_ticker_CARA', 'f22_ticker_TNDM', 'f22_ticker_HDS', 'f22_ticker_CNSL', 'f22_ticker_ETON', 'f22_ticker_MTEM', 'f22_ticker_USCR',
                  'f22_ticker_RILY', 'f22_ticker_CSX', 'f22_ticker_SUNS', 'f22_ticker_CYTK', 'f22_ticker_CBTX', 'f22_ticker_DGLY', 'f22_ticker_XONE', 'f22_ticker_CYRX',
                  'f22_ticker_FNKO', 'f22_ticker_GSBC', 'f22_ticker_SMSI', 'f22_ticker_BDSI', 'f22_ticker_BOOM', 'f22_ticker_SNES', 'f22_ticker_ARDX', 'f22_ticker_BSET',
                  'f22_ticker_AZYO', 'f22_ticker_CDLX', 'f22_ticker_DNKN', 'f22_ticker_CMTL', 'f22_ticker_BSTC', 'f22_ticker_CGNX', 'f22_ticker_AIHS', 'f22_ticker_LCNB',
                  'f22_ticker_IMUX', 'f22_ticker_PAYS', 'f22_ticker_ASUR', 'f22_ticker_ERIE', 'f22_ticker_SYRS', 'f22_ticker_WHLR', 'f22_ticker_CASY', 'f22_ticker_ESPR',
                  'f22_ticker_AQMS', 'f22_ticker_OTIC', 'f22_ticker_HWC', 'f22_ticker_ANGI', 'f22_ticker_VITL', 'f22_ticker_BIOC', 'f22_ticker_TIG', 'f22_ticker_NNOX',
                  'f22_ticker_NHTC', 'f22_ticker_MTBC', 'f22_ticker_RDNT', 'f22_ticker_IPGP', 'f22_ticker_LQDA', 'f22_ticker_FMAO', 'f22_ticker_CDZI', 'f22_ticker_BGCP',
                  'f22_ticker_LEVL', 'f22_ticker_VTGN', 'f22_ticker_DDOG', 'f22_ticker_DWSN', 'f22_ticker_TC', 'f22_ticker_OFED', 'f22_ticker_NVEC', 'f22_ticker_INTU',
                  'f22_ticker_FOXF', 'f22_ticker_OTTR', 'f22_ticker_XENE', 'f22_ticker_HSIC', 'f22_ticker_MRUS', 'f22_ticker_RGEN', 'f22_ticker_LSBK', 'f22_ticker_MDGL',
                  'f22_ticker_ITRI', 'f22_ticker_SATS', 'f22_ticker_NDRA', 'f22_ticker_SPT', 'f22_ticker_PRPO', 'f22_ticker_BCTG', 'f22_ticker_TZOO', 'f22_ticker_CVBF',
                  'f22_ticker_PCRX', 'f22_ticker_FSLR', 'f22_ticker_GILD', 'f22_ticker_ACNB', 'f22_ticker_CDEV', 'f22_ticker_KERN', 'f22_ticker_RXT', 'f22_ticker_CSWI',
                  'f22_ticker_HWCC', 'f22_ticker_AMED', 'f22_ticker_LANC', 'f22_ticker_KRYS', 'f22_ticker_NATI', 'f22_ticker_COHU', 'f22_ticker_NH', 'f22_ticker_NTES',
                  'f22_ticker_SELB', 'f22_ticker_CZWI', 'f22_ticker_LIVX', 'f22_ticker_PACB', 'f22_ticker_AFIB', 'f22_ticker_SIEB', 'f22_ticker_CPAH', 'f22_ticker_IIIN',
                  'f22_ticker_IART', 'f22_ticker_AVDL', 'f22_ticker_BYND', 'f22_ticker_HBT', 'f22_ticker_ADES', 'f22_ticker_IMXI', 'f22_ticker_ICFI', 'f22_ticker_ORBC',
                  'f22_ticker_INVE', 'f22_ticker_SNEX', 'f22_ticker_AMSWA', 'f22_ticker_PAVM', 'f22_ticker_VLDR', 'f22_ticker_CLAR', 'f22_ticker_PLSE', 'f22_ticker_USIO',
                  'f22_ticker_III', 'f22_ticker_SANW', 'f22_ticker_WIRE', 'f22_ticker_CNSP', 'f22_ticker_AMZN', 'f22_ticker_TRMK', 'f22_ticker_ATAX', 'f22_ticker_PAND',
                  'f22_ticker_GVP', 'f22_ticker_THFF', 'f22_ticker_BCML', 'f22_ticker_NDAQ', 'f22_ticker_SIC', 'f22_ticker_GNLN', 'f22_ticker_ARCT', 'f22_ticker_USWS',
                  'f22_ticker_BWFG', 'f22_ticker_LITE', 'f22_ticker_RGLS', 'f22_ticker_INFN', 'f22_ticker_GLPI', 'f22_ticker_LFUS', 'f22_ticker_RADA', 'f22_ticker_IGMS',
                  'f22_ticker_FLUX', 'f22_ticker_GNCA', 'f22_ticker_CTSO', 'f22_ticker_ZIOP', 'f22_ticker_IONS', 'f22_ticker_ON', 'f22_ticker_CSTL', 'f22_ticker_ACHC',
                  'f22_ticker_WABC', 'f22_ticker_NTIC', 'f22_ticker_CLXT', 'f22_ticker_SCHL', 'f22_ticker_TBK', 'f22_ticker_FUTU', 'f22_ticker_XPEL', 'f22_ticker_CREG',
                  'f22_ticker_NEOS', 'f22_ticker_AVAV', 'f22_ticker_XERS', 'f22_ticker_NTNX', 'f22_ticker_NWPX', 'f22_ticker_STX', 'f22_ticker_UFPT', 'f22_ticker_TACT',
                  'f22_ticker_SYBT', 'f22_ticker_ERES', 'f22_ticker_STRA', 'f22_ticker_EXPD', 'f22_ticker_RCKT', 'f22_ticker_MKTX', 'f22_ticker_AXGN', 'f22_ticker_EYEN',
                  'f22_ticker_TOTA', 'f22_ticker_WORX', 'f22_ticker_XPER', 'f22_ticker_BPTH', 'f22_ticker_INFO', 'f22_ticker_ANSS', 'f22_ticker_PVAC', 'f22_ticker_NBTB',
                  'f22_ticker_BASI', 'f22_ticker_ULTA', 'f22_ticker_MGRC', 'f22_ticker_CACC', 'f22_ticker_MBUU', 'f22_ticker_TTWO', 'f22_ticker_VNOM', 'f22_ticker_COKE',
                  'f22_ticker_KPTI', 'f22_ticker_MNKD', 'f22_ticker_DLTH', 'f22_ticker_MSBI', 'f22_ticker_OLLI', 'f22_ticker_CMPS', 'f22_ticker_AMTX', 'f22_ticker_AMRN',
                  'f22_ticker_CMLS', 'f22_ticker_AHCO', 'f22_ticker_CENT', 'f22_ticker_SEAC', 'f22_ticker_ECOL', 'f22_ticker_CRWD', 'f22_ticker_DYN', 'f22_ticker_COCP',
                  'f22_ticker_INAQ', 'f22_ticker_BILI', 'f22_ticker_SRRK', 'f22_ticker_ETNB', 'f22_ticker_SAFT', 'f22_ticker_FCFS', 'f22_ticker_SGBX', 'f22_ticker_MRNA',
                  'f22_ticker_TARA', 'f22_ticker_BMCH', 'f22_ticker_POAI', 'f22_ticker_AAXN', 'f22_ticker_ASO', 'f22_ticker_NDLS', 'f22_ticker_JAMF', 'f22_ticker_AMRS',
                  'f22_ticker_ADAP', 'f22_ticker_CPTA', 'f22_ticker_HWKN', 'f22_ticker_DLTR', 'f22_ticker_MAT', 'f22_ticker_SSPK', 'f22_ticker_FFBW', 'f22_ticker_FBIZ',
                  'f22_ticker_MBOT', 'f22_ticker_LMRK', 'f22_ticker_UEIC', 'f22_ticker_AVID', 'f22_ticker_ADMS', 'f22_ticker_GNUS', 'f22_ticker_OPES', 'f22_ticker_TSRI',
                  'f22_ticker_SPWH', 'f22_ticker_WSTG', 'f22_ticker_RDI', 'f22_ticker_AKUS', 'f22_ticker_SAVA', 'f22_ticker_GFN', 'f22_ticker_QTNT', 'f22_ticker_ENDP',
                  'f22_ticker_MSON', 'f22_ticker_KOSS', 'f22_ticker_BLDR', 'f22_ticker_STEP', 'f22_ticker_FRHC', 'f22_ticker_PEAK', 'f22_ticker_HCAT', 'f22_ticker_FTOC',
                  'f22_ticker_CHFS', 'f22_ticker_URBN', 'f22_ticker_ARTL', 'f22_ticker_INO', 'f22_ticker_CRUS', 'f22_ticker_HAIN', 'f22_ticker_SBFG', 'f22_ticker_PRNB',
                  'f22_ticker_SYNH', 'f22_ticker_INPX', 'f22_ticker_UTHR', 'f22_ticker_TCON', 'f22_ticker_LCUT', 'f22_ticker_USEG', 'f22_ticker_AXNX', 'f22_ticker_BJRI',
                  'f22_ticker_BCRX', 'f22_ticker_NYMT', 'f22_ticker_SCYX', 'f22_ticker_SUPN', 'f22_ticker_TSLA', 'f22_ticker_ATRC', 'f22_ticker_LMST', 'f22_ticker_RAPT',
                  'f22_ticker_AVEO', 'f22_ticker_RDFN', 'f22_ticker_SASR', 'f22_ticker_TRMD', 'f22_ticker_SLNO', 'f22_ticker_BBCP', 'f22_ticker_LGND', 'f22_ticker_PDCO',
                  'f22_ticker_UG', 'f22_ticker_UVSP', 'f22_ticker_PRAA', 'f22_ticker_GIFI', 'f22_ticker_POWL', 'f22_ticker_APEN', 'f22_ticker_CATY', 'f22_ticker_BANR',
                  'f22_ticker_ORRF', 'f22_ticker_ADOM', 'f22_ticker_OPOF', 'f22_ticker_ANDE', 'f22_ticker_METX', 'f22_ticker_MOXC', 'f22_ticker_COLL', 'f22_ticker_SMBC',
                  'f22_ticker_FLGT', 'f22_ticker_FTNT', 'f22_ticker_TBNK', 'f22_ticker_PACW', 'f22_ticker_YNDX', 'f22_ticker_RCKY', 'f22_ticker_PDSB', 'f22_ticker_HAS',
                  'f22_ticker_EBMT', 'f22_ticker_CBRL', 'f22_ticker_CLDX', 'f22_ticker_ODP', 'f22_ticker_TTEK', 'f22_ticker_ACST', 'f22_ticker_WASH', 'f22_ticker_BOCH',
                  'f22_ticker_MOR', 'f22_ticker_CDXS', 'f22_ticker_AXLA', 'f22_ticker_MLVF', 'f22_ticker_KLDO', 'f22_ticker_DFFN', 'f22_ticker_ACOR', 'f22_ticker_CCBG',
                  'f22_ticker_HPK', 'f22_ticker_CMRX', 'f22_ticker_EVFM', 'f22_ticker_NFLX', 'f22_ticker_SGMO', 'f22_ticker_WEYS', 'f22_ticker_EGAN', 'f22_ticker_ZG',
                  'f22_ticker_ZM', 'f22_ticker_CGIX', 'f22_ticker_SAMA', 'f22_ticker_SHYF', 'f22_ticker_VSEC', 'f22_ticker_ITCI', 'f22_ticker_ONCT', 'f22_ticker_DSPG',
                  'f22_ticker_ATXI', 'f22_ticker_IFMK', 'f22_ticker_HTBI', 'f22_ticker_HBNC', 'f22_ticker_PZZA', 'f22_ticker_APTO', 'f22_ticker_HCSG', 'f22_ticker_RGCO',
                  'f22_ticker_EIGR', 'f22_ticker_LOGI', 'f22_ticker_FITB', 'f22_ticker_KRUS', 'f22_ticker_BBIO', 'f22_ticker_ARLP', 'f22_ticker_ELOX', 'f22_ticker_SLGN',
                  'f22_ticker_ABTX', 'f22_ticker_TRCH', 'f22_ticker_HOMB', 'f22_ticker_APXT', 'f22_ticker_HGEN', 'f22_ticker_NURO', 'f22_ticker_ORGO', 'f22_ticker_PTVE',
                  'f22_ticker_MOMO', 'f22_ticker_CHRW', 'f22_ticker_BGFV', 'f22_ticker_CNET', 'f22_ticker_SWTX', 'f22_ticker_VUZI', 'f22_ticker_NKTR', 'f22_ticker_SKYW',
                  'f22_ticker_EDIT', 'f22_ticker_FWRD', 'f22_ticker_IBCP', 'f22_ticker_INVA', 'f22_ticker_AMPH', 'f22_ticker_ABUS', 'f22_ticker_BEEM', 'f22_ticker_PSEC',
                  'f22_ticker_SNBR', 'f22_ticker_UNIT', 'f22_ticker_STIM', 'f22_ticker_OXSQ', 'f22_ticker_CCMP', 'f22_ticker_DOMO', 'f22_ticker_MTCH', 'f22_ticker_IDXX',
                  'f22_ticker_NTRA', 'f22_ticker_UAL', 'f22_ticker_MEDP', 'f22_ticker_CRNX', 'f22_ticker_MGPI', 'f22_ticker_IESC', 'f22_ticker_KXIN', 'f22_ticker_ALTM',
                  'f22_ticker_LECO', 'f22_ticker_XOMA', 'f22_ticker_FROG', 'f22_ticker_FLL', 'f22_ticker_CECE', 'f22_ticker_CRNC', 'f22_ticker_UROV', 'f22_ticker_PXLW',
                  'f22_ticker_HCKT', 'f22_ticker_PLAB', 'f22_ticker_FULC', 'f22_ticker_HGBL', 'f22_ticker_CAR', 'f22_ticker_OBNK', 'f22_ticker_TRNS', 'f22_ticker_YORW',
                  'f22_ticker_AEGN', 'f22_ticker_GEC', 'f22_ticker_VRRM', 'f22_ticker_FB', 'f22_ticker_OTLK', 'f22_ticker_GBT', 'f22_ticker_ICPT', 'f22_ticker_PRGX',
                  'f22_ticker_PRCP', 'f22_ticker_IDRA', 'f22_ticker_RYTM', 'f22_ticker_LOPE', 'f22_ticker_HQI', 'f22_ticker_PLAY', 'f22_ticker_FMTX', 'f22_ticker_SLDB',
                  'f22_ticker_GRIF', 'f22_ticker_MANT', 'f22_ticker_OSW', 'f22_ticker_BLUE', 'f22_ticker_CWCO', 'f22_ticker_ALOT', 'f22_ticker_RTLR', 'f22_ticker_KZR',
                  'f22_ticker_HMNF', 'f22_ticker_POOL', 'f22_ticker_SUMO', 'f22_ticker_FLNT', 'f22_ticker_MORF', 'f22_ticker_INSG', 'f22_ticker_BBI', 'f22_ticker_XELB',
                  'f22_ticker_UONE', 'f22_ticker_PROG', 'f22_ticker_KLAC', 'f22_ticker_BEAM', 'f22_ticker_PLXP', 'f22_ticker_POLA', 'f22_ticker_FUV', 'f22_ticker_MVBF',
                  'f22_ticker_QUMU', 'f22_ticker_AKTS', 'f22_ticker_TWNK', 'f22_ticker_MCRB', 'f22_ticker_PGC', 'f22_ticker_GFED', 'f22_ticker_VBLT', 'f22_ticker_AVCT',
                  'f22_ticker_BGNE', 'f22_ticker_FMBI', 'f22_ticker_PWOD', 'f22_ticker_TENB', 'f22_ticker_NSYS', 'f22_ticker_EHTH', 'f22_ticker_GHIV', 'f22_ticker_PBYI',
                  'f22_ticker_CENX', 'f22_ticker_DADA', 'f22_ticker_NEOG', 'f22_ticker_SMMC', 'f22_ticker_SWKH', 'f22_ticker_MSEX', 'f22_ticker_TACO', 'f22_ticker_LORL',
                  'f22_ticker_CRTD', 'f22_ticker_TCPC', 'f22_ticker_SENEA', 'f22_ticker_HSDT', 'f22_ticker_VCYT', 'f22_ticker_MIST', 'f22_ticker_EYEG', 'f22_ticker_ISIG',
                  'f22_ticker_AVXL', 'f22_ticker_AAME', 'f22_ticker_RICK', 'f22_ticker_CORT', 'f22_ticker_EQIX', 'f22_ticker_FCBC', 'f22_ticker_ANY', 'f22_ticker_KNSL',
                  'f22_ticker_ADXS', 'f22_ticker_SIRI', 'f22_ticker_JAN', 'f22_ticker_IPAR', 'f22_ticker_NEO', 'f22_ticker_PTVCB', 'f22_ticker_UI', 'f22_ticker_CZR',
                  'f22_ticker_INMB', 'f22_ticker_GEOS', 'f22_ticker_CSBR', 'f22_ticker_HTBK', 'f22_ticker_STND', 'f22_ticker_FLIC', 'f22_ticker_GSIT', 'f22_ticker_GPRO',
                  'f22_ticker_INCY', 'f22_ticker_MACK', 'f22_ticker_CLSD', 'f22_ticker_JBLU', 'f22_ticker_ATEX', 'f22_ticker_GDEN', 'f22_ticker_SINT', 'f22_ticker_PRAH',
                  'f22_ticker_LULU', 'f22_ticker_TSBK', 'f22_ticker_VERI', 'f22_ticker_WSC', 'f22_ticker_OCFC', 'f22_ticker_ASTE', 'f22_ticker_SCPH', 'f22_ticker_NTCT',
                  'f22_ticker_JAZZ', 'f22_ticker_SQBG', 'f22_ticker_CERC', 'f22_ticker_RAIL', 'f22_ticker_ALLO', 'f22_ticker_MBWM', 'f22_ticker_GPP', 'f22_ticker_MRTX',
                  'f22_ticker_NVFY', 'f22_ticker_DCPH', 'f22_ticker_HMST', 'f22_ticker_SSNT', 'f22_ticker_GALT', 'f22_ticker_ZEUS', 'f22_ticker_IMBI', 'f22_ticker_ATOM',
                  'f22_ticker_LAMR', 'f22_ticker_ACAD', 'f22_ticker_HOLX', 'f22_ticker_REGN', 'f22_ticker_FCEL', 'f22_ticker_ALTR', 'f22_ticker_ZI', 'f22_ticker_RAVE',
                  'f22_ticker_SPPI', 'f22_ticker_TTEC', 'f22_ticker_BL', 'f22_ticker_CASI', 'f22_ticker_TDAC', 'f22_ticker_WBA', 'f22_ticker_CUE', 'f22_ticker_EGLE',
                  'f22_ticker_SMBK', 'f22_ticker_LXRX', 'f22_ticker_REPL', 'f22_ticker_ALT', 'f22_ticker_CALM', 'f22_ticker_HALL', 'f22_ticker_PSTI', 'f22_ticker_CTIB',
                  'f22_ticker_DMAC', 'f22_ticker_HTBX', 'f22_ticker_MGTX', 'f22_ticker_PRIM', 'f22_ticker_PTON', 'f22_ticker_HELE', 'f22_ticker_EFSC', 'f22_ticker_FANG',
                  'f22_ticker_OSIS', 'f22_ticker_KBAL', 'f22_ticker_PFIN', 'f22_ticker_DGICA', 'f22_ticker_ACIW', 'f22_ticker_AGBA', 'f22_ticker_ATEC', 'f22_ticker_UNTY',
                  'f22_ticker_PKOH', 'f22_ticker_SFBS', 'f22_ticker_CATC', 'f22_ticker_CIDM', 'f22_ticker_VACQ', 'f22_ticker_NXPI', 'f22_ticker_USAP', 'f22_ticker_WAFD',
                  'f22_ticker_CRAI', 'f22_ticker_ESCA', 'f22_ticker_LOB', 'f22_ticker_BLI', 'f22_ticker_WINA', 'f22_ticker_ZSAN', 'f22_ticker_RKDA', 'f22_ticker_AMOT',
                  'f22_ticker_RIDE', 'f22_ticker_TMDX', 'f22_ticker_MYRG', 'f22_ticker_XAIR', 'f22_ticker_HWBK', 'f22_ticker_PFMT', 'f22_ticker_WING', 'f22_ticker_CHUY',
                  'f22_ticker_MLAB', 'f22_ticker_CPSS', 'f22_ticker_NBEV', 'f22_ticker_LPCN', 'f22_ticker_ALAC', 'f22_ticker_HSON', 'f22_ticker_SSB', 'f22_ticker_ASTC',
                  'f22_ticker_PIRS', 'f22_ticker_NWLI', 'f22_ticker_AGIO', 'f22_ticker_PHUN', 'f22_ticker_VYGR', 'f22_ticker_SNGX', 'f22_ticker_<unknown>', 'f22_ticker_EBSB',
                  'f22_ticker_CRSR', 'f22_ticker_MWK', 'f22_ticker_PGEN', 'f22_ticker_CBNK', 'f22_ticker_LAZY', 'f22_ticker_UNFI', 'f22_ticker_RDVT', 'f22_ticker_GNSS',
                  'f22_ticker_KOPN', 'f22_ticker_LPTX', 'f22_ticker_RIGL', 'f22_ticker_AHPI', 'f22_ticker_PRVL', 'f22_ticker_ISTR', 'f22_ticker_IZEA', 'f22_ticker_QBAK',
                  'f22_ticker_AMTD', 'f22_ticker_FVE', 'f22_ticker_LSCC', 'f22_ticker_NFE', 'f22_ticker_LIVE', 'f22_ticker_LMB', 'f22_ticker_MRBK', 'f22_ticker_SBRA',
                  'f22_ticker_TRHC', 'f22_ticker_BRKL', 'f22_ticker_MEIP', 'f22_ticker_MASI', 'f22_ticker_SYNC', 'f22_ticker_EIDX', 'f22_ticker_LNTH', 'f22_ticker_AYTU',
                  'f22_ticker_STRO', 'f22_ticker_ADIL', 'f22_ticker_XBIO', 'f22_ticker_CYRN', 'f22_ticker_BBSI', 'f22_ticker_HCAP', 'f22_ticker_BNTC', 'f22_ticker_TXMD',
                  'f22_ticker_FIZZ', 'f22_ticker_PME', 'f22_ticker_CMPR', 'f22_ticker_CNST', 'f22_ticker_BHTG', 'f22_ticker_WW', 'f22_ticker_ROIC', 'f22_ticker_PRFT',
                  'f22_ticker_NXTC', 'f22_ticker_CERS', 'f22_ticker_CERE', 'f22_ticker_WRLD', 'f22_ticker_UEPS', 'f22_ticker_AMAG', 'f22_ticker_AEIS', 'f22_ticker_MDLZ',
                  'f22_ticker_AMBC', 'f22_ticker_CETV', 'f22_ticker_NCNO', 'f22_ticker_WSBF', 'f22_ticker_FCAP', 'f22_ticker_CSTR', 'f22_ticker_EDUC', 'f22_ticker_NETE',
                  'f22_ticker_KRTX', 'f22_ticker_EDSA', 'f22_ticker_WETF', 'f22_ticker_CTAS', 'f22_ticker_NAVI', 'f22_ticker_ACBI', 'f22_ticker_CREE', 'f22_ticker_ALNA',
                  'f22_ticker_PHIO', 'f22_ticker_ZION', 'f22_ticker_FSBW', 'f22_ticker_EYPT', 'f22_ticker_STXB', 'f22_ticker_UFPI', 'f22_ticker_GWRS', 'f22_ticker_ODFL',
                  'f22_ticker_STLD', 'f22_ticker_REFR', 'f22_ticker_CRSP', 'f22_ticker_QURE', 'f22_ticker_VRA', 'f22_ticker_JNCE', 'f22_ticker_LBAI', 'f22_ticker_SLGG',
                  'f22_ticker_VBFC', 'f22_ticker_TW', 'f22_ticker_FRME', 'f22_ticker_MLND', 'f22_ticker_OXFD', 'f22_ticker_QNST', 'f22_ticker_SHEN', 'f22_ticker_ATNI',
                  'f22_ticker_ARCB', 'f22_ticker_JOUT', 'f22_ticker_BNFT', 'f22_ticker_RCM', 'f22_ticker_KIDS', 'f22_ticker_LINK', 'f22_ticker_GRAY', 'f22_ticker_BZUN',
                  'f22_ticker_MFH', 'f22_ticker_PRTH', 'f22_ticker_DZSI', 'f22_ticker_SSKN', 'f22_ticker_SALM', 'f22_ticker_STBA', 'f22_ticker_TCX', 'f22_ticker_HYAC',
                  'f22_ticker_NWSA', 'f22_ticker_CTXS', 'f22_ticker_CYCN', 'f22_ticker_SCON', 'f22_ticker_PTI', 'f22_ticker_UHAL', 'f22_ticker_ALXN', 'f22_ticker_ONCS',
                  'f22_ticker_TURN', 'f22_ticker_MMSI', 'f22_ticker_CRIS', 'f22_ticker_ICMB', 'f22_ticker_LMNR', 'f22_ticker_NLOK', 'f22_ticker_SAGE', 'f22_ticker_UK',
                  'f22_ticker_HSKA', 'f22_ticker_LVGO', 'f22_ticker_TA', 'f22_ticker_HBCP', 'f22_ticker_REG', 'f22_ticker_OPT', 'f22_ticker_STRM', 'f22_ticker_TWOU',
                  'f22_ticker_MVIS', 'f22_ticker_CNNB', 'f22_ticker_AKBA', 'f22_ticker_REAL', 'f22_ticker_SSTI', 'f22_ticker_AMSC', 'f22_ticker_MTLS', 'f22_ticker_MANH',
                  'f22_ticker_INTZ', 'f22_ticker_COLM', 'f22_ticker_FONR', 'f22_ticker_ANAB', 'f22_ticker_SLP', 'f22_ticker_GWPH', 'f22_ticker_IAC', 'f22_ticker_BSGM',
                  'f22_ticker_HRTX', 'f22_ticker_ENPH', 'f22_ticker_OCUL', 'f22_ticker_BCPC', 'f22_ticker_DGII', 'f22_ticker_MPAA', 'f22_ticker_XLRN', 'f22_ticker_VALU',
                  'f22_ticker_PCH', 'f22_ticker_PBIP', 'f22_ticker_WSTL', 'f22_ticker_KIRK', 'f22_ticker_PFIE', 'f22_ticker_VERY', 'f22_ticker_AVRO', 'f22_ticker_AEYE',
                  'f22_ticker_BLFS', 'f22_ticker_SLCT', 'f22_ticker_BKNG', 'f22_ticker_FENC', 'f22_ticker_DLPN', 'f22_ticker_CSPI', 'f22_ticker_SPWR', 'f22_ticker_TMUS',
                  'f22_ticker_AKRO', 'f22_ticker_IRDM', 'f22_ticker_MIME', 'f22_ticker_NMRD', 'f22_ticker_MERC', 'f22_ticker_PRMW', 'f22_ticker_CRWS', 'f22_ticker_PPD',
                  'f22_ticker_ALBO', 'f22_ticker_LOAN', 'f22_ticker_CDNA', 'f22_ticker_AIRG', 'f22_ticker_GTIM', 'f22_ticker_OSS', 'f22_ticker_BOKF', 'f22_ticker_OFIX',
                  'f22_ticker_CSII', 'f22_ticker_SFNC', 'f22_ticker_MRNS', 'f22_ticker_STRL', 'f22_ticker_BSQR', 'f22_ticker_PVBC', 'f22_ticker_PDCE', 'f22_ticker_DIOD',
                  'f22_ticker_QTRX', 'f22_ticker_GOGO', 'f22_ticker_HYRE', 'f22_ticker_ICBK', 'f22_ticker_CNMD', 'f22_ticker_ALNY', 'f22_ticker_AMKR', 'f22_ticker_LAUR',
                  'f22_ticker_OPHC', 'f22_ticker_HYMC', 'f22_ticker_WDFC', 'f22_ticker_MOBL', 'f22_ticker_SNCA', 'f22_ticker_ADPT', 'f22_ticker_KALU', 'f22_ticker_UBX',
                  'f22_ticker_EVOK', 'f22_ticker_OSTK', 'f22_ticker_NMTR', 'f22_ticker_CBMG', 'f22_ticker_ALTA', 'f22_ticker_RBB', 'f22_ticker_SSSS', 'f22_ticker_TFSL',
                  'f22_ticker_MYSZ', 'f22_ticker_LMFA', 'f22_ticker_USAU', 'f22_ticker_KALA', 'f22_ticker_MPB', 'f22_ticker_LRCX', 'f22_ticker_CINF', 'f22_ticker_MESA',
                  'f22_ticker_PI', 'f22_ticker_MKSI', 'f22_ticker_SDGR', 'f22_ticker_NEON', 'f22_ticker_ALDX', 'f22_ticker_MDCA', 'f22_ticker_CCB', 'f22_ticker_LYFT',
                  'f22_ticker_MRAM', 'f22_ticker_MIK', 'f22_ticker_HEPA', 'f22_ticker_BRQS', 'f22_ticker_AOSL', 'f22_ticker_IEA', 'f22_ticker_MNST', 'f22_ticker_CPHC',
                  'f22_ticker_CRBP', 'f22_ticker_QTT', 'f22_ticker_GBCI', 'f22_ticker_ENSG', 'f22_ticker_AMAT', 'f22_ticker_ADBE', 'f22_ticker_CLIR', 'f22_ticker_HARP',
                  'f22_ticker_QCRH', 'f22_ticker_SLS', 'f22_ticker_EVLO', 'f22_ticker_MKGI', 'f22_ticker_YTEN', 'f22_ticker_FISV', 'f22_ticker_USAK', 'f22_ticker_KALV',
                  'f22_ticker_VRTX', 'f22_ticker_COHR', 'f22_ticker_LFAC', 'f22_ticker_CBMB', 'f22_ticker_FTFT', 'f22_ticker_KMPH', 'f22_ticker_MNTA', 'f22_ticker_RUSHA',
                  'f22_ticker_CPSI', 'f22_ticker_PSTV', 'f22_ticker_MFIN', 'f22_ticker_ICON', 'f22_ticker_SGLB', 'f22_ticker_JACK', 'f22_ticker_TRVN', 'f22_ticker_GAIA',
                  'f22_ticker_CDW', 'f22_ticker_RPRX', 'f22_ticker_ONTX', 'f22_ticker_FSTR', 'f22_ticker_FSFG', 'f22_ticker_NCBS', 'f22_ticker_PRFX', 'f22_ticker_WISA',
                  'f22_ticker_RBCAA', 'f22_ticker_NWBI', 'f22_ticker_AVGO', 'f22_ticker_PDLI', 'f22_ticker_SWKS', 'f22_ticker_STAF', 'f22_ticker_RARE', 'f22_ticker_BCLI',
                  'f22_ticker_VRTU', 'f22_ticker_MDNA', 'f22_ticker_RMTI', 'f22_ticker_LBTYA', 'f22_ticker_BLNK', 'f22_ticker_AMCI', 'f22_ticker_NXGN', 'f22_ticker_MRCY',
                  'f22_ticker_MARA', 'f22_ticker_RMBS', 'f22_ticker_APPF', 'f22_ticker_ENTA', 'f22_ticker_HUBG', 'f22_ticker_FLWS', 'f22_ticker_ACRX', 'f22_ticker_JRVR',
                  'f22_ticker_AMWD', 'f22_ticker_INOV', 'f22_ticker_BLCM', 'f22_ticker_MATW', 'f22_ticker_MGNX', 'f22_ticker_HOOK', 'f22_ticker_FHB', 'f22_ticker_RNST',
                  'f22_ticker_BCDA', 'f22_ticker_ACIA', 'f22_ticker_SYNL', 'f22_ticker_TCBI', 'f22_ticker_VOXX', 'f22_ticker_ASMB', 'f22_ticker_CELC', 'f22_ticker_CTXR',
                  'f22_ticker_TRMT', 'f22_ticker_INTC', 'f22_ticker_HOTH', 'f22_ticker_WATT', 'f22_ticker_CHNG', 'f22_ticker_RAND', 'f22_ticker_RFIL', 'f22_ticker_WDAY',
                  'f22_ticker_CAMP', 'f22_ticker_CLRB', 'f22_ticker_HMSY', 'f22_ticker_HHR', 'f22_ticker_LLNW', 'f22_ticker_PLUS', 'f22_ticker_NK', 'f22_ticker_SRNE',
                  'f22_ticker_EEFT', 'f22_ticker_FLEX', 'f22_ticker_TCF', 'f22_ticker_EFOI', 'f22_ticker_KNSA', 'f22_ticker_MBII', 'f22_ticker_TBLT', 'f22_ticker_OM',
                  'f22_ticker_TYHT', 'f22_ticker_UBOH', 'f22_ticker_LE', 'f22_ticker_ICCH', 'f22_ticker_GP', 'f22_ticker_SPOK', 'f22_ticker_TCFC', 'f22_ticker_TNAV',
                  'f22_ticker_GLIBA', 'f22_ticker_GOOGL', 'f22_ticker_TBBK', 'f22_ticker_ILMN', 'f22_ticker_THMO', 'f22_ticker_EXC', 'f22_ticker_SCVL', 'f22_ticker_JBHT',
                  'f22_ticker_FGEN', 'f22_ticker_CNFR', 'f22_ticker_CRVL', 'f22_ticker_NEWT', 'f22_ticker_FORM', 'f22_ticker_CASA', 'f22_ticker_LUMO', 'f22_ticker_CDXC',
                  'f22_ticker_PCSB', 'f22_ticker_CSGP', 'f22_ticker_FUNC', 'f22_ticker_BTAI', 'f22_ticker_LIVK', 'f22_ticker_HLNE', 'f22_ticker_AZPN', 'f22_ticker_GO',
                  'f22_ticker_CVCO', 'f22_ticker_MCHX', 'f22_ticker_SCHN', 'f22_ticker_TTD', 'f22_ticker_PTCT', 'f22_ticker_BKEP', 'f22_ticker_PKBK', 'f22_ticker_API',
                  'f22_ticker_ECHO', 'f22_ticker_BWEN', 'f22_ticker_KRNT', 'f22_ticker_RVNC', 'f22_ticker_GROW', 'f22_ticker_GOSS', 'f22_ticker_DCOM', 'f22_ticker_SIGA',
                  'f22_ticker_PFLT', 'f22_ticker_AGEN', 'f22_ticker_FBSS', 'f22_ticker_KLXE', 'f22_ticker_GH', 'f22_ticker_GRNQ', 'f22_ticker_GNMK', 'f22_ticker_AAOI',
                  'f22_ticker_VSAT', 'f22_ticker_TEAM', 'f22_ticker_SLRX', 'f22_ticker_SGMS', 'f22_ticker_HTGM', 'f22_ticker_HCCH', 'f22_ticker_ADSK', 'f22_ticker_FRTA',
                  'f22_ticker_ZUMZ', 'f22_ticker_SBBP', 'f22_ticker_BIDU', 'f22_ticker_LWAY', 'f22_ticker_AGFS', 'f22_ticker_ZGNX', 'f22_ticker_EXEL', 'f22_ticker_EVBG',
                  'f22_ticker_LAKE', 'f22_ticker_CRTO', 'f22_ticker_SINO', 'f22_ticker_INGN', 'f22_ticker_POWI', 'f22_ticker_AXSM', 'f22_ticker_SMTX', 'f22_ticker_HBAN',
                  'f22_ticker_SEEL', 'f22_ticker_GTHX', 'f22_ticker_ERII', 'f22_ticker_CHDN', 'f22_ticker_ROST', 'f22_ticker_SITM', 'f22_ticker_CFB', 'f22_ticker_OCSI',
                  'f22_ticker_ACCD', 'f22_ticker_SOHO', 'f22_ticker_LIFE', 'f22_ticker_PLCE', 'f22_ticker_OCC', 'f22_ticker_ROKU', 'f22_ticker_FPAY', 'f22_ticker_VBIV',
                  'f22_ticker_FTHM', 'f22_ticker_CTIC', 'f22_ticker_LTRX', 'f22_ticker_GDS', 'f22_ticker_HCAC', 'f22_ticker_SGH', 'f22_ticker_FARM', 'f22_ticker_BMRA',
                  'f22_ticker_BRP', 'f22_ticker_HBP', 'f22_ticker_SAIA', 'f22_ticker_DISCA', 'f22_ticker_QELL', 'f22_ticker_TRIP', 'f22_ticker_NMIH', 'f22_ticker_CSWC',
                  'f22_ticker_TCMD', 'f22_ticker_APEI', 'f22_ticker_XELA', 'f22_ticker_PAAS', 'f22_ticker_SHSP', 'f22_ticker_PMTS', 'f22_ticker_ACLS', 'f22_ticker_RPAY',
                  'f22_ticker_CHMG', 'f22_ticker_RMG', 'f22_ticker_SELF', 'f22_ticker_GOCO', 'f22_ticker_MHLD', 'f22_ticker_CVLT', 'f22_ticker_TBIO', 'f22_ticker_SPKE',
                  'f22_ticker_NSTG', 'f22_ticker_RMNI', 'f22_ticker_ZYXI', 'f22_ticker_UPLD', 'f22_ticker_MGLN', 'f22_ticker_WWR', 'f22_ticker_AKU', 'f22_ticker_CCLP',
                  'f22_ticker_FCNCA', 'f22_ticker_CLDB', 'f22_ticker_LMNX', 'f22_ticker_NVIV', 'f22_ticker_MTEX', 'f22_ticker_WTRH', 'f22_ticker_NTUS', 'f22_ticker_EA',
                  'f22_ticker_PFG', 'f22_ticker_CHEF', 'f22_ticker_ORLY', 'f22_ticker_MRKR', 'f22_ticker_MDB', 'f22_ticker_HNNA', 'f22_ticker_LAND', 'f22_ticker_TGLS',
                  'f22_ticker_ANIK', 'f22_ticker_DRAD', 'f22_ticker_VIRC', 'f22_ticker_SBAC', 'f22_ticker_LEGH', 'f22_ticker_CLBS', 'f22_ticker_PODD', 'f22_ticker_PMBC',
                  'f22_ticker_CYBR', 'f22_ticker_HMHC', 'f22_ticker_PETS', 'f22_ticker_QDEL', 'f22_ticker_MYGN', 'f22_ticker_NWL', 'f22_ticker_HNRG', 'f22_ticker_PCTI',
                  'f22_ticker_MAR', 'f22_ticker_SGEN', 'f22_ticker_WVE', 'f22_ticker_XP', 'f22_ticker_LPTH', 'f22_ticker_EXPO', 'f22_ticker_NVDA', 'f22_ticker_BRKR',
                  'f22_ticker_COST', 'f22_ticker_RCMT', 'f22_ticker_STNE', 'f22_ticker_EXAS', 'f22_ticker_TARS', 'f22_ticker_EXPI', 'f22_ticker_MARPS', 'f22_ticker_SCPL',
                  'f22_ticker_KHC', 'f22_ticker_OSBC', 'f22_ticker_ZNGA', 'f22_ticker_MMLP', 'f22_ticker_VISL', 'f22_ticker_SYPR', 'f22_ticker_FTDR', 'f22_ticker_BMRC',
                  'f22_ticker_SRDX', 'f22_ticker_MCBC', 'f22_ticker_BPOP', 'f22_ticker_CLBK', 'f22_ticker_AAL', 'f22_ticker_AKER', 'f22_ticker_SHOO', 'f22_ticker_HCCI',
                  'f22_ticker_ASPU', 'f22_ticker_UFCS', 'f22_ticker_SNPS', 'f22_ticker_IMKTA', 'f22_ticker_OVID', 'f22_ticker_CSGS', 'f22_ticker_NBRV', 'f22_ticker_AERI',
                  'f22_ticker_WERN', 'f22_ticker_CFFN']
    from_predict = ['industry_Beverages - Non-Alcoholic', 'category_CEF', 'location_Alberta; Canada', 'f22_sentiment_neg', 'deferredrev', 'ncf', 'scalerevenue_6 - Mega',
                    'location_Michigan; U.S.A', 'location_British Columbia; Canada', 'location_Virginia; U.S.A', 'currency_KRW', 'user_followers_count', 'currency_HKD',
                    'location_Ohio; U.S.A', 'dividends', 'table_SF1', 'nasdaq_day_roi', 'industry_Rental & Leasing Services', 'location_Oregon; U.S.A', 'close',
                    'famaindustry_Candy & Soda', 'ebitda', 'famaindustry_Defense', 'industry_Consumer Electronics', 'industry_<unknown>', 'location_Taiwan', 'industry_Silver',
                    'location_Denmark', 'scalerevenue_<unknown>', 'user_is_translation_enabled', 'category_ADR Common Stock Primary Class', 'currency_BRL',
                    'location_Puerto Rico', 'industry_Diversified Industrials', 'location_Japan', 'location_Ontario; Canada', 'close_SMA_50', 'location_Maine; U.S.A',
                    'location_Finland', 'location_Oklahoma; U.S.A', 'industry_Specialty Finance', 'famaindustry_Entertainment', 'location_Kansas; U.S.A',
                    'category_Canadian Common Stock Primary Class', 'famaindustry_Fabricated Products', 'taxliabilities', 'industry_Uranium', 'location_Quebec; Canada',
                    'debtc', 'industry_Drug Manufacturers - General', 'location_Poland', 'industry_Recreational Vehicles', 'netincdis', 'future_open', 'location_Mexico',
                    'category_Canadian Preferred Stock', 'location_New Brunswick; Canada', 'location_Marshall Islands', 'sector_Consumer Defensive',
                    'location_Nebraska; U.S.A', 'ebitdausd', 'table_<unknown>', 'location_Hong Kong', 'close_SMA_200', 'location_Georgia U.S.A.', 'location_Thailand',
                    'industry_Healthcare Plans', 'user_statuses_count', 'industry_Insurance - Life', 'industry_Diagnostics & Research', 'ppnenet',
                    'famaindustry_Shipping Containers', 'ebitdamargin', 'f22_ticker_in_text', 'industry_Furnishings', 'future_close', 'industry_Restaurants',
                    'industry_REIT - Healthcare Facilities', 'location_Illinois; U.S.A', 'currency_USD', 'industry_Insurance - Reinsurance', 'intexp',
                    'industry_Computer Hardware', 'industry_Broadcasting - TV', 'taxexp', 'sicsector_Nonclassifiable', 'user_listed_count', 'industry_Biotechnology',
                    'location_Utah; U.S.A', 'location_South Carolina; U.S.A', 'days_since', 'industry_Business Equipment', 'industry_Advertising Agencies',
                    'industry_Insurance - Specialty', 'close_SMA_100', 'industry_Software - Infrastructure', 'opinc', 'famaindustry_Recreation', 'category_ADR Stock Warrant',
                    'industry_Medical Devices', 'location_District Of Columbia; U.S.A', 'location_Republic Of Korea', 'industry_Auto & Truck Dealerships',
                    'industry_Credit Services', 'location_New York; U.S.A', 'industry_Financial Conglomerates', 'industry_Utilities - Independent Power Producers',
                    'location_Manitoba; Canada', 'industry_Software - Application', 'industry_Steel', 'industry_Medical Distribution', 'location_Nova Scotia; Canada',
                    'industry_REIT - Retail', 'famaindustry_Beer & Liquor', 'industry_Farm & Heavy Construction Machinery', 'industry_Tools & Accessories', 'category_IDX',
                    'industry_Semiconductors', 'currency_PLN', 'currency_VEF', 'industry_REIT - Industrial', 'industry_Beverages - Soft Drinks', 'industry_Apparel Stores',
                    'siccode', 'ncfbus', 'industry_Infrastructure Operations', 'industry_Airlines', 'location_Saudi Arabia', 'retearn', 'ros', 'currency_CNY', 'date',
                    'industry_Packaged Foods', 'currency_CLP', 'industry_Broadcasting', 'price', 'f22_has_cashtag', 'industry_Oil & Gas Equipment & Services',
                    'sicsector_Construction', 'currentratio', 'currency_PEN', 'currency_<unknown>', 'pe', 'famaindustry_Restaraunts Hotels Motels',
                    'industry_Auto Manufacturers', 'taxassets', 'currency_NZD', 'scalemarketcap_3 - Small', 'scalerevenue_5 - Large', 'location_New Mexico; U.S.A',
                    'fd_day_of_week', 'industry_Apparel Retail', 'industry_Industrial Metals & Minerals', 'sicsector_Retail Trade', 'industry_Luxury Goods',
                    'user_screen_name', 'user_has_extended_profile', 'famaindustry_Medical Equipment', 'industry_Consulting Services', 'industry_Health Care Plans',
                    'location_Argentina', 'location_Saint Vincent And The Grenadines', 'future_high', 'location_New Hampshire; U.S.A', 'category_Domestic Preferred Stock',
                    'user_verified', 'location_Nevada; U.S.A', 'industry_Trucking', 'industry_Staffing & Outsourcing Services', 'debtusd', 'industry_Leisure',
                    'famaindustry_Transportation', 'famaindustry_Construction Materials', 'location_Bahamas', 'industry_Media - Diversified', 'location_Hungary',
                    'industry_Travel Services', 'roe', 'location_South Dakota; U.S.A', 'cor', 'location_Delaware; U.S.A', 'location_Indonesia', 'location_Malta', 'sgna',
                    'industry_Medical Care', 'famaindustry_Insurance', 'industry_Oil & Gas Midstream', 'industry_Lumber & Wood Production', 'famaindustry_Real Estate',
                    'liabilitiesc', 'location_United Republic Of Tanzania', 'fd_day_of_year', 'investments', 'famaindustry_Tobacco Products', 'open',
                    'industry_Banks - Regional', 'sicsector_Transportation Communications Electric Gas And Sanitary Service', 'sicsector_Agriculture Forestry And Fishing',
                    'location_Vermont; U.S.A', 'industry_Farm Products', 'industry_Utilities - Regulated Water', 'industry_Specialty Retail', 'location_Isle Of Man',
                    'sharesbas', 'ebit', 'original_close_price', 'industry_REIT - Hotel & Motel', 'location_<unknown>', 'industry_Home Improvement Retail', 'location_Canada',
                    'famaindustry_Computers', 'location_Costa Rica', 'future_low', 'future_date', 'user_friends_count', 'industry_Coking Coal', 'scalerevenue_3 - Small',
                    "location_Democratic People'S Republic Of Korea", 'location_Texas; U.S.A', 'currency_INR', 'evebitda', 'roic', 'netinc', 'famaindustry_Wholesale',
                    'industry_Packaging & Containers', 'currency_JPY', 'industry_Gold', 'days_util_sale', 'close_SMA_20_days_since_under', 'location_Uruguay', 'low',
                    'famaindustry_Personal Services', 'location_Massachusetts; U.S.A', 'industry_Education & Training Services',
                    'industry_Electronics & Computer Distribution', 'assetsc', 'industry_Oil & Gas E&P', 'ebt', 'tbvps', 'industry_Waste Management',
                    'industry_Banks - Global', 'category_ADR Preferred Stock', 'location_Germany', 'location_Ireland', 'location_Colombia', 'sector_Utilities',
                    'industry_Electronic Gaming & Multimedia', 'location_Malaysia', 'famaindustry_Textiles', 'location_Norway', 'location_Louisiana; U.S.A', 'netmargin',
                    'revenueusd', 'sicsector_<unknown>', 'industry_Utilities - Regulated Electric', 'consolinc', 'famaindustry_Measuring and Control Equipment', 'payoutratio',
                    'location_Missouri; U.S.A', 'location_Cyprus', 'industry_Textile Manufacturing', 'famaindustry_Petroleum and Natural Gas', 'revenue', 'f22_compound_score',
                    'scalemarketcap_2 - Micro', 'famaindustry_Rubber and Plastic Products', 'industry_Entertainment', 'industry_Asset Management', 'sector_Healthcare',
                    'location_Georgia; U.S.A', 'location_Singapore', 'prefdivis', 'industry_REIT - Diversified', 'industry_Real Estate - General', 'industry_Marine Shipping',
                    'scalemarketcap_1 - Nano', 'famaindustry_Construction', 'epsdil', 'cashneq', 'netinccmnusd', 'location_Alaska; U.S.A', 'assetsnc',
                    'sicsector_Finance Insurance And Real Estate', 'industry_REIT - Mortgage', 'location_California; U.S.A', 'location_Austria', 'famaindustry_Communication',
                    'location_Russian Federation', 'debtnc', 'location_North Dakota; U.S.A', 'category_ETD', 'industry_Integrated Freight & Logistics',
                    'famaindustry_Aircraft', 'location_Idaho; U.S.A', 'user_protected', 'intangibles', 'industry_Beverages - Wineries & Distilleries',
                    'location_Maryland; U.S.A', 'location_Hawaii; U.S.A', 'industry_Medical Care Facilities', 'location_Venezuela', 'divyield', 'invcap', 'location_Guam',
                    'industry_Data Storage', 'gp', 'famaindustry_Printing and Publishing', 'cashnequsd', 'location_Wyoming; U.S.A', 'famasector', 'sicsector_Manufacturing',
                    'industry_Resorts & Casinos', 'f22_ticker', 'industry_Other Industrial Metals & Mining', 'industry_Other Precious Metals & Mining',
                    'location_British Virgin Islands', 'eps', 'scalemarketcap_<unknown>', 'category_<unknown>', 'location_Iowa; U.S.A', 'payables', 'location_Peru',
                    'famaindustry_<unknown>', 'location_Jordan', 'industry_Drug Manufacturers - Specialty & Generic', 'location_New Zealand', 'purchase_date',
                    'location_Israel-Jordan', 'accoci', 'table_SEP', 'famaindustry_Chemicals', 'fd_day_of_month', 'sector_Real Estate',
                    'category_Domestic Common Stock Secondary Class', 'famaindustry_Non-Metallic and Industrial Metal Mining', 'location_Greece',
                    'industry_Furnishings Fixtures & Appliances', 'industry_Medical Instruments & Supplies', 'industry_Communication Equipment', 'currency_COP',
                    'industry_Savings & Cooperative Banks', 'industry_Oil & Gas Drilling', 'category_Domestic Common Stock Primary Class', 'sicsector_Public Administration',
                    'close_SMA_200_days_since_under', 'user_geo_enabled', 'currency_CHF', 'grossmargin', 'location_Newfoundland; Canada', 'location_Chile',
                    'industry_Household & Personal Products', 'industry_Business Services', 'industry_Mortgage Finance', 'industry_Specialty Business Services',
                    'location_Cayman Islands', 'location_United States; U.S.A', 'industry_Specialty Industrial Machinery', 'industry_Semiconductor Equipment & Materials',
                    'famaindustry_Business Supplies', 'industry_Shell Companies', 'currency_PHP', 'industry_Semiconductor Memory', 'location_Canada (Federal Level)',
                    'ebitusd', 'f22_id', 'epsusd', 'sicsector_Mining', 'industry_Utilities - Renewable', 'netincnci', 'pb', 'shareswadil', 'evebit', 'location_Maldives',
                    'fcf', 'close_SMA_15_days_since_under', 'sector_Basic Materials', 'famaindustry_Utilities', 'industry_Publishing', 'industry_Industrial Distribution',
                    'famaindustry_Precious Metals', 'industry_REIT - Office', 'sector_Consumer Cyclical', 'industry_Home Improvement Stores', 'industry_Internet Retail',
                    'location_Macau', 'investmentsc', 'location_Netherlands', 'industry_Tobacco', 'industry_REIT - Residential', 'location_Panama',
                    'category_Canadian Stock Warrant', 'industry_Health Information Services', 'liabilities', 'rnd', 'currency_IDR', 'famaindustry_Apparel',
                    'industry_Insurance Brokers', 'location_Turkey', 'sector_Energy', 'location_Brazil', 'category_ETF', 'category_Canadian Common Stock',
                    'industry_Financial Exchanges', 'location_Sweden', 'industry_Conglomerates', 'location_Mississippi; U.S.A', 'industry_Oil & Gas Refining & Marketing',
                    'currency_CAD', 'assets', 'ncfdiv', 'ncff', 'industry_Department Stores', 'location_Pennsylvania; U.S.A', 'industry_Utilities - Regulated Gas',
                    'location_North Carolina; U.S.A', 'industry_Long-Term Care Facilities', 'industry_Chemicals', 'industry_Real Estate Services',
                    'close_SMA_100_days_since_under', 'location_Bermuda', 'industry_Aluminum', 'industry_Solar', 'famaindustry_Shipbuilding Railroad Equipment',
                    'currency_SEK', 'industry_Real Estate - Diversified', 'category_Domestic Stock Warrant', 'industry_Shipping & Ports', 'location_Minnesota; U.S.A',
                    'category_ADR Common Stock', 'industry_Insurance - Property & Casualty', 'industry_Metal Fabrication', 'industry_Aerospace & Defense',
                    'famaindustry_Steel Works Etc', 'high', 'inventory', 'assetturnover', 'deposits', 'equityavg', 'ncfdebt', 'industry_Oil & Gas Integrated', 'equityusd',
                    'ncfcommon', 'ncfx', 'famaindustry_Business Services', 'location_Australia', 'bvps', 'location_Tennessee; U.S.A',
                    'industry_Internet Content & Information', 'dps', 'close_SMA_20', 'famaindustry_Machinery', 'favorite_count', 'famaindustry_Trading',
                    'industry_Business Equipment & Supplies', 'location_United Kingdom', 'ev', 'industry_Gambling', 'currency_NOK', 'location_France', 'sector_Industrials',
                    'industry_Grocery Stores', 'location_Ghana', 'famaindustry_Consumer Goods', 'sps', 'possibly_sensitive', 'industry_Engineering & Construction',
                    'industry_Copper', 'industry_Building Materials', 'sicsector_Services', 'industry_Financial Data & Stock Exchanges', 'location_West Virginia; U.S.A',
                    'f22_sentiment_pos', 'f22_is_tweet_after_hours', 'sector_Communication Services', 'currency_AUD', 'location_Italy', 'f22_num_other_tickers_in_tweet',
                    'famaindustry_Electronic Equipment', 'famaindustry_Coal', 'currency_ARS', 'category_ETN', 'location_Connecticut; U.S.A', 'invcapavg',
                    'scalemarketcap_5 - Large', 'location_Philippines', 'scalerevenue_2 - Micro', 'marketcap', 'tangibles', 'location_Washington; U.S.A',
                    'location_United Arab Emirates', 'industry_Home Furnishings & Fixtures', 'ps1', 'sicsector_Wholesale Trade', 'industry_Computer Systems', 'de',
                    'industry_Paper & Paper Products', 'scalerevenue_1 - Nano', 'industry_Thermal Coal', 'location_New Jersey; U.S.A', 'ncfi', 'pe1', 'industry_Confectioners',
                    'industry_Discount Stores', 'location_Colorado; U.S.A', 'currency_EUR', 'industry_Telecom Services', 'location_China', 'location_Jersey',
                    'location_Israel-Syria', 'location_Wisconsin; U.S.A', 'location_Israel', 'location_Virgin Islands; U.S.', 'industry_Insurance - Diversified',
                    'location_Spain', 'opex', 'location_Alabama; U.S.A', 'industry_Apparel Manufacturing', 'industry_Security & Protection Services', 'location_South Africa',
                    'industry_Auto Parts', 'currency_GBP', 'f22_sentiment_compound', 'industry_Staffing & Employment Services', 'currency_DKK', 'assetsavg', 'location_Monaco',
                    'sector_Technology', 'currency_ZAR', 'industry_Banks - Diversified', 'industry_Banks - Regional - US', 'location_Czech Republic',
                    'industry_Pharmaceutical Retailers', 'location_Saskatchewan; Canada', 'roa', 'scalerevenue_4 - Mid', 'close_SMA_15', 'shareswa', 'equity', 'ncfinv',
                    'location_Switzerland', 'industry_Footwear & Accessories', 'location_Kentucky; U.S.A', 'famaindustry_Almost Nothing', 'industry_Broadcasting - Radio',
                    'location_Gibraltar', 'ncfo', 'receivables', 'workingcapital', 'currency_MYR', 'depamor', 'location_Montana; U.S.A', 'sbcomp', 'currency_TWD',
                    'f22_sentiment_neu', 'sector_Financial Services', 'fxusd', 'user_follow_request_sent', 'retweet_count', 'currency_ILS',
                    'industry_Electrical Equipment & Parts', 'location_Arkansas; U.S.A', 'industry_Lodging', 'industry_Building Products & Equipment',
                    'industry_Food Distribution', 'industry_REIT - Specialty', 'location_Iceland', 'volume', 'category_Domestic Common Stock', 'famaindustry_Food Products',
                    'netinccmn', 'industry_Information Technology Services', 'location_Belgium', 'famaindustry_Healthcare', 'industry_Farm & Construction Equipment',
                    'sharefactor', 'industry_Real Estate - Development', 'close_SMA_50_days_since_under', 'currency_RUB', 'liabilitiesnc', 'location_India',
                    'location_Arizona; U.S.A', 'currency_MXN', 'industry_Airports & Air Services', 'industry_Utilities - Diversified', 'ps',
                    'industry_Residential Construction', 'famaindustry_Pharmaceutical Products', 'industry_Electronic Components', 'closeunadj',
                    'industry_Specialty Chemicals', 'table_SFP', 'debt', 'location_Luxembourg', 'table_SF3B', 'location_Netherlands Antilles', 'location_Rhode Island; U.S.A',
                    'famaindustry_Automobiles and Trucks', 'industry_Railroads', 'industry_Beverages - Brewers', 'fcfps', 'location_Florida; U.S.A', 'famaindustry_Banking',
                    'capex', 'location_Mauritius', 'famaindustry_Electrical Equipment', 'scalemarketcap_4 - Mid', 'industry_Agricultural Inputs', 'sector_<unknown>',
                    'location_Unknown', 'location_Guernsey', 'industry_Coal', 'industry_Personal Services', 'industry_Drug Manufacturers - Major', 'famaindustry_Retail',
                    'location_Oman', 'location_Indiana; U.S.A', 'industry_Pollution & Treatment Controls', 'scalemarketcap_6 - Mega', 'famaindustry_Agriculture',
                    'category_ADR Common Stock Secondary Class', 'industry_Capital Markets', 'currency_TRY', 'investmentsnc', 'industry_Scientific & Technical Instruments',
                    'f22_day_tweet_count', 'rating', 'rank', 'rating_age_days', 'target_price', 'rank_roi', 'f22_ticker_SEDG', 'f22_ticker_LQDT', 'f22_ticker_ALGN',
                    'f22_ticker_ARVN', 'f22_ticker_CELH', 'f22_ticker_LIND', 'f22_ticker_SPSC', 'f22_ticker_PWFL', 'f22_ticker_GLAD', 'f22_ticker_JJSF', 'f22_ticker_GOOD',
                    'f22_ticker_TIPT', 'f22_ticker_ADI', 'f22_ticker_FLIR', 'f22_ticker_TGTX', 'f22_ticker_TXN', 'f22_ticker_STRT', 'f22_ticker_MBIN', 'f22_ticker_CLNE',
                    'f22_ticker_PCB', 'f22_ticker_CRTX', 'f22_ticker_AMBA', 'f22_ticker_CYCC', 'f22_ticker_CAPR', 'f22_ticker_NODK', 'f22_ticker_CVET', 'f22_ticker_FIII',
                    'f22_ticker_ELSE', 'f22_ticker_PSNL', 'f22_ticker_CARV', 'f22_ticker_DOCU', 'f22_ticker_FREQ', 'f22_ticker_NOVT', 'f22_ticker_FSRV', 'f22_ticker_LFVN',
                    'f22_ticker_UBSI', 'f22_ticker_INSM', 'f22_ticker_SONA', 'f22_ticker_QRVO', 'f22_ticker_BOWX', 'f22_ticker_FNCB', 'f22_ticker_CRDF', 'f22_ticker_MMAC',
                    'f22_ticker_QLYS', 'f22_ticker_AVNW', 'f22_ticker_BRKS', 'f22_ticker_FIVN', 'f22_ticker_PAYX', 'f22_ticker_OBLN', 'f22_ticker_INMD', 'f22_ticker_ALIM',
                    'f22_ticker_CLCT', 'f22_ticker_VCEL', 'f22_ticker_ISEE', 'f22_ticker_AEMD', 'f22_ticker_GTLS', 'f22_ticker_ETSY', 'f22_ticker_ZIXI', 'f22_ticker_CALA',
                    'f22_ticker_PRSC', 'f22_ticker_NAKD', 'f22_ticker_AAWW', 'f22_ticker_RGNX', 'f22_ticker_PCTY', 'f22_ticker_MOTS', 'f22_ticker_HAYN', 'f22_ticker_ADP',
                    'f22_ticker_ZS', 'f22_ticker_NTLA', 'f22_ticker_AESE', 'f22_ticker_FISI', 'f22_ticker_AXTI', 'f22_ticker_LUNG', 'f22_ticker_BBBY', 'f22_ticker_DBX',
                    'f22_ticker_FBMS', 'f22_ticker_JCOM', 'f22_ticker_ABMD', 'f22_ticker_HEAR', 'f22_ticker_TROW', 'f22_ticker_CD', 'f22_ticker_IHRT', 'f22_ticker_CBAT',
                    'f22_ticker_GTYH', 'f22_ticker_WSFS', 'f22_ticker_NIU', 'f22_ticker_PRPL', 'f22_ticker_ARNA', 'f22_ticker_MLAC', 'f22_ticker_MSFT', 'f22_ticker_EZPW',
                    'f22_ticker_DNLI', 'f22_ticker_KE', 'f22_ticker_DLHC', 'f22_ticker_TNXP', 'f22_ticker_SUNW', 'f22_ticker_STOK', 'f22_ticker_OPRA', 'f22_ticker_AFIN',
                    'f22_ticker_OPCH', 'f22_ticker_OVLY', 'f22_ticker_LCA', 'f22_ticker_WIX', 'f22_ticker_EPAY', 'f22_ticker_PAE', 'f22_ticker_RMR', 'f22_ticker_COWN',
                    'f22_ticker_ESSA', 'f22_ticker_CVAC', 'f22_ticker_NNBR', 'f22_ticker_CMCSA', 'f22_ticker_ASML', 'f22_ticker_PRTS', 'f22_ticker_TXRH', 'f22_ticker_ASND',
                    'f22_ticker_MPWR', 'f22_ticker_CFII', 'f22_ticker_VFF', 'f22_ticker_MTSI', 'f22_ticker_VERO', 'f22_ticker_SABR', 'f22_ticker_OTEX', 'f22_ticker_AVT',
                    'f22_ticker_PROV', 'f22_ticker_FDUS', 'f22_ticker_IIN', 'f22_ticker_SONO', 'f22_ticker_CPST', 'f22_ticker_GRPN', 'f22_ticker_MIND', 'f22_ticker_RELL',
                    'f22_ticker_VRNT', 'f22_ticker_QRHC', 'f22_ticker_CLLS', 'f22_ticker_EGOV', 'f22_ticker_TTMI', 'f22_ticker_NTGR', 'f22_ticker_ITRM', 'f22_ticker_AYRO',
                    'f22_ticker_TLGT', 'f22_ticker_AVCO', 'f22_ticker_NVAX', 'f22_ticker_FAT', 'f22_ticker_AZN', 'f22_ticker_CMLF', 'f22_ticker_XNCR', 'f22_ticker_PFPT',
                    'f22_ticker_SND', 'f22_ticker_TESS', 'f22_ticker_SG', 'f22_ticker_VERU', 'f22_ticker_TSCO', 'f22_ticker_MSTR', 'f22_ticker_REKR', 'f22_ticker_PINC',
                    'f22_ticker_BIGC', 'f22_ticker_ZBRA', 'f22_ticker_DMTK', 'f22_ticker_FTEK', 'f22_ticker_SFT', 'f22_ticker_QLGN', 'f22_ticker_FLXN', 'f22_ticker_FIVE',
                    'f22_ticker_BANF', 'f22_ticker_IRBT', 'f22_ticker_CPSH', 'f22_ticker_NRC', 'f22_ticker_CSCO', 'f22_ticker_OKTA', 'f22_ticker_IEC', 'f22_ticker_FBIO',
                    'f22_ticker_SMTC', 'f22_ticker_XLNX', 'f22_ticker_IMVT', 'f22_ticker_BLIN', 'f22_ticker_DTSS', 'f22_ticker_IOVA', 'f22_ticker_PENN', 'f22_ticker_DYNT',
                    'f22_ticker_KLIC', 'f22_ticker_NXST', 'f22_ticker_PFC', 'f22_ticker_CTSH', 'f22_ticker_AVO', 'f22_ticker_SRPT', 'f22_ticker_LMAT', 'f22_ticker_ATRI',
                    'f22_ticker_LOOP', 'f22_ticker_HA', 'f22_ticker_BEAT', 'f22_ticker_MGTA', 'f22_ticker_CG', 'f22_ticker_CDK', 'f22_ticker_ABEO', 'f22_ticker_ARWR',
                    'f22_ticker_LKQ', 'f22_ticker_CDNS', 'f22_ticker_HTLD', 'f22_ticker_NKLA', 'f22_ticker_ONEM', 'f22_ticker_ESTA', 'f22_ticker_FRPT', 'f22_ticker_BNGO',
                    'f22_ticker_ZAGG', 'f22_ticker_PEP', 'f22_ticker_FFIV', 'f22_ticker_MCRI', 'f22_ticker_SIBN', 'f22_ticker_VRSN', 'f22_ticker_TYME', 'f22_ticker_HALO',
                    'f22_ticker_EBIX', 'f22_ticker_NMRK', 'f22_ticker_VIR', 'f22_ticker_EAR', 'f22_ticker_ATOS', 'f22_ticker_IROQ', 'f22_ticker_SAL', 'f22_ticker_HOFV',
                    'f22_ticker_WDC', 'f22_ticker_NGM', 'f22_ticker_TSC', 'f22_ticker_APVO', 'f22_ticker_GSHD', 'f22_ticker_RIVE', 'f22_ticker_BPMC', 'f22_ticker_CTMX',
                    'f22_ticker_VRNS', 'f22_ticker_NICK', 'f22_ticker_QCOM', 'f22_ticker_WWD', 'f22_ticker_PATK', 'f22_ticker_KYMR', 'f22_ticker_FEYE', 'f22_ticker_CPRX',
                    'f22_ticker_VYNE', 'f22_ticker_SBUX', 'f22_ticker_TXG', 'f22_ticker_KTRA', 'f22_ticker_XEL', 'f22_ticker_SONM', 'f22_ticker_ONCR', 'f22_ticker_RVSB',
                    'f22_ticker_SREV', 'f22_ticker_KFRC', 'f22_ticker_OPTT', 'f22_ticker_TER', 'f22_ticker_BCEL', 'f22_ticker_CLGN', 'f22_ticker_ARCC', 'f22_ticker_MBIO',
                    'f22_ticker_PLUG', 'f22_ticker_ACMR', 'f22_ticker_KROS', 'f22_ticker_PGNY', 'f22_ticker_EMKR', 'f22_ticker_STTK', 'f22_ticker_UIHC', 'f22_ticker_FOXA',
                    'f22_ticker_EQ', 'f22_ticker_SEER', 'f22_ticker_ATSG', 'f22_ticker_MARK', 'f22_ticker_RNET', 'f22_ticker_ARKR', 'f22_ticker_ACTC', 'f22_ticker_VC',
                    'f22_ticker_RESN', 'f22_ticker_EVOL', 'f22_ticker_NNDM', 'f22_ticker_FIXX', 'f22_ticker_IRIX', 'f22_ticker_IVA', 'f22_ticker_AMD', 'f22_ticker_HIBB',
                    'f22_ticker_AVGR', 'f22_ticker_XSPA', 'f22_ticker_CREX', 'f22_ticker_MXIM', 'f22_ticker_AUTO', 'f22_ticker_PCVX', 'f22_ticker_NPA', 'f22_ticker_BOXL',
                    'f22_ticker_CETX', 'f22_ticker_LJPC', 'f22_ticker_MRSN', 'f22_ticker_BHF', 'f22_ticker_JVA', 'f22_ticker_SECO', 'f22_ticker_TRMB', 'f22_ticker_CSCW',
                    'f22_ticker_ABIO', 'f22_ticker_SP', 'f22_ticker_TLRY', 'f22_ticker_ADUS', 'f22_ticker_DENN', 'f22_ticker_MRTN', 'f22_ticker_RPTX', 'f22_ticker_PYPL',
                    'f22_ticker_ENTG', 'f22_ticker_ALRM', 'f22_ticker_RNWK', 'f22_ticker_BCOV', 'f22_ticker_GSKY', 'f22_ticker_OPRX', 'f22_ticker_WYNN', 'f22_ticker_EBAY',
                    'f22_ticker_TWCT', 'f22_ticker_APRE', 'f22_ticker_ITI', 'f22_ticker_VRCA', 'f22_ticker_CME', 'f22_ticker_GWGH', 'f22_ticker_KOR', 'f22_ticker_VRSK',
                    'f22_ticker_GXGX', 'f22_ticker_ASRT', 'f22_ticker_HOL', 'f22_ticker_STAY', 'f22_ticker_DORM', 'f22_ticker_MCFT', 'f22_ticker_VSTM', 'f22_ticker_CTRE',
                    'f22_ticker_APPN', 'f22_ticker_SILK', 'f22_ticker_VIVO', 'f22_ticker_JD', 'f22_ticker_NTAP', 'f22_ticker_GLUU', 'f22_ticker_CHTR', 'f22_ticker_GOVX',
                    'f22_ticker_WMG', 'f22_ticker_ARTNA', 'f22_ticker_CBIO', 'f22_ticker_CONE', 'f22_ticker_FULT', 'f22_ticker_MDRR', 'f22_ticker_IPDN', 'f22_ticker_VIRT',
                    'f22_ticker_SIVB', 'f22_ticker_CURI', 'f22_ticker_CBLI', 'f22_ticker_MICT', 'f22_ticker_LBRDA', 'f22_ticker_CATB', 'f22_ticker_BFC', 'f22_ticker_DISH',
                    'f22_ticker_FARO', 'f22_ticker_ISRG', 'f22_ticker_IDEX', 'f22_ticker_MNOV', 'f22_ticker_PTE', 'f22_ticker_BLU', 'f22_ticker_ECPG', 'f22_ticker_LTBR',
                    'f22_ticker_EXLS', 'f22_ticker_VIAV', 'f22_ticker_HRMY', 'f22_ticker_NBIX', 'f22_ticker_CEVA', 'f22_ticker_LTRPA', 'f22_ticker_EWBC', 'f22_ticker_UMBF',
                    'f22_ticker_GT', 'f22_ticker_MESO', 'f22_ticker_VERB', 'f22_ticker_DJCO', 'f22_ticker_NARI', 'f22_ticker_SPRO', 'f22_ticker_FATE', 'f22_ticker_JKHY',
                    'f22_ticker_PICO', 'f22_ticker_APTX', 'f22_ticker_IMMR', 'f22_ticker_RETA', 'f22_ticker_AMTB', 'f22_ticker_SRAC', 'f22_ticker_HOPE', 'f22_ticker_GRIL',
                    'f22_ticker_HBIO', 'f22_ticker_SHBI', 'f22_ticker_ANCN', 'f22_ticker_INFI', 'f22_ticker_DARE', 'f22_ticker_LPLA', 'f22_ticker_LBC', 'f22_ticker_ALLK',
                    'f22_ticker_IIVI', 'f22_ticker_UPWK', 'f22_ticker_TRS', 'f22_ticker_COOP', 'f22_ticker_HDSN', 'f22_ticker_ANNX', 'f22_ticker_CWBR', 'f22_ticker_NVCR',
                    'f22_ticker_RSSS', 'f22_ticker_QADB', 'f22_ticker_TTGT', 'f22_ticker_IMAC', 'f22_ticker_OTRK', 'f22_ticker_SMMT', 'f22_ticker_VXRT', 'f22_ticker_SYNA',
                    'f22_ticker_EYES', 'f22_ticker_PIXY', 'f22_ticker_ATCX', 'f22_ticker_PCYG', 'f22_ticker_PDD', 'f22_ticker_TWST', 'f22_ticker_ACEV', 'f22_ticker_ALSK',
                    'f22_ticker_CTRN', 'f22_ticker_CIIC', 'f22_ticker_DTIL', 'f22_ticker_BECN', 'f22_ticker_AAPL', 'f22_ticker_ODT', 'f22_ticker_VTVT', 'f22_ticker_DKNG',
                    'f22_ticker_STAA', 'f22_ticker_MRLN', 'f22_ticker_LUNA', 'f22_ticker_AQB', 'f22_ticker_BBQ', 'f22_ticker_CMCT', 'f22_ticker_SBGI', 'f22_ticker_ARRY',
                    'f22_ticker_OXBR', 'f22_ticker_LOAC', 'f22_ticker_PNTG', 'f22_ticker_UNAM', 'f22_ticker_AGTC', 'f22_ticker_PNFP', 'f22_ticker_AKAM', 'f22_ticker_CAKE',
                    'f22_ticker_VCTR', 'f22_ticker_MGI', 'f22_ticker_SWBI', 'f22_ticker_SRAX', 'f22_ticker_LI', 'f22_ticker_CSSE', 'f22_ticker_SURF', 'f22_ticker_SFIX',
                    'f22_ticker_SVMK', 'f22_ticker_FFNW', 'f22_ticker_RP', 'f22_ticker_FUSN', 'f22_ticker_ANGO', 'f22_ticker_VG', 'f22_ticker_SWAV', 'f22_ticker_ENG',
                    'f22_ticker_BCBP', 'f22_ticker_DRIO', 'f22_ticker_HLIO', 'f22_ticker_PLMR', 'f22_ticker_TPIC', 'f22_ticker_AGNC', 'f22_ticker_IBKR', 'f22_ticker_TWIN',
                    'f22_ticker_HCDI', 'f22_ticker_TCDA', 'f22_ticker_ALVR', 'f22_ticker_CDMO', 'f22_ticker_SSYS', 'f22_ticker_WKHS', 'f22_ticker_AAON', 'f22_ticker_MITK',
                    'f22_ticker_LOVE', 'f22_ticker_JAKK', 'f22_ticker_NUAN', 'f22_ticker_SCKT', 'f22_ticker_STMP', 'f22_ticker_HZNP', 'f22_ticker_PS', 'f22_ticker_GLSI',
                    'f22_ticker_HONE', 'f22_ticker_CFMS', 'f22_ticker_EYE', 'f22_ticker_ADMP', 'f22_ticker_PRAX', 'f22_ticker_CAC', 'f22_ticker_MORN', 'f22_ticker_AUPH',
                    'f22_ticker_RUN', 'f22_ticker_GLG', 'f22_ticker_KOD', 'f22_ticker_TRUP', 'f22_ticker_WEN', 'f22_ticker_ICAD', 'f22_ticker_THBR', 'f22_ticker_MTSC',
                    'f22_ticker_OMEX', 'f22_ticker_GBDC', 'f22_ticker_MSVB', 'f22_ticker_VRM', 'f22_ticker_OCGN', 'f22_ticker_MRVL', 'f22_ticker_VIAC', 'f22_ticker_VTNR',
                    'f22_ticker_LGIH', 'f22_ticker_XGN', 'f22_ticker_CJJD', 'f22_ticker_LPSN', 'f22_ticker_LMPX', 'f22_ticker_PBCT', 'f22_ticker_KTOS', 'f22_ticker_ATRA',
                    'f22_ticker_RMBL', 'f22_ticker_OFS', 'f22_ticker_XENT', 'f22_ticker_AMGN', 'f22_ticker_TH', 'f22_ticker_IBTX', 'f22_ticker_GRMN', 'f22_ticker_CROX',
                    'f22_ticker_THRM', 'f22_ticker_EPZM', 'f22_ticker_APPS', 'f22_ticker_CNBKA', 'f22_ticker_TTCF', 'f22_ticker_BIOL', 'f22_ticker_CASH', 'f22_ticker_SPLK',
                    'f22_ticker_ALXO', 'f22_ticker_GPRE', 'f22_ticker_SLM', 'f22_ticker_NWFL', 'f22_ticker_NBAC', 'f22_ticker_ONVO', 'f22_ticker_DTEA', 'f22_ticker_TECH',
                    'f22_ticker_CLSK', 'f22_ticker_PEGA', 'f22_ticker_WNEB', 'f22_ticker_MELI', 'f22_ticker_UXIN', 'f22_ticker_RPD', 'f22_ticker_NOVN', 'f22_ticker_ALCO',
                    'f22_ticker_REGI', 'f22_ticker_NKTX', 'f22_ticker_BAND', 'f22_ticker_SYKE', 'f22_ticker_EQOS', 'f22_ticker_OPTN', 'f22_ticker_SPTN', 'f22_ticker_APLS',
                    'f22_ticker_NERV', 'f22_ticker_KELYA', 'f22_ticker_KNDI', 'f22_ticker_CLSN', 'f22_ticker_COUP', 'f22_ticker_IGAC', 'f22_ticker_LNDC', 'f22_ticker_EXPC',
                    'f22_ticker_ACER', 'f22_ticker_MAXN', 'f22_ticker_ALPN', 'f22_ticker_CPRT', 'f22_ticker_ATVI', 'f22_ticker_ACRS', 'f22_ticker_LNT', 'f22_ticker_JAGX',
                    'f22_ticker_ALRN', 'f22_ticker_FRPH', 'f22_ticker_MU', 'f22_ticker_TPTX', 'f22_ticker_OLB', 'f22_ticker_PTGX', 'f22_ticker_COLB', 'f22_ticker_GEVO',
                    'f22_ticker_NBSE', 'f22_ticker_WINT', 'f22_ticker_LOCO', 'f22_ticker_FOLD', 'f22_ticker_GECC', 'f22_ticker_CRON', 'f22_ticker_EXPE', 'f22_ticker_GNPX',
                    'f22_ticker_PTC', 'f22_ticker_GHSI', 'f22_ticker_LYRA', 'f22_ticker_SNOA', 'f22_ticker_IBEX', 'f22_ticker_RIOT', 'f22_ticker_KIN', 'f22_ticker_CARA',
                    'f22_ticker_TNDM', 'f22_ticker_MTEM', 'f22_ticker_XONE', 'f22_ticker_CSX', 'f22_ticker_CBTX', 'f22_ticker_DGLY', 'f22_ticker_THCB', 'f22_ticker_FNKO',
                    'f22_ticker_SMSI', 'f22_ticker_MCAC', 'f22_ticker_ARDX', 'f22_ticker_CDLX', 'f22_ticker_CGNX', 'f22_ticker_AIHS', 'f22_ticker_LCNB', 'f22_ticker_PAYS',
                    'f22_ticker_ASUR', 'f22_ticker_SYRS', 'f22_ticker_WHLR', 'f22_ticker_OTIC', 'f22_ticker_ESPR', 'f22_ticker_AQMS', 'f22_ticker_NBRV', 'f22_ticker_ANGI',
                    'f22_ticker_BIOC', 'f22_ticker_TIG', 'f22_ticker_NNOX', 'f22_ticker_NUZE', 'f22_ticker_MTBC', 'f22_ticker_IPGP', 'f22_ticker_LQDA', 'f22_ticker_HROW',
                    'f22_ticker_VTGN', 'f22_ticker_DDOG', 'f22_ticker_TC', 'f22_ticker_OFED', 'f22_ticker_NVEC', 'f22_ticker_INTU', 'f22_ticker_FOXF', 'f22_ticker_HSIC',
                    'f22_ticker_MRUS', 'f22_ticker_RGEN', 'f22_ticker_ITRI', 'f22_ticker_SPT', 'f22_ticker_BCTG', 'f22_ticker_CVBF', 'f22_ticker_FSLR', 'f22_ticker_PTRS',
                    'f22_ticker_GILD', 'f22_ticker_ACNB', 'f22_ticker_KERN', 'f22_ticker_RXT', 'f22_ticker_AMED', 'f22_ticker_SMPL', 'f22_ticker_LANC', 'f22_ticker_NTES',
                    'f22_ticker_COHU', 'f22_ticker_NH', 'f22_ticker_CZWI', 'f22_ticker_LIVX', 'f22_ticker_PACB', 'f22_ticker_IART', 'f22_ticker_AMST', 'f22_ticker_BYND',
                    'f22_ticker_HBT', 'f22_ticker_ICFI', 'f22_ticker_SNEX', 'f22_ticker_VLDR', 'f22_ticker_CLAR', 'f22_ticker_PLSE', 'f22_ticker_USIO', 'f22_ticker_CNSP',
                    'f22_ticker_AMZN', 'f22_ticker_ATAX', 'f22_ticker_FCCO', 'f22_ticker_PAND', 'f22_ticker_NDAQ', 'f22_ticker_ARCT', 'f22_ticker_USWS', 'f22_ticker_BWFG',
                    'f22_ticker_LITE', 'f22_ticker_RGLS', 'f22_ticker_INFN', 'f22_ticker_IGMS', 'f22_ticker_FLUX', 'f22_ticker_ZIOP', 'f22_ticker_ON', 'f22_ticker_CSTL',
                    'f22_ticker_ACHC', 'f22_ticker_NTIC', 'f22_ticker_FUTU', 'f22_ticker_XPEL', 'f22_ticker_NEOS', 'f22_ticker_AVAV', 'f22_ticker_NTNX', 'f22_ticker_STX',
                    'f22_ticker_UFPT', 'f22_ticker_TACT', 'f22_ticker_SYBT', 'f22_ticker_STRA', 'f22_ticker_RCKT', 'f22_ticker_MKTX', 'f22_ticker_AXGN', 'f22_ticker_XPER',
                    'f22_ticker_BPTH', 'f22_ticker_GBLI', 'f22_ticker_NBTB', 'f22_ticker_BASI', 'f22_ticker_ULTA', 'f22_ticker_MBUU', 'f22_ticker_KPTI', 'f22_ticker_MNKD',
                    'f22_ticker_OLLI', 'f22_ticker_CMPS', 'f22_ticker_AMTX', 'f22_ticker_AHCO', 'f22_ticker_CRWD', 'f22_ticker_DYN', 'f22_ticker_COCP', 'f22_ticker_BILI',
                    'f22_ticker_SGBX', 'f22_ticker_MRNA', 'f22_ticker_BMCH', 'f22_ticker_POAI', 'f22_ticker_AAXN', 'f22_ticker_NDLS', 'f22_ticker_JAMF', 'f22_ticker_AMRS',
                    'f22_ticker_ADAP', 'f22_ticker_DLTR', 'f22_ticker_MAT', 'f22_ticker_FFBW', 'f22_ticker_FBIZ', 'f22_ticker_MBOT', 'f22_ticker_LMRK', 'f22_ticker_AVID',
                    'f22_ticker_GNUS', 'f22_ticker_TSRI', 'f22_ticker_SPWH', 'f22_ticker_RDI', 'f22_ticker_KBNT', 'f22_ticker_AKUS', 'f22_ticker_SAVA', 'f22_ticker_GFN',
                    'f22_ticker_QTNT', 'f22_ticker_ENDP', 'f22_ticker_MSON', 'f22_ticker_KOSS', 'f22_ticker_BLDR', 'f22_ticker_STEP', 'f22_ticker_FRHC', 'f22_ticker_HCAT',
                    'f22_ticker_FTOC', 'f22_ticker_ISSC', 'f22_ticker_URBN', 'f22_ticker_INO', 'f22_ticker_HAIN', 'f22_ticker_SBFG', 'f22_ticker_INPX', 'f22_ticker_UTHR',
                    'f22_ticker_TCON', 'f22_ticker_LCUT', 'f22_ticker_AXNX', 'f22_ticker_TSLA', 'f22_ticker_SCYX', 'f22_ticker_BCRX', 'f22_ticker_NYMT', 'f22_ticker_SUPN',
                    'f22_ticker_ATRC', 'f22_ticker_RAPT', 'f22_ticker_RDFN', 'f22_ticker_BBCP', 'f22_ticker_PDCO', 'f22_ticker_AIRT', 'f22_ticker_UG', 'f22_ticker_PRAA',
                    'f22_ticker_XCUR', 'f22_ticker_RVMD', 'f22_ticker_APEN', 'f22_ticker_ORRF', 'f22_ticker_VRAY', 'f22_ticker_METX', 'f22_ticker_FLGT', 'f22_ticker_FTNT',
                    'f22_ticker_YNDX', 'f22_ticker_CMPI', 'f22_ticker_RCKY', 'f22_ticker_PDSB', 'f22_ticker_HAS', 'f22_ticker_ODP', 'f22_ticker_ACST', 'f22_ticker_MOR',
                    'f22_ticker_CDXS', 'f22_ticker_DFFN', 'f22_ticker_ACOR', 'f22_ticker_NFLX', 'f22_ticker_EVFM', 'f22_ticker_SGMO', 'f22_ticker_WEYS', 'f22_ticker_ZG',
                    'f22_ticker_ZM', 'f22_ticker_GDRX', 'f22_ticker_SHYF', 'f22_ticker_VSEC', 'f22_ticker_NLTX', 'f22_ticker_ITCI', 'f22_ticker_ONCT', 'f22_ticker_DSPG',
                    'f22_ticker_ATXI', 'f22_ticker_IFMK', 'f22_ticker_HBNC', 'f22_ticker_PZZA', 'f22_ticker_APTO', 'f22_ticker_EIGR', 'f22_ticker_LOGI', 'f22_ticker_FITB',
                    'f22_ticker_BBIO', 'f22_ticker_ARLP', 'f22_ticker_TRCH', 'f22_ticker_HOMB', 'f22_ticker_APXT', 'f22_ticker_HGEN', 'f22_ticker_MOMO', 'f22_ticker_BGFV',
                    'f22_ticker_SWTX', 'f22_ticker_VUZI', 'f22_ticker_NKTR', 'f22_ticker_FCAC', 'f22_ticker_EDIT', 'f22_ticker_ABUS', 'f22_ticker_BEEM', 'f22_ticker_PSEC',
                    'f22_ticker_UNIT', 'f22_ticker_DOMO', 'f22_ticker_MTCH', 'f22_ticker_IDXX', 'f22_ticker_NTRA', 'f22_ticker_UAL', 'f22_ticker_RNA', 'f22_ticker_IESC',
                    'f22_ticker_KXIN', 'f22_ticker_XOMA', 'f22_ticker_FROG', 'f22_ticker_CCNC', 'f22_ticker_CRNC', 'f22_ticker_PLAB', 'f22_ticker_FULC', 'f22_ticker_HGBL',
                    'f22_ticker_CAR', 'f22_ticker_YORW', 'f22_ticker_GEC', 'f22_ticker_FB', 'f22_ticker_OTLK', 'f22_ticker_GBT', 'f22_ticker_PRGX', 'f22_ticker_RYTM',
                    'f22_ticker_IDRA', 'f22_ticker_LOPE', 'f22_ticker_HQI', 'f22_ticker_SLDB', 'f22_ticker_OSW', 'f22_ticker_BLUE', 'f22_ticker_CWCO', 'f22_ticker_NXTD',
                    'f22_ticker_HMNF', 'f22_ticker_POOL', 'f22_ticker_SUMO', 'f22_ticker_FLNT', 'f22_ticker_MORF', 'f22_ticker_INSG', 'f22_ticker_BBI', 'f22_ticker_UONE',
                    'f22_ticker_PROG', 'f22_ticker_KLAC', 'f22_ticker_BEAM', 'f22_ticker_POLA', 'f22_ticker_FUV', 'f22_ticker_MVBF', 'f22_ticker_QUMU', 'f22_ticker_MCRB',
                    'f22_ticker_PGC', 'f22_ticker_AVCT', 'f22_ticker_NHLD', 'f22_ticker_BGNE', 'f22_ticker_TENB', 'f22_ticker_GHIV', 'f22_ticker_AGLE', 'f22_ticker_PBYI',
                    'f22_ticker_DADA', 'f22_ticker_NEOG', 'f22_ticker_ROCH', 'f22_ticker_SMMC', 'f22_ticker_SWKH', 'f22_ticker_CRTD', 'f22_ticker_VCYT', 'f22_ticker_MIST',
                    'f22_ticker_STWO', 'f22_ticker_EYEG', 'f22_ticker_ISIG', 'f22_ticker_AVXL', 'f22_ticker_CORT', 'f22_ticker_ANY', 'f22_ticker_KNSL', 'f22_ticker_ADXS',
                    'f22_ticker_SIRI', 'f22_ticker_JAN', 'f22_ticker_GRCY', 'f22_ticker_NEO', 'f22_ticker_CZR', 'f22_ticker_INMB', 'f22_ticker_FLIC', 'f22_ticker_GPRO',
                    'f22_ticker_INCY', 'f22_ticker_CLSD', 'f22_ticker_JBLU', 'f22_ticker_GDEN', 'f22_ticker_PRAH', 'f22_ticker_LULU', 'f22_ticker_TSBK', 'f22_ticker_VERI',
                    'f22_ticker_WSC', 'f22_ticker_ASTE', 'f22_ticker_EML', 'f22_ticker_SCPH', 'f22_ticker_JAZZ', 'f22_ticker_SQBG', 'f22_ticker_ALLO', 'f22_ticker_MBWM',
                    'f22_ticker_MRTX', 'f22_ticker_DCPH', 'f22_ticker_JCTCF', 'f22_ticker_HMST', 'f22_ticker_SSNT', 'f22_ticker_ATOM', 'f22_ticker_ACAD', 'f22_ticker_HOLX',
                    'f22_ticker_REGN', 'f22_ticker_FCEL', 'f22_ticker_ALTR', 'f22_ticker_ZI', 'f22_ticker_RAVE', 'f22_ticker_SPPI', 'f22_ticker_TTEC', 'f22_ticker_BL',
                    'f22_ticker_CASI', 'f22_ticker_WBA', 'f22_ticker_EGLE', 'f22_ticker_LXRX', 'f22_ticker_REPL', 'f22_ticker_ALT', 'f22_ticker_PSTI', 'f22_ticker_CTIB',
                    'f22_ticker_DMAC', 'f22_ticker_HTBX', 'f22_ticker_PTON', 'f22_ticker_HELE', 'f22_ticker_OSIS', 'f22_ticker_DGICA', 'f22_ticker_AGBA', 'f22_ticker_ATEC',
                    'f22_ticker_UNTY', 'f22_ticker_SFBS', 'f22_ticker_LPRO', 'f22_ticker_CRAI', 'f22_ticker_ESCA', 'f22_ticker_LOB', 'f22_ticker_BLI', 'f22_ticker_ZSAN',
                    'f22_ticker_RKDA', 'f22_ticker_RIDE', 'f22_ticker_TMDX', 'f22_ticker_WING', 'f22_ticker_PFMT', 'f22_ticker_CHUY', 'f22_ticker_ARDS', 'f22_ticker_SSB',
                    'f22_ticker_ASTC', 'f22_ticker_AGIO', 'f22_ticker_PHUN', 'f22_ticker_VYGR', 'f22_ticker_SNGX', 'f22_ticker_<unknown>', 'f22_ticker_CRSR', 'f22_ticker_MWK',
                    'f22_ticker_PGEN', 'f22_ticker_CBNK', 'f22_ticker_RDVT', 'f22_ticker_GNSS', 'f22_ticker_KOPN', 'f22_ticker_LPTX', 'f22_ticker_PRVL', 'f22_ticker_IZEA',
                    'f22_ticker_FVE', 'f22_ticker_LSCC', 'f22_ticker_NFE', 'f22_ticker_LMB', 'f22_ticker_TRHC', 'f22_ticker_NGAC', 'f22_ticker_EIDX', 'f22_ticker_AYTU',
                    'f22_ticker_ADIL', 'f22_ticker_XBIO', 'f22_ticker_HCAP', 'f22_ticker_FIZZ', 'f22_ticker_TPCO', 'f22_ticker_CNST', 'f22_ticker_WW', 'f22_ticker_NXTC',
                    'f22_ticker_CERE', 'f22_ticker_WRLD', 'f22_ticker_UEPS', 'f22_ticker_FORD', 'f22_ticker_NCNO', 'f22_ticker_EDUC', 'f22_ticker_KRTX', 'f22_ticker_NGHC',
                    'f22_ticker_CTAS', 'f22_ticker_BFIN', 'f22_ticker_ACBI', 'f22_ticker_CREE', 'f22_ticker_OEG', 'f22_ticker_PHIO', 'f22_ticker_ZION', 'f22_ticker_EYPT',
                    'f22_ticker_UFPI', 'f22_ticker_ODFL', 'f22_ticker_STLD', 'f22_ticker_REFR', 'f22_ticker_CRSP', 'f22_ticker_VRA', 'f22_ticker_SLGG', 'f22_ticker_TW',
                    'f22_ticker_BRPA', 'f22_ticker_SHEN', 'f22_ticker_RCEL', 'f22_ticker_ATNI', 'f22_ticker_JOUT', 'f22_ticker_RCM', 'f22_ticker_LINK', 'f22_ticker_BZUN',
                    'f22_ticker_DZSI', 'f22_ticker_SVBI', 'f22_ticker_CTXS', 'f22_ticker_UHAL', 'f22_ticker_ALXN', 'f22_ticker_ONCS', 'f22_ticker_TURN', 'f22_ticker_MMSI',
                    'f22_ticker_CRIS', 'f22_ticker_LMNR', 'f22_ticker_NLOK', 'f22_ticker_SAGE', 'f22_ticker_UK', 'f22_ticker_WTER', 'f22_ticker_CHRS', 'f22_ticker_TA',
                    'f22_ticker_REG', 'f22_ticker_OPT', 'f22_ticker_STRM', 'f22_ticker_TWOU', 'f22_ticker_MVIS', 'f22_ticker_AKBA', 'f22_ticker_REAL', 'f22_ticker_SSTI',
                    'f22_ticker_AMSC', 'f22_ticker_MTLS', 'f22_ticker_MANH', 'f22_ticker_REPH', 'f22_ticker_INTZ', 'f22_ticker_ANAB', 'f22_ticker_GWPH', 'f22_ticker_IAC',
                    'f22_ticker_BSGM', 'f22_ticker_ENPH', 'f22_ticker_XLRN', 'f22_ticker_VALU', 'f22_ticker_PCH', 'f22_ticker_KIRK', 'f22_ticker_VERY', 'f22_ticker_SLCT',
                    'f22_ticker_BLFS', 'f22_ticker_BKNG', 'f22_ticker_DLPN', 'f22_ticker_CSPI', 'f22_ticker_SPWR', 'f22_ticker_TMUS', 'f22_ticker_AKRO', 'f22_ticker_IRDM',
                    'f22_ticker_MIME', 'f22_ticker_NMRD', 'f22_ticker_MERC', 'f22_ticker_PPD', 'f22_ticker_CDNA', 'f22_ticker_AIRG', 'f22_ticker_LOGC', 'f22_ticker_GTIM',
                    'f22_ticker_OSS', 'f22_ticker_SONN', 'f22_ticker_BOKF', 'f22_ticker_SFNC', 'f22_ticker_BSQR', 'f22_ticker_PVBC', 'f22_ticker_DIOD', 'f22_ticker_QTRX',
                    'f22_ticker_GOGO', 'f22_ticker_ICBK', 'f22_ticker_CNMD', 'f22_ticker_ALNY', 'f22_ticker_AMKR', 'f22_ticker_LAUR', 'f22_ticker_OPHC', 'f22_ticker_SNCA',
                    'f22_ticker_ADPT', 'f22_ticker_CBSH', 'f22_ticker_EVOK', 'f22_ticker_OSTK', 'f22_ticker_NMTR', 'f22_ticker_CBMG', 'f22_ticker_RBB', 'f22_ticker_TFSL',
                    'f22_ticker_MYSZ', 'f22_ticker_LMFA', 'f22_ticker_KALA', 'f22_ticker_MPB', 'f22_ticker_LRCX', 'f22_ticker_PI', 'f22_ticker_SDGR', 'f22_ticker_NEON',
                    'f22_ticker_ALDX', 'f22_ticker_MDCA', 'f22_ticker_CCB', 'f22_ticker_LYFT', 'f22_ticker_SPRB', 'f22_ticker_MIK', 'f22_ticker_HEPA', 'f22_ticker_MNST',
                    'f22_ticker_CRBP', 'f22_ticker_QTT', 'f22_ticker_ENSG', 'f22_ticker_AMAT', 'f22_ticker_ADBE', 'f22_ticker_HARP', 'f22_ticker_QCRH', 'f22_ticker_EVLO',
                    'f22_ticker_YTEN', 'f22_ticker_FISV', 'f22_ticker_KALV', 'f22_ticker_VRTX', 'f22_ticker_COHR', 'f22_ticker_LFAC', 'f22_ticker_FTFT', 'f22_ticker_PSTV',
                    'f22_ticker_SGLB', 'f22_ticker_JACK', 'f22_ticker_TRVN', 'f22_ticker_GAIA', 'f22_ticker_CDW', 'f22_ticker_ONTX', 'f22_ticker_NCBS', 'f22_ticker_AVGO',
                    'f22_ticker_PDLI', 'f22_ticker_GSMG', 'f22_ticker_SWKS', 'f22_ticker_STAF', 'f22_ticker_RARE', 'f22_ticker_BCLI', 'f22_ticker_VRTU', 'f22_ticker_MDNA',
                    'f22_ticker_RMTI', 'f22_ticker_BLNK', 'f22_ticker_AMCI', 'f22_ticker_MRCY', 'f22_ticker_MARA', 'f22_ticker_RMBS', 'f22_ticker_APPF', 'f22_ticker_HUBG',
                    'f22_ticker_FLWS', 'f22_ticker_ACRX', 'f22_ticker_MGNX', 'f22_ticker_BCDA', 'f22_ticker_SYNL', 'f22_ticker_TCBI', 'f22_ticker_VOXX', 'f22_ticker_CELC',
                    'f22_ticker_PHAT', 'f22_ticker_INTC', 'f22_ticker_HOTH', 'f22_ticker_WATT', 'f22_ticker_CHNG', 'f22_ticker_RAND', 'f22_ticker_RFIL', 'f22_ticker_WDAY',
                    'f22_ticker_CAMP', 'f22_ticker_CLRB', 'f22_ticker_HMSY', 'f22_ticker_PTMN', 'f22_ticker_PLUS', 'f22_ticker_NK', 'f22_ticker_SRNE', 'f22_ticker_FLEX',
                    'f22_ticker_TCF', 'f22_ticker_KNSA', 'f22_ticker_TBLT', 'f22_ticker_OM', 'f22_ticker_LE', 'f22_ticker_ICCH', 'f22_ticker_GP', 'f22_ticker_SPOK',
                    'f22_ticker_DHC', 'f22_ticker_TNAV', 'f22_ticker_GOOGL', 'f22_ticker_ILMN', 'f22_ticker_THMO', 'f22_ticker_JBHT', 'f22_ticker_MNPR', 'f22_ticker_CNFR',
                    'f22_ticker_CASA', 'f22_ticker_PCSB', 'f22_ticker_LIVK', 'f22_ticker_AZPN', 'f22_ticker_GO', 'f22_ticker_TTD', 'f22_ticker_SCHN', 'f22_ticker_PTCT',
                    'f22_ticker_BKEP', 'f22_ticker_API', 'f22_ticker_ZGYH', 'f22_ticker_BWEN', 'f22_ticker_KRNT', 'f22_ticker_RVNC', 'f22_ticker_CGRO', 'f22_ticker_GOSS',
                    'f22_ticker_SIGA', 'f22_ticker_SGC', 'f22_ticker_AGEN', 'f22_ticker_GH', 'f22_ticker_GRNQ', 'f22_ticker_GNMK', 'f22_ticker_AAOI', 'f22_ticker_VSAT',
                    'f22_ticker_TEAM', 'f22_ticker_ADSK', 'f22_ticker_FRTA', 'f22_ticker_BIDU', 'f22_ticker_LWAY', 'f22_ticker_EXEL', 'f22_ticker_LAKE', 'f22_ticker_CRTO',
                    'f22_ticker_INGN', 'f22_ticker_POWI', 'f22_ticker_AXSM', 'f22_ticker_CCNE', 'f22_ticker_GTHX', 'f22_ticker_ERII', 'f22_ticker_CHDN', 'f22_ticker_ROST',
                    'f22_ticker_ACCD', 'f22_ticker_SOHO', 'f22_ticker_LIFE', 'f22_ticker_ROKU', 'f22_ticker_OCC', 'f22_ticker_VBIV', 'f22_ticker_FPAY', 'f22_ticker_SQFT',
                    'f22_ticker_GDS', 'f22_ticker_BRP', 'f22_ticker_HBP', 'f22_ticker_SAIA', 'f22_ticker_DISCA', 'f22_ticker_QELL', 'f22_ticker_TRIP', 'f22_ticker_NMIH',
                    'f22_ticker_CSWC', 'f22_ticker_XELA', 'f22_ticker_APEI', 'f22_ticker_PAAS', 'f22_ticker_RPAY', 'f22_ticker_CHMG', 'f22_ticker_SELF', 'f22_ticker_CVLT',
                    'f22_ticker_TBIO', 'f22_ticker_ZYXI', 'f22_ticker_UPLD', 'f22_ticker_MGLN', 'f22_ticker_WWR', 'f22_ticker_AKU', 'f22_ticker_FCNCA', 'f22_ticker_NVIV',
                    'f22_ticker_LMNX', 'f22_ticker_WTRH', 'f22_ticker_EA', 'f22_ticker_PFG', 'f22_ticker_CHEF', 'f22_ticker_MRKR', 'f22_ticker_MDB', 'f22_ticker_HNNA',
                    'f22_ticker_ANIK', 'f22_ticker_PNNT', 'f22_ticker_SBAC', 'f22_ticker_GRWG', 'f22_ticker_CLBS', 'f22_ticker_PODD', 'f22_ticker_PMBC', 'f22_ticker_CYBR',
                    'f22_ticker_PETS', 'f22_ticker_QDEL', 'f22_ticker_MYGN', 'f22_ticker_HNRG', 'f22_ticker_SGEN', 'f22_ticker_MAR', 'f22_ticker_XP', 'f22_ticker_LPTH',
                    'f22_ticker_CVGW', 'f22_ticker_EXPO', 'f22_ticker_NVDA', 'f22_ticker_BRKR', 'f22_ticker_COST', 'f22_ticker_RCMT', 'f22_ticker_GDYN', 'f22_ticker_STNE',
                    'f22_ticker_EXAS', 'f22_ticker_EXPI', 'f22_ticker_KHC', 'f22_ticker_ZNGA', 'f22_ticker_MMLP', 'f22_ticker_VISL', 'f22_ticker_SRDX', 'f22_ticker_CLBK',
                    'f22_ticker_AAL', 'f22_ticker_AKER', 'f22_ticker_HSTO', 'f22_ticker_INZY', 'f22_ticker_SNPS', 'f22_ticker_OVID', 'f22_ticker_CSGS', 'f22_ticker_PAVM',
                    'f22_ticker_AERI', 'f22_ticker_CFFN']

    from_train_set = set(from_train)
    from_pred_set = set(from_predict)

    missing_from_pred = list(from_train_set - from_pred_set)
    logger.info(len(from_train))
    logger.info(missing_from_pred)


def test_multiple_searches():
    # Arrange
    ticker_1 = "AAPL"
    name_1 = "Apple"

    ticker_2 = "TSLA"
    name_2 = "Tesla"

    date_range = DateRange.from_date_strings(from_date_str="2021-01-01", to_date_str="2021-01-01")
    # Act
    query = f"\"{ticker_1}\" {name_1} OR \"{ticker_2}\" {name_2}"
    # query = f"\"{ticker_1}\" {name_1}"

    twitter_service.search_standard(query=query, tweet_raw_output_path=tweet_raw_output_path, date_range=date_range, max_count=100)
    # Assert


def test_multi_query():
    spark = get_or_create("test")
    df = spark.json(tweet_raw_output_path)

    logger.info(df.count())


def test_fetch_up_to_date_tweets():
    # Arrange
    youngest_date_str = "2020-12-24"
    # Act
    with patch("ams.services.twitter_service.search_one_day_at_a_time") as mock_search, \
        patch("ams.services.twitter_service.twitter_utils.get_youngest_tweet_date_in_system", return_value=youngest_date_str) as mock_get_youngest:
        # Act
        date_range = twitter_service.fetch_up_to_date_tweets()

        mock_search.assert_called_once()
        mock_get_youngest.assert_called_once()
        from_date_str_actual = date_utils.get_standard_ymd_format(date_range.from_date)
        to_date_str_actual = date_utils.get_standard_ymd_format(date_range.to_date)

        today_str = date_utils.get_standard_ymd_format(datetime.now())

        # Assert
        assert ("2020-12-25" == from_date_str_actual)
        assert (to_date_str_actual == today_str)


def test_split():
    # Arrange
    date_range = DateRange.from_date_strings(from_date_str="2020-12-17", to_date_str="2020-12-25")
    df_stock = ticker_service.get_ticker_eod_data("HEXO")
    df_stock = df_stock[(df_stock["date"] >= date_range.from_date_str) & (df_stock["date"] <= date_range.to_date_str)]
    df_stock.rename(columns={"ticker": "f22_ticker"}, inplace=True)
    df_stock.loc[:, "purchase_date"] = date_range.from_date_str

    df_stock_splits = stock_action_service.get_splits()[["ticker", "date", "value"]]
    print(df_stock.head())
    print(df_stock_splits.head())

    # Act
    df_rez = twitter_service.join_with_stock_splits_2(df=df_stock, date_range=date_range)

    # Assert
    print(df_rez.head())


def test_nas_roi_with_constraints():
    import pandas as pd
    df_roi_nasdaq = pd.read_parquet(str(constants.DAILY_ROI_NASDAQ_PATH))

    logger.info(df_roi_nasdaq.head(100))


def train_mlp_2(X_train, y_train, X_test, y_test):
    """ Home-made mini-batch learning
        -> not to be used in out-of-core setting!
    """
    import numpy as np
    import matplotlib.pyplot as plt

    num_input_features = X_train.shape[1]
    classes = 2  # Buy/Sell
    num_hidden_neurons = int(num_input_features / classes)

    mlp = MLPClassifier(hidden_layer_sizes=(500,), max_iter=50, alpha=0.0001,
                        solver='sgd', verbose=0, random_state=42, tol=0.000000001)

    N_TRAIN_SAMPLES = X_train.shape[0]
    N_EPOCHS = 48
    N_BATCH = 128
    N_CLASSES = np.unique(y_train)

    scores_train = []
    scores_test = []

    # EPOCH
    epoch = 0
    while epoch < N_EPOCHS:
        print('epoch: ', epoch)
        # SHUFFLING
        random_perm = np.random.permutation(X_train.shape[0])
        mini_batch_index = 0
        while True:
            # MINI-BATCH
            indices = random_perm[mini_batch_index:mini_batch_index + N_BATCH]
            mlp.partial_fit(X_train[indices], y_train[indices], classes=N_CLASSES)
            mini_batch_index += N_BATCH

            if mini_batch_index >= N_TRAIN_SAMPLES:
                break

        # SCORE TRAIN
        scores_train.append(mlp.score(X_train, y_train))

        # SCORE TEST
        score = mlp.score(X_test, y_test)
        print(f"Score: {score}")
        scores_test.append(score)

        epoch += 1

    """ Plot """
    fig, ax = plt.subplots(2, sharex=True, sharey=True)
    ax[0].plot(scores_train)
    ax[0].set_title('Train')
    ax[1].plot(scores_test)
    ax[1].set_title('Test')
    fig.suptitle("Accuracy over epochs", fontsize=14)
    plt.show()


def test_train_mlp():
    X_train, y_train, X_test, y_test = get_split_prepped_twitter_data(requires_balance=True)

    # twitter_service.train_mlp(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    train_mlp_2(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)