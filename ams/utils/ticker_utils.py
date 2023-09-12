import datetime
import math
import statistics
from collections import namedtuple
from datetime import timedelta
from typing import List, Tuple

import pandas as pd

from ams.DateRange import DateRange
from ams.config import logger_factory
from ams.services import ticker_service
from ams.utils import date_utils

logger = logger_factory.create(__name__)


def add_simple_moving_averages(df: pd.DataFrame, target_column: str, windows: List[int]):
    df_copy = df.copy()
    for w in windows:
        with pd.option_context('mode.chained_assignment', None):
            df_copy[f"{target_column}_SMA_{w}"] = df_copy[target_column].rolling(window=w).mean().astype("float64")
    return df_copy


def add_price_volatility(df: pd.DataFrame, col_price: str):
    df_copy = df.copy()
    with pd.option_context('mode.chained_assignment', None):
        df_copy[f"price_volatility"] = df_copy[col_price].rolling(window=30).std().astype("float64")
    return df_copy


def add_sma_history(df: pd.DataFrame, target_column: str, windows: List[int], tweet_date_str: str):
    max_window = max(windows)

    all_dataframes = []

    date_col = "date"

    df_g = df.groupby(by=["f22_ticker"])

    for ticker, df_group in df_g:
        df_equity = ticker_service.get_ticker_eod_data(ticker)
        dt_oldest_tweet_str = min(df_group["date"])

        if df_equity is not None:
            df_equity = df_equity[df_equity["date"] <= tweet_date_str].copy()

            df_equity.dropna(subset=[date_col], inplace=True)

            df_equity.sort_values(by=date_col, ascending=True, inplace=True)

            if df_equity is not None and df_equity.shape[0] > 0:
                # dt_youngest_ticker_str = max(df_equity["date"])
                # dt_oldest_tweet_str = dt_oldest_tweet_str if dt_oldest_tweet_str > dt_youngest_ticker_str else dt_oldest_ticker_str

                df_hist = df_equity[df_equity[date_col] < dt_oldest_tweet_str].copy()
                dt_start_str = None
                if df_hist.shape[0] > max_window:
                    with pd.option_context('mode.chained_assignment', None):
                        dt_start_str = df_hist.iloc[-max_window:][date_col].values.tolist()[0]
                elif df_hist.shape[0] > 0:
                    dt_start_str = df_hist[date_col].min()

                if dt_start_str is not None:
                    df_dated = df_equity[df_equity[date_col] >= dt_start_str].copy()

                    df_sma = add_simple_moving_averages(df=df_dated, target_column=target_column, windows=windows)
                    df_price_vol = add_price_volatility(df=df_sma, col_price=target_column)

                    all_dataframes.append(df_price_vol)

    df_all = pd.concat(all_dataframes, axis=0)

    df_all = df_all.rename(columns={"ticker": "f22_ticker"})
    df_merged = pd.merge(df, df_all, how='inner', left_on=["f22_ticker", "date"], right_on=["f22_ticker", date_col], suffixes=[None, "_drop"])

    df_dropped = df_merged.drop(columns=[c for c in df_merged.columns if c.endswith("_drop")])

    return df_dropped


days_under_sma = 0


def add_days_since_under_sma_many_tickers(df: pd.DataFrame, col_sma: str, close_col: str):
    df_g = df.groupby(by=["f22_ticker"])

    new_groups = []
    for _, df_group in df_g:
        df_group = add_days_since_under_sma_to_ticker(df_one_ticker=df_group, col_sma=col_sma, close_col=close_col)
        new_groups.append(df_group)

    df_result = df
    if len(new_groups) > 0:
        df_result = pd.concat(new_groups).reset_index(drop=True)

    return df_result


def get_count_days(row: pd.Series, col_sma: str, close_col: str):
    global days_under_sma

    close = row[close_col]
    sma = row[col_sma]

    if close < sma:
        if days_under_sma < 0:
            days_under_sma = 0
        elif days_under_sma >= 0:
            days_under_sma += 1
    else:
        if days_under_sma > 0:
            days_under_sma = 0
        elif days_under_sma <= 0:
            days_under_sma -= 1

    return days_under_sma


def add_days_since_under_sma_to_ticker(df_one_ticker: pd.DataFrame, col_sma: str, close_col: str):
    global days_under_sma
    days_under_sma = 0

    df_one_ticker.sort_values(by=["date"], inplace=True)
    df_one_ticker[f"{col_sma}_days_since_under"] = df_one_ticker.apply(lambda x: get_count_days(x, close_col=close_col, col_sma=col_sma), axis=1)

    return df_one_ticker


MDWindow = namedtuple("MDWindow", "start_year end_year start_month_day end_month_day")


def leverage_ma_return_to_mean():
    # tickers = ticker_service.get_nasdaq_tickers()["ticker_drop"].unique()
    tickers = ticker_service.get_ticker_info()["ticker"].unique()
    # tickers = ["GOOG", "NVDA", "FB", "MSFT", "AMZN", "AAPL", "DDOG"]

    # https://investorplace.com/2015/12/10-best-stocks-to-buy-for-2016/
    # ticker_tup = (2016, ["TSS", "UFS", "BWLD", "ETE", "ELLI", "AXP", "RAVE"]) # -3%

    # https://www.fool.com/investing/general/2016/01/03/7-top-stocks-to-buy-for-2016.aspx
    # ticker_tup = (2016, ["LNKD", "IBM", "TTE", "OSK", "MU", "CMG", "AGN"])  # 25%

    # https://money.usnews.com/investing/slideshows/8-stocks-to-buy-for-a-great-2016?slide=2
    # ticker_tup = (2016, ["NVDA", "SBUX", "GOOG", "AAPL", "EGRX", "CMG", "ROSS", "LUV"]) # 40% ann roi

    # https://www.nasdaq.com/articles/10-best-best-stocks-buy-2016-2016-01-04
    # ticker_tup = (2016, ["AHS", "INCR", "XRS", "ORA", "CTRP", "FIX", "FBC", "AMWD", "UFCS", "TSS"]) # 30% ann roi

    # https://www.reddit.com/r/stocks/comments/3xqsog/stock_picks_for_2016/
    # ticker_tup = (2016, ["ATVI", "T", "GORO", "ADM", "AAPL", "LLY", "VA", "CSCO", "SHOP", "FB", "FRE"])  # 45% ann roi; ann roi: -5.1%

    # https://www.fool.com/investing/general/2016/01/14/the-10-best-stocks-in-the-sp-500-in-2015.aspx
    # ticker_tup = (2016, ["NFLX", "AMZN", "ATVI", "NVDA", "HRL", "VRSN", "RAI", "SBUX", "FSLR"]) # 16.9%; ann roi: 13.5%

    # https://www.usatoday.com/story/money/markets/2015/12/31/best----and-worst----stocks-2015/78122024/
    # ticker_tup = (2016, ["NFLX", "AMZN", "ATVI", "VRSN", "HRL", "FSLR", "TSS", "GOOGL"])  # 2% ann roi; rroi: 13.5%

    # https://money.cnn.com/2015/12/22/investing/worst-stocks-2015-oil-energy/index.html
    # ticker_tup = (2016, ["CHK", "FCX", "FOSL", "NRG", "MU", "WYNN", "SPLS", "VIAB", "M", "THC"])  # 20% ann roi; rroi: -58%

    # # https://investmentu.com/best-performing-stocks/
    # ticker_tup = (2021, ["EXPI", "AMD", "FIVN", "CYRX", "NVDA", "IMMU", "ENPH", "RNG", "APPS"])  # -64% ann roi; rroi: -29%

    # https://www.marketwatch.com/story/these-are-the-20-best-performing-stocks-of-the-past-decade-and-some-of-them-will-surprise-you-2019-12-09
    # ticker_tup = (2020, ["NFLX", "MKTX", "ABMD", "TDG", "AVGO", "ALGN", "URI", "REGN", "ULTA"])  # 50% ann roi; rroi: 65%

    # https://www.csmonitor.com/Business/Saving-Money/2015/0918/The-12-best-performing-stocks-of-the-2000s
    # ticker_tup = (2016, ["AAPL", "UVE", "AMZN", "REGN", "PCLN", "NFLX", "ILMN", "ALXN", "NEU"])  # 14.7% ann roi; rroi: 12.8%

    # https://www.usatoday.com/story/sponsor-story/motley-fool/2015/12/16/top-5-stocks-past-10-years/77359024/
    # ticker_tup = (2016, ["PCLN", "NFLX", "AAPL", "CELG", "CRM"])  # 20% ann roi; rroi: 13%

    # https://www.barrons.com/articles/top-10-stock-picks-for-2017-1480748688
    # ticker_tup = (2017, ["GOOG", "AAPL", "C", "DIS", "MRK", "DAL", "DTEGY", "NVS", "UL", "TOL", "CAT"])  # 27% ann roi; rroi: 34%

    # https://www.barrons.com/articles/top-10-stock-picks-for-2017-1480748688
    # ticker_tup = (2017, ["AZO", "AMZN", "JPM", "V", "AMT", "GILD", "NOK", "CHL", "GSK", "GOOGL", "ATVI"])  # 17.9% ann roi; rroi: 1.3%

    # https://www.wallstreetphysician.com/best-stocks-2018-lets-look-back-2017-stock-picks-first/ (Two dudes)
    # ticker_tup = (2018, ["AMAT", "BRKB", "SCHW", "C", "CUB", "DLPH", "DXC", "FB", "GV", "HUBS", "ISRG"])  # -70% ann roi; rroi: -21%

    # https://www.wallstreetphysician.com/best-stocks-2018-lets-look-back-2017-stock-picks-first/ (Fortune)
    # ticker_tup = (2018, ["AMZN", "GOOGL", "MSFT", "FB", "NVDA", "AMAT", "IFNNY", "CSCO", "HON", "ROP", "FTV"])  # -89% ann roi; rroi: 45%

    # https://www.wallstreetphysician.com/best-stocks-2018-lets-look-back-2017-stock-picks-first/ (Forbs)
    # ticker_tup = (2017, ["SD", "CMP", "SCS", "AHT", "NAVI", "SCMP", "MITL", "SIEGY", "SHPG", "BABA", "SJM", "CPB"])  # -73% ann roi; rroi: -56%

    # https://www.bloomberg.com/features/year-ahead-2017/
    # ticker_tup = (2018, ["ADS", "ALKS", "ATC", "AMZN", "AEO", "ANTA", "ARIA", "ABX", "CAT", "CBS", "CC", "CMG"])  # -82% ann roi; rroi: -25%

    # https://www.reddit.com/r/stocks/comments/7hz7bs/23_years_old_looking_for_tips_on_low_priced/
    # ticker_tup = (2018, ["AMD", "SNAP", "PHM", "TCEHY", "JNUG", "CWEB", "BABA", "NVDA", "MU", "SHOP", "SQ"])  # -84% ann roi; rroi: -1.5%

    # https://www.fool.com/investing/2017/10/23/4-top-marijuana-stocks-to-buy-in-2018.aspx
    # ticker_tup = (2018, ["CGC", "ACB"])  # -60% ann roi; rroi: 667%

    # https://www.benzinga.com/general/education/17/12/10937570/7-best-marijuana-stocks-that-blazed-a-trail-in-2017
    # ticker_tup = (2018, ["GWPH", "ARNA", "EMHTF", "STZ"])  # -77% ann roi; rroi: 19%

    # https://www.kiplinger.com/slideshow/investing/t052-s001-4-lit-marijuana-stocks-to-buy/index.html
    # ticker_tup = (2019, ["CGC", "STZ", "GWPH", "IIPR"])  # -2% ann roi; rroi: -16%

    # https://seekingalpha.com/article/4127883-canadian-marijuana-stocks-u-s-investors-looking-f-stocks
    # ticker_tup = (2018, ["STZ", "STZ", "GWPH", "IIPR"])  # -2% ann roi; rroi: -16%

    # ticker_tup = (2015, ["BRK.B"])

    # https://www.lynalden.com/blue-chip-dividend-stocks/
    # ticker_tup = (2018, ["BNS", "BTI", "EPD", "AMGN"])  # -75% ann roi; rroi: -12%

    # https://www.lynalden.com/blue-chip-dividend-stocks/
    # ticker_tup = (2018, ["BSX", "UA", "HCA", "NFLX", "CMG", "AAP", "FTNT", "TRIP", "AMD", "ABMD"])  # -% ann roi; rroi: -%

    # https://www.fool.com/slideshow/10-stocks-are-crushing-market-2018/?slide=2
    # ticker_tup = (2018, ["MSFT", "INTU", "CRM", "AMZN", "SHOP", "TDOC", "TWTR", "SQ", "TTD", "NFLX"])  # -% ann roi; rroi: -%

    # ticker_tup = (2014, ["GOOG", "NVDA", "FB", "MSFT", "AMZN", "AAPL"]) # -% ann roi; rroi: -%

    # Worst of 5 years prior to 2021
    # ticker_tup = (2021, ['EMX', 'MLACW', 'TMUSR', 'BBUCQ', 'GBDCV', 'PHIOW', 'IWBB', 'GIG.R', 'CREXW', 'ENIA.R'])  # -% ann roi; rroi: -%

    # Worst of 5 years prior to 2020
    # ticker_tup = (2020, ['BBUCQ', 'PHIOW', 'IWBB', 'GIG.R', 'CREXW', 'PZG.W', 'ENIA.R', 'ALACW', 'VIAVV', 'JRCCQ'])

    # Worst of 5 years prior to 2019
    # ticker_tup = (2019, ['PABK', 'PHIOW', 'IWBB', 'GIG.R', 'SHLDR', 'CREXW', 'PZG.W', 'ALACW', 'NGHC', 'VIAVV'])

    # Worst of 5 years prior to 2018
    # ticker_tup = (2018, ['HNIN', 'IWBB', 'SHLDR', 'PZG.W', 'CMRO', 'CEXE', 'NGHC', 'VIAVV', 'MSRT', 'DEAR']) # Ann roi: -.98

    # Worst of 5 years prior to 2017
    # ticker_tup = (2017, ['SHLDR', 'PZG.W', 'NGHC', 'VIAVV', 'MSRT', 'CTZN', 'TRIDQ', 'BATRA', 'PHIO', 'CERCW']) # Ann Roi: 0.4247: range_roi: -0.0275

    # Worst of 5 years prior to 2016
    # ticker_tup = (2016, ['SHLDR', 'PZG.W', 'KNBA', 'NGHC', 'VIAVV', 'MSRT', 'PBAL1', 'PDRT', 'PHIO', 'PRPL1']) #Ann Roi: -0.8537: range_roi: 0.1666

    # Worst of 5 years prior to 2015
    # ticker_tup = (2015, ['HYPRQ', 'SHLDR', 'KNBA', 'NGHC', 'SBLUQ', 'HMNY', 'CTQN', 'PHIO', 'IMSCQ', 'ASTTY'])  # Ann Roi: 0.1103: range_roi: 196499.0000

    # Worst of 6 mos years prior to 2021
    # ticker_tup = (2021, ['MLACW', 'SSNYQ', 'GLUX', 'RAHGF', 'SMME', 'PWON', 'ADILW', 'SRCH', 'VPGI', 'ECTE'])  # Ann Roi: 72.5862: range_roi: 0.8537

    # Worst of 6 mos years prior to 2020
    # ticker_tup = (2020, ['BBUCQ', 'ENIA.R', 'PAVMW', 'ERINQ', 'INTEQ', 'MTFBW', 'ESES', 'SNDL', 'NSPR.WS', 'QDMI'])  # Ann Roi: 717.7574: range_roi: 1.4545

    # Worst of 6 mos years prior to 2019
    # ticker_tup = (2019,['PHIOW', 'EYEGW', 'CREXW', 'ALACW', 'CBRI', 'RLM', 'TBLTW', 'PRVB', 'CHUC', 'PARNF'])  # Ann Roi: 365.8594: range_roi: -0.4286

    # Worst of 6 mos years prior to 2018
    # ticker_tup = (2018, ['ILIU', 'IWBB', 'EASTW', 'CANB', 'DLPN', 'CPSL', 'POWW', 'CYHHZ', 'EDIG', 'LCLP']) # Ann Roi: 24.3702: range_roi: 365.6667

    # Worst of 6 mos years prior to 2017
    # ticker_tup = (2017, ['AXPW', 'FWMHQ', 'BIEI', 'HCGS', 'NSPR.WS', 'CANB', 'XBIT', 'SNGXW', 'VOIL', 'BICX']) # Ann Roi: 3.3919: range_roi: -0.1667

    # Worst of 6 mos years prior to 2016
    # ticker_tup = (2016, ['EQUR', 'VIAVV', 'CACH', 'PBSOQ', 'ABHD', 'CERCW', 'LLEN', 'IRLD', 'BRQSW', 'RGDX']) # Ann Roi: 1.6990: range_roi: 0.0519

    # Worst of 6 mos years prior to 2015
    # ticker_tup = (2015, ['SHLDR', 'KADR', 'LIWA', 'IHSI', 'PRQR', 'BFRE', 'BNVIQ', 'MOLGY', 'SLNO', 'ADFS'])  # Ann Roi: 0.7730: range_roi: 54.0000

    # Worst of 6 mos years prior to 2014
    # ticker_tup = (2014, ['AHMIQ', 'CPICQ', 'HOKU', 'LEU', 'FTBK', 'RIHT', 'FLUX', 'WSTC', 'CDII', 'LOGC1'])  # Ann Roi: 8.5566: range_roi: -0.4211

    # Worst of 6 mos years prior to 2013
    # ticker_tup = (2013, ['WNEA', 'CCME', 'HCGS', 'CBMC', 'WAVE', 'BICX', 'PSID', 'CFBK', 'DPTRQ', 'CFCB'])  # Ann Roi: -0.8170: range_roi: 0.5000

    # Worst of 6 mos years prior to 2012
    # ticker_tup = (2012, ['BGPIQ', 'FRGBQ', 'PABK', 'ZG', 'FPFX', 'Z', 'WARM', 'BMTL', 'SDNA', 'NCMV'])  # Ann Roi: -0.9863: range_roi: -0.9500

    # Worst of 6 mos years prior to 2011
    # ticker_tup = (2011, ['AMFIQ', 'IMMCQ', 'MBHIQ', 'STXX1', 'VIVE', 'JLIC', 'AVNA', 'BWTR', 'MDIZQ', 'ODDJ'])  # Ann Roi: 69.2123: range_roi: 6665.6667

    # Best of 6 mos years prior to 2021
    # ticker_tup = (2021, )  #

    # Of all stocks (many are penny stocks, the date of investment should match the sample day. That is, if the sample is from August -> December,
    # the returns are best the same time next year August -> Dec.

    # Best of 6 mos years prior to 2020
    # ticker_tup = (2020, ['FSRVW', 'AMRWW', 'BLIAQ', 'MNPR', 'CHUC', 'FELPQ', 'ELVG', 'PRHR', 'SUWN', 'FLLCW'])  # Ann Roi: 52.4472: range_roi: 891.0000

    # Best of 6 mos years prior to 2019
    # ticker_tup = (2019, ['BNGOW', 'AGE', 'OMTK', 'CACH', 'QTT', 'BGMD', 'CYTXW', 'RNVA', 'DDDX', 'PRHL'])  # Ann Roi: 16.6401: range_roi: -0.8810

    # Best of 6 mos years prior to 2018
    # ticker_tup = (2018, ['EMMAW', 'NDRAW', 'LGL.R', 'DARE', 'IMUC', 'LYL', 'RYMM', 'RIBS', 'GLF', 'RIBTW']) # Ann Roi: 213.8359: range_roi: -0.7800

    # Best of 6 mos years prior to 2017
    # ticker_tup = (2017, ['BGI', 'TYHT', 'LDKYQ', 'PRSNQ', 'BFRE', 'WAVXQ', 'LMFAW', 'WSGI', 'TRX', 'TTTM']) # Ann Roi: 3.4757: range_roi: -0.0769

    # Best of 6 mos years prior to 2016
    # ticker_tup = (2016, ['AROC', 'HRBR', 'SCEI', 'OCX.W', 'EYEGW', 'BDIC', 'NRDS', 'SCUEF', 'DGWIY', 'INOW']) # Ann Roi: 19.3812: range_roi: 0.1651

    # Best of 6 mos years prior to 2015
    # ticker_tup = (2015, ['AROC', 'RWLK', 'CAMH', 'LKII', 'BLIAQ', 'AAIIQ', 'APDNW', 'CAMT', 'KE', 'LBYE'])  # Ann Roi: 3.1143: range_roi: -0.8873

    # Best of 6 mos years prior to 2014
    # ticker_tup = (2014, ['AROC', 'BSTG', 'HNDI', 'SATCQ', 'FISK', 'WEB', 'CAGM', 'OGCP', 'BNVIQ', 'CBAI'])  # Ann Roi: 1954.9650: range_roi: -0.0066

    # Best of 6 mos years prior to 2013
    # ticker_tup = (2013, ['AROC', 'SCUEF', 'IBNKQ', 'INTZ', 'ZANE', 'SDTHQ', 'UAHC', 'MERR', 'NVDL', 'GLXZ'])  # Ann Roi: 2732.8815: range_roi: 0.8143

    # Best of 6 mos years prior to 2012
    # ticker_tup = (2012, ['AROC', 'LFTC', 'SBLUQ', 'HROW', 'MFCO', 'TRNT', 'WCYN', 'EDIG', 'PFIE', 'CLWY'])  # Ann Roi: 3009.3021: range_roi: 1.0161

    # Best of 6 mos years prior to 2011
    # ticker_tup = (2011, ['BOFLQ', 'QSND', 'WGNB', 'MIIX', 'LYRI', 'GWOW', 'TWAIQ', 'CORSQ', 'IRWNQ', 'WDDD'])  # Ann Roi: 389.0476: range_roi: -0.2667

    # Best Nasdaq with price * vol > 500,000

    # Best of 6 mos years prior to 2020, investment also in last 6 months of 2020
    # ticker_tup = (2020, ['MNPR', 'XBIO', 'IGMS', 'KRUS', 'PGNY', 'NBSE', 'OYST', 'SINT', 'STKL', 'FRG'])  # Ann Roi: 0.9686: range_roi: 4.3650

    # Best of 6 mos years prior to 2019, investment also in last 6 months of 2020
    # ticker_tup = (2019, ['QTT', 'CRNX', 'PRNB', 'BBOX', 'SONO', 'ONTX', 'RGSE', 'ZYNE', 'SAVA', 'GH'])  # Ann Roi: 3.6461: range_roi: 0.0068

    # Best of 6 mos years prior to 2018, investment also in last 6 months of 2020
    # ticker_tup = (2018, ['LYL', 'WINS', 'ROKU', 'NITE', 'RYTM', 'RETO', 'ABLX', 'NEWA', 'CLXT', 'AKCA']) # Ann Roi: 0.2713: range_roi: -0.7957

    # Best of 6 mos years prior to 2017, investment also in last 6 months of 2020
    # ticker_tup = (2017, ['NTNX', 'NOVN', 'LEDS', 'EVBG', 'TLND', 'BINDQ', 'POLA', 'TPIC', 'SESN', 'CHUBK']) # Ann Roi: 0.1814: range_roi: 0.3687

    # Best of 6 mos years prior to 2016, investment also in last 6 months of 2020
    # ticker_tup = (2016, ['AIMT', 'WEB', 'AAAP', 'ARAV', 'GBT', 'NETE', 'VYGR', 'NBRV', 'AVXL', 'AGRX']) # Ann Roi: -0.2242: range_roi: 0.0476

    # Best of 6 mos years prior to 2015, investment also in last 6 months of 2020
    # ticker_tup = (2015, ['RWLK', 'CAMT', 'KE', 'HABT', 'TUBE', 'NVUS', 'LOCO', 'SSRG', 'CYBR', 'MOMO'])  # Ann Roi: 3.2066: range_roi: -0.6097

    # Best of 6 mos years prior to 2014, investment also in last 6 months of 2020
    # ticker_tup = (2014, ['WEB', 'XNCR', 'KIN', 'MONT', 'ENZY', 'BNFT', 'MEET', 'PLUG', 'SFM', 'ISEE'])  # Ann Roi: -0.6459: range_roi: -0.3398

    # Best of 6 mos years prior to 2013, investment also in last 6 months of 2020
    # ticker_tup = (2013, ['WEB', 'GNMX', 'MVIS', 'QLYS', 'SVNTQ', 'APAGF', 'ONVO', 'PFMT', 'ARNA', 'JRCCQ'])  # Ann Roi: -0.6109: range_roi: 1.6223

    # Best of 6 mos years prior to 2012, investment also in last 6 months of 2020
    # ticker_tup = (2012, ['FRAN', 'CARB', 'DNKN', 'PSTV', 'SIFY', 'AXAS', 'MGAM', 'OPEN', 'HNSN', 'ZLTQ'])  # Ann Roi: 0.5846: range_roi: 0.0579

    # Best of 6 mos years prior to 2011, investment also in last 6 months of 2020
    # ticker_tup = (2011, ['SEEL', 'GAGA', 'MMYT', 'NTSP', 'RP', 'AEGR', 'KEYW', 'BBRG', 'VRA', 'CETC'])  # Ann Roi: 4.0664: range_roi: 0.8668

    # Best of NASDAQ 5 years prior with price * vol > 500,000: For ski jumps, 39th day is best. Avoid drop offs
    # on right of chart.

    #39th day; Best of 5 years prior; Ann Roi: -0.6904: range_roi: -0.2437
    # ticker_tup = (2011, ['ATSG', 'HLCSQ', 'UONEK', 'GNTA', 'JMBA', 'TSLA', 'ASPS', 'RLOC', 'GAGA', 'COSIQ'])  # Ann Roi: 0.5485: range_roi: -0.2437

    # 39th day; Best of 5 years prior; Ann Roi: 0.5084: range_roi: -0.1223
    # ticker_tup = (2012, ['ATSG', 'UONEK', 'GNTA', 'NOVT', 'JMBA', 'TSLA', 'ASPS', 'RLOC', 'GAGA', 'NSPH'])  # Ann Roi: 0.0340: range_roi: -0.2437

    # 39th day; Best of 5 years prior; Ann Roi: 2.3129: range_roi: 1.1403
    # ticker_tup = (2013, ['ARCT1', 'CZR', 'ATSG', 'IMOS1', 'FBIO', 'CLSN', 'GNTA', 'NOVT', 'JMBA', 'TSLA'])  # Ann Roi: -0.3679: range_roi: 0.5180

    # 39th day; Best of 5 years prior; Ann Roi: -0.1496: range_roi: -0.4031
    # ticker_tup = (2014, ['ARCT1', 'CZR', 'ACER', 'DXLG', 'GNTA', 'LNETQ', 'PRSC', 'XNCR', 'FBIO', 'KIN'])  # Ann Roi: -0.5632: range_roi: -0.4031

    # 39th day; Best of 5 years prior; Ann Roi: 27.3906: range_roi: -0.6097
    # ticker_tup = (2015, ['RWLK', 'ARCT1', 'CZR', 'TBPH', 'DRNA', 'XNCR', 'KE', 'FBIO', 'KIN', 'USCR'])  # Ann Roi: 2.0142: range_roi: -0.6097

    # 39th day; Best of 5 years prior; Ann Roi: 1.6046: range_roi: -0.7163
    # ticker_tup = (2016, ['RWLK', 'MCRB', 'CBAY', 'ARCT1', 'CZR', 'TBPH', 'DRNA', 'AXGT', 'XNCR', 'KE'])  # Ann Roi: 3.3888: range_roi: -0.7163

    # 39th day; Best of 5 years prior; Ann Roi: 1.0359: range_roi: -0.5577
    # ticker_tup = (2017, ['RWLK', 'MCRB', 'CBAY', 'ARCT1', 'CZR', 'TBPH', 'DRNA', 'AXGT', 'XNCR', 'NTNX'])  # Ann Roi: 2.6393: range_roi: -0.5577

    # 39th day; Best of 5 years prior: Ann Roi: 1.4353: range_roi: -0.9537
    # ticker_tup = (2018, ['SPCB', 'RWLK', 'LYL', 'MCRB', 'CBAY', 'TBPH', 'WINS', 'DRNA', 'ROKU', 'AXGT'])  # Ann Roi: -0.2086: range_roi: -0.5919

    # 39th day; Best of 5 years prior; Ann Roi: 149430.4095: range_roi: -0.3451
    # ticker_tup = (2019, ['RWLK', 'MCRB', 'CBAY', 'QTT', 'TBPH', 'DRNA', 'ROKU', 'SEED', 'AXGT', 'FAMI'])  # Ann Roi: -0.3339: range_roi: -0.9537

    # 39th day; Best of 5 years prior; Ann Roi: 149430.4095: range_roi: -0.3451
    # ticker_tup = (2020, ['CIH', 'MNPR', 'MCRB', 'QTT', 'CRTX', 'ROKU', 'JFIN', 'BYND', 'AXGT', 'FAMI'])  # Ann Roi: 6.7380: range_roi: -0.3451

    # Best of 5 years prior 20 stocks
    # ticker_tup = (2021, ['AQB', 'MNPR', 'LNSR', 'AUVI', 'QTT', 'CRTX', 'WINS', 'REKR', 'ROKU', 'JFIN'])  # 39th day: Ann Roi: 3.2216: range_roi: 0.1036

    # 39th day; Ann Roi: -0.5198: range_roi: -0.2437
    # ticker_tup = (2011, ['ATSG', 'HLCSQ', 'UONEK', 'GNTA', 'JMBA', 'TSLA', 'ASPS', 'RLOC', 'GAGA', 'COSIQ', 'GTATQ', 'EXXIQ', 'HUGH', 'ANTE', 'FDML', 'SSCCQ', 'ACAD', 'VRNT', 'PNCLQ', 'ARTC'])

    # 39th day; Ann Roi: 2.6774: range_roi: -0.1223
    # ticker_tup = (2012, ['ATSG', 'UONEK', 'GNTA', 'NOVT', 'JMBA', 'TSLA', 'ASPS', 'RLOC', 'GAGA', 'NSPH', 'COSIQ', 'GTATQ', 'EXXIQ', 'HUGH', 'ANTE', 'FDML', 'SSCCQ', 'ACAD', 'VRNT', 'XBKS'])

    # 39th day; Ann Roi: 2.3129: range_roi: 1.1403
    # ticker_tup = (2013, ['ARCT1', 'CZR', 'ATSG', 'IMOS1', 'FBIO', 'CLSN', 'GNTA', 'NOVT', 'JMBA', 'TSLA', 'ASPS', 'MITK', 'CORT', 'RLOC', 'NSPH', 'COSIQ', 'GTATQ', 'EXXIQ', 'HUGH', 'ANTE'])

    # 39th day; Ann Roi: 0.5736: range_roi: -0.4031
    # ticker_tup = (2014, ['ARCT1', 'CZR', 'ACER', 'DXLG', 'GNTA', 'LNETQ', 'PRSC', 'XNCR', 'FBIO', 'KIN', 'LAVA', 'ACAS', 'QUIK', 'MONT', 'ENZY', 'PNCLQ', 'BNFT', 'TSLA', 'MCGC', 'ASPS'])

    # 39th day;  Ann Roi: 169.4693: range_roi: -0.6097
    # ticker_tup = (2015, ['RWLK', 'ARCT1', 'CZR', 'TBPH', 'DRNA', 'XNCR', 'KE', 'FBIO', 'KIN', 'USCR', 'DGLY', 'HABT', 'MONT', 'RVNC', 'TUBE', 'NVUS', 'ENZY', 'LOCO', 'BNFT', 'TSLA'])

    # 39th day; Ann Roi: -0.4760: range_roi: -0.7163
    # ticker_tup = (2016, ['RWLK', 'MCRB', 'CBAY', 'ARCT1', 'CZR', 'TBPH', 'DRNA', 'AXGT', 'XNCR', 'KE', 'AIMT', 'FBIO', 'KIN', 'MTEM', 'NHTC', 'TANH', 'AAAP', 'HABT', 'MONT', 'RVNC'])

    # 39th day; Ann Roi: 0.7007: range_roi: -0.5577
    # ticker_tup = (2017, ['RWLK', 'MCRB', 'CBAY', 'ARCT1', 'CZR', 'TBPH', 'DRNA', 'AXGT', 'XNCR', 'NTNX', 'KE', 'AIMT', 'KIN', 'TANH', 'AAAP', 'HABT', 'CDCAQ', 'MONT', 'RVNC', 'TUBE'])

    # 39th day; Ann Roi: -0.8068: range_roi: -0.5919
    # ticker_tup = (2018, ['SPCB', 'RWLK', 'LYL', 'MCRB', 'CBAY', 'TBPH', 'WINS', 'DRNA', 'ROKU', 'AXGT', 'XNCR', 'NTNX', 'NITE', 'KE', 'AIMT', 'KIN', 'TANH', 'AAAP', 'HABT', 'ANIX'])

    # 39th day; Ann Roi: 1.5427: range_roi: -0.9537
    # ticker_tup = (2019, ['RWLK', 'MCRB', 'CBAY', 'QTT', 'TBPH', 'DRNA', 'ROKU', 'SEED', 'AXGT', 'FAMI', 'NTNX', 'NITE', 'KE', 'AIMT', 'TANH', 'AAAP', 'HABT', 'GSHD', 'RYTM', 'ARGX'])

    # 39th day; Ann Roi: 193.8315: range_roi: -0.3451
    # ticker_tup = (2020, ['CIH', 'MNPR', 'MCRB', 'QTT', 'CRTX', 'ROKU', 'JFIN', 'BYND', 'AXGT', 'FAMI', 'NTNX', 'NITE', 'AIMT', 'TIGR', 'TANH', 'IGMS', 'AAAP', 'MTC', 'KRUS', 'GSHD'])

    # 39th day; Ann Roi: 8.7491: range_roi: 0.1036
    # ticker_tup = (2021, ['AQB', 'MNPR', 'LNSR', 'AUVI', 'QTT', 'CRTX', 'WINS', 'REKR', 'ROKU', 'JFIN', 'BYND', 'FAMI', 'NTNX', 'NITE', 'HCDI', 'MREO', 'TIGR', 'SIGA', 'IGMS', 'MTC'])

    # Top Revenue prev year.

    # 39th day: Ann Roi: -0.4385: range_roi: 0.1287
    # 19th day: Ann Roi: -0.0219: range_roi: 0.1287
    # ticker_tup = (2016, ['AAPL', 'TM', 'MCK', 'ABC', 'COST', 'HMC', 'KR', 'WBA', 'CAH', 'MSFT'])

    # 39th day: Ann Roi: 2.0729: range_roi: 0.3404
    # 19th day: Ann Roi: 0.5121: range_roi: 0.3404
    # ticker_tup = (2017, ['WMT', 'TM', 'AAPL', 'MCK', 'ABC', 'HMC', 'CAH', 'COST', 'WBA', 'KR'])

    # 19th day: Ann Roi: Ann Roi: 0.5167: range_roi: 0.1068
    # ticker_tup = (2018, ['WMT', 'TM', 'AAPL', 'MCK', 'ABC', 'CAH', 'COST', 'HMC', 'WBA', 'KR'])

    # 19th day: Ann Roi: 1.4562: range_roi: 0.1293
    # ticker_tup = (2019, ['WMT', 'AAPL', 'TM', 'MCK', 'ABC', 'COST', 'CAH', 'HMC', 'WBA', 'KR'])

    # 19th day: Ann Roi: 3.2616: range_roi: 0.1413
    # ticker_tup = (2020, ['WMT', 'TM', 'AAPL', 'MCK', 'ABC', 'COST', 'CAH', 'HMC', 'WBA', 'MSFT'])

    # 19th day: Ann Roi: 1.1491: range_roi: 0.0084
    # ticker_tup = (2021, ['WMT', 'TM', 'AAPL', 'MCK', 'ABC', 'COST', 'CAH', 'MSFT', 'WBA', 'HMC'])

    ma_days = 20
    num_days_under = 40
    wait_until_long_down = 19
    start_year = 2011
    sell_day_in_fut = 1
    days_in_future = 1
    rep_year = ticker_tup[0]
    end_year = rep_year - 1
    tickers = ticker_tup[1]

    dr = DateRange.from_date_strings(from_date_str=f"{start_year}-01-01", to_date_str=f"{end_year}-12-31")

    df_ticks, num_days_under, all_tickers_range_roi = get_complex_ma(tickers=tickers,
                                                                     ma_days=ma_days,
                                                                     sample_ticker_dr=dr,
                                                                     num_days_under=num_days_under,
                                                                     md_window=None,
                                                                     days_in_future=days_in_future)

    rois = []
    for i in range(num_days_under):
        df_is_down = df_ticks[df_ticks[f"is_long_down_{i}"]]
        r = df_is_down[f"fut_day_{sell_day_in_fut}_roi"].mean()
        rois.append((r, df_is_down.shape[0]))

    # logger.info(df_ticks[["ticker", "date", "is_long_down", "fut_day_1_roi"]].head(20))
    # pv = statistics.pvariance(df_ticks.sample(frac=1.)["fut_day_1_roi"])
    # logger.info(f"Pv: {pv:.6f}")

    chart_name = tickers[0] if len(tickers) == 1 else "Multiple Stocks"
    show_hist(chart_name=chart_name, rois=rois, num_days=num_days_under)
    md_window = None # MDWindow(start_year=rep_year, end_year=rep_year, start_month_day="07-01", end_month_day="12-31")

    year_test = dr.to_date.year + 1
    dr = DateRange.from_date_strings(from_date_str=f"{year_test}-01-01", to_date_str=f"{year_test + 1}-01-01")
    df_ticks, num_days_under, all_tickers_range_roi = get_complex_ma(tickers=tickers,
                                                                     sample_ticker_dr=dr,
                                                                     num_days_under=num_days_under,
                                                                     ma_days=ma_days,
                                                                     days_in_future=days_in_future,
                                                                     md_window=md_window)

    df_is_down = df_ticks[df_ticks[f"is_long_down_{wait_until_long_down}"]]
    r = df_is_down[f"fut_day_{sell_day_in_fut}_roi"].mean()
    logger.info(f"Mean roi: {r}")

    # frac = 250/df_is_down.shape[0]
    # df_is_down = df_is_down.sample(frac=frac)
    rois = list(df_is_down[f"fut_day_{sell_day_in_fut}_roi"].values)
    num_opps = len(rois)
    init_inv = 1000
    inv = init_inv
    for r in rois:
        inv = inv + (inv * r)

    num_inv = len(rois)
    aroi = (1 + (inv - init_inv) / init_inv) ** (1 / (num_inv / 250)) - 1
    logger.info(f"Num investments: {num_opps}: End total: {inv}: Ann Roi: {aroi:.4f}: range_roi: {all_tickers_range_roi:.4f}")


def get_complex_ma(tickers: List[str],
                   sample_ticker_dr: DateRange,
                   ma_days: int = 20,
                   num_days_under: int = 21,
                   md_window: MDWindow = None,
                   days_in_future: int = 5):
    # NOTE: 2021-09-04: chris.flesche: Trying to get all the days in the range part of the MA
    buff_days = num_days_under + int((num_days_under / 7) * 2.5)
    dr_buffed = DateRange(from_date=sample_ticker_dr.from_date - timedelta(days=buff_days), to_date=sample_ticker_dr.to_date + timedelta(days=buff_days))

    df_ticks = ticker_service.get_tickers_in_range(tickers=tickers, date_range=dr_buffed)
    df_g = df_ticks.groupby("ticker")
    df_all = []

    new_cols = []

    range_rois = []
    for key, df_t in df_g:
        dt_open = df_ticks["date"].min()
        dt_close = df_ticks["date"].max()
        range_open = df_ticks[df_ticks["date"] == dt_open]["open"].values[0]
        range_close = df_ticks[df_ticks["date"] == dt_close]["close"].values[0]
        range_rois.append((range_close - range_open) / range_open)

        df_t["close_prev_1"] = df_t["close"].shift(1)
        df_t["fut_day_1_roi"] = (df_t["close"] - df_t["close_prev_1"]) / df_t["close_prev_1"]
        df_t["fut_day_1_roi"] = df_t["fut_day_1_roi"].shift(-1)
        for i in range(2, days_in_future + 1):
            col_name = f"fut_day_{i}_roi"
            df_t[col_name] = df_t[f"fut_day_{i - 1}_roi"].shift(-1)
            new_cols.append(col_name)

        df_t["was_up_0"] = df_t["close"] - df_t["close_prev_1"] > 0

        for i in range(7):
            i = i + 1
            col_name = f"was_up_1_{i}"
            df_t[col_name] = df_t[f"was_up_0"].shift(i)
            new_cols.append(col_name)

        df_t["close_ma"] = df_t["close"].transform(lambda x: x.rolling(ma_days, 1).mean())

        df_t["was_close_under_20d_ma_0"] = df_t["close_ma"] - df_t["close"] > 0

        for i in range(num_days_under):
            i = i + 1
            col_name = f"was_close_under_20d_ma_{i}"
            df_t[col_name] = df_t[f"was_close_under_20d_ma_0"].shift(i)
            new_cols.append(col_name)

        df_t = df_t.dropna(subset=[f"fut_day_1_roi"] + new_cols)
        for x in range(num_days_under):
            col = f"is_long_down_{x}"
            df_t[col] = True
            for i in range(x):
                df_t[col] = df_t[col] & df_t[f"was_close_under_20d_ma_{i}"]

        # NOTE: 2021-09-04: chris.flesche: Trims the buffered stuff off
        df_t = df_t[(df_t["date"] > sample_ticker_dr.start_date_str) & (df_t["date"] < sample_ticker_dr.end_date_str)]

        if md_window is not None:
            snaps = []
            for year in range(md_window.start_year, md_window.end_year + 1):
                df_snap = df_t[(df_t["date"] > f"{year}-{md_window.start_month_day}") & (df_t["date"] < f"{year}-{md_window.end_month_day}")]
                snaps.append(df_snap)

            df_t = pd.concat(snaps)

        df_all.append(df_t)

    df_ticks = pd.concat(df_all)

    all_tickers_roi = 0
    if len(range_rois) > 0:
        all_tickers_roi = statistics.mean(range_rois)

    return df_ticks, num_days_under, all_tickers_roi


def get_maos(tickers: List[str],
             dr: DateRange,
             ma_days: int = 20,
             num_days_under: int = 21,
             add_future_cols: bool = True):
    df_result = None
    buff_days = (1 * num_days_under) + int((num_days_under / 7) * 4.5)

    dr_buffed = DateRange(from_date=dr.from_date - timedelta(days=buff_days), to_date=dr.to_date + timedelta(buff_days))

    df_ticks = ticker_service.get_tickers_in_range(tickers=tickers, date_range=dr_buffed)

    # if df_ticks[df_ticks["date"] == dr.end_date_str].shape[0] == 0:
    #     return df_result

    df_g = df_ticks.groupby("ticker")
    df_all = []
    new_cols = []

    for key, df_t in df_g:
        df_t.loc[:, "close_prev_1"] = df_t["close"].shift(1)
        df_t["was_up_0"] = (df_t["close"] - df_t["close_prev_1"] > 0).copy()
        for i in range(7):
            i = i + 1
            col_name = f"was_up_1_{i}"
            df_t.loc[:, col_name] = df_t[f"was_up_0"].shift(i)
            new_cols.append(col_name)

        df_t.loc[:, "close_ma"] = df_t["close"].transform(lambda x: x.rolling(ma_days, 1).mean())
        df_t.loc[:, "close_ma_1_day_before"] = df_t["close"].transform(lambda x: x.rolling(ma_days - 1, 1).mean())

        if add_future_cols:
            df_t.loc[:, "tomorrow_close"] = df_t["close"].shift(-1)
            df_t.loc[:, "fut_day_1_roi"] = (df_t["tomorrow_close"] - df_t["close"]) / df_t["close"]
            for i in range(2, 7 + 1):
                col_name = f"fut_day_{i}_roi"
                df_t.loc[:, col_name] = df_t[f"fut_day_1_roi"].shift(-(i - 1))
                new_cols.append(col_name)

        df_t["was_close_under_20d_ma_0"] = (df_t["close_ma"] - df_t["close"] > 0).copy()

        for i in range(num_days_under):
            i = i + 1
            col_name = f"was_close_under_20d_ma_{i}"
            df_t.loc[:, col_name] = df_t[f"was_close_under_20d_ma_0"].shift(i)
            new_cols.append(col_name)

        df_t = df_t.dropna(subset=new_cols).copy()
        for x in range(num_days_under):
            col = f"is_long_down_{x}"
            df_t.loc[:, col] = True
            for i in range(x):
                df_t[col] = (df_t[col] & df_t[f"was_close_under_20d_ma_{i}"]).copy()

        # NOTE: 2021-09-04: chris.flesche: Trims the buffered stuff off
        # logger.info(f"Num before date trim: {df_t.shape[0]}")
        # logger.info(f"{dr.from_date_str} to {dr.to_date_str}")
        df_t = (df_t[(df_t["date"] >= dr.start_date_str) & (df_t["date"] < dr.end_date_str)]).copy()

        # logger.info(f"Dates: {df_t['date'].unique()}")
        df_all.append(df_t)

    if len(df_all) > 0:
        df_result = pd.concat(df_all, axis=0)

    return df_result


def show_hist(chart_name: str, rois: List[Tuple[float, int]], num_days: int):
    import numpy as np
    import matplotlib.pyplot as plt

    x = range(num_days)

    amounts = [amt for (r, amt) in rois]
    rois = [r for (r, amount) in rois]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('Num Days Under MA')
    ax.set_ylabel('ROI')
    ax.set_title(f'{chart_name} ROI Over Days')
    ax.set_xticks(x)
    ax.set_yticks(np.arange(-0.003, .01, .001))
    ax.bar(x, rois)

    # Iterrating over the bars one-by-one
    for ndx, bar in enumerate(ax.patches):
        # Using Matplotlib's annotate function and
        # passing the coordinates where the annotation shall be done
        # x-coordinate: bar.get_x() + bar.get_width() / 2
        # y-coordinate: bar.get_height()
        # free space to be left to make graph pleasing: (0, 8)
        # ha and va stand for the horizontal and vertical alignment

        amt = amounts[ndx]
        amt_str = str(amt)
        if amt > 1000:
            amt = math.floor(amt / 1000)
            amt_str = str(amt) + "k"
        ax.annotate(amt_str,
                    (bar.get_x() + bar.get_width() / 2,
                     bar.get_height()), ha='center', va='center',
                    size=7, xytext=(0, 4),
                    textcoords='offset points')

    for label in ax.xaxis.get_ticklabels()[::2]:
        label.set_visible(False)

    plt.show()


def get_top_roi_tickers(tickers: List[str], prev_days: int, to_date_str, top_n: int = 10, min_price_vol_metric: float = 500000):
    to_dt = date_utils.parse_std_datestring(to_date_str)
    from_dt = to_dt - timedelta(days=prev_days)
    from_date_str = date_utils.get_standard_ymd_format(from_dt)
    dr = DateRange.from_date_strings(from_date_str=from_date_str, to_date_str=to_date_str)
    dr_buffed = DateRange(from_date=dr.from_date, to_date=dr.to_date)

    df_ticks = ticker_service.get_tickers_in_range(tickers=tickers, date_range=dr_buffed)

    df_g = df_ticks.groupby("ticker")

    df_all = []
    for ticker, df_t in df_g:
        vol = df_t["volume"].mean()
        price = df_t["close"].mean()

        if vol * price < min_price_vol_metric:
            continue

        dt_open = df_t["date"].min()
        dt_close = df_t["date"].min()
        range_open = df_t[df_t["date"] == dt_open]["open"].values[0]
        range_close = df_t[df_t["date"] == dt_close]["close"].values[0]
        unit_roi = (range_close - range_open) / range_open
        df_all.append(dict(ticker=ticker, unit_roi=unit_roi))

    df_all = sorted(df_all, key=lambda i: i['unit_roi'], reverse=True)

    df_all = df_all[:top_n]

    return [item["ticker"] for item in df_all]