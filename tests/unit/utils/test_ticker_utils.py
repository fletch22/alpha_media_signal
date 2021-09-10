import statistics
from datetime import timedelta
from typing import List

import pandas as pd

from ams.DateRange import DateRange
from ams.config import logger_factory
from ams.services import ticker_service
from ams.utils import ticker_utils, date_utils

logger = logger_factory.create(__name__)

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def test_get_sma():
    # Arrange
    tweet_rows = [
        {"f22_ticker": "AAPL", "date": "2020-09-08", "close": .11},
        {"f22_ticker": "AAPL", "date": "2020-09-09", "close": .22},
        {"f22_ticker": "MSFT", "date": "2020-09-15", "close": .33},
        {"f22_ticker": "ATVI", "date": "2020-09-17", "close": .44},
        {"f22_ticker": "ATVI", "date": "2020-09-18", "close": .55},
        {"f22_ticker": "AAPL", "date": "2020-09-08", "close": .11},
        {"f22_ticker": "AAPL", "date": "2020-09-09", "close": .22},
        {"f22_ticker": "MSFT", "date": "2020-09-15", "close": .33},
        {"f22_ticker": "ATVI", "date": "2020-09-17", "close": .44},
        {"f22_ticker": "ATVI", "date": "2020-09-18", "close": .55},
    ]
    df_tweets = pd.DataFrame(tweet_rows)
    window_list = [15, 20, 50, 200]

    # Act
    df_tweets_new = ticker_utils.add_simple_moving_averages(df=df_tweets, target_column="close", windows=[2, 15, 20, 50, 200])

    # Assert
    assert (df_tweets_new["SMA_2"].fillna(0).mean() > 0)
    assert (df_tweets_new["SMA_200"].fillna(0).mean() == 0)
    logger.info(df_tweets_new.head(20))


def test_add_sma_history():
    # Arrange
    date_range = DateRange.from_date_strings("2020-09-01", "2020-10-01")
    df_equities = ticker_service.get_tickers_in_range(tickers=["NVDA", "MOMO"], date_range=date_range)
    df_equities = df_equities.rename(columns={"ticker": "f22_ticker"})

    # Act
    df = ticker_utils.add_sma_history(df=df_equities, target_column="close", windows=[20, 200])

    # Assert
    assert ("close_SMA_200" in df.columns)
    # assert(df[df["close_SMA_200"].isnull()].shape[0] == 0)

    logger.info(list(df.columns))


def test_set_num_days_under_sma():
    # Arrange
    date_range = DateRange.from_date_strings("2020-09-01", "2020-10-01")
    df_equities = ticker_service.get_tickers_in_range(tickers=["NVDA", "MOMO"], date_range=date_range)
    df_equities = df_equities.rename(columns={"ticker": "f22_ticker"})

    df = ticker_utils.add_sma_history(df=df_equities, target_column="close", windows=[20, 200])

    # Act
    df["close_SMA_200_diff"] = df["close"] - df["close_SMA_200"]


def test_ams():
    # Arrange
    date_range = DateRange.from_date_strings("2009-09-01", "2021-10-01")
    df_equities = ticker_service.get_tickers_in_range(tickers=["NVDA", "MOMO"], date_range=date_range)
    df_equities = df_equities.rename(columns={"ticker": "f22_ticker"})
    df = ticker_utils.add_sma_history(df=df_equities, target_column="close", windows=[20, 200])

    # Act
    df_ungrouped = ticker_utils.add_days_since_under_sma_many_tickers(df=df, col_sma="close_SMA_200", close_col="close")

    # Assert
    num_under = df_ungrouped[df_ungrouped["close_SMA_200_days_since_under"] > 0].shape[0]
    assert (num_under > 1)


def test_up_or_down():
    # Arrange
    df = pd.DataFrame([
        {"close": 1.0},
        {"close": 0.0},
        {"close": 3.0},
        {"close": 3.0},
        {"close": 8.0},
        {"close": 2.0}
    ])

    # Act
    df = ticker_service.prev_up_or_down(df=df)

    # Assert
    logger.info(df.head())


def test_lev_return_2_mean():
    ticker_utils.leverage_ma_return_to_mean()


def test_get_maos():
    # tickers = ticker_service.get_ticker_info()["ticker"].unique()

    # 39th day; top rev prev year: .0038
    # 29th day; top rev prev year: 0.0022
    stock_dict = dict(
        _2016=['AAPL', 'TM', 'MCK', 'ABC', 'COST', 'HMC', 'KR', 'WBA', 'CAH', 'MSFT', 'NTTYY', 'JXHLY', 'PG', 'SNE', 'PEP', 'VOD', 'INTC', 'DIS', 'HPQ', 'CSCO',
               'SYY', 'FDX',
               'BHP', 'TTM', 'MUFG', 'TSN', 'ORCL', 'DCMYY', 'SNEX', 'S', 'ACN', 'HPE', 'NKE', 'TFCF', 'DE', 'SMFG', 'RY', 'BTTGY', 'RAD', 'FLEX', 'QCOM', 'MFG',
               'TD', 'MDT',
               'ADNT', 'NGG', 'SBUX', 'DXC', 'BNS', 'IX', 'ACM', 'JBL', 'AVT', 'GIS', 'JCI', 'DEO', 'NGL', 'EMR', 'MU', 'CCL', 'WFM', 'PFGC', 'SSL', 'WBK', 'MON',
               'BMO',
               'WDC', 'IBN', 'ARMK', 'KMX', 'V', 'STX', 'NMR', 'SNX', 'SVU', 'VIAB', 'KYOCY', 'PH', 'BABA', 'TEL', 'J', 'BBBY', 'CSX', 'ODP', 'VEDL', 'WRK', 'ADP',
               'DHI',
               'EL', 'HSIC', 'CM', 'BDX', 'AZO', 'NAV', 'PCP', 'TYC', 'AMAT', 'LEN', 'HRL', 'CAG'],
        _2017=['WMT', 'TM', 'AAPL', 'MCK', 'ABC', 'CAH', 'COST', 'HMC', 'WBA', 'KR', 'NTTYY', 'MSFT', 'HD', 'TGT', 'SNE', 'PG', 'LOW', 'PEP', 'JXHLY', 'INTC',
               'DELL', 'FDX',
               'ACI', 'VOD', 'SYY', 'DIS', 'HPQ', 'CSCO', 'DCMYY', 'TTM', 'BBY', 'TSN', 'ORCL', 'ACN', 'BHP', 'NKE', 'S', 'TJX', 'RY', 'SMFG', 'BTTGY', 'DE', 'MDT',
               'SNEX',
               'HPE', 'MUFG', 'TFCF', 'TD', 'KHC', 'M', 'TECD', 'IX', 'USFD', 'FLEX', 'BABA', 'RAD', 'JCI', 'SBUX', 'QCOM', 'SHLDQ', 'DG', 'BNS', 'DLTR', 'MU',
               'MFG', 'KSS',
               'WDC', 'JBL', 'NGG', 'V', 'SPLS', 'ACM', 'AVGO', 'CCL', 'IBN', 'AVT', 'BMO', 'WBK', 'SNX', 'PFGC', 'ADNT', 'DEO', 'KMX', 'TAK', 'GIS', 'GPS', 'EMR',
               'WRK',
               'JWN', 'AMAT', 'MON', 'ARMK', 'TXT', 'DHI', 'VIAB', 'SSL', 'SWK', 'JCPNQ', 'ROST', 'K'],
        _2018=['WMT', 'AAPL', 'TM', 'MCK', 'ABC', 'COST', 'CAH', 'HMC', 'WBA', 'KR', 'MSFT', 'HD', 'JNJ', 'DELL', 'SNE', 'TGT', 'INTC', 'LOW', 'PG', 'FDX', 'VOD',
               'PEP',
               'ACI', 'DIS', 'SYY', 'HPQ', 'CSCO', 'TTM', 'BHP', 'BBY', 'ACN', 'TSN', 'BABA', 'ORCL', 'MUFG', 'DE', 'NKE', 'TJX', 'TECD', 'BTTGY', 'SMFG', 'RY', 'S',
               'HPE',
               'TFCF', 'MU', 'MDT', 'TD', 'SNEX', 'KHC', 'FLEX', 'IX', 'M', 'SBUX', 'USFD', 'DG', 'JCI', 'MFG', 'QCOM', 'DLTR', 'JBL', 'BNS', 'DXC', 'RAD', 'NGG',
               'AVGO',
               'WDC', 'V', 'LEN', 'KSS', 'SNX', 'AVT', 'CCL', 'IBN', 'PFGC', 'BMO', 'ADNT', 'EMR', 'KMX', 'AMAT', 'SHLDQ', 'WRK', 'DEO', 'DHI', 'BDX', 'GPS', 'WBK',
               'ARMK',
               'GIS', 'TAK', 'JWN', 'PH', 'SVU', 'ROST', 'TEL', 'SWK', 'TXT', 'VEDL', 'ACM', 'KYOCY'],
        _2019=['WMT', 'AAPL', 'TM', 'MCK', 'ABC', 'COST', 'CAH', 'HMC', 'WBA', 'KR', 'MSFT', 'HD', 'JNJ', 'DELL', 'SNE', 'TGT', 'INTC', 'LOW', 'PG', 'FDX', 'VOD',
               'PEP',
               'ACI', 'DIS', 'SYY', 'HPQ', 'CSCO', 'TTM', 'BHP', 'BBY', 'ACN', 'TSN', 'BABA', 'ORCL', 'MUFG', 'DE', 'NKE', 'TJX', 'TECD', 'BTTGY', 'SMFG', 'RY', 'S',
               'HPE',
               'TFCF', 'MU', 'MDT', 'TD', 'SNEX', 'KHC', 'FLEX', 'IX', 'M', 'SBUX', 'USFD', 'DG', 'JCI', 'MFG', 'QCOM', 'DLTR', 'JBL', 'BNS', 'DXC', 'RAD', 'NGG',
               'AVGO',
               'WDC', 'V', 'LEN', 'KSS', 'SNX', 'AVT', 'CCL', 'IBN', 'PFGC', 'BMO', 'ADNT', 'EMR', 'KMX', 'AMAT', 'SHLDQ', 'WRK', 'DEO', 'DHI', 'BDX', 'GPS', 'WBK',
               'ARMK',
               'GIS', 'TAK', 'JWN', 'PH', 'SVU', 'ROST', 'TEL', 'SWK', 'TXT', 'VEDL', 'ACM', 'KYOCY'],
        _2020=['WMT', 'TM', 'AAPL', 'MCK', 'ABC', 'COST', 'CAH', 'HMC', 'WBA', 'MSFT', 'KR', 'HD', 'DELL', 'JNJ', 'SNE', 'TGT', 'INTC', 'LOW', 'FDX', 'DIS', 'PG',
               'PEP',
               'ACI', 'SYY', 'HPQ', 'VOD', 'BABA', 'CSCO', 'BHP', 'TTM', 'ACN', 'BBY', 'TSN', 'ORCL', 'DE', 'NKE', 'TJX', 'TECD', 'RY', 'MUFG', 'S', 'SNEX', 'TD',
               'BTTGY',
               'MDT', 'HPE', 'SMFG', 'SBUX', 'FLEX', 'USFD', 'DG', 'JBL', 'KHC', 'M', 'QCOM', 'JCI', 'SNX', 'BNS', 'MU', 'V', 'DLTR', 'AVGO', 'UNFI', 'LEN', 'IX',
               'RAD',
               'CCL', 'DXC', 'KSS', 'NGG', 'PFGC', 'AVT', 'BMO', 'TAK', 'IBN', 'MFG', 'EMR', 'WRK', 'KMX', 'DHI', 'BDX', 'GIS', 'GPS', 'WDC', 'ADNT', 'DEO', 'ARMK',
               'JWN',
               'ROST', 'EL', 'AMAT', 'SSL', 'SWK', 'PH', 'CM', 'ADP', 'WBK', 'ACM', 'K', 'TEL'],
        _2021=['WMT', 'TM', 'AAPL', 'MCK', 'ABC', 'COST', 'CAH', 'MSFT', 'WBA', 'HMC', 'KR', 'HD', 'DELL', 'TGT', 'SNE', 'BABA', 'LOW', 'PG', 'FDX', 'DIS', 'ACI',
               'VOD',
               'SYY', 'CSCO', 'ACN', 'BBY', 'TSN', 'BHP', 'TJX', 'ORCL', 'NKE', 'TECD', 'RY', 'MUFG', 'TTM', 'TD', 'TAK', 'MDT', 'BTTGY', 'DG', 'JBL', 'UNFI',
               'PFGC', 'SMFG',
               'M', 'FLEX', 'BNS', 'DLTR', 'QCOM', 'SBUX', 'JCI', 'RAD', 'V', 'MU', 'IX', 'KMX', 'DHI', 'KSS', 'IBN', 'DXC', 'BMO', 'MFG', 'NGG', 'AVT', 'GIS',
               'WRK', 'BDX',
               'CRM', 'EMR', 'WDC', 'GPS', 'ROST', 'JWN', 'DEO', 'WBK', 'ADP', 'CM', 'EL', 'PH', 'TXT', 'J', 'ACM', 'BJ', 'LB', 'ARMK', 'INFY', 'ADNT', 'AZO',
               'AMCR', 'FOXA',
               'TEL', 'NMR', 'BERY', 'JCPNQ', 'BBBY', 'LDOS', 'VEDL', 'CAG', 'SSL', 'NVDA'])

    # Bottom pe and untradable
    # stock_dict = dict(
    #     _2016=['FRTG', 'VSAT', 'CSII', 'SHAK', 'XTEG', 'CVLT', 'AXR', 'STZ', 'PCYG', 'PLT', 'MANU', 'XTEG', 'XTEG', 'ELEC', 'CYAN', 'GSIT', 'GEOS', 'GCI', 'MTRX', 'LCI',
    #            'INVN', 'OMN', 'MIME', 'NEWM', 'ELEC', 'MXC', 'NTCT', 'GHM', 'ENR', 'CSII', 'EXTR', 'PCTY', 'CUB', 'CCN', 'MITK', 'VRTU', 'QRVO', 'NMRD', 'FC', 'HBK', 'DLNO',
    #            'ARCW', 'LITE', 'HRB', 'RENT', 'KMT', 'MIME', 'PRCP', 'KTOS', 'LPG', 'TTWO', 'GWRE', 'ATGE', 'FORM', 'DAKT', 'CVLT', 'IDTI', 'NMRD', 'MOD', 'HQI', 'DJCO',
    #            'NUAN', 'SCHL', 'CCUR', 'MG', 'AMD', 'NX', 'TAL', 'EGHT', 'PINC', 'DJCO', 'PTC', 'SPGX', 'NGL', 'TNAV', 'ECEZ', 'PZZA', 'PANW', 'SKIS', 'GWRE', 'AJRD', 'ASH',
    #            'EPAY', 'QRVO', 'NMRD', 'SRDX', 'MIME', 'JVA', 'QRVO', 'VOXX', 'CGNT', 'PRCP', 'EGAN', 'MJCO', 'TEAM', 'ENZ', 'MLAB', 'PAY', 'AGTC', 'EEI'],
    #     _2017=['FRTG', 'IINX', 'VSAT', 'CSII', 'SHAK', 'XTEG', 'CVLT', 'AXR', 'STZ', 'PCYG', 'PLT', 'MANU', 'XTEG', 'XTEG', 'ELEC', 'CYAN', 'NXEO', 'PRMW', 'GSIT', 'GEOS',
    #            'GCI', 'MTRX', 'LCI', 'INVN', 'OMN', 'MIME', 'PHUN', 'NEWM', 'CRM', 'ELEC', 'MXC', 'RH', 'NTCT', 'GHM', 'ENR', 'CSII', 'EXTR', 'PCTY', 'CUB', 'MITK', 'VRTU',
    #            'QRVO', 'NMRD', 'FC', 'HBK', 'DLNO', 'ARCW', 'LITE', 'HRB', 'KMT', 'MIME', 'PRCP', 'KTOS', 'VRNT', 'LPG', 'ADSK', 'TTWO', 'GWRE', 'GWPH', 'PRMW', 'ATGE',
    #            'DAKT', 'CVLT', 'IDTI', 'KHC', 'NMRD', 'MOD', 'HQI', 'DJCO', 'NUAN', 'SCHL', 'CCUR', 'NX', 'ARWA', 'STAA', 'TAL', 'EGHT', 'PINC', 'DJCO', 'PTC', 'SPGX', 'NGL',
    #            'TNAV', 'ECEZ', 'PZZA', 'PANW', 'SKIS', 'GWRE', 'ASH', 'EPAY', 'QRVO', 'NMRD', 'CAL', 'FOSL', 'SRDX', 'MIME', 'JVA', 'QRVO', 'VOXX', 'PRCP'],
    #     _2018=['FRTG', 'IINX', 'VSAT', 'CSII', 'SHAK', 'XTEG', 'CVLT', 'AXR', 'STZ', 'PCYG', 'PLT', 'MANU', 'XTEG', 'XTEG', 'ELEC', 'CYAN', 'NXEO', 'PRMW', 'GSIT', 'GEOS',
    #            'MTRX', 'LCI', 'OMN', 'MIME', 'CRM', 'ELEC', 'MXC', 'RH', 'NTCT', 'GHM', 'ENR', 'CSII', 'EXTR', 'PCTY', 'CUB', 'MITK', 'SGH', 'VRTU', 'QRVO', 'NMRD', 'FC',
    #            'HBK', 'DLNO', 'ARCW', 'LITE', 'HRB', 'KMT', 'MIME', 'PRCP', 'VRNT', 'LPG', 'ADSK', 'TTWO', 'GWRE', 'GWPH', 'PRMW', 'ATGE', 'FORM', 'DAKT', 'CVLT', 'IDTI',
    #            'KHC', 'NMRD', 'MOD', 'HQI', 'DJCO', 'NUAN', 'SCHL', 'CCUR', 'AMD', 'NX', 'STAA', 'TAL', 'EGHT', 'PINC', 'DJCO', 'PTC', 'SPGX', 'NGL', 'TNAV', 'ECEZ', 'PANW',
    #            'SKIS', 'GWRE', 'ASH', 'EPAY', 'QRVO', 'NMRD', 'CAL', 'FOSL', 'SRDX', 'MIME', 'JVA', 'QRVO', 'VOXX', 'PRCP', 'ITGR', 'EGAN', 'AQUA', 'MJCO'],
    #     _2019=['FAT', 'FRTG', 'IINX', 'VSAT', 'HCCH', 'WINR', 'CSII', 'SHAK', 'XTEG', 'CVLT', 'AXR', 'STZ', 'PCYG', 'PLT', 'MANU', 'XTEG', 'XTEG', 'CYAN', 'NXEO', 'PRMW',
    #            'IIIV', 'GSIT', 'GEOS', 'MTRX', 'LCI', 'OMN', 'MIME', 'NEWM', 'CRM', 'MXC', 'RH', 'NTCT', 'GHM', 'ENR', 'CSII', 'EXTR', 'PCTY', 'CUB', 'MITK', 'SGH', 'VRTU',
    #            'QRVO', 'NMRD', 'FC', 'HBK', 'ARCW', 'LITE', 'HRB', 'KMT', 'MIME', 'PRCP', 'KTOS', 'VRNT', 'LPG', 'ADSK', 'HCCH', 'TTWO', 'ZS', 'GWRE', 'GWPH', 'PRMW', 'ATGE',
    #            'FORM', 'DAKT', 'XTEG', 'CVLT', 'IDTI', 'KHC', 'NMRD', 'MOD', 'HQI', 'DJCO', 'NUAN', 'QTNA', 'SCHL', 'CCUR', 'SONO', 'AMD', 'NX', 'STAA', 'TAL', 'EGHT', 'PINC',
    #            'DJCO', 'PTC', 'NGL', 'TNAV', 'PZZA', 'PANW', 'SKIS', 'GWRE', 'ASH', 'EPAY', 'QRVO', 'NMRD', 'CAL', 'FOSL', 'SRDX', 'MIME', 'JVA'],
    #     _2020=['FAT', 'IINX', 'VSAT', 'HCCH', 'WINR', 'CSII', 'SHAK', 'XTEG', 'CVLT', 'LOAC', 'AXR', 'STZ', 'PCYG', 'PLT', 'MANU', 'XTEG', 'XTEG', 'CYAN', 'PRMW', 'IIIV',
    #            'GSIT', 'GEOS', 'MTRX', 'LCI', 'OMN', 'MIME', 'CRM', 'MXC', 'LAAB', 'RH', 'TMRR', 'NTCT', 'GHM', 'ENR', 'CSII', 'EXTR', 'PCTY', 'CUB', 'MITK', 'SGH', 'VRTU',
    #            'QRVO', 'NMRD', 'FC', 'TNCP', 'LITE', 'HRB', 'KMT', 'MIME', 'PRCP', 'KTOS', 'VRNT', 'LPG', 'ADSK', 'HCCH', 'TTWO', 'ZS', 'GWRE', 'PRMW', 'ATGE', 'FORM', 'DAKT',
    #            'XTEG', 'CVLT', 'KHC', 'NMRD', 'MOD', 'DJCO', 'NUAN', 'SCHL', 'CCUR', 'SONO', 'AMD', 'NX', 'TAL', 'EGHT', 'PINC', 'DJCO', 'PTC', 'NGL', 'TNAV', 'PZZA', 'PANW',
    #            'SKIS', 'GWRE', 'ASH', 'EPAY', 'QRVO', 'NMRD', 'CAL', 'FOSL', 'PTON', 'SRDX', 'MIME', 'JVA', 'QRVO', 'VOXX', 'PRCP', 'BILL', 'EGAN'],
    #     _2021=['IINX', 'VSAT', 'HCCH', 'WINR', 'CSII', 'CVLT', 'LOAC', 'AXR', 'STZ', 'PCYG', 'PLT', 'MANU', 'CYAN', 'IIIV', 'GSIT', 'BWMY', 'GEOS', 'MTRX', 'LCI', 'FEDU',
    #            'MIME', 'ECRP', 'CRM', 'MXC', 'LAAB', 'RH', 'LSAC', 'NTCT', 'GHM', 'ENR', 'CSII', 'EXTR', 'PCTY', 'CUB', 'SGH', 'VRTU', 'QRVO', 'NMRD', 'FC', 'TNCP', 'LITE',
    #            'HRB', 'KMT', 'MIME', 'PRCP', 'VRNT', 'LPG', 'ADSK', 'HCCH', 'TTWO', 'ZS', 'GWRE', 'ATGE', 'DAKT', 'CVLT', 'NMRD', 'MOD', 'NUAN', 'SCHL', 'CCUR', 'SONO',
    #            'STAA', 'TAL', 'EGHT', 'PINC', 'PTC', 'NGL', 'TNAV', 'PANW', 'GWRE', 'ASH', 'EPAY', 'QRVO', 'NMRD', 'CAL', 'PTON', 'SRDX', 'MIME', 'QRVO', 'VOXX', 'PRCP',
    #            'BILL', 'EGAN', 'AQUA', 'MJCO', 'TEAM', 'ENZ', 'EQOS', 'LOAC', 'MLAB', 'MTA', 'AGTC', 'LTRX', 'IIIV', 'NLOK', 'DRI', 'POST', 'EGHT', 'ANGO', 'DCT']
    # )

    # 39th day; above 30 pe + tradeable; 0.0078
    # 29th day; above 30 pe + tradeable; 0.0055
    stock_dict = dict(
        _2016=['POST', 'BNED', 'CSC', 'ASH', 'CSCO', 'LRN', 'MTSI', 'RAD', 'TEAM', 'CVLT', 'PTC', 'CVLT', 'CRS', 'MLAB', 'CUB', 'EPC', 'ABM', 'NYT', 'MCK', 'CSII', 'AVX',
               'VIAV', 'RAMP', 'PZZA', 'MPSX', 'ZOES', 'MYGN', 'BDX', 'LNN', 'NUAN', 'MCK', 'GWRE', 'BRC', 'GWRE', 'SJR', 'EGHT', 'ODP', 'ASEI', 'NXGN', 'PCTY', 'LGF.B',
               'DECK', 'HAIN', 'GWRE', 'GWRE', 'CIEN', 'ZOES', 'MYCC', 'RAMP', 'PTC', 'MOG.A', 'RXN', 'LOGI', 'HABT', 'EPAY', 'AVAV', 'PRGO', 'EPAY', 'KRNY', 'FNSR', 'DLB',
               'POST', 'VSAT', 'NXGN', 'CUB', 'AIR', 'NTAP', 'VSAT', 'KTOS', 'AMD', 'MSGN', 'PLAY', 'GIMO', 'CMD', 'PLXS', 'SPH', 'EZPW', 'AIR', 'NAV', 'CRS', 'VRTU', 'HAE',
               'JJSF', 'GPN', 'KLIC', 'MNR', 'WING', 'SYY', 'CUDA', 'PCTY', 'MLAB', 'RGS', 'CALM', 'AIR', 'ABMD', 'ISLE', 'GGG', 'SHAK', 'HAIN', 'SHAK'],
        _2017=['TAST', 'WWW', 'POST', 'CSC', 'ASH', 'CSCO', 'LRN', 'MTSI', 'RAD', 'SUM', 'CRM', 'TEAM', 'CVLT', 'PTC', 'SUM', 'CVLT', 'CRS', 'CHS', 'MLAB', 'CUB', 'EPC',
               'ABM', 'NYT', 'MCK', 'CSII', 'AVX', 'SUM', 'VIAV', 'IMBI', 'RAMP', 'PZZA', 'ZOES', 'HAWK', 'MYGN', 'BDX', 'LNN', 'NUAN', 'MCK', 'GWRE', 'BRC', 'GWRE', 'SJR',
               'EGHT', 'ASEI', 'MRVL', 'NXGN', 'PCTY', 'LGF.B', 'DECK', 'HAIN', 'NORD', 'GWRE', 'GWRE', 'JNJ', 'CIEN', 'STAA', 'SUM', 'ZOES', 'MYCC', 'SLAB', 'RAMP', 'LNCE',
               'COKE', 'PTC', 'MOG.A', 'RXN', 'ISIL', 'EBF', 'LOGI', 'CRM', 'HABT', 'EPAY', 'CY', 'AVAV', 'EPAY', 'ADSK', 'ANF', 'RH', 'ANF', 'ICHR', 'KRNY', 'FNSR', 'DLB',
               'CRM', 'POST', 'VSAT', 'NXGN', 'CUB', 'CONN', 'AIR', 'NTAP', 'VSAT', 'ITGR', 'KTOS', 'MSGN', 'PLAY', 'CMD', 'PLXS', 'SPH', 'EZPW'],
        _2018=['WWW', 'POST', 'BNED', 'ASH', 'CSCO', 'LRN', 'MTSI', 'RAD', 'HLNE', 'WINS', 'COHU', 'SUM', 'CRM', 'TEAM', 'CVLT', 'PTC', 'SUM', 'CVLT', 'CRS', 'CHS', 'MLAB',
               'CUB', 'EPC', 'ABM', 'MCK', 'CSII', 'PRMW', 'AVX', 'SUM', 'VIAV', 'IMBI', 'RAMP', 'ZOES', 'HAWK', 'MYGN', 'BDX', 'LNN', 'NUAN', 'MCK', 'GWRE', 'BRC', 'GWRE',
               'SJR', 'EGHT', 'ODP', 'MRVL', 'NXGN', 'PCTY', 'LGF.B', 'DECK', 'HAIN', 'AQUA', 'GWRE', 'GWRE', 'CIEN', 'STAA', 'SUM', 'ZOES', 'SLAB', 'PSDO', 'RAMP', 'LNCE',
               'PTC', 'MOG.A', 'RXN', 'LOGI', 'CRM', 'HABT', 'EPAY', 'AVAV', 'EPAY', 'ADSK', 'WINS', 'ANF', 'RH', 'ANF', 'ICHR', 'KRNY', 'FNSR', 'DLB', 'CRM', 'POST', 'VSAT',
               'NXGN', 'CUB', 'CONN', 'AIR', 'NTAP', 'VSAT', 'ITGR', 'AMD', 'MSGN', 'PLAY', 'ACB', 'CMD', 'PLXS', 'SPH', 'EZPW', 'AIR', 'MITK'],
        _2019=['TAST', 'WWW', 'POST', 'BNED', 'ASH', 'CSCO', 'LRN', 'MTSI', 'SIGM', 'RAD', 'HLNE', 'COHU', 'SUM', 'CRM', 'TEAM', 'CVLT', 'PTC', 'SUM', 'CVLT', 'CRS', 'CHS',
               'MLAB', 'CUB', 'EPC', 'ABM', 'NYT', 'MCK', 'CSII', 'PRMW', 'AVX', 'SUM', 'VIAV', 'IMBI', 'RAMP', 'PZZA', 'SRDX', 'MYGN', 'BDX', 'LNN', 'NUAN', 'MCK', 'GWRE',
               'BRC', 'GWRE', 'SJR', 'EGHT', 'ODP', 'MRVL', 'NXGN', 'PCTY', 'LGF.B', 'DECK', 'HAIN', 'AQUA', 'GWRE', 'GWRE', 'JNJ', 'CIEN', 'STAA', 'CIVI', 'SUM', 'SLAB',
               'PSDO', 'RAMP', 'COKE', 'PTC', 'MOG.A', 'RXN', 'LOGI', 'CRM', 'HABT', 'EPAY', 'CY', 'AVAV', 'EPAY', 'ADSK', 'ANF', 'RH', 'ANF', 'ICHR', 'KRNY', 'FNSR', 'DLB',
               'DAVA', 'QTNA', 'CRM', 'POST', 'VSAT', 'NXGN', 'CUB', 'CONN', 'AIR', 'NTAP', 'VSAT', 'ITGR', 'KTOS', 'AMD', 'MSGN', 'PLAY', 'BGFV'],
        _2020=['TAST', 'WWW', 'POST', 'BNED', 'ASH', 'CSCO', 'EPAC', 'LRN', 'MTSI', 'RAD', 'HLNE', 'COHU', 'SUM', 'CRM', 'TEAM', 'CVLT', 'PTC', 'ZM', 'SUM', 'CVLT', 'CRS',
               'CHS', 'MLAB', 'CUB', 'EPC', 'ABM', 'NYT', 'MCK', 'CSII', 'PRMW', 'MSGS', 'AVX', 'SUM', 'VIAV', 'IMBI', 'RAMP', 'PZZA', 'NLOK', 'SRDX', 'MYGN', 'BDX', 'LNN',
               'NUAN', 'MCK', 'GWRE', 'BRC', 'GWRE', 'SJR', 'EGHT', 'ODP', 'MRVL', 'NXGN', 'PCTY', 'LGF.B', 'DECK', 'HAIN', 'AQUA', 'GWRE', 'GWRE', 'JNJ', 'CIEN', 'SUM',
               'SLAB', 'PSDO', 'RAMP', 'COKE', 'PTC', 'MOG.A', 'RXN', 'LOGI', 'CRM', 'EPAY', 'CY', 'AVAV', 'EPAY', 'ADSK', 'ANF', 'RH', 'ANF', 'GO', 'ICHR', 'KRNY', 'FNSR',
               'DLB', 'DAVA', 'CRM', 'POST', 'VSAT', 'NXGN', 'CUB', 'CONN', 'IFMK', 'AIR', 'BBW', 'NTAP', 'VSAT', 'KTOS', 'AMD', 'MSGN', 'PLAY'],
        _2021=['POST', 'BNED', 'ASH', 'CSCO', 'EPAC', 'LRN', 'MTSI', 'RAD', 'HLNE', 'CRM', 'TEAM', 'CVLT', 'PTC', 'ZM', 'CVLT', 'CRS', 'CHS', 'MLAB', 'CUB', 'EPC', 'MCK',
               'CSII', 'MSGS', 'VIAV', 'RAMP', 'NLOK', 'SRDX', 'MYGN', 'BDX', 'LNN', 'NUAN', 'MCK', 'SMRTQ', 'GWRE', 'BRC', 'GWRE', 'SJR', 'EGHT', 'AVCT', 'MRVL', 'NXGN',
               'PCTY', 'LGF.B', 'DECK', 'HAIN', 'AQUA', 'GWRE', 'GWRE', 'STAA', 'LAKE', 'RAMP', 'ELF', 'PTC', 'MOG.A', 'RXN', 'LOGI', 'CRM', 'AGTC', 'DGII', 'EPAY', 'AVAV',
               'EPAY', 'ADSK', 'WINS', 'ANF', 'RH', 'ANF', 'KRNY', 'DLB', 'DAVA', 'CRM', 'POST', 'VSAT', 'NXGN', 'CUB', 'CONN', 'IFMK', 'AIR', 'BBW', 'NTAP', 'VSAT', 'MSGN',
               'PLAY', 'ACB', 'CMD', 'PLXS', 'SPH', 'AIR', 'CRS', 'VRTU', 'VRNT', 'HAE', 'JJSF', 'VRNT', 'KLIC', 'MNR', 'TAK', 'MANU', 'SYY', 'PCTY']
    )

    # 39th day; High netinc; 0.0013
    # 29th day; High netinc; 0.0010;
    # stock_dict = dict(
    #     _2016=['TLK', 'KEP', 'CIB', 'SHG', 'TM', 'KB', 'MUFG', 'SKM', 'LPL', 'SMFG', 'BCH', 'KT', 'NTTYY', 'HMC', 'BSAC', 'DCMYY', 'TSM', 'CAJ', 'PKX', 'CCU', 'TTM',
    #            'IBN', 'CHL', 'HDB', 'WIT', 'AAPL', 'MBT', 'CHT', 'PTR', 'NVO', 'LFC', 'BIDU', 'SNP', 'SSL', 'ITUB', 'BABA', 'BRK.B', 'WFC', 'JPM', 'RDY', 'PHI',
    #            'CEO', 'ASX', 'BBD', 'GILD', 'VZ', 'NVS', 'FMX', 'C', 'GOOGL', 'XOM', 'BAC', 'ERIC', 'HSBC', 'T', 'IBM', 'ABEV', 'MSFT', 'INTC', 'FNMA', 'TV', 'CHU',
    #            'KOF', 'ORCL', 'RY', 'GM', 'YNDX', 'SPIL', 'CSCO', 'GSK', 'DIS', 'TFCF', 'BUD', 'CMCSA', 'WBK', 'TD', 'DD', 'AAL', 'F', 'KO', 'UAL', 'MDLZ', 'PG',
    #            'BNS', 'PFE', 'AMGN', 'PM', 'NTES', 'FMCC', 'V', 'UBS', 'MS', 'GS', 'VOD', 'USB', 'UNH', 'PRU', 'PEP', 'MET', 'QCOM'],
    #     _2017=['TLK', 'KEP', 'CIB', 'SHG', 'EC', 'TM', 'KB', 'SKM', 'PKX', 'LPL', 'SMFG', 'MUFG', 'KT', 'NTTYY', 'BCH', 'DCMYY', 'BSAC', 'EOCCY', 'HMC', 'TSM', 'IX',
    #            'CAJ', 'SNE', 'CCU', 'HDB', 'CHL', 'IBN', 'TTM', 'NJDCY', 'WIT', 'BABA', 'MBT', 'SNP', 'AAPL', 'CHT', 'NVO', 'FMX', 'BRK.B', 'JPM', 'WFC', 'ITUB',
    #            'ASX', 'MSFT', 'RDY', 'PHI', 'GOOGL', 'LFC', 'BBD', 'BAC', 'JNJ', 'AGN', 'C', 'WMT', 'MO', 'GILD', 'SSL', 'VZ', 'T', 'ABEV', 'FNMA', 'IBM', 'BIDU',
    #            'NTES', 'CSCO', 'KOF', 'PG', 'RY', 'INTC', 'FB', 'EBR', 'SPIL', 'GM', 'DIS', 'ORCL', 'TD', 'CMCSA', 'PTR', 'XOM', 'FMCC', 'AMGN', 'GE', 'WBK', 'GS',
    #            'EBAY', 'PFE', 'BNS', 'UNH', 'HD', 'PM', 'YNDX', 'LN', 'NVS', 'BMA', 'KO', 'PEP', 'TOT', 'RAI', 'V', 'MS', 'ABBV'],
    #     _2018=['TLK', 'EC', 'KB', 'SHG', 'PKX', 'CIB', 'SKM', 'LPL', 'TM', 'KEP', 'NTTYY', 'DCMYY', 'SMFG', 'HMC', 'BCH', 'BSAC', 'KT', 'EOCCY', 'TSM', 'CAJ',
    #            'MUFG', 'CCU', 'HDB', 'CHL', 'IBN', 'WIT', 'SNE', 'TTM', 'MBT', 'VEDL', 'SNP', 'AAPL', 'BRK.B', 'BABA', 'CHT', 'NVO', 'BTI', 'FMX', 'LFC', 'VZ', 'T',
    #            'MSFT', 'CEO', 'ASX', 'PTR', 'ITUB', 'CMCSA', 'JPM', 'WFC', 'PFE', 'SSL', 'XOM', 'BIDU', 'BAC', 'BBD', 'FB', 'PG', 'TEO', 'WMT', 'PHI', 'GRVY',
    #            'RDS.A', 'YPF', 'GOOGL', 'RDY', 'RY', 'KHC', 'HSBC', 'UNP', 'NTES', 'UNH', 'TD', 'MO', 'CHTR', 'CSCO', 'INTC', 'ORCL', 'YNDX', 'CVX', 'BMA', 'DIS',
    #            'TGS', 'PAM', 'RIO', 'TOT', 'BA', 'SNY', 'LN', 'BNS', 'WBK', 'HD', 'PRU', 'EDN', 'NGG', 'F', 'NVS', 'BUD', 'ABEV', 'V', 'UMC'],
    #     _2019=['TLK', 'EC', 'SHG', 'SKM', 'KB', 'CIB', 'TM', 'PKX', 'MUFG', 'HMC', 'SMFG', 'KT', 'BCH', 'BSAC', 'SNE', 'TSM', 'CCU', 'IX', 'CAJ', 'TAK', 'HDB',
    #            'CHL', 'WIT', 'IBN', 'TTM', 'BABA', 'SNP', 'AAPL', 'PTR', 'CEO', 'YNDX', 'NVO', 'YPF', 'CHT', 'FMX', 'GOOGL', 'JPM', 'BAC', 'BIDU', 'ASX', 'CEPU',
    #            'ITUB', 'RDS.A', 'WFC', 'FB', 'CHA', 'INTC', 'XOM', 'T', 'C', 'TGS', 'PAM', 'BBD', 'MSFT', 'FNMA', 'VZ', 'JNJ', 'KOF', 'CVX', 'MU', 'VEDL', 'HSBC',
    #            'RIO', 'EBR', 'NVS', 'DIS', 'PEP', 'RY', 'AVGO', 'UNH', 'CMCSA', 'TOT', 'LFC', 'TD', 'PFE', 'ABEV', 'BA', 'GS', 'V', 'CHU', 'AMZN', 'WMT', 'RDY',
    #            'PG', 'BP', 'UL', 'UN', 'BMA', 'VIV', 'MS', 'SSL', 'IBM', 'HD', 'BNS', 'AMGN', 'TEO', 'WBK', 'GM', 'PM', 'NGG'],
    #     _2020=['TLK', 'EC', 'SHG', 'KB', 'CIB', 'TM', 'PKX', 'SNE', 'SKM', 'MUFG', 'KT', 'BSAC', 'HMC', 'SMFG', 'TSM', 'IX', 'HDB', 'CCU', 'TAK', 'CAJ', 'CHL',
    #            'WIT', 'BABA', 'BRK.B', 'CEO', 'LFC', 'SNP', 'AAPL', 'MBT', 'VEDL', 'PTR', 'IBN', 'MSFT', 'NVO', 'JPM', 'GOOGL', 'CHT', 'BAC', 'ITUB', 'NTES', 'INTC',
    #            'BBD', 'WFC', 'C', 'VZ', 'RDY', 'FB', 'ASX', 'PFE', 'RDS.A', 'JNJ', 'XOM', 'FNMA', 'T', 'UNH', 'CMCSA', 'RY', 'YNDX', 'TGS', 'PAM', 'KOF', 'JD', 'V',
    #            'ABEV', 'NVS', 'TD', 'CSCO', 'AMZN', 'LBTYA', 'CHU', 'TOT', 'HD', 'ORCL', 'DIS', 'PBR', 'MRK', 'IBM', 'MS', 'KO', 'BUD', 'GS', 'WUBA', 'BNS', 'LLY',
    #            'BHP', 'MA', 'RIO', 'ABBV', 'AMGN', 'EBR', 'HSBC', 'PEP', 'FMCC', 'COP', 'PM', 'TCOM', 'USB', 'WBK', 'AXP', 'GM'],
    #     _2021=['TM', 'SNE', 'HMC', 'MUFG', 'IX', 'HDB', 'SMFG', 'BABA', 'WIT', 'IBN', 'AAPL', 'MSFT', 'TAK', 'RDY', 'WMT', 'PG', 'TD', 'RY', 'HD', 'CSCO', 'V',
    #            'ORCL', 'BHP', 'BNS', 'VMW', 'QCOM', 'ACN', 'BMO', 'MDT', 'DELL', 'LOW', 'COST', 'NLOK', 'CM', 'TGT', 'TJX', 'EA', 'NVDA', 'MU', 'NKE', 'ADP', 'DHI',
    #            'INFY', 'WBK', 'LRCX', 'GIS', 'TSN', 'EMR', 'APD', 'INTU', 'AZO', 'DG', 'ROST', 'KR', 'CPB', 'MRVL', 'BBY', 'DEO', 'FDX', 'NGG', 'ADI', 'KLAC', 'PH',
    #            'HOLX', 'PAYX', 'ROK', 'STX', 'FOXA', 'CLX', 'SBUX', 'HRL', 'MCK', 'KMX', 'CTAS', 'BDX', 'CAG', 'DLTR', 'BF.B', 'NTAP', 'RJF', 'TXT', 'SWKS', 'BEN',
    #            'XLNX', 'SJM', 'ULTA', 'CPRT', 'TDG', 'KSS', 'SJR', 'EL', 'VFC', 'LDOS', 'MXIM', 'RYAAY', 'LULU', 'JCI', 'RMD', 'AMCR', 'ATO']
    # )

    # 39th day; highest above 30pe+ and tradeable fixed 1; 0.0035
    # 29th day; highest above 30pe+ and tradeable fixed 1; 0.0039
    stock_dict = dict(
        _2016=['GERN', 'BSFT', 'CSC', 'INCY', 'PENN', 'PCRX', 'MITT', 'FTI', 'TRU', 'SGRY', 'PEN', 'FTNT', 'YRCW', 'REXR', 'EGRX', 'AMZN', 'SNCR', 'ZOES', 'ATHN',
               'GWRE', 'BRC', 'NFLX', 'ULTI', 'EGHT', 'ODP', 'ASEI', 'CX', 'CCOI', 'ALXN', 'FNV', 'CIEN', 'BKI', 'SPSC', 'MINI', 'FCPT', 'ENV', 'LOGI', 'AEM',
               'HABT', 'MDSO', 'AVAV', 'FWONA', 'PRGO', 'RKUS', 'MOMO', 'ORC', 'EXAM', 'KRNY', 'FNSR', 'CXO', 'OLED', 'XXIA', 'ACOR', 'PNM', 'NUE', 'SSNC', 'PLAY',
               'VIRT', 'GIMO', 'PRTY', 'KMI', 'UNVR', 'ZBH', 'PRE', 'KRG', 'AAWW', 'HAE', 'ATML', 'NBHC', 'MBLY', 'GPRE', 'CCRN', 'BLKB', 'NPTN', 'AVID', 'LOGM',
               'SINA', 'IMMR', 'AIR', 'SCU', 'ISLE', 'WB', 'DOC', 'LYG', 'IRC', 'ITRI', 'PAYC', 'HTA', 'CCJ', 'BCR', 'OAK', 'HSC', 'RGEN', 'TVPT', 'ACAT', 'EQT',
               'EQIX', 'WES', 'EDR', 'QTS'],
        _2017=['TAST', 'MPLX', 'PEIX', 'ZLTQ', 'KRG', 'MTSI', 'BSFT', 'SEMG', 'TEAM', 'CVLT', 'BP', 'LOGM', 'KW', 'CHS', 'CUB', 'CEO', 'CC', 'TEVA', 'CMG', 'ACHC',
               'ECHO', 'HAWK', 'TRP', 'SUI', 'ARRS', 'CHU', 'GWRE', 'CQH', 'NFLX', 'WEB', 'SUM', 'ZOES', 'MYCC', 'IQV', 'GKOS', 'RAMP', 'LNCE', 'ELF', 'SPSC',
               'SALE', 'RMBS', 'TRCO', 'ERIC', 'HSTM', 'OFIX', 'BIO', 'SWC', 'OFC', 'WEX', 'GTT', 'CG', 'EQIX', 'ATHN', 'MORE', 'CONE', 'ICHR', 'INCY', 'ULTI',
               'DEA', 'ANIP', 'BSM', 'NXPI', 'AXTA', 'SBAC', 'WAGE', 'FSP', 'VSAT', 'NXGN', 'FTNT', 'SHPG', 'PCH', 'HPP', 'ITGR', 'MSGN', 'AMZN', 'SITC', 'SU',
               'SNCR', 'ERII', 'RP', 'SPH', 'ORC', 'GPT', 'BUD', 'MITK', 'AG', 'CHEF', 'PEN', 'CRS', 'KMPR', 'VRNT', 'LN', 'BKI', 'OMCL', 'ALRM', 'CCOI', 'TGE',
               'HSBC', 'PTR', 'PTHN'],
        _2018=['IONS', 'FTAI', 'WWW', 'RP', 'ASH', 'IT', 'SPSC', 'QTS', 'FLS', 'NSA', 'LRN', 'AM', 'KMI', 'RAD', 'HLNE', 'MCRN', 'VIRT', 'SID', 'EURN', 'BIP', 'PTC',
               'MELI', 'SUM', 'PTEN', 'WAT', 'EPC', 'ABM', 'NYT', 'PEN', 'IMAX', 'RAMP', 'ACIW', 'ULTI', 'CARG', 'BSIG', 'REI', 'GRA', 'LX', 'MRVL', 'IR', 'CCOI',
               'PCTY', 'LGF.B', 'DECK', 'BSX', 'ZBRA', 'NCSM', 'GWRE', 'JNJ', 'AAXN', 'USM', 'TREE', 'ALB', 'PSDO', 'TTEC', 'SRCL', 'AMPH', 'FTNT', 'BCO', 'LGND',
               'GOL', 'AXTA', 'ORC', 'DEA', 'ABT', 'GRPN', 'WINS', 'RH', 'ANF', 'HSC', 'AMZN', 'SBAC', 'QGEN', 'ATVI', 'EQC', 'HR', 'CRY', 'CRM', 'WPM', 'POST',
               'OFIX', 'KO', 'VSAT', 'EQIX', 'OPI', 'NFLX', 'BGFV', 'MTW', 'CCI', 'APPF', 'CHCT', 'SWIR', 'VRTX', 'DD', 'NAV', 'KRG', 'DLR', 'PUMP', 'COG', 'CWH'],
        _2019=['AXS', 'CONE', 'EHTH', 'WP', 'CSCO', 'PODD', 'GNL', 'PDCE', 'CEVA', 'PEN', 'CSII', 'ATUS', 'AVX', 'SUM', 'SAIL', 'IMBI', 'PZZA', 'BDX', 'MCK', 'SWCH',
               'SJR', 'NSA', 'ENV', 'SBAC', 'NXGN', 'PEGA', 'UNIT', 'TGP', 'HAIN', 'AQUA', 'CATM', 'PGRE', 'ACIA', 'NBIX', 'MEET', 'ALXN', 'STAA', 'AMH', 'APA',
               'PTC', 'CXP', 'Y', 'CRM', 'SSRM', 'ADSW', 'ZNGA', 'EPAY', 'PAAS', 'EIGI', 'NUVA', 'BEP', 'FTAI', 'ANF', 'NEO', 'CWST', 'TERP', 'LBRDA', 'DLB', 'QTNA',
               'SMLP', 'SYNH', 'DEA', 'ICUI', 'CUB', 'CONN', 'TPIC', 'NTAP', 'TDC', 'GDDY', 'ACB', 'PLXS', 'SAND', 'FARO', 'RGEN', 'OSPN', 'SJI', 'ELAN', 'PRNB',
               'RP', 'AYX', 'TCMD', 'BRK.B', 'ALTR', 'INCY', 'CHCT', 'RGS', 'ULTI', 'GCP', 'ALRM', 'CARB', 'ABMD', 'VMW', 'HSKA', 'INXN', 'SHAK', 'JBGS', 'SSNC',
               'ANGI', 'RCII', 'CMPR'],
        _2020=['CWK', 'CEVA', 'CIIC', 'AAXN', 'PRA', 'ALUS', 'NOK', 'SUM', 'SBSW', 'SFTW', 'QTS', 'DRQ', 'ONTO', 'PODD', 'CVLT', 'F', 'MCK', 'PRMW', 'BAND', 'CRY',
               'MSGS', 'VIAV', 'DOYU', 'NLOK', 'MYGN', 'LNN', 'GSHD', 'GWRE', 'SWCH', 'PEAK', 'INOV', 'NEO', 'PNTG', 'WTTR', 'E', 'BANC', 'BLKB', 'MMSI', 'SWI',
               'FIS', 'NBIX', 'BIP', 'AMG', 'CXP', 'SLAB', 'FMX', 'UMH', 'DEA', 'COKE', 'AYX', 'RXN', 'RGEN', 'TREE', 'VRT', 'CY', 'HTA', 'BKR', 'EPAY', 'FOE',
               'UEIC', 'DXCM', 'CSTL', 'CVA', 'GO', 'LBRDA', 'GSX', 'UNIT', 'AUDC', 'SBAC', 'CONE', 'LINX', 'LXFR', 'SSTI', 'ELAN', 'MNRL', 'IQV', 'BCO', 'WPM',
               'KTOS', 'AMD', 'OPRT', 'BIDU', 'NXPI', 'FSP', 'GMRE', 'EZPW', 'AIR', 'COLD', 'ZNGA', 'HPP', 'KRUS', 'VRTU', 'VICR', 'HPK', 'FWONA', 'GKOS', 'VFF',
               'GPN', 'KLIC', 'ATUS'],
        _2021=['POST', 'EPAC', 'CRM', 'ZM', 'CRS', 'MLAB', 'SRDX', 'NUAN', 'MOG.A', 'ADSK', 'DAVA', 'AIR', 'CMD', 'VRNT', 'JJSF', 'TAK', 'SYY', 'PCTY', 'CALM',
               'HQY', 'SLP', 'STAA', 'SMTC', 'SBUX', 'MSGE', 'AIT', 'EL', 'VIAV', 'JBL', 'DSGX', 'NXGN', 'BDX', 'AMSWA', 'MTN', 'PTC', 'VEEV', 'WBA', 'SMPL', 'HAE',
               'CTLT', 'REX', 'NEOG', 'NKE', 'VAR', 'CMTL', 'NVDA', 'DGII', 'BRKS', 'FICO', 'MRCY', 'SLQT', 'NSSC', 'TDG', 'TFSL', 'OTEX', 'DEO', 'FLEX', 'JCI',
               'JKHY', 'EGAN', 'HLNE', 'WDFC', 'EDU', 'LULU', 'LRN', 'RMD', 'EXPO', 'TECH', 'INTU', 'LITE', 'THR', 'RGLD', 'GBDC', 'COST', 'SR', 'FIVE', 'BF.B',
               'ADI', 'V', 'FDS', 'EAT', 'SXI', 'AVAV', 'APD', 'MSFT', 'BRBR', 'EXP', 'KEM', 'LAKE', 'VFC', 'AAPL', 'FRHC', 'ETH', 'GMS', 'TTWO', 'ABMD', 'WGO',
               'RAVN', 'RPM', 'PAHC']
    )

    num_days_under = 40
    min_days_under_sought = 29
    num_hold_days = 1
    all_rois = []
    for _year in sorted(stock_dict.keys()):
        year = int(_year[1:])
        tickers = stock_dict[_year]
        year_rois = []

        start_dt_str = f"{year}-01-01"
        start_dt = date_utils.parse_std_datestring(start_dt_str)

        end_dt = start_dt + timedelta(days=365)
        dr = DateRange(from_date=start_dt, to_date=end_dt)

        df = ticker_utils.get_maos(tickers=tickers, dr=dr, num_days_under=num_days_under)

        tar_dts = []
        for k in range(365):
            target_dt = start_dt + timedelta(days=k) + timedelta(hours=11)

            is_closed, _ = date_utils.is_stock_market_closed(target_dt)
            if is_closed:
                continue
            else:
                tar_dts.append(target_dt)

        for target_dt in tar_dts:
            target_dt_str = date_utils.get_standard_ymd_format(target_dt)

            df_one_day = df[df["date"] == target_dt_str]

            df_all = []
            col_days_down = "days_down_in_a_row"
            df_one_day[col_days_down] = -1
            for x in range(num_days_under - 1, 0, -1):
                col = f"is_long_down_{x}"

                df_true = df_one_day[df_one_day[col] == True].copy()
                if df_true.shape[0] > 0:
                    df_true[col_days_down] = x
                    df_all.append(df_true)

                df_one_day = df_one_day[df_one_day[col] == False].copy()

            if len(df_all) > 0:
                df_one_day = pd.concat(df_all)

                df_one_day = df_one_day.sort_values(col_days_down, ascending=False)

                mean_under_today = df_one_day[col_days_down].mean()

                df_one_day = df_one_day[df_one_day[col_days_down] >= min_days_under_sought].copy()

                import numpy as np
                for ndx, row in df_one_day.iterrows():
                    ticker = row["ticker"]
                    roi = row[f"fut_day_{num_hold_days}_roi"]
                    if not np.isnan(roi) and roi is not None:
                        mean_roi = 0
                        if len(year_rois) > 0:
                            mean_roi = statistics.mean(year_rois)
                        logger.info(f"{target_dt_str}: {ticker} roi: {roi:.4f}: mean_under_today: {mean_under_today:.1f}; mean roi so far: {mean_roi:.4f}")
                        year_rois.append(roi)

        if len(year_rois) > 0:
            logger.info(f"{year}: Num inv: {len(year_rois)}; Mean roi: {statistics.mean(year_rois):.4f}")
            all_rois += year_rois

    if len(all_rois) > 0:
        init_inv = 100
        real_ret = get_real_return(all_rois, init_inv)
        logger.info(f"Initial: 100: End with: {real_ret:,.2f}: overall mean roi: {statistics.mean(all_rois):.4f}")


def get_real_return(all_rois: List[float], init_inv: float):
    inv = init_inv
    for r in all_rois:
        inv = inv + (inv * r)
    return inv


def test_get_days_under():
    tickers = ['IBM', 'NVDA', 'GOOGL', 'ABC', 'GE', 'FB']

    # NOTE: 2021-09-09: chris.flesche: # 39th day; above 30 pe + tradeable; 0.0078
    tickers = ['POST', 'BNED', 'ASH', 'CSCO', 'EPAC', 'LRN', 'MTSI', 'RAD', 'HLNE', 'CRM', 'TEAM', 'CVLT', 'PTC', 'ZM', 'CVLT', 'CRS', 'CHS', 'MLAB', 'CUB', 'EPC', 'MCK',
               'CSII', 'MSGS', 'VIAV', 'RAMP', 'NLOK', 'SRDX', 'MYGN', 'BDX', 'LNN', 'NUAN', 'MCK', 'SMRTQ', 'GWRE', 'BRC', 'GWRE', 'SJR', 'EGHT', 'AVCT', 'MRVL', 'NXGN',
               'PCTY', 'LGF.B', 'DECK', 'HAIN', 'AQUA', 'GWRE', 'GWRE', 'STAA', 'LAKE', 'RAMP', 'ELF', 'PTC', 'MOG.A', 'RXN', 'LOGI', 'CRM', 'AGTC', 'DGII', 'EPAY', 'AVAV',
               'EPAY', 'ADSK', 'WINS', 'ANF', 'RH', 'ANF', 'KRNY', 'DLB', 'DAVA', 'CRM', 'POST', 'VSAT', 'NXGN', 'CUB', 'CONN', 'IFMK', 'AIR', 'BBW', 'NTAP', 'VSAT', 'MSGN',
               'PLAY', 'ACB', 'CMD', 'PLXS', 'SPH', 'AIR', 'CRS', 'VRTU', 'VRNT', 'HAE', 'JJSF', 'VRNT', 'KLIC', 'MNR', 'TAK', 'MANU', 'SYY', 'PCTY']

    date_str = "2021-09-01"
    start_dt = date_utils.parse_std_datestring(date_str)
    start_dt = start_dt + timedelta(hours=10)

    num_days_to_test = 30
    num_days_under = 40
    for i in range(num_days_to_test):
        start_dt = start_dt + timedelta(days=1)
        start_dt_str = date_utils.get_standard_ymd_format(start_dt)
        is_closed, _ = date_utils.is_stock_market_closed(start_dt)
        if is_closed:
            logger.info(f"Stock market closed {start_dt_str}")
            continue

        end_dt = start_dt + timedelta(days=1)
        end_dt_str = date_utils.get_standard_ymd_format(end_dt)
        dr = DateRange.from_date_strings(from_date_str=start_dt_str, to_date_str=end_dt_str)

        logger.info(f"Looking at {start_dt_str} to {end_dt_str}")

        df = ticker_utils.get_maos(tickers=tickers, dr=dr, num_days_under=num_days_under, add_future_cols=False)

        # logger.info(f"Size of df: {df.shape[0]}")

        df = df.sort_values(by=["ticker", "date"])

        # logger.info(df[["ticker", "date"]].head(100))

        # cols = [f"is_long_down_{x}" for x in range(num_days_under)]
        # cols.append("ticker")
        # logger.info(df[cols].head())

        df_all = []
        col_days_down = "days_down_in_a_row"
        df[col_days_down] = -1
        for x in range(num_days_under - 1, 0, -1):
            col = f"is_long_down_{x}"

            df_true = df[df[col] == True].copy()
            if df_true.shape[0] > 0:
                df_true[col_days_down] = x
                df_all.append(df_true)

            df = df[df[col] == False].copy()

        if len(df_all) > 0:
            df = pd.concat(df_all)

        df = df.sort_values(col_days_down, ascending=False)

        logger.info(df[["ticker", col_days_down]].head())
