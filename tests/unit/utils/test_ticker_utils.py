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
        _2016=['AAPL', 'TM', 'MCK', 'ABC', 'COST', 'HMC', 'KR', 'WBA', 'CAH', 'MSFT', 'NTTYY', 'JXHLY', 'PG', 'SNE', 'PEP', 'VOD', 'INTC', 'DIS', 'HPQ', 'CSCO', 'SYY', 'FDX',
               'BHP', 'TTM', 'MUFG', 'TSN', 'ORCL', 'DCMYY', 'SNEX', 'S', 'ACN', 'HPE', 'NKE', 'TFCF', 'DE', 'SMFG', 'RY', 'BTTGY', 'RAD', 'FLEX', 'QCOM', 'MFG', 'TD', 'MDT',
               'ADNT', 'NGG', 'SBUX', 'DXC', 'BNS', 'IX', 'ACM', 'JBL', 'AVT', 'GIS', 'JCI', 'DEO', 'NGL', 'EMR', 'MU', 'CCL', 'WFM', 'PFGC', 'SSL', 'WBK', 'MON', 'BMO',
               'WDC', 'IBN', 'ARMK', 'KMX', 'V', 'STX', 'NMR', 'SNX', 'SVU', 'VIAB', 'KYOCY', 'PH', 'BABA', 'TEL', 'J', 'BBBY', 'CSX', 'ODP', 'VEDL', 'WRK', 'ADP', 'DHI',
               'EL', 'HSIC', 'CM', 'BDX', 'AZO', 'NAV', 'PCP', 'TYC', 'AMAT', 'LEN', 'HRL', 'CAG'],
        _2017=['WMT', 'TM', 'AAPL', 'MCK', 'ABC', 'CAH', 'COST', 'HMC', 'WBA', 'KR', 'NTTYY', 'MSFT', 'HD', 'TGT', 'SNE', 'PG', 'LOW', 'PEP', 'JXHLY', 'INTC', 'DELL', 'FDX',
               'ACI', 'VOD', 'SYY', 'DIS', 'HPQ', 'CSCO', 'DCMYY', 'TTM', 'BBY', 'TSN', 'ORCL', 'ACN', 'BHP', 'NKE', 'S', 'TJX', 'RY', 'SMFG', 'BTTGY', 'DE', 'MDT', 'SNEX',
               'HPE', 'MUFG', 'TFCF', 'TD', 'KHC', 'M', 'TECD', 'IX', 'USFD', 'FLEX', 'BABA', 'RAD', 'JCI', 'SBUX', 'QCOM', 'SHLDQ', 'DG', 'BNS', 'DLTR', 'MU', 'MFG', 'KSS',
               'WDC', 'JBL', 'NGG', 'V', 'SPLS', 'ACM', 'AVGO', 'CCL', 'IBN', 'AVT', 'BMO', 'WBK', 'SNX', 'PFGC', 'ADNT', 'DEO', 'KMX', 'TAK', 'GIS', 'GPS', 'EMR', 'WRK',
               'JWN', 'AMAT', 'MON', 'ARMK', 'TXT', 'DHI', 'VIAB', 'SSL', 'SWK', 'JCPNQ', 'ROST', 'K'],
        _2018=['WMT', 'AAPL', 'TM', 'MCK', 'ABC', 'COST', 'CAH', 'HMC', 'WBA', 'KR', 'MSFT', 'HD', 'JNJ', 'DELL', 'SNE', 'TGT', 'INTC', 'LOW', 'PG', 'FDX', 'VOD', 'PEP',
               'ACI', 'DIS', 'SYY', 'HPQ', 'CSCO', 'TTM', 'BHP', 'BBY', 'ACN', 'TSN', 'BABA', 'ORCL', 'MUFG', 'DE', 'NKE', 'TJX', 'TECD', 'BTTGY', 'SMFG', 'RY', 'S', 'HPE',
               'TFCF', 'MU', 'MDT', 'TD', 'SNEX', 'KHC', 'FLEX', 'IX', 'M', 'SBUX', 'USFD', 'DG', 'JCI', 'MFG', 'QCOM', 'DLTR', 'JBL', 'BNS', 'DXC', 'RAD', 'NGG', 'AVGO',
               'WDC', 'V', 'LEN', 'KSS', 'SNX', 'AVT', 'CCL', 'IBN', 'PFGC', 'BMO', 'ADNT', 'EMR', 'KMX', 'AMAT', 'SHLDQ', 'WRK', 'DEO', 'DHI', 'BDX', 'GPS', 'WBK', 'ARMK',
               'GIS', 'TAK', 'JWN', 'PH', 'SVU', 'ROST', 'TEL', 'SWK', 'TXT', 'VEDL', 'ACM', 'KYOCY'],
        _2019=['WMT', 'AAPL', 'TM', 'MCK', 'ABC', 'COST', 'CAH', 'HMC', 'WBA', 'KR', 'MSFT', 'HD', 'JNJ', 'DELL', 'SNE', 'TGT', 'INTC', 'LOW', 'PG', 'FDX', 'VOD', 'PEP',
               'ACI', 'DIS', 'SYY', 'HPQ', 'CSCO', 'TTM', 'BHP', 'BBY', 'ACN', 'TSN', 'BABA', 'ORCL', 'MUFG', 'DE', 'NKE', 'TJX', 'TECD', 'BTTGY', 'SMFG', 'RY', 'S', 'HPE',
               'TFCF', 'MU', 'MDT', 'TD', 'SNEX', 'KHC', 'FLEX', 'IX', 'M', 'SBUX', 'USFD', 'DG', 'JCI', 'MFG', 'QCOM', 'DLTR', 'JBL', 'BNS', 'DXC', 'RAD', 'NGG', 'AVGO',
               'WDC', 'V', 'LEN', 'KSS', 'SNX', 'AVT', 'CCL', 'IBN', 'PFGC', 'BMO', 'ADNT', 'EMR', 'KMX', 'AMAT', 'SHLDQ', 'WRK', 'DEO', 'DHI', 'BDX', 'GPS', 'WBK', 'ARMK',
               'GIS', 'TAK', 'JWN', 'PH', 'SVU', 'ROST', 'TEL', 'SWK', 'TXT', 'VEDL', 'ACM', 'KYOCY'],
        _2020=['WMT', 'TM', 'AAPL', 'MCK', 'ABC', 'COST', 'CAH', 'HMC', 'WBA', 'MSFT', 'KR', 'HD', 'DELL', 'JNJ', 'SNE', 'TGT', 'INTC', 'LOW', 'FDX', 'DIS', 'PG', 'PEP',
               'ACI', 'SYY', 'HPQ', 'VOD', 'BABA', 'CSCO', 'BHP', 'TTM', 'ACN', 'BBY', 'TSN', 'ORCL', 'DE', 'NKE', 'TJX', 'TECD', 'RY', 'MUFG', 'S', 'SNEX', 'TD', 'BTTGY',
               'MDT', 'HPE', 'SMFG', 'SBUX', 'FLEX', 'USFD', 'DG', 'JBL', 'KHC', 'M', 'QCOM', 'JCI', 'SNX', 'BNS', 'MU', 'V', 'DLTR', 'AVGO', 'UNFI', 'LEN', 'IX', 'RAD',
               'CCL', 'DXC', 'KSS', 'NGG', 'PFGC', 'AVT', 'BMO', 'TAK', 'IBN', 'MFG', 'EMR', 'WRK', 'KMX', 'DHI', 'BDX', 'GIS', 'GPS', 'WDC', 'ADNT', 'DEO', 'ARMK', 'JWN',
               'ROST', 'EL', 'AMAT', 'SSL', 'SWK', 'PH', 'CM', 'ADP', 'WBK', 'ACM', 'K', 'TEL'],
        _2021=['WMT', 'TM', 'AAPL', 'MCK', 'ABC', 'COST', 'CAH', 'MSFT', 'WBA', 'HMC', 'KR', 'HD', 'DELL', 'TGT', 'SNE', 'BABA', 'LOW', 'PG', 'FDX', 'DIS', 'ACI', 'VOD',
               'SYY', 'CSCO', 'ACN', 'BBY', 'TSN', 'BHP', 'TJX', 'ORCL', 'NKE', 'TECD', 'RY', 'MUFG', 'TTM', 'TD', 'TAK', 'MDT', 'BTTGY', 'DG', 'JBL', 'UNFI', 'PFGC', 'SMFG',
               'M', 'FLEX', 'BNS', 'DLTR', 'QCOM', 'SBUX', 'JCI', 'RAD', 'V', 'MU', 'IX', 'KMX', 'DHI', 'KSS', 'IBN', 'DXC', 'BMO', 'MFG', 'NGG', 'AVT', 'GIS', 'WRK', 'BDX',
               'CRM', 'EMR', 'WDC', 'GPS', 'ROST', 'JWN', 'DEO', 'WBK', 'ADP', 'CM', 'EL', 'PH', 'TXT', 'J', 'ACM', 'BJ', 'LB', 'ARMK', 'INFY', 'ADNT', 'AZO', 'AMCR', 'FOXA',
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

    # 39th day; above 30 pe + tradeable; 0.0017
    # 29th day; above 30 pe + tradeable; 0.0025
    # stock_dict = dict(
    #     _2015=['POST', 'BNED', 'CSC', 'ASH', 'CSCO', 'LRN', 'MTSI', 'RAD', 'TEAM', 'CVLT', 'PTC', 'CVLT', 'CRS', 'MLAB', 'CUB', 'EPC', 'ABM', 'NYT', 'MCK', 'CSII', 'AVX',
    #            'VIAV', 'RAMP', 'PZZA', 'MPSX', 'ZOES', 'MYGN', 'BDX', 'LNN', 'NUAN', 'MCK', 'GWRE', 'BRC', 'GWRE', 'SJR', 'EGHT', 'ODP', 'ASEI', 'NXGN', 'PCTY', 'LGF.B',
    #            'DECK', 'HAIN', 'GWRE', 'GWRE', 'CIEN', 'ZOES', 'MYCC', 'RAMP', 'PTC', 'MOG.A', 'RXN', 'LOGI', 'HABT', 'EPAY', 'AVAV', 'PRGO', 'EPAY', 'KRNY', 'FNSR', 'DLB',
    #            'POST', 'VSAT', 'NXGN', 'CUB', 'AIR', 'NTAP', 'VSAT', 'KTOS', 'AMD', 'MSGN', 'PLAY', 'GIMO', 'CMD', 'PLXS', 'SPH', 'EZPW', 'AIR', 'NAV', 'CRS', 'VRTU', 'HAE',
    #            'JJSF', 'GPN', 'KLIC', 'MNR', 'WING', 'SYY', 'CUDA', 'PCTY', 'MLAB', 'RGS', 'CALM', 'AIR', 'ABMD', 'ISLE', 'GGG', 'SHAK', 'HAIN', 'SHAK'],
    #     _2016=['TAST', 'WWW', 'POST', 'CSC', 'ASH', 'CSCO', 'LRN', 'MTSI', 'RAD', 'SUM', 'CRM', 'TEAM', 'CVLT', 'PTC', 'SUM', 'CVLT', 'CRS', 'CHS', 'MLAB', 'CUB', 'EPC',
    #            'ABM', 'NYT', 'MCK', 'CSII', 'AVX', 'SUM', 'VIAV', 'IMBI', 'RAMP', 'PZZA', 'ZOES', 'HAWK', 'MYGN', 'BDX', 'LNN', 'NUAN', 'MCK', 'GWRE', 'BRC', 'GWRE', 'SJR',
    #            'EGHT', 'ASEI', 'MRVL', 'NXGN', 'PCTY', 'LGF.B', 'DECK', 'HAIN', 'NORD', 'GWRE', 'GWRE', 'JNJ', 'CIEN', 'STAA', 'SUM', 'ZOES', 'MYCC', 'SLAB', 'RAMP', 'LNCE',
    #            'COKE', 'PTC', 'MOG.A', 'RXN', 'ISIL', 'EBF', 'LOGI', 'CRM', 'HABT', 'EPAY', 'CY', 'AVAV', 'EPAY', 'ADSK', 'ANF', 'RH', 'ANF', 'ICHR', 'KRNY', 'FNSR', 'DLB',
    #            'CRM', 'POST', 'VSAT', 'NXGN', 'CUB', 'CONN', 'AIR', 'NTAP', 'VSAT', 'ITGR', 'KTOS', 'MSGN', 'PLAY', 'CMD', 'PLXS', 'SPH', 'EZPW'],
    #     _2017=['WWW', 'POST', 'BNED', 'ASH', 'CSCO', 'LRN', 'MTSI', 'RAD', 'HLNE', 'WINS', 'COHU', 'SUM', 'CRM', 'TEAM', 'CVLT', 'PTC', 'SUM', 'CVLT', 'CRS', 'CHS', 'MLAB',
    #            'CUB', 'EPC', 'ABM', 'MCK', 'CSII', 'PRMW', 'AVX', 'SUM', 'VIAV', 'IMBI', 'RAMP', 'ZOES', 'HAWK', 'MYGN', 'BDX', 'LNN', 'NUAN', 'MCK', 'GWRE', 'BRC', 'GWRE',
    #            'SJR', 'EGHT', 'ODP', 'MRVL', 'NXGN', 'PCTY', 'LGF.B', 'DECK', 'HAIN', 'AQUA', 'GWRE', 'GWRE', 'CIEN', 'STAA', 'SUM', 'ZOES', 'SLAB', 'PSDO', 'RAMP', 'LNCE',
    #            'PTC', 'MOG.A', 'RXN', 'LOGI', 'CRM', 'HABT', 'EPAY', 'AVAV', 'EPAY', 'ADSK', 'WINS', 'ANF', 'RH', 'ANF', 'ICHR', 'KRNY', 'FNSR', 'DLB', 'CRM', 'POST', 'VSAT',
    #            'NXGN', 'CUB', 'CONN', 'AIR', 'NTAP', 'VSAT', 'ITGR', 'AMD', 'MSGN', 'PLAY', 'ACB', 'CMD', 'PLXS', 'SPH', 'EZPW', 'AIR', 'MITK'],
    #     _2018=['TAST', 'WWW', 'POST', 'BNED', 'ASH', 'CSCO', 'LRN', 'MTSI', 'SIGM', 'RAD', 'HLNE', 'COHU', 'SUM', 'CRM', 'TEAM', 'CVLT', 'PTC', 'SUM', 'CVLT', 'CRS', 'CHS',
    #            'MLAB', 'CUB', 'EPC', 'ABM', 'NYT', 'MCK', 'CSII', 'PRMW', 'AVX', 'SUM', 'VIAV', 'IMBI', 'RAMP', 'PZZA', 'SRDX', 'MYGN', 'BDX', 'LNN', 'NUAN', 'MCK', 'GWRE',
    #            'BRC', 'GWRE', 'SJR', 'EGHT', 'ODP', 'MRVL', 'NXGN', 'PCTY', 'LGF.B', 'DECK', 'HAIN', 'AQUA', 'GWRE', 'GWRE', 'JNJ', 'CIEN', 'STAA', 'CIVI', 'SUM', 'SLAB',
    #            'PSDO', 'RAMP', 'COKE', 'PTC', 'MOG.A', 'RXN', 'LOGI', 'CRM', 'HABT', 'EPAY', 'CY', 'AVAV', 'EPAY', 'ADSK', 'ANF', 'RH', 'ANF', 'ICHR', 'KRNY', 'FNSR', 'DLB',
    #            'DAVA', 'QTNA', 'CRM', 'POST', 'VSAT', 'NXGN', 'CUB', 'CONN', 'AIR', 'NTAP', 'VSAT', 'ITGR', 'KTOS', 'AMD', 'MSGN', 'PLAY', 'BGFV'],
    #     _2019=['TAST', 'WWW', 'POST', 'BNED', 'ASH', 'CSCO', 'EPAC', 'LRN', 'MTSI', 'RAD', 'HLNE', 'COHU', 'SUM', 'CRM', 'TEAM', 'CVLT', 'PTC', 'ZM', 'SUM', 'CVLT', 'CRS',
    #            'CHS', 'MLAB', 'CUB', 'EPC', 'ABM', 'NYT', 'MCK', 'CSII', 'PRMW', 'MSGS', 'AVX', 'SUM', 'VIAV', 'IMBI', 'RAMP', 'PZZA', 'NLOK', 'SRDX', 'MYGN', 'BDX', 'LNN',
    #            'NUAN', 'MCK', 'GWRE', 'BRC', 'GWRE', 'SJR', 'EGHT', 'ODP', 'MRVL', 'NXGN', 'PCTY', 'LGF.B', 'DECK', 'HAIN', 'AQUA', 'GWRE', 'GWRE', 'JNJ', 'CIEN', 'SUM',
    #            'SLAB', 'PSDO', 'RAMP', 'COKE', 'PTC', 'MOG.A', 'RXN', 'LOGI', 'CRM', 'EPAY', 'CY', 'AVAV', 'EPAY', 'ADSK', 'ANF', 'RH', 'ANF', 'GO', 'ICHR', 'KRNY', 'FNSR',
    #            'DLB', 'DAVA', 'CRM', 'POST', 'VSAT', 'NXGN', 'CUB', 'CONN', 'IFMK', 'AIR', 'BBW', 'NTAP', 'VSAT', 'KTOS', 'AMD', 'MSGN', 'PLAY'],
    #     _2020=['POST', 'BNED', 'ASH', 'CSCO', 'EPAC', 'LRN', 'MTSI', 'RAD', 'HLNE', 'CRM', 'TEAM', 'CVLT', 'PTC', 'ZM', 'CVLT', 'CRS', 'CHS', 'MLAB', 'CUB', 'EPC', 'MCK',
    #            'CSII', 'MSGS', 'VIAV', 'RAMP', 'NLOK', 'SRDX', 'MYGN', 'BDX', 'LNN', 'NUAN', 'MCK', 'SMRTQ', 'GWRE', 'BRC', 'GWRE', 'SJR', 'EGHT', 'AVCT', 'MRVL', 'NXGN',
    #            'PCTY', 'LGF.B', 'DECK', 'HAIN', 'AQUA', 'GWRE', 'GWRE', 'STAA', 'LAKE', 'RAMP', 'ELF', 'PTC', 'MOG.A', 'RXN', 'LOGI', 'CRM', 'AGTC', 'DGII', 'EPAY', 'AVAV',
    #            'EPAY', 'ADSK', 'WINS', 'ANF', 'RH', 'ANF', 'KRNY', 'DLB', 'DAVA', 'CRM', 'POST', 'VSAT', 'NXGN', 'CUB', 'CONN', 'IFMK', 'AIR', 'BBW', 'NTAP', 'VSAT', 'MSGN',
    #            'PLAY', 'ACB', 'CMD', 'PLXS', 'SPH', 'AIR', 'CRS', 'VRTU', 'VRNT', 'HAE', 'JJSF', 'VRNT', 'KLIC', 'MNR', 'TAK', 'MANU', 'SYY', 'PCTY']
    # )

    # 39th day; High netinc; 0.0048; rerun with new list of tickers
    # 29th day; High netinc; 0.0034; rerun with new list of tickers
    # stock_dict = dict(
    # _2015 = ['TM', 'NTTYY', 'DCMYY', 'SNE', 'HMC', 'MUFG', 'HDB', 'SMFG', 'BABA', 'WIT', 'IBN', 'AAPL', 'MSFT', 'INTC', 'RDY', 'PG', 'TD', 'RY', 'CSCO', 'V', 'ORCL', 'BHP',
    #          'PEP', 'BNS', 'QCOM', 'ACN', 'BMO', 'MDT', 'TFCF', 'COST', 'CM', 'CSX', 'DE', 'HPQ', 'EA', 'CCL', 'ADBE', 'AVGO', 'AMAT', 'MU', 'NKE', 'ADP', 'DHI', 'INFY',
    #          'WBK', 'MON', 'LRCX', 'AMTD', 'GIS', 'TSN', 'EMR', 'APD', 'LEN', 'INTU', 'AZO', 'KR', 'CPB', 'VIAB', 'PCP', 'DEO', 'YUM', 'FDX', 'NGG', 'GIB', 'ADI', 'KLAC',
    #          'PH', 'HOLX', 'PAYX', 'A', 'HPE', 'COL', 'ROK', 'STX', 'GRMN', 'LHX', 'CLX', 'SBUX', 'HRL', 'MCK', 'KMX', 'CTAS', 'BDX', 'CAG', 'BF.B', 'NTAP', 'RJF', 'SWKS',
    #          'BEN', 'XLNX', 'SJM', 'CPRT', 'TDG', 'HSIC', 'SJR', 'EL', 'MKC', 'MXIM', 'RYAAY', 'JCI'],
    # _2016 = ['NJDCY', 'STJ', 'WFM', 'LLTC', 'SNDK', 'HAR', 'VAL1', 'ARG', 'ATW', 'CSC', 'IM', 'BRCD', 'PNRA', 'CLC', 'MESG', 'BWLD', 'GK', 'TFM', 'MFRM', 'ISIL', 'HW', 'NORD',
    #          'QLGC', 'ISLE', 'PLKI', 'KKD', 'PTHN', 'NEWP', 'AEPI', 'TIVO1', 'NILE', 'MYCC', 'ACAT', 'SGI', 'BLOX', 'INVN', 'AMCC', 'APIC', 'JOY', 'RDEN', 'APOL'],
    # _2017 = ['NTTYY', 'DCMYY', 'MON', 'CSRA', 'SPB1', 'FGL', 'MSCC', 'MENT', 'LNCE', 'BOBE', 'PRXL', 'CAFD', 'SHLM', 'XCRA', 'IXYS', 'AFAM', 'CUDA', 'ZOES', 'SHOR', 'BV1',
    #          'XTLY', 'DDC', 'STRP', 'ALOG', 'HAWK', 'NMBL', 'PAY', 'SPLS'],
    # _2018 = ['TFCF', 'COL', 'CA', 'ISCA', 'VSM', 'APU', 'ESIO', 'SONC', 'ESL', 'KMG', 'OCLR', 'LXFT', 'PERY', 'KLXI', 'CRCM', 'SVU', 'NXEO', 'ABAX', 'NEWM', 'FINL', 'QTNA',
    #          'VSI', 'VCON', 'IDTI', 'KANG', 'LAYN', 'SIGM', 'TNTR'],
    # _2019 = ['INTC', 'JNJ', 'PEP', 'DE', 'HPQ', 'CCL', 'ADBE', 'AVGO', 'AMAT', 'AMTD', 'KHC', 'LEN', 'VIAB', 'MFGP', 'GIB', 'A', 'HPE', 'ILMN', 'CDNS', 'JEF', 'K', 'SWK',
    #          'GRMN', 'LHX', 'HSIC', 'SNA', 'GTES', 'MKC', 'KEYS', 'HBI', 'TOL', 'TSCO', 'SNPS', 'CERN', 'HAS', 'INFO', 'SNX', 'AAP', 'DOX', 'COO', 'PPC', 'RHT', 'TDY', 'DPZ',
    #          'EV', 'LEVI', 'USFD', 'MIDD', 'GGG', 'AMD', 'NDSN', 'HEI', 'DAR', 'AVY', 'TTC', 'AVX', 'KBH', 'CRI', 'AZPN', 'GIL', 'CIEN', 'CRL', 'DNKN', 'RBC', 'LSTR', 'PKI',
    #          'NAV', 'MASI', 'IAA', 'UFPI', 'GEF', 'FLO', 'VMI', 'FND', 'ZAYO', 'SFM', 'NYT', 'LFUS', 'WEN', 'FUL', 'BLMN', 'WWW', 'ABM', 'WW', 'DENN', 'KELYA', 'HNI', 'SAM',
    #          'MLI', 'USNA', 'ODP', 'KTB', 'IRBT', 'SNEX', 'DORM', 'SNBR', 'IMKTA', 'TILE', 'SITE', 'BGS'],
    # _2020 = ['TM', 'SNE', 'HMC', 'MUFG', 'IX', 'HDB', 'SMFG', 'BABA', 'WIT', 'IBN', 'AAPL', 'MSFT', 'TAK', 'RDY', 'WMT', 'PG', 'TD', 'RY', 'HD', 'CSCO', 'V', 'ORCL', 'BHP',
    #          'BNS', 'VMW', 'QCOM', 'ACN', 'BMO', 'MDT', 'DELL', 'LOW', 'COST', 'NLOK', 'CM', 'TGT', 'TJX', 'EA', 'NVDA', 'MU', 'NKE', 'ADP', 'DHI', 'INFY', 'WBK', 'LRCX',
    #          'GIS', 'TSN', 'EMR', 'APD', 'INTU', 'AZO', 'DG', 'ROST', 'KR', 'CPB', 'MRVL', 'BBY', 'DEO', 'FDX', 'NGG', 'ADI', 'KLAC', 'PH', 'HOLX', 'PAYX', 'ROK', 'STX',
    #          'FOXA', 'CLX', 'SBUX', 'HRL', 'MCK', 'KMX', 'CTAS', 'BDX', 'CAG', 'DLTR', 'BF.B', 'NTAP', 'RJF', 'TXT', 'SWKS', 'BEN', 'XLNX', 'SJM', 'ULTA', 'CPRT', 'TDG',
    #          'KSS', 'SJR', 'EL', 'VFC', 'LDOS', 'MXIM', 'RYAAY', 'LULU', 'JCI', 'RMD', 'AMCR', 'ATO']
    # )

    # 39th day; highest above 30pe+ and tradeable fixed 1; 0.0038
    # 29th day; highest above 30pe+ and tradeable fixed 1; 0.0022
    # stock_dict = dict(
    #     _2015=['GERN', 'BSFT', 'INCY', 'PENN', 'PEN', 'PCRX', 'MITT', 'TRU', 'FTNT', 'REXR', 'SGRY', 'ZOES', 'AMZN', 'FNV', 'ATHN', 'YRCW', 'EGRX', 'EGHT', 'GWRE', 'ODP',
    #            'BRC', 'NFLX', 'AEM', 'CCOI', 'FTI', 'ASEI', 'LOGI', 'CIEN', 'AVAV', 'PRGO', 'BKI', 'ULTI', 'ALXN', 'EXAM', 'MINI', 'ENV', 'ORC', 'FCPT', 'CX', 'KRNY', 'FWONA',
    #            'RKUS', 'DOC', 'CXO', 'HABT', 'FNSR', 'PNM', 'PLAY', 'SPSC', 'OLED', 'ISLE', 'XXIA', 'KMI', 'PRTY', 'MOMO', 'VIRT', 'WB', 'MDSO', 'GIMO', 'KRG', 'PRE', 'UNVR',
    #            'SINA', 'ACOR', 'SSNC', 'ZBH', 'NPTN', 'ITRI', 'ATML', 'HAE', 'EDR', 'AAWW', 'NBHC', 'ELLI', 'EQIX', 'AFFX', 'PKX', 'IRC', 'LYG', 'AIR', 'HTA', 'AVID', 'BETR',
    #            'BLKB', 'CCJ', 'EQT', 'BCR', 'SE1', 'ECHO', 'OAK', 'TVPT', 'MBLY', 'RGEN', 'QTS', 'ADPTQ', 'DPLO', 'GLOG', 'PTC', 'ACAT', 'LOGM'],
    #     _2016=['TAST', 'CVLT', 'MPLX', 'ZLTQ', 'OMCL', 'PEIX', 'LOGM', 'MTSI', 'KRG', 'BSFT', 'TEAM', 'SEMG', 'BP', 'KW', 'CHS', 'CC', 'CUB', 'CX', 'ACHC', 'AGRO', 'NGD',
    #            'CEO', 'TEVA', 'CMG', 'HAWK', 'TRP', 'ERII', 'ECHO', 'CHU', 'SUI', 'GKOS', 'ICHR', 'NFLX', 'MYCC', 'CQH', 'GWRE', 'ARRS', 'SUM', 'LNCE', 'RAMP', 'WEB', 'ZOES',
    #            'ATHN', 'ELF', 'TRCO', 'INCY', 'OFIX', 'DEA', 'SALE', 'SWC', 'CG', 'CONE', 'EQIX', 'HSTM', 'BIO', 'RMBS', 'FTNT', 'GTT', 'CLNY', 'HPP', 'ULTI', 'ITGR', 'SBAC',
    #            'MORE', 'ORC', 'NXPI', 'MSGN', 'SPSC', 'RP', 'AXTA', 'BSM', 'SHPG', 'AMZN', 'PEN', 'PCH', 'IQV', 'TGE', 'VSAT', 'SU', 'CRS', 'AG', 'ANIP', 'FSP', 'BUD', 'GPT',
    #            'ALRM', 'SITC', 'NXGN', 'LN', 'KMPR', 'WAGE', 'MLNX', 'RDCM', 'BKI', 'CCOI', 'PTHN', 'HSBC', 'OCLR', 'CORT', 'SPH'],
    #     _2017=['RDNT', 'RP', 'JELD', 'FTAI', 'WWW', 'CVLT', 'ASH', 'IT', 'VIRT', 'FLS', 'NSA', 'HLNE', 'LRN', 'KMI', 'MCRN', 'QTS', 'MELI', 'SID', 'AQUA', 'PTC', 'RAD', 'BIP',
    #            'EURN', 'NYT', 'SUM', 'GOL', 'WAT', 'PEN', 'PTEN', 'ABM', 'IMAX', 'LX', 'EPC', 'AM', 'ACIW', 'ULTI', 'RAMP', 'CARG', 'REI', 'ZBRA', 'GRA', 'BSIG', 'AAXN',
    #            'DECK', 'NCSM', 'MRVL', 'BSX', 'IR', 'PCTY', 'CCOI', 'LGF.B', 'CRM', 'RH', 'PSDO', 'AMD', 'TREE', 'GWRE', 'FTNT', 'JNJ', 'AKS', 'LGND', 'USM', 'SOI', 'AMZN',
    #            'SRCL', 'TTEC', 'ABT', 'BCO', 'CRY', 'VER', 'AMPH', 'AXTA', 'CASA', 'ALB', 'ATVI', 'HSC', 'WINS', 'ORC', 'ANF', 'NFLX', 'DEA', 'QGEN', 'GRPN', 'NCMI', 'SBAC',
    #            'EQC', 'GCI', 'VSAT', 'OFIX', 'APPF', 'WPM', 'VRTX', 'POST', 'KO', 'HR', 'NAV', 'GOOS', 'SWIR', 'CHU', 'SFUN'],
    #     _2018=['AXS', 'EHTH', 'CONE', 'WP', 'CSCO', 'PODD', 'GNL', 'PDCE', 'CEVA', 'RH', 'PZZA', 'SUM', 'ATUS', 'PEN', 'CSII', 'CRM', 'SAIL', 'SWCH', 'GOOD', 'AVX', 'ENV',
    #            'TGP', 'NXGN', 'PEGA', 'MCK', 'SBAC', 'NSA', 'BDX', 'IMBI', 'ACIA', 'PGRE', 'CATM', 'NBIX', 'MEET', 'ALXN', 'STAA', 'NURO', 'APA', 'ZNGA', 'HAIN', 'EPAY',
    #            'NEO', 'CXP', 'AMH', 'SSRM', 'NTAP', 'CWST', 'ADSW', 'Y', 'FTAI', 'LBRDA', 'NUVA', 'ANF', 'QTNA', 'BEP', 'EIGI', 'UNIT', 'TCMD', 'TERP', 'SJR', 'PAAS', 'OSPN',
    #            'SYNH', 'PTC', 'TDC', 'DEA', 'AQUA', 'TPIC', 'ALTR', 'PRNB', 'ICUI', 'SMLP', 'SAND', 'GDDY', 'RGEN', 'RP', 'AYX', 'INCY', 'ULTI', 'CHCT', 'ABMD', 'FARO',
    #            'ELLI', 'PLXS', 'SSNC', 'SJI', 'CONN', 'CUB', 'GCP', 'IIPR', 'ALRM', 'HLNE', 'TME', 'INXN', 'SHAK', 'JBGS', 'HABT', 'ELAN', 'BRK.B', 'RCII'],
    #     _2019=['CEVA', 'CWK', 'CIIC', 'AAXN', 'DKNG', 'PRA', 'ALUS', 'SBSW', 'NOK', 'SUM', 'SFTW', 'QTS', 'PODD', 'ONTO', 'BAND', 'MYGN', 'MCK', 'F', 'PRMW', 'DRQ', 'CVLT',
    #            'VRT', 'CRY', 'VIAV', 'MSGS', 'VVNT', 'LNN', 'DOYU', 'INOV', 'GSHD', 'PEAK', 'GWRE', 'SWCH', 'NLOK', 'AYX', 'MMSI', 'NEO', 'FIS', 'BLKB', 'PNTG', 'BANC', 'SWI',
    #            'GSX', 'BIP', 'CXP', 'WTTR', 'RXN', 'NBIX', 'DEA', 'HTA', 'E', 'SLAB', 'AMG', 'DXCM', 'SBAC', 'CY', 'NDLS', 'FMX', 'COKE', 'LBRDA', 'RGEN', 'TREE', 'GO', 'AIR',
    #            'SSTI', 'CVA', 'EPAY', 'CONE', 'AUDC', 'BKR', 'GMRE', 'AMD', 'CSTL', 'IQV', 'KTOS', 'KRNT', 'ELAN', 'FOE', 'ZNGA', 'UNIT', 'WPM', 'GPN', 'WING', 'KRUS', 'SHAK',
    #            'TW', 'BCO', 'COLD', 'HPP', 'OPRT', 'NXPI', 'MNRL', 'KLIC', 'MNR', 'CHCT', 'CDAY', 'ATUS', 'BKS', 'RCM', 'HPK'],
    #     _2020=['POST', 'ZM', 'EPAC', 'CRM', 'MLAB', 'CRS', 'NUAN', 'MOG.A', 'DAVA', 'JJSF', 'AIR', 'TAK', 'ADSK', 'SLP', 'CMD', 'SYY', 'SBUX', 'EL', 'CALM', 'PCTY', 'VIAV',
    #            'MSGE', 'AIT', 'JBL', 'VRNT', 'MTN', 'STAA', 'NXGN', 'HQY', 'PTC', 'CTLT', 'BDX', 'DSGX', 'VEEV', 'BRKS', 'LRN', 'AMSWA', 'EAT', 'WBA', 'HAE', 'SMTC', 'NVDA',
    #            'NEOG', 'HLNE', 'NKE', 'TFSL', 'EDU', 'DGII', 'VAR', 'TDG', 'EGAN', 'FICO', 'FLEX', 'OTEX', 'WDFC', 'SMPL', 'JCI', 'CMTL', 'MRCY', 'INTU', 'NSSC', 'ELF', 'DEO',
    #            'RGLD', 'LITE', 'EXPO', 'FRHC', 'SR', 'SLQT', 'GMS', 'ABMD', 'APPS', 'JKHY', 'GBDC', 'TECH', 'AVAV', 'LAKE', 'ADI', 'ETH', 'RMD', 'CLCT', 'TTWO', 'COST',
    #            'QNST', 'EXP', 'LULU', 'KEM', 'BF.B', 'DLB', 'MCHP', 'TTEK', 'V', 'CTAS', 'CPRT', 'KLIC', 'VFC', 'QRVO', 'BR', 'RMR', 'SXI']
    # )

    all_rois = []
    for _year in sorted(stock_dict.keys()):
        year = int(_year[1:])
        tickers = stock_dict[_year]
        year_rois = []
        num_days_under = 40
        min_days_under_sought = 29
        num_hold_days = 1
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