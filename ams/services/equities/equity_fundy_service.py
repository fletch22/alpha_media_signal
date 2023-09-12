from collections import namedtuple
from typing import List

import pandas as pd
from dateutil.relativedelta import relativedelta

from ams.DateRange import DateRange
from ams.config import constants, logger_factory
from ams.services import ticker_service
from ams.services.equities.EquityFundaDimension import EquityFundaDimension

logger = logger_factory.create(__name__)

DrPeriod = namedtuple("DrPeriod", "date_range period")


def get_equity_fundies():
    return pd.read_csv(constants.SHAR_CORE_FUNDY_FILE_PATH)


def get_most_recent_quarter_data():
    df = get_equity_fundies()

    df_fil = df[df["dimension"] == EquityFundaDimension.AsReportedQuarterly.value]

    return df_fil.sort_values(by=["ticker", "calendardate"]) \
        .drop_duplicates(subset=["ticker"], keep="last")


def get_all_quarterly_data():
    df = get_equity_fundies()

    df_fil = df[df["dimension"] == EquityFundaDimension.AsReportedQuarterly.value].copy()

    df_fil.sort_values(by=["ticker", "calendardate"], inplace=True)

    return df_fil


def filter_by_dimension(df: pd.DataFrame, efd: EquityFundaDimension):
    return df[df["dimension"] == efd.value].copy()


def get_top_by_attribute(indicator: str, dr_period_list: List[DrPeriod], is_low_good: bool, efd: EquityFundaDimension = EquityFundaDimension.MostRecentAnnual):
    df = get_equity_fundies()
    df_ti = ticker_service.get_ticker_info()
    right_exch = df_ti[df_ti["exchange"].isin(["NASDAQ", "NYSE"])]["ticker"].unique()

    df = df[df["ticker"].isin(right_exch)]

    df = filter_by_dimension(df=df, efd=efd)

    assert df.shape[0] > 0

    top_n = 100
    min_pe = 30

    l_or_h = "low" if is_low_good else "high"
    print(f"\n# 39th day: {l_or_h} {indicator}: ")
    print(f"# 29th day: {l_or_h} {indicator}: ")
    print(f"stock_dict_{l_or_h}_{indicator} = dict(")

    for drp in dr_period_list:
        dr = drp.date_range

        df_period = df[(df["reportperiod"] >= dr.start_date_str) & (df["reportperiod"] <= dr.end_date_str)]

        df_period = df_period.sort_values(by=["ticker", "reportperiod"])

        tickers = df_period["ticker"].unique()

        if len(tickers) == 0:
            continue
        df_tickers = ticker_service.get_tickers_in_range(tickers, date_range=dr)

        df_tickers = df_tickers.set_index(["ticker"])
        mean_thing = df_tickers.groupby("ticker")["volume"].mean()
        df_tickers["mean_vol"] = mean_thing
        df_tickers = df_tickers.reset_index()

        df_tickers = df_tickers.sort_values(["ticker", "date"], ascending=False)
        des_cols = ["ticker"]
        df_tickers = df_tickers.drop_duplicates(subset=des_cols, keep="first")[["ticker", "mean_vol"]].copy()

        df_period = df_period.sort_values(["ticker", "reportperiod"])
        df_period = df_period.drop_duplicates(subset=des_cols, keep="last").copy()

        df_enh = pd.merge(left=df_period, right=df_tickers, on="ticker")

        if indicator == "pe":
            df_enh = df_enh[(df_enh[indicator] >= min_pe)]
        df_enh = df_enh[(df_enh["price"] * df_enh["mean_vol"]) > (10 * 250000)]

        tickers = df_enh.sort_values(by=[indicator], ascending=is_low_good)["ticker"].values.tolist()

        dr_tar = DateRange(from_date=dr.from_date + relativedelta(years=1), to_date=dr.to_date + relativedelta(years=1))

        year = dr.from_date.year
        print(
            f"\t_{year + 1}_p{drp.period}={{'start_dt': '{dr_tar.start_date_str}', 'end_dt': '{dr_tar.end_date_str}', 'period': '{drp.period}', 'tickers': {tickers[:top_n]}}},")
    print(")")


def explain_fundy_fields():
    fundies = ['ticker', 'dimension', 'calendardate', 'datekey', 'reportperiod', 'lastupdated', 'accoci', 'assets', 'assetsavg', 'assetsc', 'assetsnc',
               'assetturnover',
               'bvps', 'capex', 'cashneq', 'cashnequsd', 'cor', 'consolinc', 'currentratio', 'de', 'debt', 'debtc', 'debtnc', 'debtusd', 'deferredrev', 'depamor',
               'deposits',
               'divyield', 'dps', 'ebit', 'ebitda', 'ebitdamargin', 'ebitdausd', 'ebitusd', 'ebt', 'eps', 'epsdil', 'epsusd', 'equity', 'equityavg', 'equityusd',
               'ev',
               'evebit', 'evebitda', 'fcf', 'fcfps', 'fxusd', 'gp', 'grossmargin', 'intangibles', 'intexp', 'invcap', 'invcapavg', 'inventory', 'investments',
               'investmentsc',
               'investmentsnc', 'liabilities', 'liabilitiesc', 'liabilitiesnc', 'marketcap', 'ncf', 'ncfbus', 'ncfcommon', 'ncfdebt', 'ncfdiv', 'ncff', 'ncfi',
               'ncfinv',
               'ncfo', 'ncfx', 'netinc', 'netinccmn', 'netinccmnusd', 'netincdis', 'netincnci', 'netmargin', 'opex', 'opinc', 'payables', 'payoutratio', 'pb', 'pe',
               'pe1',
               'ppnenet', 'prefdivis', 'price', 'ps', 'ps1', 'receivables', 'retearn', 'revenue', 'revenueusd', 'rnd', 'roa', 'roe', 'roic', 'ros', 'sbcomp', 'sgna',
               'sharefactor', 'sharesbas', 'shareswa', 'shareswadil', 'sps', 'tangibles', 'taxassets', 'taxexp', 'taxliabilities', 'tbvps', 'workingcapital']

    df = pd.read_csv(constants.SHAR_INDICATORS_CSV)

    # print(list(df.columns))

    df = df[df["indicator"].isin(fundies)]

    print()
    for ndx, row in df.iterrows():
        title = row["title"]
        desc = row["description"]
        ind = row["indicator"]
        info = f"{ind}: {title} - {desc}"
        print(info)
