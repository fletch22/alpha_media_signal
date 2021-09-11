import pandas as pd

from ams.DateRange import DateRange
from ams.config import constants, logger_factory
from ams.services import ticker_service
from ams.services.equities.EquityFundaDimension import EquityFundaDimension

logger = logger_factory.create(__name__)


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


def get_top_by_attribute(indicator: str, is_low_good: bool ):
    df = get_equity_fundies()

    df = filter_by_dimension(df=df, efd=EquityFundaDimension.MostRecentAnnual)

    assert df.shape[0] > 0

    # logger.info(list(df.columns))

    top_n = 100
    min_pe = 30

    easy_trade = True

    l_or_h = "low" if is_low_good else "high"
    print(f"\n# 39th day: {l_or_h} {indicator}: ")
    print(f"# 29th day: {l_or_h} {indicator}: ")
    print(f"stock_dict_{l_or_h}_{indicator} = dict(")

    year_range = list(range(2015, 2021))
    for year in year_range:
        start_rep_year = f"{year}-01-01"
        end_rep_year = f"{year}-12-31"

        df_year = df[(df["reportperiod"] >= start_rep_year) & (df["reportperiod"] <= end_rep_year)]

        tickers = df_year["ticker"].unique()

        dr = DateRange.from_date_strings(from_date_str=start_rep_year, to_date_str=end_rep_year)
        df_tickers = ticker_service.get_tickers_in_range(tickers, date_range=dr)

        # df_tickers["mean_vol"] = df_tickers["volume"].mean()
        # des_cols = ["ticker", "mean_vol"]
        # df_tickers = df_tickers.drop_duplicates(subset=des_cols)[des_cols].copy()

        df_tickers = df_tickers.set_index(["ticker"])
        mean_thing = df_tickers.groupby("ticker")["volume"].mean()
        df_tickers["mean_vol"] = mean_thing
        df_tickers = df_tickers.reset_index()

        df_tickers = df_tickers.sort_values(["ticker", "date"])
        des_cols = ["ticker"]
        df_tickers = df_tickers.drop_duplicates(subset=des_cols, keep="first")[["ticker", "mean_vol"]].copy()

        # logger.info(df_tickers[["ticker", "mean_vol"]].head())

        df_year = df_year.sort_values(["ticker", "reportperiod"])
        df_year = df_year.drop_duplicates(subset=des_cols, keep="last").copy()

        df_enh = pd.merge(left=df_year, right=df_tickers, on="ticker")

        if easy_trade:
            if indicator == "pe":
                df_enh = df_enh[(df_enh[indicator] >= min_pe)]
            df_enh = df_enh[(df_enh["price"] * df_enh["mean_vol"]) > (10 * 250000)]

        tickers = df_enh.sort_values(by=[indicator], ascending=is_low_good)["ticker"].values.tolist()

        print(f"\t_{year + 1}={tickers[:top_n]},")
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
