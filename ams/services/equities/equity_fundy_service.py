import pandas as pd

from ams.config import constants
from ams.services.equities.EquityFundaDimension import EquityFundaDimension


def get_equity_fundies():
    return pd.read_csv(constants.SHAR_CORE_FUNDY_FILE_PATH)


def get_most_recent_quarter_data():
    df = get_equity_fundies()

    df_fil = df[df["dimension"] == EquityFundaDimension.AsReportedQuarterly.value]

    return df_fil.sort_values(by=["ticker", "calendardate"]) \
        .drop_duplicates(subset=["ticker"], keep="last")


def get_all_quarterly_data():
    df = get_equity_fundies()

    df_fil = df[df["dimension"] == EquityFundaDimension.AsReportedQuarterly.value]

    return df_fil.sort_values(by=["ticker", "calendardate"])

