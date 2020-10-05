import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ams.config import constants
from ams.services import ticker_service
from ams.services.equities.EquityFundaDimension import EquityFundaDimension


def get_equity_fundies():
    return pd.read_csv(constants.SHAR_CORE_FUNDY_FILE_PATH)


def get_most_recent_quarter_data():
    df = get_equity_fundies()

    df_fil = df[df["dimension"] == EquityFundaDimension.AsReportedQuarterly.value]

    return df_fil.sort_values(by=["ticker", "calendardate"]) \
        .drop_duplicates(subset=["ticker"], keep="last")


def get_nasdaq_tickers_std_and_cat():
    df_nasdaq = ticker_service.get_nasdaq_info()

    df_dropped = df_nasdaq.drop(columns=["firstpricedate", "lastpricedate", "firstquarter", "lastquarter",
                                         "secfilings", "companysite", "lastupdated", "cusips",
                                         "isdelisted", "name", "exchange", "firstadded"])

    df_all_tickers = ticker_service.get_ticker_info()

    df_rem = df_all_tickers[df_dropped.columns]

    columns = [c for c in df_rem.columns if str(df_rem[c].dtype) == "object"]
    columns.remove("ticker")

    all_ohe = {}
    for c in columns:
        unique_values = df_all_tickers[c].unique().tolist()
        unique_values.append("<unknown>")

        ohe = OneHotEncoder(sparse=False, handle_unknown="ignore")
        all_ohe[c] = ohe
        ohe.fit(np.array(unique_values).reshape(-1, 1))

        df_dropped[c] = df_dropped[c].fillna("<unknown>")
        col_values = df_dropped[c].values
        t_values = ohe.transform(col_values.reshape(-1, 1))
        s_values = pd.Series(t_values.tolist())

        df_dropped[c] = s_values

    num_cols = [c for c in df_dropped.columns if str(df_dropped[c].dtype) == "float64"]  # need logic to get numeric
    standard_scaler = StandardScaler()
    for c in num_cols:
        median = df_dropped[c].median()
        median = 0 if str(median) == 'nan' else median
        df_dropped[c] = df_dropped[c].fillna(median)
        df_dropped[c] = standard_scaler.fit_transform(df_dropped[[c]])

    return df_dropped
