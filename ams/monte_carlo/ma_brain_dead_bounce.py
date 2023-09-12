from random import shuffle
from typing import List, Tuple

import pandas as pd

from ams.config import logger_factory
from ams.services import ticker_service as ts
from ams.utils import ticker_utils

logger = logger_factory.create(__name__)


def process():
    # INCORREC RESULTS! RERUN!! inc_to_wait_4: .1, ma=150; Volume * price == 500k
    # roi 3.67
    tickers_2010 = ['TTNP', 'DDRX', 'SNBR', 'VNDA', 'CAR', 'HGSI', 'PPC', 'IMBI', 'DRWI', 'KERX']
    # roi -0.53
    tickers_2011 = ['CPWM', 'RDCM', 'AMRN', 'HNH', 'MEET', 'APKT', 'ISLN', 'TGA', 'WWR', 'TPCG']
    # roi -.52
    tickers_2012 = ['SIMO', 'INHX', 'ADLR', 'LFVN', 'ICGN', 'PTE', 'GENE', 'GLNG', 'QCOR', 'MDVN']
    # roi .58
    tickers_2013 = ['CDCAQ', 'SRPT', 'ARNA', 'BDSI', 'CLSN', 'INFI', 'ACAD', 'SHF', 'PCYC', 'RPRX']
    # roi .-61
    tickers_2014 = ['DZSI', 'SGLB', 'CSIQ', 'TAOP', 'CMGE', 'PBYI', 'VISN', 'ACAD', 'HIMX', 'ADEP']
    # roi -.9
    tickers_2015 = ['GRBK', 'SUNW', 'RDNT', 'AVNR', 'ITMN', 'AGIO', 'MLND', 'RDUS', 'BLUE', 'RCPT']
    # roi 95.43
    tickers_2016 = ['NYMX', 'AVXL', 'VLTC', 'EGRX', 'PRTA', 'ANAC', 'EXEL', 'DSKX', 'ITCI', 'NHTC']
    # roi 9.17
    tickers_2017 = ['TELL', 'CPXX', 'CDEVW', 'GLBS', 'CRBP', 'SINO', 'CLCD', 'TBRA', 'AMD', 'CYBE']
    # roi 2.25
    tickers_2018 = ['GRVY', 'MRNS', 'XOMA', 'RIOT', 'PTE', 'ANAB', 'MDGL', 'HLG', 'ESPR', 'RXDX']
    # roi 20.33
    tickers_2019 = ['BIMI', 'TNDM', 'NIHD', 'HEAR', 'ECYT', 'PRQR', 'AGMH', 'TLRY', 'CDNA', 'AMRN']
    # roi 12.09
    tickers_2020 = ['RLMD', 'PHIO', 'CYCC', 'SQBG', 'SNES', 'CGIX', 'AXSM', 'GRPN', 'SCYX', 'AMRH']
    # roi
    tickers_2021 = ['BLNKW', 'NBACW', 'NKLAW', 'HGEN', 'BTBT', 'MGEN', 'HCCHW', 'NVAX', 'BLNK', 'RMG.WS']

    # No volume limitations; Results: Good; All but one roi 1.0+; but one loser was .93
    # inc_to_wait_4: .1, ma=150; Volume 0+;
    # tickers_2014 = ['MRK', 'PFE', 'MSFT', 'NKE', 'V', "CSCO", "UNH", "BA", "VZ", "KO"]
    # tickers_2015 = ['VITK', 'ATHYQ', 'ENTP', 'WMGIZ', 'NEST', 'GRBK', 'DWRI', 'LINK', 'ODDJ', 'SUNW']
    # tickers_2016 = ['OPNT', 'KEYP', 'CFCB', 'CALL1', 'HYPRQ', 'MAXW', 'SBKC', 'WGLF', 'ZPLSQ', 'NYMX']
    # tickers_2017 = ['INVC', 'TELL', 'CPXX', 'BCDA', 'HNIN', 'LPHIQ', 'CDEVW', 'GLBS', 'RPRXW', 'CRBP']
    # tickers_2018 = ['ZYXI', 'QUSA', 'GRVY', 'MRNS', 'XOMA', 'RIOT', 'PTE', 'ANAB', 'ANDAW2', 'VKTXW']
    # tickers_2019 = ['SKBI', 'BIMI', 'TNDM', 'NIHD', 'HEAR', 'ECYT', 'FLUX', 'PRQR', 'LIME', 'PAYS']
    # tickers_2020 = ['RLMD', 'PHIO', 'CYCC', 'SQBG', 'SNES', 'CGIX', 'AXSM', 'GRPN', 'VRME', 'AIRTP']
    # tickers_2021 = ['BLNKW', 'NBACW', 'NKLAW', 'HGEN', 'BTBT', 'MGEN', 'PRPLW', 'BRPAW', 'HCCHW', 'NVAX']
    tickers = tickers_2017
    year = 2017
    inc_to_wait_4 = .1
    ma = 150

    shuffle(tickers)

    process_tickers_in_year(tickers=tickers, year=year, ma=ma, inc_to_wait_4=inc_to_wait_4)


def process_tickers_in_year(tickers, year, ma, inc_to_wait_4):
    date_from_str = f"{year}-01-04"
    date_to_str = f"{year}-12-31"
    all_df = []
    target_col = "close"
    ma_col = f"{target_col}_SMA_{ma}"

    for ndx, t in enumerate(tickers):
        df = ts.get_ticker_eod_data(ticker=t)
        df = df.dropna(subset=[target_col])
        df = ticker_utils.add_simple_moving_averages(df=df, target_column=target_col, windows=[ma])
        if df is not None:
            logger.info(f"Processing ticker: {t}")
            df = df[df["date"] >= date_from_str].copy()
            df = df[df["date"] <= date_to_str].copy()
            df = df.dropna(subset=[ma_col])
            if df is not None and df.shape[0] > 0:
                all_df.append(df)

    if len(all_df) > 0:
        df_mas = pd.concat(all_df, axis=0)
        print(f"{df_mas.shape[0]}")
        monte_carlo(df_mas, ma_col=ma_col, inc_to_wait_4=inc_to_wait_4)


def monte_carlo(df, ma_col, inc_to_wait_4: float):
    df_mas = df.sort_values("date")
    df_g = df_mas.groupby("ticker")
    cash = 10000
    initial_cash = cash
    purchases = []
    num_purch = 0
    ticker_cash = cash / 10

    date_range_open = None
    for ticker, df_t in df_g:
        df_t = df_t.sort_values("date")
        num_rows = df_t.shape[0]
        row_ndx = 0
        last_price = 0
        equity = 0

        for _, r in df_t.iterrows():
            action_price = r["close"]
            if row_ndx == 0:
                date_range_open = action_price
            if row_ndx == num_rows - 1:
                date_range_close = action_price
                passive_roi = (date_range_close - date_range_open) / date_range_open
                print(f"{ticker} passive roi: {passive_roi:,.2f}")
            ma = r[ma_col]
            if ma > action_price:
                # equity = get_equity(sell_price=action_price, purchases=purchases)
                # if equity < max_cash_inv:
                num_stocks_to_purchase = int(ticker_cash / action_price)
                if num_stocks_to_purchase > 0:
                    purchase_total = num_stocks_to_purchase * action_price
                    ticker_cash -= purchase_total
                    purchases.append((num_stocks_to_purchase, action_price))
                    num_purch += 1
            if action_price > ma:
                ticker_cash, purchases = redeem_purchases(cash=ticker_cash, purchases=purchases, sell_price=action_price, inc_to_wait_4=inc_to_wait_4)
            last_price = action_price
            row_ndx += 1
            equity = get_equity(sell_price=last_price, purchases=purchases)

        purchases = []
        cash = cash + ticker_cash + equity
        if cash <= 0:
            print("You broke!")
            break

    roi = (cash - initial_cash) / initial_cash
    print(f"Cash: {cash:,.2f}; equity: {equity:,.2f}: roi: {roi:,.2f}; num_purch: {num_purch}")


def get_equity(sell_price, purchases):
    value = 0
    for num, _ in purchases:
        value += num * sell_price
    return value


def redeem_purchases(cash: float, purchases: List[Tuple[int, float]], sell_price: float, inc_to_wait_4: float):
    retained_equity_purchs = []
    for num, price in purchases:
        roi = (sell_price - price) / price
        if roi > inc_to_wait_4:
            cash += num * sell_price
        else:
            retained_equity_purchs.append((num, price))

    return cash, retained_equity_purchs


if __name__ == '__main__':
    # get_top_tickers()
    process()
