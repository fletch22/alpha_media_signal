from datetime import datetime
from pathlib import Path
from random import shuffle
from statistics import mean
from typing import Union, List

import pandas as pd

from ams.config import constants, logger_factory
from ams.config.constants import ensure_dir
from ams.models import xgb_reg
from ams.pipes.p_make_prediction.DayPredictionInfo import DayPredictionInfo, ImportantFeatures
from ams.pipes.p_make_prediction.SplitDataFrames import SplitDataFrames
from ams.pipes.p_make_prediction.TrainingBag import TrainingBag
from ams.pipes.p_stock_merge.sm_process import STOCKS_MERGED_FILENAME, get_stock_merge_trainer_params
from ams.services import slack_service
from ams.twitter import skip_day_predictor
from ams.twitter.TrainAndPredictionParams import TrainAndPredictionParams, PredictionMode
from ams.twitter.pred_perf_testing import get_days_roi_from_prediction_table
from ams.utils.date_utils import get_next_market_day_no_count_closed_days

logger = logger_factory.create(__name__)

PREDICTIONS_CSV = "predictions.csv"
MONEY_PREDICTIONS_CSV = "real_money_predictions.csv"


def split_train_test(df: pd.DataFrame, tweet_date_str: str) -> (pd.DataFrame, pd.DataFrame):
    logger.info(f"Splitting on tweet date {tweet_date_str}")

    df_train = df[df["date"] < tweet_date_str].copy()
    df_test = df[df["date"] == tweet_date_str].copy()

    max_test = df_test['future_date'].max()
    df_train = df_train[df_train["future_date"] < max_test].copy()

    # FIXME: 2021-04-09: chris.flesche: For testing only. This avoids predicting when
    # we have less than n rows.
    if df_train.shape[0] < 10000:
        return None, None

    logger.info(f"Test data size: {df_test.shape[0]}")
    logger.info(f"Train data size: {df_train.shape[0]}")
    logger.info(f"Oldest date of train data (future_date): {df_train['future_date'].max()}")
    logger.info(f"Oldest date of test data (future_date): {df_test['future_date'].max()}")

    return df_train, df_test


def get_real_money_preds_path(output_path: Path):
    return Path(output_path, MONEY_PREDICTIONS_CSV)


def persist_predictions(df_buy: pd.DataFrame, tapp: TrainAndPredictionParams, output_path: Path):
    df_preds = None

    pred_path = Path(output_path, PREDICTIONS_CSV)
    if tapp.prediction_mode == PredictionMode.RealMoneyStockRecommender:
        pred_path = get_real_money_preds_path(output_path)

    is_new = not pred_path.exists()

    if not is_new:
        df_preds = pd.read_csv(pred_path)
        df_preds = df_preds[~((df_preds["purchase_date"] == tapp.purchase_date_str)
                              & (df_preds["num_hold_days"] == tapp.num_hold_days))]

    if df_buy is None or df_buy.shape[0] == 0:
        df_combined = df_preds
    else:
        if df_preds is not None:
            df_combined = pd.concat([df_preds, df_buy], axis=0)
        else:
            df_combined = df_buy

    if df_combined is not None:
        logger.info(f"Added {df_buy.shape[0]} predictions to {df_combined.shape[0]} {tapp.prediction_mode.name} output file '{str(pred_path)}'")
        df_combined.to_csv(pred_path, index=False)


def get_adjusted_buy(df: pd.DataFrame):
    top_divisor = 0
    if top_divisor != 0 and df.shape[0] >= top_divisor:
        df.sort_values(by=["prediction"], ascending=False, inplace=True)

        num_rows = df.shape[0]
        quint = int(num_rows / top_divisor)

        min_predict = df["prediction"].to_list()[quint]

        logger.info(f"Finding predictions above min: {min_predict}")
        df_buy = df[df["prediction"] > min_predict].copy()
    else:
        df_buy = df[df["prediction"] > 0].copy()

    return df_buy


def predict_day(dpi: DayPredictionInfo, persist_results: bool = True) -> (Union[None, float], List[str]):
    df_train, df_test = split_train_test(df=dpi.df, tweet_date_str=dpi.tapp.tweet_date_str)

    if df_test is None or df_test.shape[0] == 0:
        return None

    df_train = df_train.fillna(value=0)

    all_mod_preds = xgb_reg.train_predict(df_train=df_train,
                                          df_test=df_test.copy(),
                                          narrow_cols=list(df_train.columns),
                                          dpi=dpi,
                                          label_col="stock_val_change",
                                          require_balance=False,
                                          buy_thresh=0)

    all_buy_dfs = []
    for amp in all_mod_preds:
        col_preds = "prediction"
        df_tmp = df_test.copy()
        df_tmp.loc[:, col_preds] = amp
        df_buy = get_adjusted_buy(df=df_tmp)
        all_buy_dfs.append(df_buy)

    df_buy = pd.concat(all_buy_dfs, axis=0)
    logger.info(f"df_predict num rows: {df_test.shape[0]}: buy predictions: {df_buy.shape[0]}")

    roi, df_buy = handle_buy_predictions(df_buy, tapp=dpi.tapp)

    if persist_results:
        persist_predictions(df_buy=df_buy, tapp=dpi.tapp, output_path=dpi.output_path)

    return roi


def handle_buy_predictions(df_buy, tapp: TrainAndPredictionParams):
    df_buy = df_buy[["f22_ticker", "purchase_date", "future_date", "prediction"]].copy()

    # df_buy = get_buys_by_popularity(df=df_buy)
    # FIXME: 2021-04-10: chris.flesche: For testing only
    df_buy = get_buys_by_score(df=df_buy, frac=.03, score_min=None)

    roi = None
    num_hold_days = 1
    if df_buy is None or df_buy.shape[0] == 0:
        logger.info("No buy predictions.")
    else:
        df_buy.loc[:, "num_hold_days"] = num_hold_days
        df_buy.loc[:, "run_timestamp"] = datetime.timestamp(datetime.now())

        logger.info(f"Purchase dts: {df_buy['purchase_date'].unique()}")

        if tapp.prediction_mode == PredictionMode.DevelopmentAndTraining:
            purchase_date_str = df_buy["purchase_date"].to_list()[0]
            roi = get_days_roi_from_prediction_table(df_preds=df_buy,
                                                     purchase_date_str=purchase_date_str,
                                                     num_hold_days=num_hold_days,
                                                     min_price=0.,
                                                     addtl_hold_days=1)

    return roi, df_buy


def get_buys_by_popularity(df: pd.DataFrame):
    df = df.groupby(["f22_ticker", "purchase_date"]).size().reset_index(name='counts')

    df_buy_3 = df[df["counts"] == 3]
    df_buy_2 = df[df["counts"] == 2]
    df_buy_1 = df[df["counts"] == 1]

    logger.info(f"Found {df_buy_3.shape[0]} with 3 counts.")
    logger.info(f"Found {df_buy_2.shape[0]} with 2 counts.")
    logger.info(f"Found {df_buy_1.shape[0]} with 1 counts.")

    df = df_buy_3
    if df.shape[0] == 0:
        df = df_buy_2
    if df.shape[0] == 0:
        df = df_buy_1

    return df


def get_buys_by_score(df: pd.DataFrame, frac: float = None, score_min: float = None):
    df = df.groupby(["f22_ticker", "purchase_date"])["prediction"].mean().reset_index(name='prediction_mean_score')
    df = df.drop_duplicates(subset=["f22_ticker", "purchase_date"])

    df.sort_values(by=["prediction_mean_score"], ascending=False, inplace=True)

    if frac is not None:
        df = df.head(int(df.shape[0] * frac))
    elif score_min is not None:
        df = df[df["prediction_mean_score"] > score_min]

    return df


def filter_columns(df: pd.DataFrame):
    cols = list(df.columns)
    cols = [c for c in cols if not c.startswith("location_")]
    cols = [c for c in cols if not c.startswith("currency_")]
    cols = [c for c in cols if not c.startswith("industry_")]
    cols = [c for c in cols if not c.startswith("famaindustry_")]
    cols = [c for c in cols if not c.startswith("category_")]
    cols = [c for c in cols if not c.startswith("sector_")]
    cols = [c for c in cols if not c.startswith("scalerevenue_")]
    cols = [c for c in cols if not c.startswith("table_")]
    cols = [c for c in cols if not c.startswith("sicsector_")]
    cols = [c for c in cols if not c.startswith("scalemarketcap_")]

    return df[cols]


def reduce_columns_test(df):
    cols = set(list(df.columns))
    # omit_cols = {}
    # ret_cols = cols.difference(omit_cols)
    # return df[ret_cols].copy()

    # df.loc[:, "f22_ticker"] = "AAAA"
    new_cols = []
    for c in cols:
        if not c.startswith("f22_ticker_"):
            new_cols.append(c)

    print(new_cols)

    return df[new_cols]


unimportant_cols = {"location_Democratic People'S Republic Of Korea", 'f22_ticker_SLGN', 'f22_ticker_ULTA', 'f22_ticker_CATY', 'location_Uruguay', 'f22_ticker_ACEV',
                    'f22_ticker_UVSP', 'f22_ticker_TITN', 'f22_ticker_RBCAA', 'f22_ticker_SAL', 'f22_ticker_STX', 'f22_ticker_SVMK', 'f22_ticker_LKQ', 'f22_ticker_TVTY',
                    'f22_ticker_LOVE', 'location_Japan', 'f22_ticker_PTMN', 'f22_ticker_BLPH', 'currency_TWD', 'f22_ticker_FNLC', 'f22_ticker_HAFC', 'f22_ticker_AKTS',
                    'f22_ticker_JJSF', 'f22_ticker_CVGW', 'f22_ticker_IEP', 'location_Israel-Syria', 'f22_ticker_SELF', 'f22_ticker_SVVC', 'f22_ticker_AKRXQ',
                    'f22_ticker_ASUR', 'f22_ticker_FCBC', 'industry_Beverages - Brewers', 'f22_ticker_PAHC', 'currency_SEK', 'f22_ticker_TRST', 'f22_ticker_SVC',
                    'f22_ticker_in_text', 'f22_ticker_WSC', 'f22_ticker_CTG', 'f22_ticker_SBT', 'f22_ticker_ABCB', 'f22_ticker_ENVI', 'location_Arizona; U.S.A',
                    'f22_ticker_HOMB', 'f22_ticker_MTBC', 'f22_ticker_HSAQ', 'f22_ticker_HRZN', 'f22_ticker_CFFN', 'f22_ticker_EXLS', 'f22_ticker_BBSI', 'f22_ticker_NWLI',
                    'f22_ticker_PAYS', 'f22_ticker_TRMD', 'f22_ticker_OMER', 'f22_ticker_GARS', 'f22_ticker_UEPS', 'f22_ticker_CATC', 'f22_ticker_DCT',
                    'location_South Carolina; U.S.A', 'f22_ticker_FSFG', 'location_Ghana', 'f22_ticker_ANIP', 'f22_ticker_PRIM', 'f22_ticker_CCOI', 'f22_ticker_ORGS',
                    'f22_ticker_LVGO', 'f22_ticker_PFIN', 'f22_ticker_JRSH', 'f22_ticker_RMBI', 'f22_ticker_ALTM', 'f22_ticker_PLPC', 'f22_ticker_HVBC',
                    'famaindustry_Shipping Containers', 'scalemarketcap_6 - Mega', 'f22_ticker_PDCO', 'f22_ticker_QRVO', 'f22_ticker_ARDX', 'f22_ticker_WWD',
                    'f22_ticker_COLL', 'location_Maine; U.S.A', 'f22_ticker_NBLX', 'f22_ticker_SLM', 'f22_ticker_ACNB', 'f22_ticker_EVOP', 'f22_ticker_VERX',
                    'f22_ticker_HSIC', 'f22_ticker_PDLB', 'f22_ticker_RNDB', 'f22_ticker_FVAM', 'currency_PLN', 'f22_ticker_VECO', 'f22_ticker_GT', 'category_ETD',
                    'f22_ticker_HWC', 'f22_ticker_BOKF', 'f22_ticker_XBIT', 'f22_ticker_ARCB', 'f22_ticker_HMNF', 'f22_ticker_PEBO', 'f22_ticker_APXT', 'f22_ticker_HBAN',
                    'location_Jersey', 'f22_ticker_MNCL', 'f22_ticker_HQI', 'f22_ticker_SBBP', 'industry_Staffing & Outsourcing Services', 'f22_ticker_FLIC',
                    'f22_ticker_CHMG', 'f22_ticker_BDSI', 'f22_ticker_MTCR', 'f22_ticker_TECH', 'f22_ticker_PCYO', 'f22_ticker_SMED', 'f22_ticker_EWBC', 'f22_ticker_LCY',
                    'industry_Health Care Plans', 'f22_ticker_FCRD', 'f22_ticker_PTI', 'f22_ticker_BLDR', 'currency_CNY', 'f22_ticker_THBR', 'f22_ticker_WERN',
                    'f22_ticker_CSSE', 'f22_ticker_RBKB', 'f22_ticker_FARO', 'f22_ticker_SCHN', 'scalemarketcap_5 - Large', 'f22_ticker_RCKT', 'f22_ticker_KRNY',
                    'f22_ticker_OMCL', 'f22_ticker_HFWA', 'f22_ticker_COMM', 'f22_ticker_CFBK', 'f22_ticker_PTVCB', 'industry_REIT - Mortgage', 'location_Denmark',
                    'location_Bahamas', 'f22_ticker_DORM', 'f22_ticker_MAR', 'category_ADR Common Stock Secondary Class', 'f22_ticker_NGAC', 'f22_ticker_NXPI',
                    'f22_ticker_FISI', 'f22_ticker_ICCC', 'f22_ticker_FVCB', 'f22_ticker_NTAP', 'f22_ticker_KRBP', 'f22_ticker_NWL', 'f22_ticker_SBCF', 'f22_ticker_SBGI',
                    'f22_ticker_SG', 'f22_ticker_LAMR', 'f22_ticker_CFII', 'f22_ticker_FBMS', 'roa', 'f22_ticker_MOFG', 'f22_ticker_MLAB', 'f22_ticker_AVO', 'f22_ticker_MELI',
                    'f22_ticker_WSFS', 'location_Peru', 'f22_ticker_LILA', 'f22_ticker_ATRC', 'f22_ticker_PEBK', 'f22_ticker_GLRE', 'f22_ticker_KTOS', 'f22_ticker_ACIW',
                    'category_Canadian Preferred Stock', 'f22_ticker_EGRX', 'f22_ticker_INWK', 'f22_ticker_FEIM', 'f22_ticker_LPTX', 'f22_ticker_NXGN', 'f22_ticker_DLTH',
                    'f22_ticker_APYX', 'f22_ticker_GWPH', 'f22_ticker_TXRH', 'f22_ticker_LPLA', 'f22_ticker_ZAGG', 'f22_ticker_DAKT', 'industry_Thermal Coal',
                    'f22_ticker_MKSI', 'location_Indonesia', 'f22_ticker_HOLX', 'f22_ticker_SCSC', 'location_Rhode Island; U.S.A', 'f22_ticker_HFBL', 'f22_ticker_NURO',
                    'f22_ticker_BMRC', 'f22_ticker_NBN', 'f22_ticker_MRSN', 'industry_Savings & Cooperative Banks', 'f22_ticker_AIMT', 'f22_ticker_ROCH', 'f22_ticker_UG',
                    'f22_ticker_CVLT', 'f22_ticker_SLRC', 'industry_Utilities - Regulated Electric', 'f22_ticker_PLRX', 'f22_ticker_RCHG', 'f22_ticker_VIRC',
                    'f22_ticker_ICCH', 'f22_ticker_WAFD', 'f22_ticker_TOTA', 'currency_HKD', 'industry_Insurance - Specialty', 'f22_ticker_VG', 'f22_ticker_CXDO',
                    'f22_ticker_TCX', 'f22_ticker_LUNG', 'f22_ticker_LGIH', 'f22_ticker_WLDN', 'f22_ticker_IIIV', 'f22_ticker_BDTX', 'f22_ticker_FCFS', 'f22_ticker_JYNT',
                    'f22_ticker_BANF', 'f22_ticker_MNSB', 'f22_ticker_SAGE', 'f22_ticker_ENTA', 'f22_ticker_NMMC', 'f22_ticker_LBAI', 'f22_ticker_PNBK',
                    'industry_REIT - Hotel & Motel', 'f22_ticker_PCAR', 'f22_ticker_LFUS', 'f22_ticker_PTAC', 'f22_ticker_ESXB', 'f22_ticker_CAMP', 'location_Panama',
                    'currency_GBP', 'location_Manitoba; Canada', 'f22_ticker_ELGXQ', 'f22_ticker_CALB', 'f22_ticker_PBHC', 'f22_ticker_SBAC', 'f22_ticker_OXSQ',
                    'f22_ticker_NUZE', 'f22_ticker_ADBE', 'f22_ticker_CRSA', 'f22_ticker_MDB', 'f22_ticker_JBHT', 'f22_ticker_CFB', 'f22_ticker_SPSC', 'f22_ticker_MACK',
                    'location_Netherlands Antilles', 'currency_BRL', 'f22_ticker_VTRU', 'f22_ticker_BOMN', 'f22_ticker_STNE', 'f22_ticker_NTRS', 'f22_ticker_CABA',
                    'f22_ticker_TBLT', 'f22_ticker_SLCT', 'f22_ticker_AKAM', 'f22_ticker_ESGR', 'f22_ticker_BLKB', 'f22_ticker_PLXS', 'f22_ticker_HLIT', 'f22_ticker_ISNS',
                    'f22_ticker_RSSS', 'f22_ticker_STRA', 'location_Jordan', 'f22_ticker_CTRN', 'f22_ticker_QCOM', 'f22_ticker_FSRV', 'f22_ticker_MSEX', 'f22_ticker_THMO',
                    'f22_ticker_GTHX', 'f22_ticker_CENT', 'f22_ticker_REYN', 'f22_ticker_OPCH', 'location_West Virginia; U.S.A', 'f22_ticker_CLFD', 'f22_ticker_CRMT',
                    'f22_ticker_SAIA', 'f22_ticker_SMBK', 'f22_ticker_CFFI', 'f22_ticker_CASY', 'f22_ticker_LRCX', 'f22_ticker_CBTX', 'f22_ticker_NSIT', 'f22_ticker_ENOB',
                    'f22_ticker_NHTC', 'f22_ticker_EDIT', 'f22_ticker_USAP', 'f22_ticker_CEVA', 'location_Oman', 'f22_ticker_MBWM', 'f22_ticker_RRR', 'f22_ticker_PCTY',
                    'f22_ticker_SURG', 'f22_ticker_XNCR', 'f22_ticker_BCPC', 'category_Canadian Stock Warrant', 'f22_ticker_BROG', 'f22_ticker_CMCSA', 'f22_ticker_MBIO',
                    'f22_ticker_LIVN', 'f22_ticker_SMCI', 'f22_ticker_PYPL', 'f22_ticker_RBNC', 'category_Canadian Common Stock Primary Class', 'f22_ticker_OTTR',
                    'f22_ticker_INSM', 'f22_ticker_LNT', 'f22_ticker_SWKH', 'f22_ticker_ZNTL', 'industry_Chemicals', 'f22_ticker_GBDC', 'f22_ticker_ZBRA', 'f22_ticker_SQFT',
                    'f22_ticker_HHR', 'f22_ticker_MIND', 'category_Domestic Stock Warrant', 'f22_ticker_TWCT', 'f22_ticker_IPGP', 'f22_ticker_AHPI',
                    'industry_Capital Markets', 'f22_ticker_BSRR', 'f22_ticker_FIVE', 'f22_ticker_SBSI', 'f22_ticker_TRMB', 'f22_ticker_NODK', 'f22_ticker_AZPN',
                    'f22_ticker_UROV', 'f22_ticker_ILMN', 'f22_ticker_ATVI', 'f22_ticker_AINV', 'f22_ticker_IIN', 'f22_ticker_SONA', 'f22_ticker_CAAS', 'f22_ticker_PATK',
                    'f22_ticker_RIGL', 'f22_ticker_NSEC', 'location_United Republic Of Tanzania', 'f22_ticker_MPAA', 'f22_ticker_BKNG', 'f22_ticker_OTEL', 'category_ETF',
                    'f22_ticker_SIRI', 'f22_ticker_KEQU', 'f22_ticker_UNIT', 'f22_ticker_FSEA', 'f22_ticker_BGCP', 'f22_ticker_TTWO', 'f22_ticker_EXPD', 'f22_ticker_RILY',
                    'f22_ticker_PGNY', 'location_United Arab Emirates', 'f22_ticker_RGLS', 'assetsavg', 'f22_ticker_ACAM', 'f22_ticker_GNTX', 'f22_ticker_WMG', 'equityavg',
                    'industry_Specialty Finance', 'f22_ticker_UBSI', 'f22_ticker_FFNW', 'f22_ticker_CPHC', 'f22_ticker_NICK', 'f22_ticker_LMST', 'f22_ticker_SYBT',
                    'f22_ticker_PTSI', 'f22_ticker_ESSC', 'f22_ticker_SSNC', 'f22_ticker_HTLF', 'f22_ticker_ETAC', '3_down_in_row', 'f22_ticker_KSMT', 'f22_ticker_LPCN',
                    'f22_ticker_VREX', 'f22_ticker_OPTN', 'f22_ticker_PLUS', 'f22_ticker_FFBW', 'f22_ticker_BCTG', 'f22_ticker_CUE', 'f22_ticker_ETFC', 'f22_ticker_AAWW',
                    'f22_ticker_SAMG', 'f22_ticker_SFBC', 'f22_ticker_BMTC', 'f22_ticker_LCA', 'f22_ticker_MANH', 'f22_ticker_ROIC', 'f22_ticker_PRNB', 'f22_ticker_OYST',
                    'f22_ticker_RCEL', 'f22_ticker_INTU', 'f22_ticker_EPAY', 'f22_ticker_EXC', 'f22_ticker_CHCO', 'f22_ticker_MSBI', 'f22_ticker_SCPH', 'f22_ticker_VSPR',
                    'f22_ticker_PAE', 'f22_ticker_FCAC', 'f22_ticker_GENC', 'f22_ticker_FGBI', 'f22_ticker_HOPE', 'f22_ticker_CERS', 'f22_ticker_AACQ', 'f22_ticker_KOR',
                    'f22_ticker_EBMT', 'f22_ticker_SPOK', 'f22_ticker_HOFT', 'f22_ticker_FDUS', 'industry_REIT - Industrial', 'f22_ticker_RMR',
                    'industry_Financial Conglomerates', 'f22_ticker_ZNGA', 'famaindustry_Steel Works Etc', 'f22_ticker_TBBK', 'f22_ticker_DFPH', 'f22_ticker_ECOL',
                    'f22_ticker_LMAT', 'f22_ticker_GEOS', 'f22_ticker_CHRS', 'f22_ticker_CTXS', 'f22_ticker_AZN', 'f22_ticker_CSBR', 'f22_ticker_LBC', 'f22_ticker_PSMT',
                    'f22_ticker_CWBC', 'f22_ticker_COUP', 'f22_ticker_TNDM', 'f22_ticker_GFN', 'f22_ticker_TMUS', 'table_SEP', 'f22_ticker_PEP', 'f22_ticker_BCBP',
                    'f22_ticker_BRP', 'f22_ticker_KBAL', 'f22_ticker_AIMC', 'f22_ticker_APEI', 'famaindustry_Coal', 'f22_ticker_CJJD', 'currency_AUD', 'f22_ticker_KURA',
                    'f22_ticker_FRPT', 'f22_ticker_VBLT', 'f22_ticker_PWOD', 'f22_ticker_PTC', 'f22_ticker_SNEX', 'f22_ticker_ALRS', 'f22_ticker_ITRI', 'f22_ticker_NHIC',
                    'user_has_extended_profile', 'f22_ticker_THFF', 'f22_ticker_TIPT', 'f22_ticker_MCBS', 'industry_Insurance - Life', 'f22_ticker_ESSA', 'f22_ticker_GIII',
                    'f22_ticker_SATS', 'f22_ticker_CFAC', 'f22_ticker_STKL', 'f22_ticker_FELE', 'f22_ticker_VLY', 'f22_ticker_HTBK', 'f22_ticker_NDAQ', 'roe', 'f22_ticker_MU',
                    'f22_ticker_GERN', 'f22_ticker_VRSN', 'f22_ticker_FSTR', 'f22_ticker_INVA', 'f22_ticker_PIH', 'f22_ticker_MGEE', 'f22_ticker_CSTR', 'f22_ticker_DBX',
                    'f22_ticker_SHBI', 'f22_ticker_NRC', 'f22_ticker_OSBC', 'f22_ticker_FIXX', 'f22_ticker_GPP', 'f22_ticker_LTRPA', 'f22_ticker_LIQT', 'f22_ticker_PBCT',
                    'f22_ticker_STBA', 'f22_ticker_PRGS', 'f22_ticker_PNNT', 'f22_ticker_AWH', 'f22_ticker_SPLK', 'f22_ticker_XENT', 'f22_ticker_SIC', 'f22_ticker_MGYR',
                    'f22_ticker_KRTX', 'f22_ticker_ITAC', 'f22_ticker_ISBC', 'f22_ticker_METC', 'f22_ticker_MTEX', 'f22_ticker_OTEX', 'f22_ticker_OPRA', 'f22_ticker_LIVK',
                    'f22_ticker_CYBR', 'f22_ticker_VRNS', 'f22_ticker_UPLCQ', 'f22_ticker_FDBC', 'f22_ticker_AXGN', 'f22_ticker_AMTD', 'f22_ticker_PFIE', 'f22_ticker_RXDX',
                    'f22_ticker_BWB', 'f22_ticker_PZZA', 'f22_ticker_SWTX', 'f22_ticker_FONR', 'f22_ticker_CINF', 'f22_ticker_RELV', 'sicsector_Construction',
                    'f22_ticker_ISTR', 'f22_ticker_GNTY', 'f22_ticker_CCB', 'location_North Dakota; U.S.A', 'f22_ticker_VRTU', 'f22_ticker_NBL', 'f22_ticker_AMOT',
                    'f22_ticker_WHF', 'f22_ticker_AXLA', 'location_Alaska; U.S.A', 'f22_ticker_CCBG', 'f22_ticker_CRUS', 'f22_ticker_EQIX', 'f22_ticker_VRTX',
                    'f22_ticker_RVSB', 'f22_ticker_EDTK', 'f22_ticker_SOHO', 'industry_Copper', 'f22_ticker_EBC', 'f22_ticker_JBSS', 'f22_ticker_ONEW', 'f22_ticker_NTUS',
                    'f22_ticker_ALCO', 'f22_ticker_UBFO', 'location_Monaco', 'f22_ticker_IART', 'f22_ticker_WVFC', 'f22_ticker_SHEN', 'f22_ticker_ALOT', 'f22_ticker_PFC',
                    'f22_ticker_STAY', 'f22_ticker_DSPG', 'f22_ticker_AGNC', 'f22_ticker_ILPT', 'f22_ticker_LOPE', 'f22_ticker_BTAQ', 'f22_ticker_ANDA', 'f22_ticker_POWI',
                    'f22_ticker_SFST', 'f22_ticker_VBFC', 'f22_ticker_FBIZ', 'f22_ticker_THRM', 'f22_ticker_ACLS', 'f22_ticker_MCHP', 'scalerevenue_6 - Mega',
                    'f22_ticker_KINS', 'f22_ticker_LPSN', 'f22_ticker_CHRW', 'f22_ticker_NTGR', 'f22_ticker_COWN', 'industry_Drug Manufacturers - Major', 'currency_KRW',
                    'f22_ticker_CSGP', 'f22_ticker_MKTX', 'f22_ticker_NVDA', 'f22_ticker_PLBC', 'f22_ticker_IBOC', 'f22_ticker_ASFI', 'location_Thailand', 'f22_ticker_HCAP',
                    'f22_ticker_EIDX', 'f22_ticker_DDOG', 'f22_ticker_BANR', 'f22_ticker_WTBA', 'f22_ticker_CDXS', 'industry_Home Furnishings & Fixtures', 'f22_ticker_CETV',
                    'f22_ticker_FUSB', 'f22_ticker_BOCH', 'f22_ticker_STLD', 'f22_ticker_OSIS', 'f22_ticker_PINC', 'f22_ticker_PCTI', 'f22_ticker_ACAD', 'f22_ticker_INAQ',
                    'f22_ticker_JKHY', 'f22_ticker_ROST', 'location_Israel-Jordan', 'f22_ticker_FCCY', 'f22_ticker_CFBI', 'f22_ticker_PPC', 'f22_ticker_EMCF',
                    'f22_ticker_JBLU', 'f22_ticker_MIDD', 'f22_ticker_PGC', 'f22_ticker_BCML', 'industry_Department Stores', 'f22_ticker_FTOC', 'f22_ticker_LYTS',
                    'f22_ticker_VMAC', 'f22_ticker_RCM', 'currency_INR', 'f22_ticker_UMPQ', 'f22_ticker_FORR', 'sicsector_Public Administration', 'f22_ticker_ZS',
                    'f22_ticker_MEIP', 'f22_ticker_SONN', 'f22_ticker_RP', 'f22_ticker_GRSV', 'f22_ticker_CSGS', 'f22_ticker_INTC', 'f22_ticker_FAST', 'f22_ticker_MHLD',
                    'f22_ticker_ICBK', 'f22_ticker_MATW', 'f22_ticker_WEN', 'location_Nova Scotia; Canada', 'f22_ticker_NKTX', 'f22_ticker_LARK', 'f22_ticker_SGMA',
                    'user_time_zone', 'f22_ticker_PTCT', 'f22_ticker_OKTA', 'f22_ticker_RTLR', 'f22_ticker_INDB', 'f22_ticker_PESI', 'f22_ticker_LOGM', 'f22_ticker_HYAC',
                    'f22_ticker_SFNC', 'f22_ticker_TRMT', 'f22_ticker_RBB', 'f22_ticker_MLND', 'f22_ticker_EGBN', 'f22_ticker_FMAO', 'f22_ticker_MLAC',
                    'industry_Infrastructure Operations', 'f22_ticker_CLBK', 'f22_ticker_CHNG', 'f22_ticker_SAFM', 'f22_ticker_CHEF', 'f22_ticker_JCS', 'f22_ticker_BCDA',
                    'f22_ticker_BNR', 'f22_ticker_HSON', 'f22_ticker_UBOH', 'industry_Industrial Metals & Minerals', 'famaindustry_Textiles', 'roic',
                    'famaindustry_Beer & Liquor', 'f22_ticker_WBA', 'f22_ticker_WVVI', 'f22_ticker_FFIN', 'f22_ticker_BECN', 'f22_ticker_SCVL', 'f22_ticker_CNBKA',
                    'location_New Hampshire; U.S.A', 'f22_ticker_URGN', 'f22_ticker_PRSC', 'f22_ticker_BPMC', 'industry_Banks - Regional - US', 'f22_ticker_DGICA',
                    'f22_ticker_FCAP', 'f22_ticker_PVBC', 'f22_ticker_AEGN', 'f22_ticker_FRBK', 'f22_ticker_GHIV', 'f22_ticker_TEAM', 'f22_ticker_MTCH', 'f22_ticker_NEON',
                    'f22_ticker_TW', 'f22_ticker_CERN', 'f22_ticker_ERIE', 'f22_ticker_AKCA', 'f22_ticker_CTBI', 'f22_ticker_ACBI', 'f22_ticker_RUTH',
                    'industry_Broadcasting - Radio', 'f22_ticker_BOTJ', 'f22_ticker_TRS', 'f22_ticker_HBP', 'f22_ticker_HALO', 'f22_ticker_PODD', 'f22_ticker_ISEE',
                    'f22_ticker_XP', 'f22_ticker_MBUU', 'f22_ticker_GBCI', 'f22_ticker_UNAM', 'industry_Mortgage Finance', 'f22_ticker_TAYD', 'f22_ticker_PSTX',
                    'f22_ticker_LANC', 'location_Hungary', 'currency_CHF', 'f22_ticker_KRUS', 'f22_ticker_CNDT', 'f22_ticker_STOK', 'f22_ticker_AUTL', 'f22_ticker_CGNX',
                    'f22_ticker_FFIC', 'industry_Banks - Diversified', 'f22_ticker_PROV', 'f22_ticker_ENSG', 'f22_ticker_CRVL', 'f22_ticker_ATXI', 'f22_ticker_RFIL',
                    'f22_ticker_LOB', 'f22_ticker_RBCN', 'f22_ticker_GRMN', 'industry_<unknown>', 'f22_ticker_PTVE', 'f22_ticker_FB', 'f22_ticker_RADI', 'f22_ticker_GRCY',
                    'f22_ticker_WINMQ', 'f22_ticker_VBTX', 'f22_ticker_HYMC', 'f22_ticker_SYKE', 'f22_ticker_PBYI', 'f22_ticker_FMBH', 'f22_ticker_ASO', 'f22_ticker_VNOM',
                    'category_CEF', 'f22_ticker_UPLD', 'f22_ticker_CBAN', 'f22_ticker_RVLT', 'f22_ticker_CCAP', 'f22_ticker_ARKR', 'f22_ticker_HGSH', 'f22_ticker_AMSWA',
                    'location_Norway', 'f22_ticker_ATCX', 'f22_ticker_NTWK', 'f22_ticker_ABUS', 'f22_ticker_TTEK', 'f22_ticker_IMKTA', 'f22_ticker_SPNE', 'f22_ticker_GO',
                    'invcapavg', 'f22_ticker_TSBK', 'f22_ticker_GBLI', 'f22_ticker_RPD', 'f22_ticker_ADI', 'f22_ticker_RDVT', 'location_Republic Of Korea', 'f22_ticker_NXST',
                    'f22_ticker_SPKE', 'f22_ticker_CDNS', 'f22_ticker_UCBI', 'f22_ticker_ACCD', 'f22_ticker_EBTC', 'f22_ticker_POOL', 'f22_ticker_CVLG', 'f22_ticker_MTSC',
                    'f22_ticker_NFLX', 'industry_Media - Diversified', 'industry_Household & Personal Products', 'location_Costa Rica', 'f22_ticker_RGCO', 'f22_ticker_MMAC',
                    'f22_ticker_BLU', 'f22_ticker_GRNV', 'currency_ARS', 'f22_ticker_COGT', 'f22_ticker_HBT', 'f22_ticker_BL', 'industry_Lodging', 'f22_ticker_HRMY',
                    'f22_ticker_FBSS', 'f22_ticker_FTIV', 'f22_ticker_UTMD', 'f22_ticker_AVGO', 'f22_ticker_CGRO', 'f22_ticker_BSET', 'f22_ticker_AMED', 'f22_ticker_LHCG',
                    'f22_ticker_SAII', 'f22_ticker_CTAS', 'f22_ticker_EML', 'f22_ticker_ANDE', 'f22_ticker_MOHO', 'f22_ticker_COKE', 'f22_ticker_KFFB', 'f22_ticker_PRDO',
                    'industry_Furnishings', 'f22_ticker_CZWI', 'f22_ticker_SOHU', 'f22_ticker_TGTX', 'f22_ticker_ENPH', 'f22_ticker_REG', 'f22_ticker_FMBI', 'f22_ticker_HUBG',
                    'f22_ticker_ROAD', 'f22_ticker_SMTC', 'f22_ticker_EXEL', 'f22_ticker_GSBC', 'f22_ticker_BRKL', 'f22_ticker_DISH', 'f22_ticker_TZOO', 'f22_ticker_JRVR',
                    'f22_ticker_ASML', 'sicsector_Services', 'table_<unknown>', 'industry_Other Precious Metals & Mining', 'f22_ticker_FTNT', 'industry_Confectioners',
                    'f22_ticker_LSTR', 'f22_ticker_PRAA', 'f22_ticker_JAZZ', 'f22_ticker_DJCO', 'f22_ticker_ARCC', 'f22_ticker_CBFV', 'f22_ticker_TDAC', 'f22_ticker_CHTR',
                    'location_Michigan; U.S.A', 'f22_ticker_PNFP', 'f22_ticker_IDCC', 'f22_ticker_CPTA', 'f22_ticker_XLRN', 'f22_ticker_DMLP', 'industry_Discount Stores',
                    'f22_ticker_SWKS', 'f22_ticker_LAWS', 'f22_ticker_QH', 'f22_ticker_EFSC', 'f22_ticker_SEIC', 'f22_ticker_JACK', 'f22_ticker_MOR', 'f22_ticker_YNDX',
                    'f22_ticker_PFLT', 'f22_ticker_IRMD', 'f22_ticker_SGA', 'f22_ticker_MYRG', 'f22_ticker_FUSN', 'f22_ticker_FHB', 'f22_ticker_GVP', 'f22_ticker_KRYS',
                    'f22_ticker_CBMG', 'f22_ticker_STRS', 'f22_ticker_SNFCA', 'f22_ticker_EZPW', 'f22_ticker_LSAC', 'f22_ticker_AMCI', 'f22_ticker_SYNH', 'f22_ticker_CSII',
                    'f22_ticker_IBKR', 'f22_ticker_PPIH', 'f22_ticker_OXFD', 'f22_ticker_STND', 'f22_ticker_GNRS', 'industry_Semiconductor Memory', 'f22_ticker_OZK',
                    'category_Domestic Common Stock Secondary Class', 'f22_ticker_JAMF', 'f22_ticker_STWO', 'f22_ticker_SRRA', 'category_<unknown>', 'f22_ticker_ZUMZ',
                    'location_Turkey', 'f22_ticker_PCVX', 'f22_ticker_USLM', 'f22_ticker_EA', 'f22_ticker_OPOF', 'f22_ticker_UHAL', 'f22_ticker_AVID', 'f22_ticker_SANM',
                    'f22_ticker_AGRX', 'f22_ticker_FLXN', 'f22_ticker_ISSC', 'f22_ticker_HEES', 'f22_ticker_CVBF', 'f22_ticker_FSBW', 'famaindustry_Aircraft',
                    'f22_ticker_SRCL', 'f22_ticker_ECPG', 'f22_ticker_SMTX', 'f22_ticker_OMEX', 'f22_ticker_GLDD', 'f22_ticker_QELL', 'f22_ticker_SMMF',
                    'famaindustry_Tobacco Products', 'f22_ticker_COLB', 'f22_ticker_SSB', 'f22_ticker_LCAP', 'f22_ticker_HSKA', 'f22_ticker_GXGX', 'location_Venezuela',
                    'location_Unknown', 'f22_ticker_CIVB', 'f22_ticker_ARAV', 'f22_ticker_PACW', 'f22_ticker_SRDX', 'f22_ticker_LMRK', 'location_Philippines',
                    'f22_ticker_SGEN', 'f22_ticker_VCEL', 'f22_ticker_NTNX', 'f22_ticker_FIII', 'user_geo_enabled', 'industry_Coal', 'category_IDX', 'f22_ticker_NVEC',
                    'f22_ticker_CBRL', 'f22_ticker_AGYS', 'f22_ticker_CSCO', 'f22_ticker_KNSL', 'f22_ticker_PFIS', 'f22_ticker_OCUL', 'f22_ticker_XBIO', 'f22_ticker_KE',
                    'location_Russian Federation', 'f22_ticker_SMPL', 'location_Cayman Islands', 'f22_ticker_NBTB', 'f22_ticker_VRM', 'f22_ticker_AMRB', 'f22_ticker_SIVB',
                    'f22_ticker_GWRS', 'f22_ticker_TARA', 'f22_ticker_SENEA', 'f22_ticker_HQY', 'industry_Healthcare Plans', 'industry_Airports & Air Services',
                    'f22_ticker_PRCP', 'location_Netherlands', 'f22_ticker_EVBG', 'f22_ticker_ALRM', 'f22_ticker_SASR', 'f22_ticker_XRAY', 'industry_Financial Exchanges',
                    'f22_ticker_ONB', 'industry_REIT - Healthcare Facilities', 'f22_ticker_SYNC', 'f22_ticker_FRAF', 'f22_ticker_UFPT',
                    'industry_Insurance - Property & Casualty', 'f22_ticker_NEO', 'f22_ticker_CBSH', 'industry_Steel', 'f22_ticker_PETQ', 'f22_ticker_FFWM', 'f22_ticker_FENC',
                    'f22_ticker_TGLS', 'f22_ticker_SKYW', 'f22_ticker_ARTNA', 'f22_ticker_CECE', 'f22_ticker_GOOGL', 'industry_Lumber & Wood Production', 'f22_ticker_EKSO',
                    'f22_ticker_ANSS', 'f22_ticker_HSTM', 'f22_ticker_PTGX', 'f22_ticker_FSDC', 'f22_ticker_PAAS', 'f22_ticker_LCNB', 'f22_ticker_HWCC', 'f22_ticker_OLED',
                    'location_Canada', 'f22_ticker_NGHC', 'f22_ticker_GLBZ', 'f22_ticker_GURE', 'f22_ticker_DFHT', 'f22_ticker_HLIO', 'f22_ticker_RGEN', 'f22_ticker_MANT',
                    'f22_ticker_NWSA', 'industry_Conglomerates', 'f22_ticker_TXN', 'f22_ticker_MGRC', 'industry_Business Equipment', 'f22_ticker_WASH', 'f22_ticker_AOUT',
                    'f22_ticker_CPAH', 'f22_ticker_AXSM', 'f22_ticker_KIDS', 'f22_ticker_NDSN', 'f22_ticker_NGM', 'f22_ticker_MFNC', 'f22_ticker_AEIS', 'f22_ticker_BIDU',
                    'industry_Beverages - Wineries & Distilleries', 'f22_ticker_ABMD', 'famaindustry_Defense', 'f22_ticker_SSP', 'f22_ticker_PAYX',
                    'category_Domestic Preferred Stock', 'f22_ticker_HONE', 'f22_ticker_FRG', 'industry_Business Services', 'f22_ticker_HCKT', 'f22_ticker_TSLA',
                    'industry_Packaging & Containers', 'f22_ticker_SNPS', 'f22_ticker_PCYG', 'f22_ticker_NATI', 'f22_ticker_RYTM', 'f22_ticker_BSVN', 'f22_ticker_BJRI',
                    'f22_ticker_COHR', 'f22_ticker_HNNA', 'f22_ticker_PEGA', 'f22_ticker_NESR', 'f22_ticker_FIVN', 'f22_ticker_ADTN', 'f22_ticker_EBAY',
                    'sector_Communication Services', 'f22_ticker_LXEH', 'f22_ticker_MDRX', 'industry_Food Distribution', 'f22_ticker_ULBI', 'f22_ticker_PAIC',
                    'f22_ticker_KFRC', 'f22_ticker_AMGN', 'f22_ticker_JCTCF', 'f22_ticker_ALSK', 'f22_ticker_IESC', 'f22_ticker_CSWC', 'f22_ticker_ZGYH', 'f22_ticker_ALKS',
                    'f22_ticker_ULH', 'industry_Medical Care', 'industry_Utilities - Regulated Water', 'f22_ticker_PCSA', 'f22_ticker_NLTX', 'f22_ticker_AKUS',
                    'f22_ticker_<unknown>', 'f22_ticker_CALM', 'f22_ticker_BGNE', 'f22_ticker_MBOT', 'f22_ticker_VRTS', 'f22_ticker_ATRO', 'f22_ticker_ALJJ',
                    'location_Luxembourg', 'f22_ticker_VRSK', 'f22_ticker_NMRK', 'currency_ILS', 'f22_ticker_LTRN', 'f22_ticker_NBIX', 'f22_ticker_ANAT', 'f22_ticker_SRCE',
                    'f22_ticker_CAC', 'f22_ticker_CMPR', 'f22_ticker_PLCE', 'f22_ticker_ACGL', 'f22_ticker_CBPO', 'industry_Beverages - Non-Alcoholic', 'f22_ticker_HTLD',
                    'f22_ticker_CIGI', 'f22_ticker_IPAR', 'f22_ticker_ICMB', 'f22_ticker_PFSW', 'f22_ticker_MCBC', 'f22_ticker_ROLL', 'f22_ticker_MXIM', 'f22_ticker_RIVE',
                    'f22_ticker_SIGA', 'f22_ticker_DLTR', 'f22_ticker_CROX', 'f22_ticker_VSTA', 'f22_ticker_CRAI', 'f22_ticker_KELYA', 'f22_ticker_HLNE', 'f22_ticker_TCPC',
                    'f22_ticker_SAMA', 'f22_ticker_AIRG', 'f22_ticker_CRWD', 'f22_ticker_SHSP', 'f22_ticker_DHIL', 'f22_ticker_AMSF', 'f22_ticker_EYE',
                    'category_ADR Preferred Stock', 'f22_ticker_CNNB', 'f22_ticker_KLAC', 'f22_ticker_HMST', 'f22_ticker_AMHC', 'f22_ticker_PBIP', 'f22_ticker_SREV',
                    'f22_ticker_NYMT', 'f22_ticker_MRLN', 'f22_ticker_OCSL', 'f22_ticker_INOV', 'f22_ticker_DOCU', 'f22_ticker_OPHC', 'industry_Medical Distribution',
                    'f22_ticker_BFST', 'f22_ticker_QCRH', 'f22_ticker_DKNG', 'f22_ticker_INMD', 'f22_ticker_GRIF', 'f22_ticker_HCAC', 'f22_ticker_NERV', 'f22_ticker_PRFT',
                    'f22_ticker_CORE', 'f22_ticker_AAXN', 'f22_ticker_WETF', 'f22_ticker_TILE', 'f22_ticker_INGN', 'f22_ticker_LMNR', 'f22_ticker_MEDP', 'f22_ticker_APLS',
                    'f22_ticker_ELOX', 'f22_ticker_RVNC', 'f22_ticker_AMAT', 'f22_ticker_AAON', 'industry_Long-Term Care Facilities', 'f22_ticker_THCA',
                    'industry_Beverages - Soft Drinks', 'industry_Diversified Industrials', 'f22_ticker_ABTX', 'f22_ticker_WSBC', 'f22_ticker_IGIC', 'f22_ticker_VC',
                    'f22_ticker_ALBO', 'industry_REIT - Office', 'f22_ticker_ATLO', 'location_Italy', 'f22_ticker_MCFT', 'f22_ticker_SGMO', 'location_Missouri; U.S.A',
                    'f22_ticker_BMRA', 'f22_ticker_KPTI', 'f22_ticker_STFC', 'location_Saskatchewan; Canada', 'f22_ticker_RNLX', 'f22_ticker_PRVB', 'f22_ticker_ASND',
                    'f22_ticker_XENE', 'f22_ticker_GTLS', 'f22_ticker_FCBP', 'f22_ticker_MCAC', 'f22_ticker_FOXA', 'f22_ticker_NFBK', 'f22_ticker_GWGH', 'f22_ticker_TRMK',
                    'f22_ticker_CERC', 'f22_ticker_SPFI', 'f22_ticker_AMBC', 'industry_Home Improvement Stores', 'f22_ticker_AMZN', 'f22_ticker_NTRA', 'f22_ticker_FITB',
                    'f22_ticker_CDW', 'f22_ticker_LEDS', 'f22_ticker_UEIC', 'f22_ticker_TER', 'f22_ticker_FWONA', 'f22_ticker_STXB', 'f22_ticker_OSW', 'f22_ticker_SANW',
                    'industry_Railroads', 'f22_ticker_NOVT', 'f22_ticker_HEC', 'f22_ticker_VICR', 'location_Cyprus', 'f22_ticker_ECHO', 'f22_ticker_FMNB', 'f22_ticker_MLVF',
                    'f22_ticker_CMCO', 'f22_ticker_NTRP', 'f22_ticker_WABC', 'f22_ticker_NMCI', 'f22_ticker_AMEH', 'f22_ticker_MSVB', 'f22_ticker_WINA', 'f22_ticker_HURC',
                    'f22_ticker_EGOV', 'location_Delaware; U.S.A', 'f22_ticker_REAL', 'f22_ticker_EYEN', 'f22_ticker_RMBS', 'f22_ticker_ESPR', 'f22_ticker_BCOR',
                    'f22_ticker_ATNI', 'f22_ticker_OVBC', 'industry_Aluminum', 'table_SF1', 'f22_ticker_WLFC', 'f22_ticker_VERY', 'f22_ticker_SLP', 'location_Switzerland',
                    'f22_ticker_MRBK', 'f22_ticker_EVER', 'f22_ticker_SVBI', 'f22_ticker_HFFG', 'f22_ticker_WHLM', 'f22_ticker_SBRA', 'f22_ticker_OPRT', 'f22_ticker_IEC',
                    'f22_ticker_WTFC', 'f22_ticker_ATSG', 'f22_ticker_CSTL', 'industry_Apparel Stores', 'f22_ticker_ODFL', 'f22_ticker_CLAR', 'f22_ticker_FANG',
                    'f22_ticker_GOOD', 'f22_ticker_CDK', 'f22_ticker_CMCT', 'currency_USD', 'f22_ticker_CPRT', 'f22_ticker_NWPX', 'f22_ticker_LOAC',
                    'industry_Farm & Construction Equipment', 'f22_ticker_TCF', 'f22_ticker_DYNT', 'f22_ticker_TCBK', 'f22_ticker_UFPI', 'f22_ticker_FATE', 'f22_ticker_GH',
                    'f22_ticker_BIMI', 'f22_ticker_SAFT', 'f22_ticker_IDXX', 'f22_ticker_DHC', 'f22_ticker_FTDR', 'category_ETN', 'f22_ticker_QADB', 'f22_ticker_RGNX',
                    'f22_ticker_BAND', 'currency_MYR', 'f22_ticker_EBSB', 'f22_ticker_BDGE', 'f22_ticker_TROW', 'f22_ticker_TCFC', 'f22_ticker_FFBC', 'currency_<unknown>',
                    'f22_ticker_ADSK', 'f22_ticker_SFBS', 'f22_ticker_UNB', 'f22_ticker_SBUX', 'f22_ticker_NAVI', 'f22_ticker_NKTR', 'f22_ticker_SMBC', 'f22_ticker_BRLI',
                    'f22_ticker_CME', 'f22_ticker_CGNT', 'location_Virgin Islands; U.S.', 'f22_ticker_VSAT', 'f22_ticker_VERO', 'f22_ticker_PFPT', 'f22_ticker_SCHL',
                    'location_Puerto Rico', 'f22_ticker_CNST', 'f22_ticker_NRIM', 'f22_ticker_UAL', 'f22_ticker_WYNN', 'f22_ticker_VLGEA', 'f22_ticker_MCRI', 'f22_ticker_CG',
                    'f22_ticker_NSSC', 'f22_ticker_BKSC', 'location_Iowa; U.S.A', 'f22_ticker_NVIV', 'f22_ticker_DCPH', 'f22_ticker_JOUT', 'f22_ticker_IIIN',
                    'f22_ticker_ALXN', 'f22_ticker_HSII', 'f22_ticker_PICO', 'f22_ticker_UFCS', 'f22_ticker_DXPE', 'f22_ticker_WSBF', 'f22_ticker_GLAD',
                    'industry_REIT - Diversified', 'f22_ticker_ISRG', 'industry_Shipping & Ports', 'f22_ticker_EYEG', 'location_Louisiana; U.S.A', 'location_Guam',
                    'f22_ticker_OBNK', 'location_Mississippi; U.S.A', 'f22_ticker_AYLA', 'f22_ticker_GDS', 'scalerevenue_5 - Large', 'f22_ticker_UMBF', 'f22_ticker_PRAH',
                    'f22_ticker_RRBI', 'currency_JPY', 'f22_ticker_RPRX', 'location_Chile', 'f22_ticker_SFM', 'f22_ticker_OPNT', 'f22_ticker_OVLY', 'f22_ticker_NMFC',
                    'f22_ticker_PBFS', 'f22_ticker_SITM', 'f22_ticker_CHMA', 'f22_ticker_KVHI', 'industry_Industrial Distribution', 'f22_ticker_CREE', 'f22_ticker_MPWR',
                    'f22_ticker_CAKE', 'f22_ticker_ALNY', 'f22_ticker_DRAD', 'f22_ticker_TFSL', 'f22_ticker_LAKE', 'f22_ticker_POWL', 'f22_ticker_ICFI', 'f22_ticker_IOSP',
                    'f22_ticker_WOOF', 'f22_ticker_HAYN', 'f22_ticker_SIGI', 'f22_ticker_CNMD', 'location_Iceland', 'f22_ticker_SPPI', 'f22_ticker_AROW', 'f22_ticker_CHUY',
                    'f22_ticker_BRKS', 'f22_ticker_LKFN', 'f22_ticker_ONEM', 'f22_ticker_KZR', 'f22_ticker_CHPM', 'f22_ticker_CLMT', 'f22_ticker_NWBI',
                    'location_Marshall Islands', 'f22_ticker_REGN', 'f22_ticker_ATAX', 'f22_ticker_BWFG', 'f22_ticker_INCY', 'f22_ticker_FRME', 'f22_ticker_BBCP', 'ros',
                    'f22_ticker_PPBI', 'f22_ticker_GAIN', 'f22_ticker_MGLN', 'f22_ticker_LAUR', 'f22_ticker_LITE', 'f22_ticker_PCH', 'f22_ticker_TPIC', 'f22_ticker_COST',
                    'location_Saint Vincent And The Grenadines', 'f22_ticker_CPSI', 'f22_ticker_AUB', 'industry_Oil & Gas Integrated', 'f22_ticker_CNOB', 'f22_ticker_RLMD',
                    'f22_ticker_RACA', 'f22_ticker_HWKN', 'f22_ticker_QNST', 'f22_ticker_CNXN', 'f22_ticker_EPSN', 'f22_ticker_LEGH', 'industry_Computer Systems',
                    'f22_ticker_ROCK', 'f22_ticker_LSBK', 'f22_ticker_AGFS', 'f22_ticker_EXPE', 'f22_ticker_XEL', 'f22_ticker_ACHC', 'f22_ticker_PKBK', 'f22_ticker_SLAB',
                    'f22_ticker_RDNT', 'f22_ticker_CWCO', 'f22_ticker_CDZI', 'f22_ticker_PHAT', 'f22_ticker_MCMJ', 'location_Austria',
                    'industry_Financial Data & Stock Exchanges', 'f22_ticker_DOMO', 'f22_ticker_FCNCA', 'f22_ticker_NTCT', 'industry_Tobacco', 'f22_ticker_AAL',
                    'f22_ticker_ZGNX', 'f22_ticker_MRCC', 'category_ADR Stock Warrant', 'location_Mexico', 'f22_ticker_ERES', 'f22_ticker_PSEC', 'f22_ticker_ORMP',
                    'f22_ticker_AMPH', 'f22_ticker_PENN', 'f22_ticker_VACQ', 'f22_ticker_SCOR', 'f22_ticker_BRID', 'location_North Carolina; U.S.A', 'f22_ticker_NKSH',
                    'f22_ticker_ADUS', 'f22_ticker_ORLY', 'f22_ticker_BMCH', 'f22_ticker_RADA', 'industry_Telecom Services', 'f22_ticker_CPIX', 'f22_ticker_BMRN',
                    'location_Macau', 'f22_ticker_WMGI', 'f22_ticker_PTRS', 'f22_ticker_GCBC', 'f22_ticker_WDAY', 'f22_ticker_BZUN', 'f22_ticker_QLGN', 'f22_ticker_TRVI',
                    'f22_ticker_ENTG', 'location_Malaysia', 'f22_ticker_LATN', 'f22_ticker_BREW', 'f22_ticker_LYFT', 'f22_ticker_SSSS', 'f22_ticker_ETSY', 'f22_ticker_FRGI',
                    'f22_ticker_LSCC', 'f22_ticker_ATRS', 'f22_ticker_JCOM', 'f22_ticker_MBIN', 'f22_ticker_USCR', 'location_Colombia', 'f22_ticker_JD', 'f22_ticker_KRNT',
                    'user_verified', 'f22_ticker_CCRN', 'f22_ticker_PLXP', 'f22_ticker_CBMB', 'f22_ticker_NWFL', 'f22_ticker_SGMS', 'industry_Uranium', 'scalerevenue_4 - Mid',
                    'f22_ticker_BELFA', 'f22_ticker_ITIC', 'f22_ticker_LFAC', 'f22_ticker_PDFS', 'location_New Brunswick; Canada', 'f22_ticker_PFHD', 'industry_Data Storage',
                    'f22_ticker_FRLN', 'industry_Utilities - Independent Power Producers', 'f22_ticker_MGTX', 'f22_ticker_SHOO', 'location_Czech Republic', 'f22_ticker_OESX',
                    'f22_ticker_BPFH', 'f22_ticker_MRTN', 'f22_ticker_FREE', 'f22_ticker_ARTW', 'f22_ticker_IBCP', 'f22_ticker_WIRE', 'f22_ticker_NCNA', 'f22_ticker_ICUI',
                    'f22_ticker_LOAN', 'f22_ticker_KRMD', 'f22_ticker_CZR', 'f22_ticker_MNTA', 'f22_ticker_ORRF', 'f22_ticker_CRWS', 'industry_Real Estate - General',
                    'currency_CLP', 'assetturnover', 'f22_ticker_NCBS', 'f22_ticker_MNRO', 'f22_ticker_SHYF', 'f22_ticker_TWNK', 'f22_ticker_HCCI', 'f22_ticker_APPF',
                    'f22_ticker_IRDM', 'sector_Financial Services', 'f22_ticker_KHC', 'f22_ticker_ANIK', 'f22_ticker_VIAV', 'f22_ticker_WING', 'f22_ticker_FBNC',
                    'f22_ticker_LFVN', 'f22_ticker_MIRM', 'f22_ticker_OPES', 'f22_ticker_HRTX', 'f22_ticker_GECC', 'f22_ticker_AAPL', 'f22_ticker_CPRX', 'f22_ticker_CTSH',
                    'location_Montana; U.S.A', 'f22_ticker_LPRO', 'currency_CAD', 'f22_ticker_FRPH', 'f22_ticker_CSX', 'f22_ticker_FIBK', 'f22_ticker_WIX', 'f22_ticker_CCNE',
                    'f22_ticker_MAT', 'f22_ticker_RNST', 'f22_ticker_CIZN', 'f22_ticker_CZNC', 'f22_ticker_STXS', 'f22_ticker_MRVL', 'f22_ticker_OPI', 'f22_ticker_FWRD',
                    'f22_ticker_MEET', 'f22_ticker_HNRG', 'f22_ticker_MBCN', 'location_Maldives', 'f22_ticker_QURE', 'f22_ticker_BUSE', 'industry_REIT - Specialty',
                    'location_Alabama; U.S.A', 'f22_ticker_MDGL', 'f22_ticker_LEVL', 'f22_ticker_ASRV', 'f22_ticker_TTMI', 'f22_ticker_ALTA', 'f22_ticker_VCTR',
                    'f22_ticker_FFIV', 'f22_ticker_GABC', 'f22_ticker_CTSO', 'f22_ticker_NATR', 'f22_ticker_VSEC', 'f22_ticker_BEAT', 'f22_ticker_INTG', 'f22_ticker_BFC',
                    'f22_ticker_CONN', 'f22_ticker_CGBD', 'f22_ticker_JAKK', 'f22_ticker_OCSI', 'f22_ticker_GILD', 'f22_ticker_HBNC', 'location_Poland', 'f22_ticker_OFLX',
                    'f22_ticker_MMSI', 'f22_ticker_PFBI', 'f22_ticker_MASI', 'f22_ticker_RVMD', 'f22_ticker_HAS', 'f22_ticker_BPOP', 'f22_has_cashtag', 'f22_ticker_GLIBA',
                    'f22_ticker_CWST', 'f22_ticker_SSPK', 'sicsector_Nonclassifiable', 'f22_ticker_SMMC', 'f22_ticker_BOWX', 'f22_ticker_FOXF', 'f22_ticker_HBMD',
                    'f22_ticker_ASPS', 'f22_ticker_SUNS', 'location_Newfoundland; Canada', 'f22_ticker_HWBK', 'currency_MXN', 'f22_ticker_YORW', 'f22_ticker_MYFW',
                    'f22_ticker_FOCS', 'location_India', 'f22_ticker_CVLY', 'f22_ticker_BKCC', 'f22_ticker_RAVN', 'f22_ticker_CBNK', 'f22_ticker_FCCO', 'f22_ticker_FISV',
                    'f22_ticker_RGLD', 'f22_ticker_PFG', 'f22_ticker_TSCO', 'f22_ticker_FUNC', 'f22_ticker_EEFT', 'f22_ticker_FXNC', 'f22_ticker_UK', 'f22_ticker_HTBI',
                    'f22_ticker_LECO', 'f22_ticker_LBTYA', 'f22_ticker_MYSZ', 'f22_ticker_STRL', 'f22_ticker_CLRB', 'location_Idaho; U.S.A', 'f22_ticker_BSBK',
                    'f22_ticker_ALGT', 'f22_ticker_CMLF', 'location_Wyoming; U.S.A', 'f22_ticker_ETNB', 'f22_ticker_FEYE', 'f22_ticker_FPAY', 'f22_ticker_LBRDA',
                    'f22_ticker_AVT', 'f22_ticker_MDLZ', 'f22_ticker_VKTX', 'f22_ticker_SIBN', 'f22_ticker_HURN', 'f22_ticker_COFS', 'f22_ticker_LORL', 'f22_ticker_PCSB',
                    'location_Spain', 'industry_Broadcasting - TV', 'f22_ticker_TBNK', 'industry_Electronics & Computer Distribution', 'f22_ticker_TZAC', 'f22_ticker_LULU',
                    'f22_ticker_SYNA', 'f22_ticker_WLTW', 'f22_ticker_ETTX', 'f22_ticker_HBCP', 'f22_ticker_INFN', 'f22_ticker_TENB', 'f22_ticker_RCII', 'f22_ticker_HAIN',
                    'f22_ticker_LNTH', 'f22_ticker_IBTX', 'f22_ticker_PLAB', 'f22_ticker_MRAM', 'f22_ticker_MNST', 'f22_ticker_ALTR', 'industry_REIT - Residential',
                    'f22_ticker_PCB', 'f22_ticker_IONS', 'f22_ticker_BRKR', 'f22_ticker_UNTY', 'currency_EUR', 'f22_ticker_AGBA', 'f22_ticker_GLPI', 'f22_ticker_NDLS',
                    'f22_ticker_RELL', 'location_Finland', 'f22_ticker_CTRE', 'f22_ticker_PPD', 'f22_ticker_EXPO', 'f22_ticker_NHLD', 'f22_ticker_IAC', 'f22_ticker_OCFC',
                    'f22_ticker_MSFT', 'f22_ticker_AMNB', 'sicsector_<unknown>', 'f22_ticker_CXDC', 'f22_ticker_ARNA', 'f22_ticker_NCNO', 'industry_Tools & Accessories',
                    'f22_ticker_AMYT', 'f22_ticker_XGN', 'f22_ticker_NEWT', 'f22_ticker_BFIN', 'f22_ticker_NLOK', 'f22_ticker_ADP', 'f22_ticker_OFIX', 'f22_ticker_SCWX',
                    'f22_ticker_SMIT', 'location_Argentina', 'f22_ticker_BHF', 'f22_ticker_CONE', 'f22_ticker_BNFT', 'f22_ticker_TSC', 'f22_ticker_TTEC'}


def start(src_path: Path, dest_path: Path, prediction_mode: PredictionMode, purchase_date_str: str, send_msgs: bool = True):
    ensure_dir(dest_path)

    logger.info(f"Getting files from {src_path}")

    df = get_stocks_merged(stock_merge_drop_path=src_path)

    logger.info(f"Stock-merged size: {df.shape[0]}")

    tapp = get_stock_merge_trainer_params(stock_merge_drop_path=src_path)
    tapp.prediction_mode = prediction_mode

    split_dfs, _ = get_split_dfs(df, purchase_date_str, tapp)

    training_bag = TrainingBag()
    roi_all = []
    if tapp.prediction_mode == PredictionMode.RealMoneyStockRecommender:
        dpi = DayPredictionInfo(tapp=tapp, training_bag=training_bag, output_path=dest_path, max_models=len(split_dfs.dfs))
        for ndx, sd in enumerate(split_dfs.dfs):
            dpi.df = sd

            persist_results = False
            if ndx == len(split_dfs.dfs) - 1:
                persist_results = True

            predict_day(dpi=dpi, persist_results=persist_results)

        tickers = inspect_real_pred_results(output_path=dest_path, purchase_date_str=tapp.purchase_date_str)

        if send_msgs:
            slack_service.send_direct_message_to_chris(f"{tapp.purchase_date_str}: {str(tickers)}")
    else:
        roi = train_skipping_data(output_path=dest_path, tapp=tapp, training_bag=training_bag, split_dfs=split_dfs)
        if roi is not None:
            roi_all.append(roi)

    if tapp.prediction_mode == PredictionMode.DevelopmentAndTraining:
        twitter_root_path = src_path.parent.parent
        TrainingBag.persist(training_bag=training_bag, twitter_root=twitter_root_path)

    overall_roi = None
    if len(roi_all) > 0:
        overall_roi = mean(roi_all)

    if overall_roi is not None:
        logger.info(f"Overall roi: {overall_roi}")

    return overall_roi


def get_split_dfs(df: pd.DataFrame, purchase_date_str: str, tapp: TrainAndPredictionParams):
    nth_sell_day = 1 + tapp.num_days_until_purchase + tapp.num_hold_days
    mode_offset = 0 if tapp.prediction_mode == PredictionMode.DevelopmentAndTraining else 2
    tweet_date_str = get_next_market_day_no_count_closed_days(date_str=purchase_date_str, num_days=-(nth_sell_day - mode_offset))

    all_dfs = []
    df.sort_values(by=["date"], inplace=True)
    for i in range(nth_sell_day):
        tapp.tweet_date_str = tweet_date_str
        tapp.max_date_str = tapp.tweet_date_str

        dates = skip_day_predictor.get_every_nth_tweet_date(nth_sell_day=nth_sell_day, skip_start_days=i)
        dates = list(set(dates) | {tapp.tweet_date_str})
        all_dfs.append(df[df["date"].isin(dates)].copy())
    split_dfs = SplitDataFrames(split_dfs=all_dfs)

    return split_dfs, tweet_date_str


def get_stocks_merged(stock_merge_drop_path: Path):
    stocks_merged_path = Path(stock_merge_drop_path, STOCKS_MERGED_FILENAME)
    df = pd.read_parquet(stocks_merged_path)
    return df


def find_unimportant_features(important_feats: List[ImportantFeatures], narrow_cols: List[str]):
    result = set()
    for im_fe in important_feats:
        result |= im_fe.feature_columns

    return set(narrow_cols) - result


def train_skipping_data(output_path: Path,
                        tapp: TrainAndPredictionParams,
                        training_bag: TrainingBag.persist,
                        split_dfs: SplitDataFrames) -> float:
    has_more_days = True
    roi_all = []
    overall_roi = None

    dates = split_dfs.get_dates()

    # FIXME: 2021-03-30: chris.flesche: Temporary?
    dates = [d for d in dates if d >= tapp.tweet_date_str]

    dpi = DayPredictionInfo(tapp=tapp, training_bag=training_bag, output_path=output_path, max_models=len(split_dfs.dfs))

    count = 0
    while has_more_days:
        df = split_dfs.get_dataframe(date_str=tapp.tweet_date_str)
        if df is not None:

            dpi.set_df(df=df)

            roi = predict_day(dpi=dpi, persist_results=True)

            if roi is not None:
                roi_all.append(roi)
                logger.info(f"Ongoing roi: {mean(roi_all)}")

        dates = dates[1:]
        tapp.tweet_date_str = dates[0]
        has_more_days = len(dates) > 1

        count += 1

    if len(roi_all) > 0:
        overall_roi = mean(roi_all)

    # unimportant_feats = find_unimportant_features(important_feats=dpi.important_feats,
    #                                               narrow_cols=list(dpi.narrow_cols))
    # if unimportant_feats is not None:
    #     logger.info(f"Unimportant features: {unimportant_feats}")

    return overall_roi


def inspect_real_pred_results(output_path: Path, purchase_date_str: str):
    real_mon_path = get_real_money_preds_path(output_path)
    df = pd.read_csv(str(real_mon_path))

    logger.info(f"Looking for purchase date str: {purchase_date_str} in CSV {str(real_mon_path)}")

    # TODO: 2021-03-16: chris.flesche:  Not yet known if this is a good strategy
    # df_g = df.groupby(by=["f22_ticker", "purchase_date"]) \
    #             .agg(F.mean(F.col("prediction").alias("mean_prediction")))

    df = df[df["purchase_date"] == purchase_date_str]
    tickers = df["f22_ticker"].to_list()
    shuffle(tickers)

    logger.info(f"Num: {len(tickers)}: {tickers}")

    return set(tickers)


if __name__ == '__main__':
    twit_root_path = constants.TWITTER_OUTPUT_RAW_PATH
    # twit_root_path = Path(constants.TEMP_PATH, "twitter")
    src_path = Path(twit_root_path, "stock_merge_drop", "main")
    dest_path = Path(twit_root_path, "prediction_bucket")

    pred_path = Path(dest_path, PREDICTIONS_CSV)
    if pred_path.exists():
        pred_path.unlink()

    prediction_mode = PredictionMode.DevelopmentAndTraining  # PredictionMode.RealMoneyStockRecommender

    # purchase_date_str = date_utils.get_standard_ymd_format(date=datetime.now())
    purchase_date_str = "2020-08-10"

    roi = start(src_path=src_path,
                dest_path=dest_path,
                prediction_mode=prediction_mode,
                send_msgs=False,
                purchase_date_str=purchase_date_str)

    slack_service.send_direct_message_to_chris(f"Roi: {roi}")