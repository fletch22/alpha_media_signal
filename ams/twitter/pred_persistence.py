from datetime import datetime

import pandas as pd

from ams.config import constants, logger_factory
from ams.twitter.PredictionParams import PredictionParams, PredictionMode

logger = logger_factory.create(__name__)


def clean_prediction_file(pp: PredictionParams):
    if pp.prediction_mode == PredictionMode.DevelopmentAndTraining:
        is_new = not constants.TWITTER_TRAINING_PREDICTIONS_FILE_PATH.exists()
        if not is_new:
            df_preds = pd.read_csv(constants.TWITTER_TRAINING_PREDICTIONS_FILE_PATH)
            df_preds = df_preds[df_preds["num_hold_days"] != pp.num_hold_days]

            df_preds.to_csv(constants.TWITTER_TRAINING_PREDICTIONS_FILE_PATH, index=False)


def persist_predictions(df_buy, pp: PredictionParams):
    df_preds = None

    pred_path = constants.TWITTER_TRAINING_PREDICTIONS_FILE_PATH
    if pp.prediction_mode == PredictionMode.RealMoneyStockRecommender:
        pred_path = constants.TWITTER_REAL_MONEY_PREDICTIONS_FILE_PATH

    is_new = not pred_path.exists()

    if not is_new:
        df_preds = pd.read_csv(pred_path)
        df_preds = df_preds[~((df_preds["purchase_date"] == pp.purchase_date_str) & (df_preds["num_hold_days"] == pp.num_hold_days))]

        logger.info(f"Old rows found: {df_preds.shape[0]}")

    if df_preds is not None:
        df_combined = pd.concat([df_preds, df_buy], axis=0)
    else:
        df_combined = df_buy

    logger.info(f"Writing predictions to output {str(pred_path)}")
    df_combined.to_csv(pred_path, index=False)


def save_predictions(df_predict: pd.DataFrame, pp: PredictionParams) -> pd.DataFrame:
    df_buy_full = df_predict[df_predict["prediction"] == 1].copy()
    df_buy = df_buy_full[["f22_ticker", "purchase_date", "future_date", "marketcap"]].copy()

    if df_buy is None or df_buy.shape[0] == 0:
        logger.info("No buy predictions.")
    else:
        df_buy.loc[:, "num_hold_days"] = pp.num_hold_days
        df_buy.loc[:, "run_timestamp"] = datetime.timestamp(datetime.now())

        persist_predictions(df_buy=df_buy, pp=pp)

    return df_buy