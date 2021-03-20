import warnings
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import List

import pandas as pd
import xgboost as xgb

from ams.config import constants, logger_factory
from ams.services import file_services
from ams.twitter.TwitterStackingModel import TwitterStackingModel
from ams.twitter.pred_perf_testing import get_days_roi_from_prediction_table
from ams.twitter.twitter_ml_utils import transform_to_numpy, get_data_for_predictions

logger = logger_factory.create(__name__)


def train_predict(df_train: pd.DataFrame,
                  df_predict: pd.DataFrame,
                  narrow_cols: List[str],
                  label_col: str = "buy_sell",
                  require_balance: bool = True,
                  buy_thresh: float = 0.):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="invalid value encountered in true_divide")
        X_train, y_train, standard_scaler = transform_to_numpy(df=df_train,
                                                               narrow_cols=narrow_cols,
                                                               label_col=label_col,
                                                               require_balance=require_balance)

        if X_train is None or X_train.shape[0] == 0 or y_train is None:
            logger.info("Not enough training data.")
            return None

        xgb_args = dict(seed=42, max_depth=8)

        if not require_balance:
            num_buy = df_train[df_train["stock_val_change"] > buy_thresh].shape[0]
            num_sell = df_train[df_train["stock_val_change"] <= buy_thresh].shape[0]

            balance_ratio = num_sell / num_buy

            xgb_args["scale_pos_weight"] = balance_ratio

        # grid_search(X_train=X_train, y_train=y_train)

        logger.info(f"Using XGB Args: {xgb_args}")
        model = xgb.XGBRegressor(**xgb_args)

        model.fit(X_train,
                  y_train)

    X_predict = get_data_for_predictions(df=df_predict, narrow_cols=narrow_cols, standard_scaler=standard_scaler)

    logger.info("Invoking model prediction ...")
    prediction = model.predict(X_predict)

    df_predict.loc[:, "prediction"] = prediction

    return df_predict, model


def process(df_train: pd.DataFrame, df_test: pd.DataFrame, is_first: bool, twitter_stacking_model: TwitterStackingModel):
    df_train = df_train.fillna(value=0)

    print(f"Num rows: {df_train.shape[0]}")

    label_col = "stock_val_change"

    # buy_thresh = df_train["stock_val_change"].mean()
    df_predict, model = train_predict(df_train=df_train,
                                      df_predict=df_test,
                                      narrow_cols=list(df_train.columns),
                                      label_col=label_col,
                                      require_balance=False,
                                      buy_thresh=0)

    if is_first:
        twitter_stacking_model.add_trained_model(model)

    df_buy = df_predict[df_predict["prediction"] > 0].copy()

    # NOTE: 2021-03-11: chris.flesche: This divides the predictions in equal parts
    # and chooses top-part. E.g., if top_divisor=5, then the value of top of the
    # 1st quintile is selected
    top_divisor = 7
    if top_divisor != 0 and df_buy.shape[0] >= top_divisor:
        df_buy.sort_values(by=["prediction"], ascending=False, inplace=True)

        num_rows = df_buy.shape[0]
        quint = int(num_rows / top_divisor)

        min_predict = df_buy["prediction"].to_list()[quint]

        logger.info(f"Finding predictions above min: {min_predict}")
        df_buy = df_buy[df_buy["prediction"] > min_predict]

    df_buy = df_buy[["f22_ticker", "purchase_date", "future_date", "marketcap"]].copy()

    min_price = 5.
    num_hold_days = 1
    roi = None
    if df_buy is None or df_buy.shape[0] == 0:
        logger.info("No buy predictions.")
    else:
        df_buy.loc[:, "num_hold_days"] = num_hold_days
        df_buy.loc[:, "run_timestamp"] = datetime.timestamp(datetime.now())

        purchase_date_str = df_predict["purchase_date"].to_list()[0]
        roi = get_days_roi_from_prediction_table(df_preds=df_buy,
                                                 purchase_date_str=purchase_date_str,
                                                 num_hold_days=num_hold_days,
                                                 min_price=min_price,
                                                 addtl_hold_days=1)
    return roi


def run_model():
    twitter_stacking_model = TwitterStackingModel(data_dirname="model_excercise")
    par_path = Path(constants.TWIT_STOCK_MERGE_DROP_PATH, "twitter_ml")
    rev_folders = file_services.list_child_folders(parent_path=par_path)

    # twitter_ml_utils_train_37.parquet
    roi_all = []
    for rev in rev_folders:
        date_folders = file_services.list_child_folders(parent_path=rev)
        date_folders = sorted(date_folders, reverse=True)
        for ndx, dt_fld in enumerate(date_folders):

            files = file_services.list_files(parent_path=dt_fld, starts_with="twitter_ml_utils_train_", ends_with="parquet", use_dir_recursion=False)
            if len(files) == 0:
                continue
            assert (len(files) == 1)
            df_train = pd.read_parquet(path=files[0])

            files = file_services.list_files(parent_path=dt_fld, starts_with="twitter_ml_utils_test_", ends_with="parquet", use_dir_recursion=False)
            assert (len(files) == 1)
            df_test = pd.read_parquet(path=files[0])

            is_first = (ndx == 0)
            roi = process(df_train=df_train, df_test=df_test, is_first=is_first, twitter_stacking_model=twitter_stacking_model)

            if roi is not None:
                roi_all.append(roi)

            logger.info(f"Overall Mean ROI: {mean(roi_all)}")

    logger.info(f"All Revolutions Mean ROI: {mean(roi_all)}")

    TwitterStackingModel.persist(twitter_stacking_model=twitter_stacking_model)


if __name__ == '__main__':
    run_model()