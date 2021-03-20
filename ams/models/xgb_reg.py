from typing import List

import pandas as pd
import xgboost as xgb

from ams.config import logger_factory
from ams.twitter.twitter_ml_utils import transform_to_numpy, get_data_for_predictions

logger = logger_factory.create(__name__)


def train_predict(df_train: pd.DataFrame, df_test: pd.DataFrame, narrow_cols: List[str], label_col: str = "buy_sell", require_balance: bool = True, buy_thresh: float = 0.):
    import warnings

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

            logger.info(f"Train Sell: {num_sell} / Buy: {num_buy}; ratio: {balance_ratio}")

            xgb_args["scale_pos_weight"] = balance_ratio

        # grid_search(X_train=X_train, y_train=y_train)

        logger.info(f"Using XGB Args: {xgb_args}")
        model = xgb.XGBRegressor(**xgb_args)

        model.fit(X_train,
                  y_train)

    X_predict = get_data_for_predictions(df=df_test, narrow_cols=narrow_cols, standard_scaler=standard_scaler)

    logger.info("Invoking model prediction ...")
    prediction = model.predict(X_predict)

    df_test.loc[:, "prediction"] = prediction

    return df_test, model