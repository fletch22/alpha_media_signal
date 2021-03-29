from pathlib import Path

import pandas as pd

from ams.config import constants
from ams.pipes.p_make_prediction.mp_process import PREDICTIONS_CSV


def test_3_model_perf():
    # Arrange
    output_path = constants.TWITTER_OUTPUT_RAW_PATH
    pred_path = Path(output_path, "prediction_bucket", )
    df = pd.read_csv(pred_path)

    ModelBag

    print(df.shape[0])
    print(list(df.columns))

    df.sort_values(by=["purchase_date"], inplace=True)

    df_grouped = df.groupby(by=["purchase_date"])

    # for key, df_g_pd in df_grouped:

    while True:
        df = df[:-1]
        row_count = df.shape[0]
        print(row_count)
        if row_count <= 0:
            break


    # Act

    # Assert