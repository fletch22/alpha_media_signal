import time
from pathlib import Path

import pandas as pd

from ams.config import logger_factory, constants

logger = logger_factory.create(__name__)


def process():
    start_time = time.time()
    df = pd.read_csv(Path(constants.DAILY_STOCK_DIR, 'SHARADAR_SEP.csv'))
    logger.info(f'Elapsed time: {time.time() - start_time:.2f}')

    df_g = df.groupby('ticker')

    for ndx, (ticker, df_ticker) in enumerate(df_g):
        logger.info(f'{ticker}: {df_ticker.shape[0]}')
        ticker_path = constants.SHAR_SPLIT_EQUITY_EOD_DIR / f'{ticker}.parquet'
        logger.info(f'{ticker_path}')
        ticker_path.parent.mkdir(parents=True, exist_ok=True)
        df_ticker.reset_index().to_parquet(ticker_path)


if __name__ == '__main__':
    process()
