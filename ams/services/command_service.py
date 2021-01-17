import os
from enum import Enum
from pathlib import Path

from ams.config import constants


class StockPredictorPaths(Enum):
    command_equity_daily_svc_path = Path(constants.STOCK_PREDICTOR_ROOT, "sedft.cmd")
    command_equity_fun_svc_path = Path(constants.STOCK_PREDICTOR_ROOT, "sefs.cmd")


def invoke_command_script(command_path: Path):
    command = f"cmd.exe /c {command_path}\n"
    return True if os.system(command) == 0 else False


def start(command_path: Path):
    was_success = invoke_command_script(command_path=command_path)

    if not was_success:
        raise Exception(f"Encountered problem running script '{str(command_path)}'.")


def get_equity_daily_data():
    start(command_path=StockPredictorPaths.command_equity_daily_svc_path.value)


def get_equity_fundamentals_data():
    start(command_path=StockPredictorPaths.command_equity_daily_svc_path.value)


if __name__ == '__main__':
    command_path_test = Path(constants.STOCK_PREDICTOR_ROOT, "test.cmd")
    start(command_path=command_path_test)