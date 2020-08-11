import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from alpha_media_signal import config


class LoggerFactory():
    rot_handler = None
    all_loggers = []

    def __init__(self):
        self.formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        self.handler_stream = logging.StreamHandler(sys.stdout)
        self.handler_stream.setFormatter(self.formatter)
        self.handler_stream.setLevel(logging.INFO)

        print(f'Will use logging path: {config.LOGGING_PATH}')

        log_path = Path(config.LOGGING_PATH, 'alpha_media_signal.log')

        self.rot_handler = RotatingFileHandler(str(log_path), maxBytes=200000, backupCount=10)
        self.rot_handler.setFormatter(self.formatter)

    def create_logger(self, name: str):
        logger = logging.getLogger(name)
        logger.addHandler(self.handler_stream)
        logger.addHandler(self.rot_handler)

        logger.setLevel(logging.INFO)
        self.all_loggers.append(logger)

        return logger
