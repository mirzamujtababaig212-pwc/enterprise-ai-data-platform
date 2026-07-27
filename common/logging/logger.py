import logging
import sys

from common.config.settings import Settings


def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(Settings.env.LOG_LEVEL)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)
    return logger
