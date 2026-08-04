import logging

from common.logger import logger


def test_logger_exists():
    assert logger is not None


def test_logger_name():
    assert logger.name == "EnterpriseAIPlatform"


def test_logger_type():
    assert isinstance(logger, logging.Logger)
