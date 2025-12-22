import logging


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    logging.getLogger(__name__).info('Logging is starting.')


def get_logger() -> logging.Logger:
    return logging.getLogger(__name__)


logger = get_logger()
