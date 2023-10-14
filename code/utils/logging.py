import logging


def get_default_logger(name: str, log_file: str = None) -> logging.Logger:
    """create a Logger object

    Args:
        name (str): logger name
        log_file (str): log file path

    Returns:
        logging.Logger
    """
    logger = logging.getLogger(name)
    logging.basicConfig(
        filename=log_file,
        level=logging.DEBUG,
        format='%(asctime)s [%(levelname)s] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    return logger
