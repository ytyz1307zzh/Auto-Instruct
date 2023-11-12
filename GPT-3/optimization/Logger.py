import logging

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S'
)
logger = logging.getLogger(__name__)
# logger.addHandler(logging.StreamHandler())


def set_log_path_and_level(filepath, log_level):
    # create file handler which logs even debug messages
    fh = logging.FileHandler(filepath)

    # add the handlers to the logger
    logger.addHandler(fh)

    # Set logging level
    logger.setLevel(log_level)
