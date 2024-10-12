import logging

logger = logging.getLogger(__name__)


class CustomFormatter(logging.Formatter):
    """Logging colored formatter, adapted from
    https://stackoverflow.com/a/56944256/3638629"""

    def __init__(self, fmt):
        super().__init__()
        grey = '\x1b[38;21m'
        blue = '\x1b[38;5;39m'
        yellow = "\x1b[33;20m"
        red = '\x1b[38;5;196m'
        bold_red = '\x1b[31;1m'
        reset = '\x1b[0m'

        self.FORMATS = {
            logging.DEBUG: grey + fmt + reset,
            logging.INFO: blue + fmt + reset,
            logging.WARNING: yellow + fmt + reset,
            logging.ERROR: red + fmt + reset,
            logging.CRITICAL: bold_red + fmt + reset
        }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)
