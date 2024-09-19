import logging

from modules.shared import opts


class CustomFormatter(logging.Formatter):
    grey = "\x1b[38m"
    yellow = "\x1b[33m"
    red = "\x1b[31m"
    bold_red = "\x1b[5;31m"
    cyan = "\x1b[36m"
    white = "\x1b[97m"
    reset = "\x1b[0m"
    log_format = "{1}[%(name)s]{0} - {2}%(levelname)8s{0} :: %(message)s"

    FORMATS = {
        logging.DEBUG: log_format.format(reset, cyan, reset),
        logging.INFO: log_format.format(reset, cyan, white),
        logging.WARNING: log_format.format(reset, cyan, yellow),
        logging.ERROR: log_format.format(reset, cyan, red),
        logging.CRITICAL: log_format.format(reset, cyan, bold_red),
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        return logging.Formatter(log_fmt).format(record)


is_debug = getattr(opts, "is_debug", False)

logger = logging.getLogger("CivitaiAssistant")
logger.setLevel(logging.INFO if not is_debug else logging.DEBUG)
logger.propagate = False

if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO if not is_debug else logging.DEBUG)
    console_handler.setFormatter(CustomFormatter())

logger.addHandler(console_handler)
