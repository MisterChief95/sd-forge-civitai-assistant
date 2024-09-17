import logging

from modules.shared import opts


is_debug = getattr(opts, "is_debug", False)

formatter = logging.Formatter("[%(name)s] - %(levelname)s :: %(message)s")

logger = logging.getLogger("CivitaiAssistant")
logger.setLevel(logging.INFO if not is_debug else logging.DEBUG)
logger.propagate = False

if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO if not is_debug else logging.DEBUG)
    console_handler.setFormatter(formatter)

logger.addHandler(console_handler)
