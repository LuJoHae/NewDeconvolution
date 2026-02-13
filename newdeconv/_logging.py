import logging

logging.getLogger().handlers.clear()
logger = logging.getLogger(__name__)
logger.propagate = False

handler = logging.StreamHandler()
formatter = logging.Formatter("[%(name)s] %(levelname)s: %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)