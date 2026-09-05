"""One logger for the package, INFO on stderr with timestamps. CLIs call configure()."""

import logging
import sys

logger = logging.getLogger("advisor")


def configure(level: int = logging.INFO) -> None:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S"))
    logger.handlers[:] = [handler]
    logger.setLevel(level)
    logger.propagate = False
