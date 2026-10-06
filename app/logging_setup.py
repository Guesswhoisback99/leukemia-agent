import logging
import sys


def setup_logging(level: str) -> None:
    """Root logging to stdout, one line per record."""
    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
        force=True,
    )
    # Qdrant client logs every HTTP request at INFO.
    logging.getLogger("httpx").setLevel(logging.WARNING)
