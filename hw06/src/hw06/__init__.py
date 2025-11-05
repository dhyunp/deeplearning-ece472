import structlog

from .logging import configure_logging
from .config import load_settings
from .testing import run_tests

"""
Discussion:
Model design inspired by https://www.youtube.com/watch?v=kCc8FmEb1nY
and its corresponding github https://github.com/karpathy/ng-video-lecture/blob/master/gpt.py
"""


def main() -> None:
    """CLI entry point."""
    configure_logging()
    log = structlog.get_logger()
    log.info("Hello from hw06!")

    settings = load_settings()
    log.info("Settings loaded", settings=settings.model_dump())

    run_tests()
