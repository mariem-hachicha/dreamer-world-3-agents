# core/utils.py — Utility functions for Dreamer World Family Demo

import logging
import sys


def setup_logger(name: str = "DreamerWorld") -> logging.Logger:
    """
    Configure and return a formatted logger.

    Args:
        name: Logger name (default: 'DreamerWorld').

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="[%(levelname)s] %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def print_header(title: str, width: int = 62) -> None:
    """
    Print a styled section header to stdout.

    Args:
        title: The section title to display.
        width: Total width of the separator line.
    """
    separator = "═" * width
    print(f"\n{separator}")
    print(f"  {title}")
    print(f"{separator}")


def print_field(label: str, value: str, indent: int = 4) -> None:
    """
    Print a key-value field with aligned formatting.

    Args:
        label: Field label.
        value: Field value.
        indent: Left padding in spaces.
    """
    padding = " " * indent
    print(f"{padding}{label:<16}: {value}")
