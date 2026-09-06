"""
Structured logging setup for the application.
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Any, Optional

import structlog
from structlog.types import Processor

from .config import LoggingConfig


def setup_logging(config: LoggingConfig, log_dir: Optional[Path] = None) -> None:
    """
    Configure structured logging for the application.

    Args:
        config: Logging configuration
        log_dir: Optional directory for log files. If None, uses current directory.
    """
    # Ensure log directory exists
    if log_dir is None:
        log_dir = Path.cwd()
    log_file = log_dir / config.file
    log_file.parent.mkdir(parents=True, exist_ok=True)

    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, config.level.upper(), logging.INFO),
    )

    # Configure file handler with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=config.max_size_mb * 1024 * 1024,
        backupCount=config.backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(getattr(logging, config.level.upper(), logging.INFO))

    # Add file handler to root logger
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)

    # Configure structlog processors
    processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if config.format == "json":
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, config.level.upper(), logging.INFO)
        ),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.BoundLogger:
    """
    Get a structured logger for the given module name.

    Args:
        name: Module name (typically __name__)

    Returns:
        Structured logger instance
    """
    return structlog.get_logger(name)


def bind_session_context(logger: structlog.BoundLogger, session_id: str) -> structlog.BoundLogger:
    """
    Bind session context to logger.

    Args:
        logger: Logger to bind context to
        session_id: Session ID to bind

    Returns:
        Logger with session context bound
    """
    return logger.bind(session_id=session_id)