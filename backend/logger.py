"""
logger.py — Structured JSON logging for StressMitigate backend.

Replaces print() statements with proper Python logging.
Outputs JSON-formatted log lines for easy parsing by monitoring tools.

Usage:
    from backend.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Server started", extra={"port": 8000})
"""
import logging
import json
import sys
from datetime import datetime, timezone


class JSONFormatter(logging.Formatter):
    """Format log records as JSON for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Include extra fields if provided
        if hasattr(record, "extra_data"):
            log_entry["data"] = record.extra_data

        # Include exception info for errors
        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
            }

        # Include source location for debug/error
        if record.levelno >= logging.WARNING:
            log_entry["source"] = {
                "file": record.pathname,
                "line": record.lineno,
                "function": record.funcName,
            }

        return json.dumps(log_entry, default=str)


class PrettyFormatter(logging.Formatter):
    """Human-readable formatter for development mode."""

    LEVEL_ICONS = {
        "DEBUG": "🔍",
        "INFO": "✅",
        "WARNING": "⚠️",
        "ERROR": "❌",
        "CRITICAL": "🔥",
    }

    def format(self, record: logging.LogRecord) -> str:
        icon = self.LEVEL_ICONS.get(record.levelname, "📝")
        timestamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
        msg = record.getMessage()

        parts = [f"[{timestamp}] {icon} {msg}"]

        if record.exc_info and record.exc_info[0] is not None:
            parts.append(f"   Exception: {record.exc_info[0].__name__}: {record.exc_info[1]}")

        return "\n".join(parts)


def get_logger(name: str, json_mode: bool = False) -> logging.Logger:
    """
    Get a configured logger instance.

    Args:
        name: Logger name (usually __name__)
        json_mode: If True, use JSON formatter. If False, use pretty formatter.
    """
    logger = logging.getLogger(name)

    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)

    if json_mode:
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(PrettyFormatter())

    logger.addHandler(handler)
    return logger
