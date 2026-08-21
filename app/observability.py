"""Structured logging and request correlation.

Clinical pipelines fail in ways that matter — a bad transcription or a silently
empty extraction is a patient-facing defect. Plain text logs make that hard to
trace, so every line is emitted as JSON with a request id that ties the upload,
the transcription, the extraction and the response together.
"""

from __future__ import annotations

import json
import logging
import sys
import time
import uuid
from contextvars import ContextVar

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")

# Attributes LogRecord always carries; anything else was passed via `extra=`
# and belongs in the structured payload.
_STANDARD_ATTRS = set(
    logging.LogRecord("", 0, "", 0, "", None, None).__dict__
) | {"asctime", "message", "taskName"}

_CONTEXT_KEY = "_ctx"

# Keys the log line owns. A context field of the same name is kept, but moved
# aside, so a caller can never make a record misreport its own level or message.
_ENVELOPE_KEYS = frozenset({"ts", "level", "logger", "request_id", "message", "exception"})


def install_request_id_factory() -> None:
    """Stamp the current request id onto every LogRecord as it is created.

    The formatter could read the ContextVar itself, but that attributes the line
    to whichever request is current when it is *written* — the wrong one as soon
    as a handler defers, batches, or hands off to another thread. Binding at
    creation time makes the id travel with the record.
    """
    current = logging.getLogRecordFactory()
    if getattr(current, "_stamps_request_id", False):
        return

    def factory(*args, **kwargs):
        record = current(*args, **kwargs)
        record.request_id = request_id_var.get()
        return record

    factory._stamps_request_id = True  # type: ignore[attr-defined]
    logging.setLogRecordFactory(factory)


def log_context(**fields: object) -> dict[str, object]:
    """Build an ``extra=`` payload that cannot collide with LogRecord internals.

    ``logging`` raises ``KeyError`` if an ``extra=`` key shadows an existing
    LogRecord attribute — and it does so lazily, only once the level is
    actually enabled, so a field named ``filename`` or ``module`` sails through
    a test suite and explodes in production. Nesting the fields under one key
    makes that impossible by construction; the formatter flattens them back out.
    """
    return {_CONTEXT_KEY: fields}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "request_id": getattr(record, "request_id", None) or request_id_var.get(),
            "message": record.getMessage(),
        }

        # Extras passed directly (e.g. by uvicorn) land as record attributes.
        for key, value in record.__dict__.items():
            if key not in _STANDARD_ATTRS and key not in (_CONTEXT_KEY, "request_id"):
                payload[key] = value

        for key, value in getattr(record, _CONTEXT_KEY, {}).items():
            payload[f"{key}_" if key in _ENVELOPE_KEYS else key] = value

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str)


def configure_logging(level: int = logging.INFO) -> None:
    install_request_id_factory()

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)

    # uvicorn installs its own handlers; let them bubble up to ours instead.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logger = logging.getLogger(name)
        logger.handlers = []
        logger.propagate = True


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Tag each request with an id, echo it back, and log how it went."""

    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
        token = request_id_var.set(request_id)
        logger = logging.getLogger("app.request")
        started = time.perf_counter()

        try:
            response = await call_next(request)
        except Exception:
            logger.exception(
                "request failed",
                extra=log_context(
                    method=request.method,
                    path=request.url.path,
                    duration_ms=round((time.perf_counter() - started) * 1000, 1),
                ),
            )
            request_id_var.reset(token)
            raise

        response.headers["X-Request-ID"] = request_id
        logger.info(
            "request completed",
            extra=log_context(
                method=request.method,
                path=request.url.path,
                status=response.status_code,
                duration_ms=round((time.perf_counter() - started) * 1000, 1),
            ),
        )
        request_id_var.reset(token)
        return response
