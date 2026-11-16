"""Structured (JSON) logging with a per-request id that follows every log line."""

import json
import logging
import sys
import time
from contextvars import ContextVar

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")


class RequestIdFilter(logging.Filter):
    def filter(self, record):
        record.request_id = request_id_var.get()
        return True


class JsonFormatter(logging.Formatter):
    RESERVED = set(vars(logging.makeLogRecord({})).keys()) | {"request_id", "message", "asctime"}

    def format(self, record):
        out = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
                  + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "request_id": getattr(record, "request_id", "-"),
            "msg": record.getMessage(),
        }
        # anything passed with logger.info(..., extra={...})
        out.update({k: v for k, v in vars(record).items() if k not in self.RESERVED})
        if record.exc_info:
            out["exc"] = self.formatException(record.exc_info)
        return json.dumps(out, default=str)


def setup_logging(level: str = "INFO", as_json: bool = True):
    handler = logging.StreamHandler(sys.stdout)
    handler.addFilter(RequestIdFilter())
    handler.setFormatter(JsonFormatter() if as_json else logging.Formatter(
        "%(asctime)s %(levelname)s [%(request_id)s] %(name)s: %(message)s"))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level.upper())
    for noisy in ("uvicorn.access", "httpx", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
