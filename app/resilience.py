"""Retries with exponential backoff, call timeouts, and a thread-safe TTL/LRU cache."""

import concurrent.futures
import functools
import logging
import random
import threading
import time
from collections import OrderedDict
from typing import Any, Callable, Hashable, Optional, Tuple, Type

log = logging.getLogger(__name__)


def retry(attempts: int = 3, base_delay: float = 0.5, max_delay: float = 8.0,
          retry_on: Tuple[Type[BaseException], ...] = (Exception,),
          on_retry: Optional[Callable[[BaseException, int], None]] = None):
    """Retries a function on the given exceptions: delays of base, 2x base, 4x base...
    (capped, with jitter so many workers don't retry in lockstep)."""
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            for attempt in range(1, attempts + 1):
                try:
                    return fn(*args, **kwargs)
                except retry_on as e:
                    if attempt == attempts:
                        raise
                    delay = min(max_delay, base_delay * 2 ** (attempt - 1)) * random.uniform(0.8, 1.2)
                    log.warning("retrying %s after error", fn.__name__,
                                extra={"attempt": attempt, "delay_s": round(delay, 2),
                                       "error": f"{type(e).__name__}: {e}"})
                    if on_retry:
                        on_retry(e, attempt)
                    time.sleep(delay)
        return wrapper
    return deco


_pool = concurrent.futures.ThreadPoolExecutor(max_workers=8, thread_name_prefix="timeout")


def call_with_timeout(fn: Callable, timeout_s: float, *args, **kwargs):
    """Runs fn in a worker thread; raises TimeoutError if it takes longer than timeout_s.
    (The worker isn't killed, but the request stops waiting for it.)"""
    fut = _pool.submit(fn, *args, **kwargs)
    try:
        return fut.result(timeout=timeout_s)
    except concurrent.futures.TimeoutError:
        raise TimeoutError(f"{getattr(fn, '__name__', 'call')} exceeded {timeout_s}s")


class TTLCache:
    def __init__(self, maxsize: int = 512, ttl_s: float = 600):
        self.maxsize, self.ttl_s = maxsize, ttl_s
        self._data: "OrderedDict[Hashable, Tuple[float, Any]]" = OrderedDict()
        self._lock = threading.Lock()
        self.hits = self.misses = 0

    def get(self, key: Hashable):
        with self._lock:
            item = self._data.get(key)
            if item and item[0] > time.monotonic():
                self._data.move_to_end(key)
                self.hits += 1
                return item[1]
            if item:
                del self._data[key]
            self.misses += 1
            return None

    def set(self, key: Hashable, value: Any):
        if self.ttl_s <= 0 or self.maxsize <= 0:
            return
        with self._lock:
            self._data[key] = (time.monotonic() + self.ttl_s, value)
            self._data.move_to_end(key)
            while len(self._data) > self.maxsize:
                self._data.popitem(last=False)

    def clear(self):
        with self._lock:
            self._data.clear()

    def __len__(self):
        return len(self._data)
