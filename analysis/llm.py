"""
LLM client for Part D.

    ANALYSIS_LLM=auto     (default) OpenAI if OPENAI_API_KEY is set and `openai` is installed,
                          otherwise no LLM (extractive mode)
    ANALYSIS_LLM=openai   same, but log an error and add a warning to every answer if unavailable
    ANALYSIS_LLM=none     never call an LLM (CI, offline demos)
    ANALYSIS_MODEL        model name (default gpt-4o-mini)
    OPENAI_BASE_URL       any OpenAI-compatible endpoint (e.g. Gemini's) works too

Nothing here raises at import or construction time: Part E constructs AnalysisAgent()
at startup and only tolerates ImportError/AttributeError.
"""

import json
import logging
import os
import re
from typing import Callable, List, Optional, Protocol, Tuple, Union

log = logging.getLogger(__name__)

DEFAULT_MODEL = "gpt-4o-mini"


class LLMError(Exception):
    """Any failure talking to the model: network, timeout, refusal, unparseable JSON."""


class LLMClient(Protocol):
    name: str

    def complete_json(self, system: str, messages: List[dict], timeout_s: float) -> dict: ...


def parse_json(text: str) -> dict:
    """Parses a JSON object, tolerating ```json fences or prose around it."""
    t = (text or "").strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    try:
        obj = json.loads(t)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", t, re.S)
        if not m:
            raise LLMError("model did not return JSON")
        try:
            obj = json.loads(m.group())
        except json.JSONDecodeError as e:
            raise LLMError(f"model returned invalid JSON: {e}") from e
    if not isinstance(obj, dict):
        raise LLMError("model returned JSON that is not an object")
    return obj


class OpenAIClient:
    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None,
                 base_url: Optional[str] = None):
        import openai  # lazy: package is optional

        self._openai = openai
        self.model = model or os.getenv("ANALYSIS_MODEL") or DEFAULT_MODEL
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url, max_retries=0)
        self.name = f"openai:{self.model}"
        # Some models reject temperature / response_format; drop them after the first 400.
        self._use_temperature = True
        self._use_json_mode = True

    def complete_json(self, system: str, messages: List[dict], timeout_s: float) -> dict:
        if timeout_s <= 0:
            raise LLMError("no time left in the analysis budget")
        for _ in range(3):
            kwargs = {"model": self.model,
                      "messages": [{"role": "system", "content": system}, *messages],
                      "timeout": timeout_s}
            if self._use_temperature:
                kwargs["temperature"] = 0
            if self._use_json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            try:
                resp = self.client.chat.completions.create(**kwargs)
                return parse_json(resp.choices[0].message.content or "")
            except self._openai.BadRequestError as e:
                msg = str(e).lower()
                if self._use_temperature and "temperature" in msg:
                    self._use_temperature = False
                    continue
                if self._use_json_mode and "response_format" in msg:
                    self._use_json_mode = False
                    continue
                raise LLMError(f"OpenAI rejected the request: {e}") from e
            except LLMError:
                raise
            except Exception as e:  # timeouts, connection errors, rate limits, auth
                raise LLMError(f"{type(e).__name__}: {e}") from e
        raise LLMError("OpenAI request kept failing on unsupported parameters")


class FakeLLM:
    """Scripted client for tests and offline demos. `responses` is a list consumed in
    order (dicts, JSON strings, or exceptions to raise), or a callable(system, messages)."""

    def __init__(self, responses: Union[list, Callable], name: str = "fake", delay_s: float = 0.0):
        self.responses = responses
        self.name = name
        self.delay_s = delay_s
        self.calls: List[Tuple[str, List[dict]]] = []

    def complete_json(self, system, messages, timeout_s):
        import time
        self.calls.append((system, messages))
        if self.delay_s:
            time.sleep(self.delay_s)
        if callable(self.responses):
            r = self.responses(system, messages)
        else:
            if not self.responses:
                raise LLMError("FakeLLM has no responses left")
            r = self.responses.pop(0)
        if isinstance(r, BaseException):
            raise r
        return parse_json(r) if isinstance(r, str) else r


def llm_from_env() -> Tuple[Optional[LLMClient], Optional[str]]:
    """Returns (client or None, problem message or None). Never raises."""
    mode = (os.getenv("ANALYSIS_LLM") or "auto").strip().lower()
    if mode in ("none", "off", "false", "0", "extractive"):
        return None, None
    if mode not in ("auto", "openai"):
        return None, f"Unknown ANALYSIS_LLM={mode!r}; running without an LLM."
    if not os.getenv("OPENAI_API_KEY"):
        return None, ("ANALYSIS_LLM=openai but OPENAI_API_KEY is not set; running without an LLM."
                      if mode == "openai" else None)
    try:
        return OpenAIClient(base_url=os.getenv("OPENAI_BASE_URL") or None), None
    except ImportError:
        msg = "The `openai` package is not installed; running without an LLM."
        if mode == "openai":
            return None, msg
        log.info(msg)
        return None, None
    except Exception as e:
        return None, f"Could not create the OpenAI client ({type(e).__name__}: {e}); running without an LLM."
