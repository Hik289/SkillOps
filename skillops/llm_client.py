"""
Single-provider OpenAI Chat client with disk cache, token counting, and a
hard budget cap.

The client is deliberately minimal: there is no fallback to other providers,
no automatic retry, and no SDK-side rate limiter. API errors are written to
the call log and re-raised so the caller can decide what to do.

Typical use::

    from skillops.llm_client import LLMClient

    client = LLMClient(model="gpt-4o-mini", data_dir="./.skillops_data")
    out = client.chat(messages=[{"role": "user", "content": "hi"}])
    print(out.text, out.cost_usd, out.cumulative_cost_usd)

Environment
-----------
``OPENAI_API_KEY`` must be set. The client never reads any other key sources.
"""
from __future__ import annotations

import dataclasses
import fcntl
import hashlib
import json
import os
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Pricing and budget
# ---------------------------------------------------------------------------

PRICE_TABLE_USD_PER_MTOK: Dict[str, Dict[str, float]] = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o-mini-2024-07-18": {"input": 0.15, "output": 0.60},
    # Test stub charged at zero.
    "stub-zero": {"input": 0.0, "output": 0.0},
}

BUDGET_HARD_CAP_USD = 100.0
BUDGET_WARN_USD = 80.0


class BudgetExceededError(RuntimeError):
    """Raised when the hard cap is reached. Callers should not catch silently."""


@dataclasses.dataclass
class LLMResponse:
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float
    cumulative_cost_usd: float
    cache_hit: bool
    latency_ms: float
    raw: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Cost helper
# ---------------------------------------------------------------------------


def cost_of(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Return the USD cost for a call. Raises on unknown models."""
    if model not in PRICE_TABLE_USD_PER_MTOK:
        raise ValueError(
            f"model {model!r} not in PRICE_TABLE_USD_PER_MTOK; register it first"
        )
    p = PRICE_TABLE_USD_PER_MTOK[model]
    return (prompt_tokens / 1_000_000.0) * p["input"] + (completion_tokens / 1_000_000.0) * p["output"]


# ---------------------------------------------------------------------------
# BudgetTracker (persistent, file-locked)
# ---------------------------------------------------------------------------


class BudgetTracker:
    """File-locked JSON state for cumulative cost across processes."""

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.data_dir / "budget.json"
        self.alert_path = self.data_dir / "BUDGET_WARNING.flag"
        self.halt_path = self.data_dir / "BUDGET_HALT.flag"
        if not self.path.exists():
            self._write_state({"cumulative_cost_usd": 0.0, "calls": 0, "warned": False, "halted": False})

    def _read_state(self) -> Dict[str, Any]:
        with self.path.open("r") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_SH)
            try:
                return json.load(f)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def _write_state(self, state: Dict[str, Any]) -> None:
        tmp = self.path.with_suffix(".tmp")
        with tmp.open("w") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                json.dump(state, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        tmp.replace(self.path)

    def get_cumulative(self) -> float:
        return float(self._read_state().get("cumulative_cost_usd", 0.0))

    def state(self) -> Dict[str, Any]:
        return self._read_state()

    def precheck(self) -> None:
        s = self._read_state()
        if s.get("halted") or s.get("cumulative_cost_usd", 0.0) >= BUDGET_HARD_CAP_USD:
            raise BudgetExceededError(
                f"hard cap ${BUDGET_HARD_CAP_USD:.2f} already reached "
                f"(cumulative=${s.get('cumulative_cost_usd', 0):.4f})"
            )

    def add(self, cost_usd: float) -> Dict[str, Any]:
        with self.path.open("r+") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.seek(0)
                state = json.load(f)
                state["cumulative_cost_usd"] = float(state.get("cumulative_cost_usd", 0.0)) + float(cost_usd)
                state["calls"] = int(state.get("calls", 0)) + 1
                cum = state["cumulative_cost_usd"]
                if cum >= BUDGET_WARN_USD and not state.get("warned"):
                    state["warned"] = True
                    self._raise_alert(cum)
                if cum >= BUDGET_HARD_CAP_USD and not state.get("halted"):
                    state["halted"] = True
                    self._raise_halt(cum)
                f.seek(0)
                f.truncate()
                json.dump(state, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
                return state
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def _raise_alert(self, cum: float) -> None:
        msg = f"SkillOps budget warning: cumulative ${cum:.4f} >= ${BUDGET_WARN_USD:.2f}"
        warnings.warn(msg, RuntimeWarning, stacklevel=2)
        try:
            self.alert_path.write_text(json.dumps({"ts": time.time(), "cumulative_cost_usd": cum}, indent=2))
        except OSError:
            pass

    def _raise_halt(self, cum: float) -> None:
        try:
            self.halt_path.write_text(json.dumps({"ts": time.time(), "cumulative_cost_usd": cum}, indent=2))
        except OSError:
            pass


# ---------------------------------------------------------------------------
# DiskCache
# ---------------------------------------------------------------------------


class DiskCache:
    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def make_key(payload: Dict[str, Any]) -> str:
        canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _path(self, key: str) -> Path:
        return self.cache_dir / key[:2] / f"{key}.json"

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        p = self._path(key)
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            return None

    def put(self, key: str, value: Dict[str, Any]) -> None:
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
        tmp.replace(p)


# ---------------------------------------------------------------------------
# LLMClient
# ---------------------------------------------------------------------------


class LLMClient:
    """Single-provider OpenAI Chat client with cache + budget guardrails.

    Parameters
    ----------
    model : str
        OpenAI Chat Completions model. Must be present in
        :data:`PRICE_TABLE_USD_PER_MTOK`.
    data_dir : str or Path
        Directory for the cache, call log, and budget state.
    api_key : str, optional
        Defaults to ``os.environ['OPENAI_API_KEY']``. Raises at first chat call
        if missing.
    cache : bool
        Enable on-disk caching keyed by the canonical request payload.
    transport : Callable, optional
        Inject a callable ``f(payload) -> dict`` for tests. When ``None``, the
        real OpenAI SDK is used.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        data_dir: Optional[os.PathLike] = None,
        api_key: Optional[str] = None,
        cache: bool = True,
        transport=None,
    ) -> None:
        self.model = model
        self.data_dir = Path(data_dir) if data_dir else Path(".skillops_data")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.cache_enabled = cache
        self.cache = DiskCache(self.data_dir / "llm_cache")
        self.budget = BudgetTracker(self.data_dir)
        self.log_path = self.data_dir / "llm_calls.jsonl"
        self._transport = transport
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._openai_client = None

    # -------- log helper --------
    def _append_log(self, record: Dict[str, Any]) -> None:
        try:
            with self.log_path.open("a") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass

    # -------- real OpenAI invocation --------
    def _ensure_openai(self):
        if self._openai_client is not None:
            return
        if not self._api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:
            raise RuntimeError("openai package is not installed: pip install openai") from exc
        self._openai_client = OpenAI(api_key=self._api_key)

    def _call_openai(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        self._ensure_openai()
        resp = self._openai_client.chat.completions.create(**payload)
        return {
            "id": resp.id,
            "model": resp.model,
            "choices": [
                {
                    "index": c.index,
                    "message": {"role": c.message.role, "content": c.message.content},
                    "finish_reason": c.finish_reason,
                }
                for c in resp.choices
            ],
            "usage": {
                "prompt_tokens": resp.usage.prompt_tokens,
                "completion_tokens": resp.usage.completion_tokens,
                "total_tokens": resp.usage.total_tokens,
            },
        }

    # -------- public API --------
    def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        seed: Optional[int] = 42,
        response_format: Optional[Dict[str, Any]] = None,
        bypass_cache: bool = False,
        **extra: Any,
    ) -> LLMResponse:
        """Send one Chat Completions request.

        Cache key includes ``model``, ``messages``, ``temperature``,
        ``max_tokens``, ``seed``, ``response_format`` and any ``extra`` kwargs.
        """
        self.budget.precheck()
        m = model or self.model
        params: Dict[str, Any] = {"model": m, "messages": messages, "temperature": temperature, "seed": seed}
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        if response_format is not None:
            params["response_format"] = response_format
        params.update(extra)

        cache_key = DiskCache.make_key(params)
        cache_hit = False
        cached = None
        if self.cache_enabled and not bypass_cache:
            cached = self.cache.get(cache_key)

        t0 = time.time()
        if cached is not None:
            api_resp = cached
            cache_hit = True
        else:
            try:
                if self._transport is not None:
                    api_resp = self._transport(params)
                else:
                    api_resp = self._call_openai(params)
            except BaseException as exc:
                latency_ms = (time.time() - t0) * 1000.0
                err_record = {
                    "ts": time.time(),
                    "model": m,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "cost_usd": 0.0,
                    "cumulative_cost_usd": self.budget.get_cumulative(),
                    "latency_ms": latency_ms,
                    "cache_hit": False,
                    "cache_key": cache_key,
                    "n_messages": len(messages),
                    "error": True,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc)[:500],
                }
                self._append_log(err_record)
                raise
            if self.cache_enabled:
                self.cache.put(cache_key, api_resp)
        latency_ms = (time.time() - t0) * 1000.0

        usage = api_resp.get("usage", {}) or {}
        prompt_tokens = int(usage.get("prompt_tokens", 0))
        completion_tokens = int(usage.get("completion_tokens", 0))
        total_tokens = int(usage.get("total_tokens", prompt_tokens + completion_tokens))

        cost_usd = 0.0
        if not cache_hit:
            cost_usd = cost_of(m, prompt_tokens, completion_tokens)
            self.budget.add(cost_usd)

        cum = self.budget.get_cumulative()

        text = ""
        try:
            text = api_resp["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError):
            text = ""

        record = {
            "ts": time.time(),
            "model": m,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cost_usd": cost_usd,
            "cumulative_cost_usd": cum,
            "latency_ms": latency_ms,
            "cache_hit": cache_hit,
            "cache_key": cache_key,
            "n_messages": len(messages),
            "error": False,
        }
        self._append_log(record)

        return LLMResponse(
            text=text,
            model=m,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost_usd=cost_usd,
            cumulative_cost_usd=cum,
            cache_hit=cache_hit,
            latency_ms=latency_ms,
            raw=api_resp,
        )
