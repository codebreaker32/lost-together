"""OpenAI-compatible chat client (urllib only) with retries, a request-hash cache and three modes.

  live    call the server (cache hits are served from the cache; misses are recorded)
  replay  cache only; a miss raises CacheMiss (used for offline re-runs of recorded GPU sessions)
  stub    a deterministic fake model (StubModel) for tests and offline development

Default target: qwen3-vl:30b on Ollama behind an SSH tunnel at http://localhost:11434/v1, thinking off
(`reasoning_effort: "none"`). The cache is jsonl under results/agent_cache/, keyed by sha256 of the
canonical request body (+ a sample index for k-repeat variance runs). Each record keeps the response,
the real latency and token usage, so replays report the latency measured on the GPU. The API key, if any,
is read from an environment variable named in the config and is never logged or cached.
"""
from __future__ import annotations

import hashlib
import http.client
import json
import os
import random
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE = ROOT / "results" / "agent_cache"


class LLMError(RuntimeError):
    pass


class CacheMiss(LLMError):
    pass


class LLMTimeout(LLMError):
    pass


@dataclass
class LLMConfig:
    mode: str = "stub"                               # live | replay | stub
    base_url: str = "http://localhost:11434/v1"
    model: str = "qwen3-vl:30b-a3b-instruct"         # the bare 30b tag is the thinking variant; think=false is ignored
    api_key_env: str | None = None                   # e.g. "GROQ_KEY"; Ollama needs none
    timeout_s: float = 60.0
    retries: int = 2                                 # extra attempts after the first
    backoff_s: float = 1.0                           # exponential: backoff * 2**attempt, plus jitter
    temperature: float = 0.0
    max_tokens: int = 400
    reasoning_effort: str | None = "none"            # Ollama: disables qwen3 thinking
    image_in_tool: bool = True                       # False: images go in a follow-up user message (Groq)
    cache_dir: Path | str | None = DEFAULT_CACHE     # None disables the cache
    sample: int = 0                                  # k-repeat index; part of the cache key
    usd_per_hour: float = 2.235                      # g6e.xlarge on-demand, ap-south-1
    usd_per_mtok_in: float = 0.0                     # per-token pricing for hosted fallbacks
    usd_per_mtok_out: float = 0.0

    @classmethod
    def from_env(cls, **overrides) -> "LLMConfig":
        """LT_LLM_MODE / LT_LLM_BASE_URL / LT_LLM_MODEL / LT_LLM_KEY_ENV / LT_LLM_IMAGE_IN_TOOL override defaults."""
        env = {"mode": os.environ.get("LT_LLM_MODE"), "base_url": os.environ.get("LT_LLM_BASE_URL"),
               "model": os.environ.get("LT_LLM_MODEL"), "api_key_env": os.environ.get("LT_LLM_KEY_ENV")}
        cfg = cls(**{k: v for k, v in env.items() if v})
        if os.environ.get("LT_LLM_IMAGE_IN_TOOL") in ("0", "false", "no"):
            cfg.image_in_tool = False
        for k, v in overrides.items():
            if v is not None:
                setattr(cfg, k, v)
        return cfg

    def cost(self, latency_s: float, tok_in: int, tok_out: int) -> float:
        return (latency_s * self.usd_per_hour / 3600.0
                + tok_in * self.usd_per_mtok_in / 1e6 + tok_out * self.usd_per_mtok_out / 1e6)


@dataclass
class ChatResponse:
    message: dict                                    # the assistant message: {role, content, tool_calls?}
    latency_s: float                                 # measured (live) or recorded (cache hit)
    usage: dict = field(default_factory=dict)        # prompt_tokens / completion_tokens when known
    cached: bool = False
    key: str = ""
    attempts: int = 1


def request_key(body: dict, sample: int = 0) -> str:
    canon = json.dumps({"body": body, "sample": sample}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canon.encode()).hexdigest()


class ResponseCache:
    """Append-only jsonl, one file per model. Stores the response, latency and usage, never the request
    (requests carry base64 images) and never credentials."""

    def __init__(self, cache_dir: Path | str, model: str):
        slug = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in model)
        self.path = Path(cache_dir) / f"{slug}.jsonl"
        self._mem: dict[str, dict] = {}
        self._lock = threading.Lock()
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                try:
                    rec = json.loads(line)
                    self._mem[rec["key"]] = rec
                except (json.JSONDecodeError, KeyError):
                    continue                          # a torn last line from an interrupted run

    def get(self, key: str) -> dict | None:
        return self._mem.get(key)

    def put(self, key: str, rec: dict) -> None:
        rec = {"key": key, **rec}
        with self._lock:
            self._mem[key] = rec
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as f:
                f.write(json.dumps(rec) + "\n")

    def __len__(self) -> int:
        return len(self._mem)


StubFn = Callable[[list[dict], list[dict] | None], dict]


class LLMClient:
    def __init__(self, cfg: LLMConfig | None = None, stub: StubFn | None = None,
                 opener: Callable[..., Any] | None = None):
        self.cfg = cfg or LLMConfig()
        if self.cfg.mode not in ("live", "replay", "stub"):
            raise ValueError(f"unknown mode {self.cfg.mode!r}")
        self.cache = ResponseCache(self.cfg.cache_dir, self.cfg.model) if self.cfg.cache_dir else None
        self.stub = stub
        if self.cfg.mode == "stub" and self.stub is None:
            from .stub import StubModel
            self.stub = StubModel()
        self._open = opener or urllib.request.urlopen
        self.calls = 0

    def body(self, messages: list[dict], tools: list[dict] | None, json_mode: bool = False) -> dict:
        b: dict[str, Any] = {"model": self.cfg.model, "messages": messages, "temperature": self.cfg.temperature,
                             "max_tokens": self.cfg.max_tokens, "stream": False}
        if tools:
            b["tools"] = tools
            b["tool_choice"] = "auto"
        if json_mode:
            b["response_format"] = {"type": "json_object"}
        if self.cfg.reasoning_effort:
            b["reasoning_effort"] = self.cfg.reasoning_effort
        return b

    def chat(self, messages: list[dict], tools: list[dict] | None = None, json_mode: bool = False,
             deadline: float | None = None) -> ChatResponse:
        self.calls += 1
        if self.cfg.mode == "stub":
            t0 = time.perf_counter()
            msg = self.stub(messages, tools)          # may raise LLMTimeout / LLMError to simulate faults
            return ChatResponse(message=msg, latency_s=time.perf_counter() - t0, usage=_estimate_usage(messages, msg))
        body = self.body(messages, tools, json_mode)
        key = request_key(body, self.cfg.sample)
        if self.cache is not None:
            rec = self.cache.get(key)
            if rec is not None:
                return ChatResponse(message=rec["message"], latency_s=rec["latency_s"], usage=rec.get("usage", {}),
                                    cached=True, key=key)
        if self.cfg.mode == "replay":
            raise CacheMiss(f"no cached response for request {key[:12]} (model {self.cfg.model})")
        resp = self._post(body, deadline)
        resp.key = key
        if self.cache is not None:
            self.cache.put(key, {"model": self.cfg.model, "message": resp.message, "latency_s": resp.latency_s,
                                 "usage": resp.usage, "ts": time.time(), "sample": self.cfg.sample})
        return resp

    def _post(self, body: dict, deadline: float | None) -> ChatResponse:
        url = self.cfg.base_url.rstrip("/") + "/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.cfg.api_key_env:
            key = os.environ.get(self.cfg.api_key_env)
            if not key:
                raise LLMError(f"environment variable {self.cfg.api_key_env} is not set")
            headers["Authorization"] = f"Bearer {key}"
        data = json.dumps(body).encode()
        last: Exception | None = None
        for attempt in range(self.cfg.retries + 1):
            timeout = self.cfg.timeout_s
            if deadline is not None:
                timeout = min(timeout, deadline - time.monotonic())
                if timeout <= 0.05:
                    raise LLMTimeout("decision deadline reached before the model call")
            req = urllib.request.Request(url, data=data, headers=headers, method="POST")
            t0 = time.perf_counter()
            try:
                with self._open(req, timeout=timeout) as r:
                    payload = json.loads(r.read().decode())
                latency = time.perf_counter() - t0
                choice = payload["choices"][0]["message"]
                msg = {"role": "assistant", "content": choice.get("content")}
                if choice.get("tool_calls"):
                    msg["tool_calls"] = choice["tool_calls"]
                return ChatResponse(message=msg, latency_s=latency, usage=payload.get("usage") or {},
                                    attempts=attempt + 1)
            except urllib.error.HTTPError as e:
                last = LLMError(f"HTTP {e.code}")
                if e.code not in (408, 409, 429, 500, 502, 503, 504):
                    raise last from None             # 4xx (bad request, auth) won't get better on retry
            except TimeoutError as e:
                last = LLMTimeout(f"timed out after {timeout:.1f}s")
            except urllib.error.URLError as e:
                last = LLMTimeout(f"timed out: {e.reason}") if isinstance(e.reason, TimeoutError) \
                    else LLMError(f"connection failed: {e.reason}")
            except (KeyError, IndexError, TypeError, json.JSONDecodeError, UnicodeDecodeError) as e:
                last = LLMError(f"malformed response: {type(e).__name__}")
            except (OSError, http.client.HTTPException) as e:   # reset / remote closed / incomplete read
                last = LLMError(f"connection failed: {type(e).__name__}")
            if attempt < self.cfg.retries:
                pause = self.cfg.backoff_s * 2 ** attempt * (1 + 0.25 * random.random())
                if deadline is not None and time.monotonic() + pause >= deadline:
                    break
                time.sleep(pause)
        raise last or LLMError("model call failed")


def _estimate_usage(messages: list[dict], msg: dict) -> dict:
    """Rough token estimate for the stub (4 chars/token, ~800 tokens per 512px image)."""
    chars, images = 0, 0
    for m in messages:
        c = m.get("content")
        if isinstance(c, str):
            chars += len(c)
        elif isinstance(c, list):
            for part in c:
                if part.get("type") == "text":
                    chars += len(part.get("text", ""))
                elif part.get("type") == "image_url":
                    images += 1
    out = len(json.dumps(msg))
    return {"prompt_tokens": chars // 4 + 800 * images, "completion_tokens": out // 4, "estimated": True}
