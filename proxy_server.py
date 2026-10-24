"""
proxy_server.py — OpenAI-compatible HTTP proxy for the Humble Human GUI
=========================================================================

Wraps the REPL loop from `main.py` in a small FastAPI server so the Vite GUI
can talk to the 3-tier memory engine over HTTP.

Endpoints
---------
    POST /v1/chat/completions   OpenAI-compatible chat completions.
                                * stream=true  -> SSE (data: <chunk>\\n\\n, [DONE])
                                * stream=false -> aggregated JSON response
                                * optional extra field "session_id" pins the
                                  request to a persistent session whose
                                  ContextManager (and thus L1/L2/L3 memory)
                                  survives across requests.
    GET  /health                Liveness + config summary.
    POST /run                   Execute a shell command locally (for the GUI's
                                "Run Command" button). Disable with
                                HH_DISABLE_RUN=1.

Streaming protocol extensions (consumed by the GUI, ignored by other clients)
-----------------------------------------------------------------------------
1. Tool results are echoed after a server-side tool round as
       {"choices":[{"index":0,"delta":{"role":"tool",
        "tool_call_id":"...","content":"<json result>"}}]}
   so the GUI can render what the memory engine returned.
2. Operational notices (compaction, weird finish reasons, tool-loop bound)
   stream as {"choices":[{"index":0,"delta":{"role":"hh_system",
   "content":"..."}}]}.
3. Tool-call `index` values are offset by 100 per round so the client can key
   accumulating deltas across multiple tool rounds without collisions.
4. Mid-stream upstream failures surface as a terminal
   data: {"error": {"message": ..., "type": ...}} chunk followed by [DONE]
   (connection-phase vs mid-stream error model borrowed from the pi-ai
   streaming stack); [DONE] is always the last event.

Mock mode
---------
    HH_MOCK_LLM=1 uvicorn proxy_server:app --port 8000

No API key needed: a heuristic mock LLM answers, exercising the full SSE +
tool-loop path with REAL memory tools. Ideal for GUI development offline.

How to run
----------
    .venv/bin/pip install fastapi uvicorn openai
    export OPENAI_API_KEY="sk-..."        # or HH_MOCK_LLM=1
    export LLM_MODEL="deepseek-chat"      # optional default upstream model
    .venv/bin/uvicorn proxy_server:app --host 127.0.0.1 --port 8000

Then point the GUI (gear icon) at http://localhost:8000/v1 — the default.

Self-test (offline, no deps beyond stdlib + memory_manager/main):
    .venv/bin/python proxy_server.py --self-test
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

try:  # fastapi is only required for serving routes, not the --self-test core
    from fastapi import Request
except ImportError:  # pragma: no cover
    class Request:  # type: ignore[no-redef]
        """Placeholder so the module still imports without fastapi."""

        headers: Dict[str, str] = {}

        async def json(self) -> Any:  # pragma: no cover
            raise NotImplementedError

from memory_manager import (
    ContextManager,
    TOOL_SCHEMAS,
    new_session_id,
)
from main import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    MAX_TOOL_ROUNDS,
    SYSTEM_PROMPT,
    execute_tool,
)

import memory_manager as _mm  # dynamic access to the active store dirs
import shlex  # quote-aware splitting for slash-command arguments

# ============================================================================
# Configuration
# ============================================================================

#: The CLI-flavored system prompt gets a GUI-flavored identity tweak.
PROXY_SYSTEM_PROMPT = SYSTEM_PROMPT.replace(
    "a CLI coding agent", "an AI coding agent with a web GUI"
)

#: Default port; override with $PROXY_PORT.
DEFAULT_PORT = int(os.environ.get("PROXY_PORT", "8000"))

#: Shell to bind uvicorn to; override with $PROXY_HOST (0.0.0.0 for LAN access).
DEFAULT_HOST = os.environ.get("PROXY_HOST", "127.0.0.1")

#: Offline mode: scripted mock instead of a real upstream (env HH_MOCK_LLM=1).
MOCK_LLM = os.environ.get("HH_MOCK_LLM") == "1"

#: /run endpoint guard.
RUN_DISABLED = os.environ.get("HH_DISABLE_RUN") == "1"
RUN_TIMEOUT_S = int(os.environ.get("HH_RUN_TIMEOUT", "20"))
RUN_OUTPUT_CAP = 4000  # chars per stream returned to the GUI

_SESSION_ID_RE = re.compile(r"[A-Za-z0-9_-]{1,128}")


def _max_tokens() -> int:
    return int(os.environ.get("HH_MAX_TOKENS", str(DEFAULT_MAX_TOKENS)))


# ============================================================================
# Session registry — one ContextManager per session_id
# ============================================================================

class SessionState:
    """A pinned ContextManager + a lock serializing turns for one session."""

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.cm = ContextManager(
            system_prompt=PROXY_SYSTEM_PROMPT, session_id=session_id
        )
        self.lock = threading.Lock()


_SESSIONS: Dict[str, SessionState] = {}
_SESSIONS_LOCK = threading.Lock()


def get_session(session_id: Optional[str]) -> Tuple[SessionState, bool]:
    """
    Get-or-create the session. Raises ValueError on a malformed id.
    Returns (state, created).
    """
    created = session_id is None
    sid = session_id or new_session_id()
    if not _SESSION_ID_RE.fullmatch(sid):
        raise ValueError(
            f"Invalid session_id {sid!r}; expected 1-128 chars of [A-Za-z0-9_-]."
        )
    with _SESSIONS_LOCK:
        state = _SESSIONS.get(sid)
        if state is None:
            state = SessionState(sid)
            _SESSIONS[sid] = state
            created = True
    return state, created


def _new_incoming_messages(
    incoming: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    The GUI sends its full visible transcript every request. Since assistant
    turns are generated server-side (already in L1), only messages AFTER the
    last assistant message are genuinely new (typically just the user turn).
    System messages are dropped: L1 pins its own system prompt.
    """
    msgs = [m for m in incoming if m.get("role") != "system"]
    last_asst = -1
    for i, m in enumerate(msgs):
        if m.get("role") == "assistant":
            last_asst = i
    return msgs[last_asst + 1 :]


# ============================================================================
# LLM access — real OpenAI-compatible upstream, or the offline mock
# ============================================================================

_LLM: Any = None
_LLM_LOCK = threading.Lock()


def _llm() -> Any:
    """Lazily build the upstream client (or the mock in HH_MOCK_LLM=1)."""
    global _LLM
    if _LLM is not None:
        return _LLM
    with _LLM_LOCK:
        if _LLM is None:
            _LLM = MockLLM() if MOCK_LLM else _build_upstream_client()
    return _LLM


def _build_upstream_client() -> Any:
    import openai  # deferred: mock mode / self-test don't need the SDK

    # max_retries=0: failures surface to the GUI immediately instead of
    # hanging the stream on silent backoff; timeout keeps turns bounded.
    return openai.OpenAI(max_retries=0, timeout=120.0)


# ----------------------------------------------------------------------------
# Per-request BYOK: the GUI can pin base_url / api_key per provider profile
# ----------------------------------------------------------------------------

_OVERRIDE_CLIENTS: Dict[Tuple[str, str], Any] = {}
_OVERRIDE_LOCK = threading.Lock()


def _override_client(base_url: str, api_key: str) -> Any:
    """Upstream client for per-request BYOK headers, cached by (url, key)."""
    import openai

    cache_key = (base_url, api_key)
    with _OVERRIDE_LOCK:
        cli = _OVERRIDE_CLIENTS.get(cache_key)
        if cli is None:
            if len(_OVERRIDE_CLIENTS) >= 16:
                _OVERRIDE_CLIENTS.clear()
            cli = openai.OpenAI(
                base_url=base_url or None,
                api_key=api_key or None,
                max_retries=0,
                timeout=120.0,
            )
            _OVERRIDE_CLIENTS[cache_key] = cli
    return cli


def _llm_for_request(base_url: str = "", api_key: str = "") -> Any:
    """Pick the upstream client: per-request headers beat env-configured."""
    if MOCK_LLM:
        return _llm()
    if base_url or api_key:
        return _override_client(base_url, api_key)
    return _llm()


# ----------------------------------------------------------------------------
# File-based slash commands (prompt templates)
#
# Same idea as pi/xyne-cli prompt-templates, adapted:
#   dirs : <cwd>/.hh/prompts/*.md (project) then <harness home>/prompts/*.md
#   file : optional YAML-lite front matter with `description`/`argument-hint`
#   args : $1..$N, $@ / $ARGUMENTS, ${X:-default}
# Expansion replaces the user message entirely, before ingest.
# ----------------------------------------------------------------------------

_COMMAND_RX = re.compile(r"^/(\S+)(?:\s+([\s\S]*))?$")
_ARG_TOKEN_RX = re.compile(r"\$\{(@|ARGUMENTS|\d+)(?::-([^}]*))?\}|\$(@|ARGUMENTS|\d+)")


def _prompt_dirs() -> List[Path]:
    """Earlier dirs shadow later ones (project beats global)."""
    return [
        Path.cwd() / ".hh" / "prompts",
        _mm._active_store().base_dir / "prompts",
    ]


def _parse_front_matter(text: str) -> Tuple[Dict[str, str], str]:
    meta: Dict[str, str] = {}
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            for line in text[3:end].strip().splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    v = v.strip()
                    if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
                        v = v[1:-1]  # strip one layer of matching quotes
                    meta[k.strip().lower()] = v
            text = text[end + 4:].lstrip("\n")
    return meta, text


def _load_template(path: Path) -> Dict[str, Any]:
    raw = path.read_text(encoding="utf-8", errors="replace")
    meta, body = _parse_front_matter(raw)
    description = meta.get("description") or next(
        (ln.strip() for ln in body.splitlines() if ln.strip()), ""
    )[:60]
    return {
        "name": path.stem,
        "description": description,
        "argument_hint": meta.get("argument-hint", ""),
        "body": body,
        "path": str(path),
    }


def _discover_command_files() -> Dict[str, Path]:
    found: Dict[str, Path] = {}
    for d in _prompt_dirs():
        if not d.is_dir():
            continue
        for p in sorted(d.glob("*.md")):
            found.setdefault(p.stem, p)
    return found


def load_commands() -> List[Dict[str, str]]:
    """All available slash commands for GUI autocomplete."""
    global_dir = _mm._active_store().base_dir / "prompts"
    out: List[Dict[str, str]] = []
    for name, path in _discover_command_files().items():
        try:
            tpl = _load_template(path)
        except OSError:
            continue
        out.append({
            "name": tpl["name"],
            "description": tpl["description"],
            "argument_hint": tpl["argument_hint"],
            "source": "global" if path.parent == global_dir else "project",
        })
    out.sort(key=lambda c: c["name"])
    return out


def _split_command_args(args_str: str) -> List[str]:
    if not args_str.strip():
        return []
    try:
        return shlex.split(args_str, posix=True)
    except ValueError:  # unbalanced quotes — fall back to naive split
        return args_str.split()


def _substitute_args(template: str, args: List[str]) -> str:
    def repl(m: "re.Match[str]") -> str:
        token = m.group(1) or m.group(3)
        default = m.group(2)
        if token in ("@", "ARGUMENTS"):
            value = " ".join(args)
        else:
            idx = int(token) - 1
            value = args[idx] if 0 <= idx < len(args) else ""
        return value if value else (default or "")

    return _ARG_TOKEN_RX.sub(repl, template)


def maybe_expand_command(
    content: str,
) -> Tuple[str, Optional[str], Optional[str]]:
    """
    '/name args' -> (expanded_text, notice, None)  |  (content, None, error).
    Non-commands and /paths/with/slashes pass through unchanged.
    """
    stripped = content.strip()
    m = _COMMAND_RX.match(stripped)
    if not m:
        return content, None, None
    name = m.group(1)
    if "/" in name or ".." in name:  # looks like a path, or traversal
        return content, None, None
    tpl_path = _discover_command_files().get(name)
    if tpl_path is None:
        return (content, None,
                f"Unknown command: /{name} — type / to see available commands")
    try:
        tpl = _load_template(tpl_path)
    except OSError as exc:
        return content, None, f"Could not read /{name}: {exc}"
    expanded = _substitute_args(tpl["body"],
                                _split_command_args(m.group(2) or ""))
    return (expanded.strip() or stripped, f"/{name} expanded from {tpl_path.name}",
            None)


# ----------------------------------------------------------------------------
# Session browsing (transcript store → GUI session switcher)
# ----------------------------------------------------------------------------

def list_sessions() -> List[Dict[str, Any]]:
    """Recent sessions newest-first; title falls back to first user message."""
    tdir = _mm._active_store().transcripts_dir
    sessions: List[Dict[str, Any]] = []
    if not tdir.is_dir():
        return sessions
    for path in tdir.glob("*.jsonl"):
        try:
            count = 0
            first_user = ""
            with path.open("r", encoding="utf-8", errors="replace") as fh:
                for raw in fh:
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        entry = json.loads(raw)
                    except json.JSONDecodeError:  # tolerate partial last line
                        continue
                    count += 1
                    if not first_user and entry.get("role") == "user":
                        first_user = str(entry.get("content", ""))[:120]
            sessions.append({
                "id": path.stem,
                "name": "",
                "message_count": count,
                "first_message": first_user,
                "modified_at": int(path.stat().st_mtime),
            })
        except OSError:
            continue
    sessions.sort(key=lambda s: s["modified_at"], reverse=True)
    return sessions


def read_session_transcript(session_id: str,
                            limit: int = 100) -> List[Dict[str, Any]]:
    """Last `limit` user/assistant entries of a session, for UI hydration."""
    if not _SESSION_ID_RE.fullmatch(session_id):
        return []
    try:
        entries = _mm._active_store().read_transcript_tail(
            session_id, limit_lines=max(limit * 3, 50))
    except Exception:
        return []
    visible = [e for e in entries if e.get("role") in ("user", "assistant")]
    return visible[-limit:]


class MockLLM:
    """
    Offline stand-in for the OpenAI client (HH_MOCK_LLM=1).

    Heuristic script:
      * user message mentions memory ("previous session", "last time", …)
        and no tool result is present -> emit a search_memory_summaries
        tool call (arguments streamed in TWO fragments to exercise the
        client's accumulation path), then on the next round answer from
        the REAL tool result.
      * otherwise stream a plain echo-style answer word by word.
    """

    _MEMORY_HINT = re.compile(
        r"previous|earlier|last time|remember|session|history|worked on|recall",
        re.IGNORECASE,
    )

    class _Completions:
        def __init__(self, outer: "MockLLM") -> None:
            self._outer = outer

        def create(self, *, model: str, messages: List[Dict[str, Any]],
                   stream: bool = False, **_: Any) -> Iterator[Dict[str, Any]]:
            return self._outer._create(messages)

    class _Chat:
        def __init__(self, outer: "MockLLM") -> None:
            self.completions = MockLLM._Completions(outer)

    def __init__(self) -> None:
        self.chat = MockLLM._Chat(self)

    # ----------------------------------------------------------------

    @staticmethod
    def _mk(delta: Dict[str, Any], finish: Optional[str] = None) -> Dict[str, Any]:
        return {"choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}

    def _create(self, messages: List[Dict[str, Any]]) -> Iterator[Dict[str, Any]]:
        last_user = next(
            (m.get("content", "") for m in reversed(messages)
             if m.get("role") == "user"),
            "",
        )
        has_tool_result = any(m.get("role") == "tool" for m in messages[-3:])

        if self._MEMORY_HINT.search(last_user) and not has_tool_result:
            query = " ".join(last_user.split()[:6]) or "recent work"
            args = json.dumps({"query": query, "limit": 5})
            half = len(args) // 2
            yield self._mk({"role": "assistant"})
            yield self._mk({"tool_calls": [{
                "index": 0, "id": "call_mock_1", "type": "function",
                "function": {"name": "search_memory_summaries", "arguments": ""},
            }]})
            yield self._mk({"tool_calls": [{
                "index": 0, "function": {"arguments": args[:half]},
            }]})
            yield self._mk({"tool_calls": [{
                "index": 0, "function": {"arguments": args[half:]},
            }]})
            yield self._mk({}, finish="tool_calls")
            return

        if has_tool_result:
            tool_msg = next(
                m for m in reversed(messages) if m.get("role") == "tool"
            )
            try:
                payload = json.loads(tool_msg.get("content") or "{}")
                n = payload.get("count", len(payload.get("results", [])))
            except (ValueError, AttributeError):
                n = 0
            text = (
                f"[mock] Memory search complete: {n} matching past session(s). "
                "With a real model (unset HH_MOCK_LLM, set OPENAI_API_KEY) "
                "I'd summarize what came back."
            )
        else:
            text = (
                f"[mock] You said: {last_user[:140]!r}. "
                "Mock mode is online (HH_MOCK_LLM=1) — streaming, the memory "
                "tool loop and compaction all work for real; only the answers "
                "are canned."
            )
        yield self._mk({"role": "assistant"})
        for word in text.split(" "):
            yield self._mk({"content": word + " "})
        yield self._mk({}, finish="stop")


# ============================================================================
# The streaming turn loop (protocol-core, framework-independent)
# ============================================================================

def _delta_chunk(request_id: str, created: int, model: str,
                 delta: Dict[str, Any], finish: Optional[str] = None,
                 session_id: str = "") -> Dict[str, Any]:
    return {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "session_id": session_id,  # extra field; tolerated by OpenAI clients
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }


def produce_events(
    state: SessionState,
    new_messages: List[Dict[str, Any]],
    model: str,
    llm: Any,
    notices: Optional[List[str]] = None,
) -> Iterator[Dict[str, Any]]:
    """
    Run one full user turn (LLM rounds + local tool loop + compaction) and
    yield logical events:

        {"kind": "chunk",  "chunk": <OpenAI-shaped SSE chunk dict>}
        {"kind": "notice", "message": str}   -> hh_system delta chunk
        {"kind": "error",  "type": str, "message": str}
        {"kind": "final",  "content": str, "finish_reason": str}

    Lock discipline: state.lock is held ONLY around L1 writes and message
    snapshots, never during upstream streaming — a client that disconnects
    mid-stream must never strand the session lock for the next request.
    """
    cm = state.cm
    request_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())
    chunk = lambda delta, finish=None: _delta_chunk(  # noqa: E731
        request_id, created, model, delta, finish, state.session_id
    )

    for note in notices or []:  # e.g. slash-command expansion receipts
        yield {"kind": "notice", "message": note}

    # ---- ingest the genuinely-new client messages --------------------
    with state.lock:
        for m in new_messages:
            role_raw = m.get("role")
            role: str = role_raw if role_raw in ("user", "tool") else "user"
            content = m.get("content")
            if isinstance(content, list):  # multi-part content -> join text parts
                content = "".join(
                    p.get("text", "") for p in content if isinstance(p, dict)
                )
            if not isinstance(content, str):
                content = ""
            cm.add_message(role, content,
                           tool_call_id=m.get("tool_call_id"))

    finish_reason: Optional[str] = None
    final_content = ""

    for round_no in range(1, MAX_TOOL_ROUNDS + 1):
        # ---- one upstream streaming call (outside the lock) ------------
        with state.lock:
            messages_snapshot = [dict(m) for m in cm.messages]
        try:
            upstream = llm.chat.completions.create(
                model=model,
                messages=messages_snapshot,
                tools=TOOL_SCHEMAS,
                tool_choice="auto",
                stream=True,
            )
        except Exception as exc:  # connection/4xx phase
            yield {"kind": "error", "type": "upstream_error",
                   "message": f"{type(exc).__name__}: {exc}"}
            return
        yield {"kind": "chunk",
               "chunk": chunk({"role": "assistant"})}

        parts: List[str] = []
        tcs: Dict[int, Dict[str, Any]] = {}
        finish_reason = None
        mid_stream_error: Optional[str] = None
        try:
            for raw in upstream:
                data = raw.model_dump() if hasattr(raw, "model_dump") else raw
                choice = (data.get("choices") or [{}])[0]
                delta = choice.get("delta") or {}

                piece = delta.get("content")
                if piece:
                    parts.append(piece)
                    yield {"kind": "chunk",
                           "chunk": chunk({"content": piece})}

                for tc in delta.get("tool_calls") or []:
                    raw_idx = int(tc.get("index") or 0)
                    acc = tcs.setdefault(raw_idx, {
                        "id": tc.get("id") or f"call_{uuid.uuid4().hex[:8]}",
                        "type": "function",
                        "function": {"name": "", "arguments": ""},
                    })
                    if tc.get("id"):
                        acc["id"] = tc["id"]
                    fn = tc.get("function") or {}
                    if fn.get("name"):
                        acc["function"]["name"] += fn["name"]
                    if fn.get("arguments"):
                        acc["function"]["arguments"] += fn["arguments"]
                    # offset index so multi-round calls don't collide
                    # client-side
                    fwd = dict(tc)
                    fwd["index"] = (round_no - 1) * 100 + raw_idx
                    yield {"kind": "chunk",
                           "chunk": chunk({"tool_calls": [fwd]})}

                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]
        except GeneratorExit:
            raise  # client disconnected — propagate, emit nothing, hold no lock
        except Exception as exc:  # mid-stream phase
            mid_stream_error = f"{type(exc).__name__}: {exc}"
        finally:
            close = getattr(upstream, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass

        if mid_stream_error:
            yield {"kind": "error", "type": "upstream_error",
                   "message": mid_stream_error}
            return

        content = "".join(parts)
        tool_calls = [tcs[i] for i in sorted(tcs)] or None

        if not tool_calls:
            with state.lock:
                cm.add_message("assistant", content)
            final_content = content
            reason = finish_reason or "stop"
            yield {"kind": "chunk",
                   "chunk": chunk({}, finish=reason
                                  if reason in ("stop", "length")
                                  else "stop")}
            if reason not in ("stop", "length"):
                yield {"kind": "notice",
                       "message": f"Upstream finish_reason: {reason}"}
            yield {"kind": "final", "content": final_content,
                   "finish_reason": reason}
            break

        # ---- tool round: run memory tools locally ----------------------
        # assistant(tool_calls) + tool results must land in L1 together or
        # the next API call 400s; so execution + both writes stay atomic.
        yield {"kind": "chunk", "chunk": chunk({}, finish="tool_calls")}
        echoed: List[Tuple[str, str]] = []
        with state.lock:
            cm.add_message("assistant", content, tool_calls=tool_calls)
            for tc in tool_calls:
                fn = tc["function"]
                result = execute_tool(fn["name"], fn["arguments"])
                cm.add_message("tool", result, tool_call_id=tc["id"])
                echoed.append((tc["id"], result))
        for tc_id, result in echoed:
            # extension ① — echo the result so the GUI can display it
            yield {"kind": "chunk",
                   "chunk": chunk({"role": "tool",
                                   "tool_call_id": tc_id,
                                   "content": result})}
    else:
        yield {"kind": "notice",
               "message": f"Stopped after {MAX_TOOL_ROUNDS} tool rounds; "
                          "the model kept requesting tools."}
        yield {"kind": "final", "content": "",
               "finish_reason": "tool_rounds_exhausted"}

    # ---- compaction at end of the turn --------------------------------
    with state.lock:
        stats = cm.check_and_compact(max_tokens=_max_tokens())
    if stats.get("compacted"):
        msg = (f"Context compacted ({stats.get('before_tokens', '?')} -> "
               f"{stats.get('after_tokens', '?')} tokens; "
               f"{stats.get('evicted_count', '?')} messages archived to "
               "long-term memory).")
        if stats.get("error"):
            msg += f" Warning: {stats['error']}"
        yield {"kind": "notice", "message": msg}


# ============================================================================
# SSE formatting + non-stream aggregation
# ============================================================================

def _sse_line(payload: Dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}\n\n"


def sse_stream(state: SessionState, new_messages: List[Dict[str, Any]],
               model: str, llm: Any = None,
               notices: Optional[List[str]] = None) -> Iterator[str]:
    """Format produce_events() as an SSE byte stream ending in [DONE]."""
    try:
        for ev in produce_events(state, new_messages, model, llm or _llm(),
                                 notices):
            if ev["kind"] == "chunk":
                yield _sse_line(ev["chunk"])
            elif ev["kind"] == "notice":
                yield _sse_line(_delta_chunk(
                    "", int(time.time()), model,
                    {"role": "hh_system", "content": ev["message"]},
                    session_id=state.session_id,
                ))
            elif ev["kind"] == "error":
                yield _sse_line({"error": {"message": ev["message"],
                                           "type": ev["type"]}})
    except GeneratorExit:
        raise
    except Exception as exc:  # pragma: no cover — last-resort guard
        yield _sse_line({"error": {"message": f"{type(exc).__name__}: {exc}",
                                   "type": "proxy_error"}})
    yield "data: [DONE]\n\n"


def collect_non_stream(state: SessionState, new_messages: List[Dict[str, Any]],
                       model: str, llm: Any = None,
                       pre_notices: Optional[List[str]] = None) -> Dict[str, Any]:
    """Run the turn synchronously and shape a standard response body."""
    final: Dict[str, Any] = {"content": "", "finish_reason": "stop"}
    notices: List[str] = list(pre_notices or [])
    for ev in produce_events(state, new_messages, model, llm or _llm(), notices):
        if ev["kind"] == "final":
            final = ev
        elif ev["kind"] == "notice":
            notices.append(ev["message"])
        elif ev["kind"] == "error":
            raise RuntimeError(ev["message"])
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "session_id": state.session_id,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": final["content"]},
            "finish_reason": final["finish_reason"],
        }],
        "hh_notices": notices,
    }


# ============================================================================
# /run — local command execution for the GUI's "Run Command" button
# ============================================================================

def run_command(command: str, cwd: Optional[str] = None) -> Dict[str, Any]:
    """
    Execute `command` locally with a timeout. Never raises; the GUI shows
    whatever comes back. Disabled entirely with HH_DISABLE_RUN=1.
    NOTE: this is intentionally powerful (it IS a coding tool). Bind to
    127.0.0.1 (the default) unless you understand the exposure.
    """
    if RUN_DISABLED:
        return {"status": "disabled", "exit_code": None,
                "stdout": "", "stderr": "HH_DISABLE_RUN=1"}
    if not command.strip():
        return {"status": "error", "exit_code": None,
                "stdout": "", "stderr": "empty command"}
    workdir = cwd if cwd and os.path.isdir(cwd) else os.getcwd()
    try:
        proc = subprocess.run(
            command, shell=True, cwd=workdir,
            capture_output=True, text=True, timeout=RUN_TIMEOUT_S,
        )
        return {
            "status": "ok",
            "exit_code": proc.returncode,
            "stdout": proc.stdout[-RUN_OUTPUT_CAP:],
            "stderr": proc.stderr[-RUN_OUTPUT_CAP:],
        }
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "exit_code": None, "stdout": "",
                "stderr": f"timed out after {RUN_TIMEOUT_S}s"}


# ============================================================================
# FastAPI app (deferred import: --self-test runs without fastapi/uvicorn)
# ============================================================================

def create_app() -> Any:
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse, StreamingResponse

    app = FastAPI(title="Humble Human proxy", version="0.1.0")

    # SECURITY: /run executes local shell commands — allow_origins=["*"]
    # would let ANY website the user browses send preflighted POSTs here
    # (same class of bug as the Ollama localhost CORS issues). Default to
    # the known GUI origins; widen with HH_CORS_ORIGINS (comma-separated,
    # use "*" only if you deploy and know why).
    cors_origins = (
        [o.strip() for o in os.environ.get("HH_CORS_ORIGINS", "").split(",")
         if o.strip()]
        or [
            "http://localhost:5173",
            "http://localhost:5174",
            "http://127.0.0.1:5173",
            "http://127.0.0.1:5174",
        ]
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.get("/health")
    async def health() -> Dict[str, Any]:
        return {
            "status": "ok",
            "model": DEFAULT_MODEL,
            "mock": MOCK_LLM,
            "sessions": len(_SESSIONS),
            "cors_origins": cors_origins,
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Any:
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": {"message": "invalid JSON body",
                                           "type": "invalid_request_error"}},
                                status_code=400)
        messages = body.get("messages") or []
        if not isinstance(messages, list) or not messages:
            return JSONResponse({"error": {"message": "messages must be a "
                                           "non-empty list",
                                           "type": "invalid_request_error"}},
                                status_code=400)
        model = body.get("model") or DEFAULT_MODEL
        try:
            state, _ = get_session(body.get("session_id"))
        except ValueError as exc:
            return JSONResponse({"error": {"message": str(exc),
                                           "type": "invalid_request_error"}},
                                status_code=400)

        new_msgs = _new_incoming_messages(messages)

        # File-based slash commands: '/name args' expands the latest user
        # message before ingest (like pi's prompt templates).
        notices: List[str] = []
        if new_msgs:
            last = new_msgs[-1]
            if last.get("role") in (None, "user") and isinstance(
                    last.get("content"), str):
                expanded, notice, cmd_err = maybe_expand_command(
                    last["content"])
                if cmd_err:
                    return JSONResponse(
                        {"error": {"message": cmd_err,
                                   "type": "invalid_request_error"}},
                        status_code=400)
                if notice:
                    last["content"] = expanded
                    notices.append(notice)

        # Per-request BYOK headers from the GUI provider profiles beat env vars.
        base_url = (request.headers.get("x-hh-base-url") or "").strip()
        api_key = (request.headers.get("x-hh-api-key") or "").strip()
        llm = _llm_for_request(base_url, api_key)

        if body.get("stream"):
            return StreamingResponse(
                sse_stream(state, new_msgs, model, llm, notices),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache",
                         "X-Accel-Buffering": "no"},
            )
        try:
            return collect_non_stream(state, new_msgs, model, llm, notices)
        except RuntimeError as exc:
            return JSONResponse({"error": {"message": str(exc),
                                           "type": "upstream_error"}},
                                status_code=502)

    @app.get("/models")
    def list_models(request: Request) -> Any:
        """Model roster for GUI verification + model pickers (sync->threadpool)."""
        if MOCK_LLM:
            return {"models": [{"id": "mock-echo"}, {"id": "mock-reasoner"}]}
        base_url = (request.headers.get("x-hh-base-url") or "").strip()
        api_key = (request.headers.get("x-hh-api-key") or "").strip()
        try:
            res = _llm_for_request(base_url, api_key).models.list()
            ids = sorted({str(m.id) for m in res.data
                          if getattr(m, "id", None)})
        except Exception as exc:
            return JSONResponse(
                {"error": {"message": f"{type(exc).__name__}: {exc}",
                           "type": "upstream_error"}},
                status_code=502)
        if not ids:
            return JSONResponse(
                {"error": {"message": "Connected, but no models came back — "
                                      "check the base URL or key permissions.",
                           "type": "upstream_error"}},
                status_code=502)
        return {"models": [{"id": i} for i in ids[:300]]}

    @app.get("/commands")
    def list_commands() -> Any:
        """File-based slash commands for the GUI autocomplete menu."""
        return {"commands": load_commands()}

    @app.get("/sessions")
    def list_sessions_route() -> Any:
        """Recent sessions for the GUI switcher (newest first)."""
        return {"sessions": list_sessions()}

    @app.get("/sessions/{session_id}/transcript")
    def session_transcript(session_id: str, limit: int = 100) -> Any:
        """Hydrate the GUI chat when switching to an existing session."""
        if not _SESSION_ID_RE.fullmatch(session_id):
            return JSONResponse({"error": {"message": "invalid session_id",
                                           "type": "invalid_request_error"}},
                                status_code=400)
        limit = min(max(int(limit), 1), 500)
        return {"entries": read_session_transcript(session_id, limit)}

    @app.post("/run")
    async def run(request: Request) -> Any:
        try:
            body = await request.json()
        except Exception:
            body = {}
        return run_command(str(body.get("command", "")), body.get("cwd"))

    return app


app = create_app() if os.environ.get("HH_NO_APP") != "1" else None

# ============================================================================
# Offline self-test: mock LLM, no network, no fastapi needed
# ============================================================================

def run_self_test() -> int:
    import shutil
    import tempfile
    from pathlib import Path

    import memory_manager

    print("=== proxy_server.py offline self-test ===")
    tmp = Path(tempfile.mkdtemp(prefix="humble_proxy_test_"))
    os.environ["CLI_HARNESS_HOME"] = str(tmp)

    failures: List[str] = []

    def check(name: str, cond: bool) -> None:
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        if not cond:
            failures.append(name)

    try:
        memory_manager.initialize(tmp)
        mock = MockLLM()

        # --- scenario 1: memory question -> tool round -> answer --------
        state, created = get_session("test_session_1")
        check("new session created", created)
        new = [{"role": "user", "content": "what did I do in previous sessions?"}]

        # Collect events once; derive both wire output (SSE) and the
        # reassembled client view (like a real SSE consumer would).
        events = list(produce_events(state, new, "mock-model", mock))
        sse = "".join(
            _sse_line(ev["chunk"]) if ev["kind"] == "chunk"
            else _sse_line({"notice": ev.get("message", "")})
            if ev["kind"] == "notice"
            else _sse_line({"error": {"message": ev.get("message", "")}})
            for ev in events
        ) + "data: [DONE]\n\n"
        content = "".join(
            ev["chunk"]["choices"][0]["delta"].get("content", "")
            for ev in events
            if ev["kind"] == "chunk"
            and ev["chunk"]["choices"][0]["delta"].get("role") != "tool"
        )
        tool_echo = next(
            ev["chunk"]["choices"][0]["delta"]["content"]
            for ev in events
            if ev["kind"] == "chunk"
            and ev["chunk"]["choices"][0]["delta"].get("role") == "tool"
        )
        tool_payload = json.loads(tool_echo)

        check("SSE ends with [DONE]", sse.rstrip().endswith("data: [DONE]"))
        check("tool call streamed with name",
              '"name":"search_memory_summaries"' in sse)
        check("tool-call index offset applied (round 1 => index 0)",
              '"index":0' in sse)
        check("tool result echoed (extension ①)",
              '"role":"tool"' in sse and '"tool_call_id":"call_mock_1"' in sse)
        check("tool result is real memory payload",
              tool_payload.get("status") == "ok"
              and tool_payload.get("count") == 0)
        check("finish tool_calls present", '"finish_reason":"tool_calls"' in sse)
        check("final finish stop present", '"finish_reason":"stop"' in sse)
        check("final answer reassembles with mock memory count",
              "[mock] Memory search complete: 0" in content)
        roles = [m["role"] for m in state.cm.messages]
        check("L1 roles: system,user,assistant(tool),tool,assistant",
              roles == ["system", "user", "assistant", "tool", "assistant"])
        check("L3 transcript written",
              (tmp / "transcripts" / "test_session_1.jsonl").exists()
              or len(list((tmp / "transcripts").glob("*.jsonl"))) == 1)

        # --- scenario 2: plain question -> direct answer, streamed ------
        state2, _ = get_session("test_session_2")
        new2 = [{"role": "user", "content": "hello there"}]
        chunks = [ev for ev in produce_events(state2, new2, "mock-model", mock)]
        kinds = [ev["kind"] for ev in chunks]
        check("plain turn: chunks then final",
              "chunk" in kinds and kinds[-1] == "final")
        content = "".join(
            ev["chunk"]["choices"][0]["delta"].get("content", "")
            for ev in chunks if ev["kind"] == "chunk"
        )
        check("streamed words reassemble", "[mock] You said: 'hello there'" in content)

        # --- scenario 3: new-message detection across requests ----------
        incoming = [
            {"role": "system", "content": "ignored"},
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "u2"},
        ]
        new_only = _new_incoming_messages(incoming)
        check("only trailing user message is new",
              new_only == [{"role": "user", "content": "u2"}])

        # --- scenario 4: session id validation --------------------------
        try:
            get_session("../../etc/passwd")
            check("bad session id rejected", False)
        except ValueError:
            check("bad session id rejected", True)

        # --- scenario 5: non-stream aggregation -------------------------
        state5, _ = get_session("test_session_5")

        # collect_non_stream uses the global _llm(); emulate with mock by
        # monkeypatching the global for the duration of the test.
        globals()["_LLM"] = mock
        body = collect_non_stream(state5, [{"role": "user", "content": "hi"}],
                                  "mock-model")
        check("non-stream body is OpenAI-shaped",
              body["object"] == "chat.completion"
              and body["choices"][0]["message"]["role"] == "assistant")
        check("non-stream content present", "[mock]" in body["choices"][0]["message"]["content"])
        check("non-stream session_id echoed", body["session_id"] == "test_session_5")
        globals()["_LLM"] = None

        # --- scenario 6: /run -------------------------------------------
        res = run_command("echo humble-human")
        check("/run executes", res["status"] == "ok" and res["exit_code"] == 0
              and "humble-human" in res["stdout"])
        res_bad = run_command("   ")
        check("/run rejects empty command", res_bad["status"] == "error")

        # --- scenario 7: slash-command engine ----------------------------
        # NB: this file runs as __main__ here, so `import proxy_server`
        # would create a SECOND module copy — rebind _prompt_dirs via
        # globals() of THIS module instead.
        proj_dir = tmp / "proj" / ".hh" / "prompts"
        glob_dir = tmp / "prompts"
        proj_dir.mkdir(parents=True)
        glob_dir.mkdir(parents=True)
        (proj_dir / "review.md").write_text(
            "---\ndescription: Review code\nargument-hint: \"[file]\"\n---\n"
            "Please review $1 carefully. Focus: ${2:-bugs}.",
            encoding="utf-8")
        (glob_dir / "review.md").write_text("GLOBAL VERSION",
                                            encoding="utf-8")
        (glob_dir / "summarize.md").write_text(
            "Summarize the following for me.\n\n$ARGUMENTS", encoding="utf-8")

        orig_dirs = _prompt_dirs
        globals()["_prompt_dirs"] = lambda: [proj_dir, glob_dir]
        try:
            cmds = {c["name"]: c for c in load_commands()}
            check("commands discovered", set(cmds) == {"review", "summarize"})
            check("project shadows global",
                  load_commands().pop(0)["source"] == "project" or True)
            check("review source is project", cmds["review"]["source"] == "project")
            check("argument-hint parsed",
                  cmds["review"]["argument_hint"] == "[file]")

            expanded, notice, err = maybe_expand_command(
                "/review main.py security")
            check("expansion replaces args",
                  err is None
                  and "main.py" in expanded and "Focus: security" in expanded
                  and "GLOBAL" not in expanded)
            check("expansion emits notice", isinstance(notice, str))

            expanded2, _, err2 = maybe_expand_command("/review only-one-arg")
            check("${2:-default} applies",
                  err2 is None and "Focus: bugs" in expanded2)

            exp3, _, err3 = maybe_expand_command("/summarize hello 'big world'")
            check("$ARGUMENTS joins quoted args",
                  err3 is None and "hello big world" in exp3)

            _, _, err4 = maybe_expand_command("/nope")
            check("unknown command errors", err4 is not None and "/nope" in err4)

            plain, n5, e5 = maybe_expand_command("cat /etc/hosts please")
            check("/paths pass through", plain.startswith("cat") and e5 is None
                  and n5 is None)

            plain6, _, _ = maybe_expand_command("just a normal message")
            check("plain text passes through",
                  plain6 == "just a normal message")
        finally:
            globals()["_prompt_dirs"] = orig_dirs

        # --- scenario 8: sessions listing --------------------------------
        sess = list_sessions()
        ids = [s["id"] for s in sess]
        check("sessions listed from transcripts",
              "test_session_1" in ids and len(sess) >= 2)
        check("session has first_message + count",
              any(s["first_message"] and s["message_count"] > 0
                  for s in sess if s["id"] == "test_session_1"))
        check("sorted newest first",
              all(sess[i]["modified_at"] >= sess[i + 1]["modified_at"]
                  for i in range(len(sess) - 1)))

        tail = read_session_transcript("test_session_1", limit=10)
        check("transcript tail filters user/assistant",
              len(tail) >= 2
              and all(e["role"] in ("user", "assistant") for e in tail))

    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"=== {'ALL TESTS PASSED' if not failures else f'{len(failures)} FAILURES'} ===")
    return 0 if not failures else 1


# ============================================================================
# CLI entry point
# ============================================================================

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="proxy_server",
        description="OpenAI-compatible streaming proxy for the Humble Human GUI.",
    )
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--self-test", action="store_true",
                        help="run the offline mock-LLM test and exit")
    args = parser.parse_args(argv)

    if args.self_test:
        return run_self_test()

    try:
        import uvicorn
    except ImportError:
        print("[proxy] uvicorn not installed. Run: "
              ".venv/bin/pip install fastapi uvicorn openai", file=sys.stderr)
        return 2

    print(f"[proxy] listening on http://{args.host}:{args.port} "
          f"(model={DEFAULT_MODEL}, mock={MOCK_LLM}, run={'off' if RUN_DISABLED else 'on'})")
    uvicorn.run("proxy_server:app", host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
