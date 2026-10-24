"""
inspector.py — single-session Memory Inspector for the L1/L2/L3 engine.

Lean diagnostic testbed that wraps `memory_manager.py` + the tool loop
from `main.py` in a monochrome split-panel UI:

    left   = chat window (single session)
    right  = live telemetry for L1 / L2 / L3 + event feed

Launch:

    .venv/bin/pip install streamlit openai
    .venv/bin/streamlit run inspector.py

Config via env (OPENAI_API_KEY, OPENAI_BASE_URL, LLM_MODEL) — anything
typed into the UI fields overrides the env for that session.
"""

from __future__ import annotations

import json
import os

import streamlit as st

import memory_manager as mm
from memory_manager import ContextManager, TOOL_SCHEMAS
from main import SYSTEM_PROMPT, MAX_TOOL_ROUNDS, call_llm, execute_tool

st.set_page_config(page_title="HH Memory Inspector", layout="wide")

st.markdown(
    """
    <style>
    html, body, .stApp, [data-testid="stHeader"] {
        background: #000000; color: #E5E5E5;
        font-family: "JetBrains Mono", ui-monospace, Menlo, Consolas, monospace;
    }
    [data-testid="stSidebar"] { background: #050505; }
    .stApp p, .stApp label, .stApp span, .stApp div { color: #E5E5E5; }
    [data-baseweb="input"], .stTextInput input, [data-baseweb="base-input"] {
        background: #0A0A0A !important; border: 1px solid #333 !important;
        color: #FFFFFF !important;
    }
    .stButton button {
        background: #000000; color: #FFFFFF; border: 1px solid #333333;
        border-radius: 0; font-family: inherit; width: 100%;
    }
    .stButton button:hover { border-color: #FFFFFF; color: #FFFFFF; }
    [data-testid="stExpander"] { border: 1px solid #333333; background: #050505; }
    [data-testid="stProgress"] > div > div { background: #FFFFFF; }
    .stSlider > div > div > div[data-baseweb="slider"] div { background: #333333; }
    hr { border-color: #333333; }
    [data-testid="stChatMessage"] { background: #050505; border: 1px solid #222; }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------
# Session state + persistence across browser reloads
# --------------------------------------------------------------------------

CONFIG_PATH = mm.BASE_DIR / "inspector_config.json"


def _load_config() -> dict:
    try:
        return json.loads(CONFIG_PATH.read_text())
    except (OSError, ValueError):
        return {}


def _save_config() -> None:
    cfg = {k: st.session_state.get(k)
           for k in ("api_key", "base_url", "model", "max_tokens")
           if st.session_state.get(k)}
    try:
        CONFIG_PATH.write_text(json.dumps(cfg))
    except OSError:
        pass


def _restore_latest_session():
    """Pick up the most recent L3 transcript; rebuild L1's text from it."""
    store = mm._active_store()
    files = sorted(store.transcripts_dir.glob("*.jsonl"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        return None, []
    sid = files[0].stem
    msgs = [
        {"role": t["role"], "content": t.get("content", "")}
        for t in store.read_transcript_tail(sid, limit_lines=200)
        if t.get("role") in ("user", "assistant")
        and (t.get("role") == "user" or t.get("content", "").strip())
    ]
    return sid, msgs


def log_event(msg: str) -> None:
    st.session_state.events.append(msg)
    del st.session_state.events[:-200]


def _init_state() -> None:
    if "events" not in st.session_state:
        st.session_state.events = []
    if "cm" in st.session_state:
        return
    cfg = _load_config()
    st.session_state.widget_defaults = cfg
    sid, msgs = _restore_latest_session()
    cm = ContextManager(system_prompt=SYSTEM_PROMPT, **({"session_id": sid} if sid else {}))
    cm.messages.extend(msgs)  # bypass add_message: these turns are already in L3
    st.session_state.cm = cm
    if msgs:
        log_event(f"[EVENT]: restored session `{sid[:8]}…` from L3 ({len(msgs)} messages)")
        budget = cfg.get("max_tokens", 4000)
        if cm.total_tokens() > budget:
            stats = cm.check_and_compact(max_tokens=budget)
            if stats.get("compacted"):
                log_event(f"[EVENT]: restored context over budget ({budget:,}); "
                          f"compacted {stats['evicted_count']} messages to L2")


_init_state()


# --------------------------------------------------------------------------
# LLM turn (mirrors main.HumbleHumanAgent.handle_user_turn, echo -> events)
# --------------------------------------------------------------------------

def _build_client():
    import openai

    kwargs: dict = {}
    api_key = st.session_state.get("api_key") or os.environ.get("OPENAI_API_KEY")
    base_url = st.session_state.get("base_url") or os.environ.get("OPENAI_BASE_URL")
    if api_key:
        kwargs["api_key"] = api_key
    if base_url:
        kwargs["base_url"] = base_url
    _save_config()
    return openai.OpenAI(**kwargs)


def run_compact(max_tokens: int, force: bool = False) -> None:
    """Compact until actually under budget (each pass evicts only ~50%)."""
    cm = st.session_state.cm
    evicted_total, rounds = 0, 0
    before = cm.total_tokens()
    stats: dict = {}
    prev_after: int | None = None
    while True:
        stats = cm.check_and_compact(max_tokens=max_tokens)
        if not stats.get("compacted"):
            break
        rounds += 1
        evicted_total += stats["evicted_count"]
        if prev_after is not None and stats["after_tokens"] >= prev_after:
            break  # irreducible tail: surviving messages alone exceed budget
        prev_after = stats["after_tokens"]
    if rounds:
        log_event(
            f"[EVENT]: Compaction triggered! Evicted {evicted_total} messages "
            f"in {rounds} round(s) ({before - stats['after_tokens']:,} tokens). "
            "Summary saved to L2."
            + (" Still over budget: surviving messages alone exceed the limit."
               if cm.total_tokens() > max_tokens else "")
        )
    elif stats.get("error"):
        log_event(f"[EVENT]: Compaction FAILED: {stats['error']}")
    elif force:
        log_event(f"[EVENT]: Force compaction: no eviction "
                  f"({stats.get('note', 'under budget')})")


def run_turn(client, model: str, max_tokens: int) -> None:
    cm = st.session_state.cm

    for _ in range(MAX_TOOL_ROUNDS):
        reply = call_llm(client, model, cm.messages, TOOL_SCHEMAS)
        if reply["tool_calls"]:
            cm.add_message("assistant", reply["content"] or "",
                           tool_calls=reply["tool_calls"])
            for tc in reply["tool_calls"]:
                fn = tc.get("function", {})
                name, args_json = fn.get("name", ""), fn.get("arguments", "{}")
                result = execute_tool(name, args_json)
                suffix = ""
                try:
                    payload = json.loads(result)
                    if name == "search_memory_summaries":
                        suffix = f" -> {payload.get('count', 0)} hits"
                    elif name == "inspect_session_transcript":
                        suffix = f" -> {payload.get('returned', 0)} turns"
                except (ValueError, AttributeError):
                    pass
                log_event(f"[TOOL CALL]: LLM called {name}({args_json}){suffix}")
                cm.add_message("tool", result, tool_call_id=tc.get("id", ""))
            continue  # auto-reprompt with tool results in L1
        cm.add_message("assistant", reply["content"] or "")
        break
    else:
        log_event(f"[EVENT]: max tool rounds ({MAX_TOOL_ROUNDS}) reached; "
                  "model kept requesting tools")

    run_compact(max_tokens)


# --------------------------------------------------------------------------
# Telemetry helpers
# --------------------------------------------------------------------------

def fmt_size(n: int) -> str:
    return f"{n / 1024:.1f} KB" if n >= 1024 else f"{n} B"


def l2_rows():
    store = mm._active_store()
    with store._lock, store._conn:
        cur = store._conn.execute(
            "SELECT session_id, timestamp, goal, summary "
            "FROM l2_summaries ORDER BY rowid DESC"
        )
        return cur.fetchall()


# --------------------------------------------------------------------------
# Layout
# --------------------------------------------------------------------------

left, right = st.columns(2, gap="medium")

# ============================== LEFT: chat ================================
with left:
    cm = st.session_state.cm
    st.markdown("### CHAT")
    st.caption(f"session `{cm.session_id}`")

    # Render conversation (skip pinned system prompt; show pointers)
    for m in cm.messages[1:]:
        role = m.get("role")
        if role == "system":  # compaction pointer
            st.markdown(
                f"<div style='color:#888;font-size:0.8em;"
                f"border-left:2px solid #333;padding-left:8px'>"
                f"{m['content']}</div>",
                unsafe_allow_html=True,
            )
        elif role in ("user", "assistant"):
            with st.chat_message(role):
                st.markdown(m.get("content") or "")
        # role == "tool": chat stays clean; see event feed on the right

    prompt = st.chat_input("Message Humble Human...")

# ============================ RIGHT: telemetry ============================
with right:
    st.markdown("### MEMORY TELEMETRY")

    # --- config ----------------------------------------------------------
    st.text_input("API key (blank = $OPENAI_API_KEY)", type="password",
                  key="api_key",
                  value=st.session_state.widget_defaults.get("api_key", ""))
    st.text_input("Base URL (blank = $OPENAI_BASE_URL)", key="base_url",
                  value=st.session_state.widget_defaults.get("base_url", ""))
    st.text_input("Model", key="model",
                  value=st.session_state.widget_defaults.get(
                      "model", os.environ.get("LLM_MODEL", "deepseek-chat")))
    max_tokens = st.slider(
        "Max L1 context tokens", 1000, 16000,
        st.session_state.widget_defaults.get("max_tokens", 4000), 250,
        key="max_tokens")

    b1, b2 = st.columns(2)
    force = b1.button("[ Force Compaction ]")
    purge = b2.button("[ Clear Session & Purge DB ]")
    st.divider()

    if prompt is None:
        if force:
            run_compact(max_tokens=1, force=True)
            st.rerun()
        if purge:
            store = mm._active_store()
            with store._lock, store._conn:
                store._conn.execute("DELETE FROM l2_summaries")
            for p in store.transcripts_dir.glob("*.jsonl"):
                p.unlink(missing_ok=True)
            st.session_state.cm = ContextManager(system_prompt=SYSTEM_PROMPT)
            st.session_state.events = []
            log_event("[EVENT]: Session cleared; L2 rows + L3 transcripts purged. "
                      "New session started.")
            st.rerun()

    # --- L1 --------------------------------------------------------------
    st.markdown("**L1 — ACTIVE CONTEXT**")
    cur, total = cm.total_tokens(), max_tokens
    pct = min(1.0, cur / total)
    st.progress(pct, text=f"{cur:,} / {total:,} tokens ({pct * 100:.1f}%)")
    st.caption(f"{len(cm.messages)} active messages")
    with st.expander("L1 message weights"):
        rows = [
            {
                "idx": i,
                "role": m.get("role"),
                "tokens": ContextManager.estimate_tokens(m.get("content") or "")
                + sum(
                    ContextManager.estimate_tokens(json.dumps(tc.get("function", {})))
                    for tc in m.get("tool_calls") or []
                ),
                "preview": (m.get("content") or "")[:60].replace("\n", " "),
            }
            for i, m in enumerate(cm.messages)
        ]
        st.dataframe(rows, hide_index=True, width="stretch")

    # --- L2 --------------------------------------------------------------
    st.markdown("**L2 — WORKING SUMMARIES (SQLite FTS5)**")
    rows = l2_rows()
    st.caption(f"{len(rows)} summary records in memory.db")
    with st.expander("L2 summaries"):
        if rows:
            st.dataframe(
                [{"session_id": r[0][:8], "timestamp": r[1], "goal": r[2],
                  "summary": r[3]} for r in rows],
                hide_index=True, width="stretch",
            )
        else:
            st.caption("(empty)")

    # --- L3 --------------------------------------------------------------
    st.markdown("**L3 — DEEP LOGS (JSONL)**")
    store = mm._active_store()
    tpath = store.transcript_path(cm.session_id)
    tsize = tpath.stat().st_size if tpath.exists() else 0
    tlines = sum(1 for _ in open(tpath, "rb")) if tpath.exists() else 0
    st.caption(f"transcripts/{cm.session_id}.jsonl — {fmt_size(tsize)}")
    st.caption(f"{tlines} raw turns recorded (this session)")
    with st.expander("L3 tail (last 5 turns)"):
        tail = store.read_transcript_tail(cm.session_id, limit_lines=5)
        if tail:
            st.code("\n".join(json.dumps(t, ensure_ascii=False) for t in tail))
        else:
            st.caption("(no transcript yet)")

    # --- events ----------------------------------------------------------
    st.markdown("**EVENT / TOOL-CALL LOG**")
    st.code("\n".join(st.session_state.events) or "(no events yet)")

# ------------------------------ turn runner -------------------------------
if prompt:
    cm.add_message("user", prompt)  # echo first, so the next rerun shows it
    st.session_state.pending_turn = True
    st.rerun()

if st.session_state.pop("pending_turn", False):
    with left, st.spinner("Thinking..."):
        try:
            run_turn(_build_client(), st.session_state.model, max_tokens)
        except Exception as exc:  # surface API errors in the feed, not a crash
            log_event(f"[ERROR]: {type(exc).__name__}: {exc}")
    st.rerun()
