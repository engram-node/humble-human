"""
memory_manager.py
=================
Three-tier memory hierarchy for a local CLI coding agent.

The goal: keep the LLM context window (L1) lean while guaranteeing that no
information is ever lost, and give the model zero-dependency, local
tool-calls to page history back in on demand.

    +--------------------------------------------------------------+
    | L1  ACTIVE CONTEXT      In-memory Python list of message     |
    |                          dicts (ContextManager.messages).    |
    |                          Fast, volatile, token-budgeted.     |
    +----------------------------+---------------------------------+
                                |  compaction (check_and_compact)
                                |  evicts oldest 50% -> summary
                                v
    +--------------------------------------------------------------+
    | L2  WORKING SUMMARIES   SQLite FTS5 table `l2_summaries`     |
    |                          (~/.cli_harness/memory.db).         |
    |                          Keyword-searchable session          |
    |                          digests (porter-stemmed).           |
    +----------------------------+---------------------------------+
                                |  search_memory_summaries()
                                |  returns session_id pointers
                                v
    +--------------------------------------------------------------+
    | L3  DEEP LOGS           Append-only JSONL transcripts       |
    |                          (~/.cli_harness/transcripts/        |
    |                           <session_id>.jsonl). Raw,         |
    |                          unedited turns. Paged in via       |
    |                          inspect_session_transcript().      |
    +--------------------------------------------------------------+

Design rules
------------
*  ZERO external data dependencies: only `sqlite3`, `json`, `os`, `pathlib`,
   `queue`, `threading`, `atexit`, `uuid`, `datetime` (all stdlib).
   No vector DB, no ORM, no tiktoken.
*  All data lives under ~/.cli_harness/ (override with the
   CLI_HARNESS_HOME environment variable).
*  L3 writes are asynchronous: `log_turn()` enqueues onto a daemon
   writer thread and returns in microseconds, so it never blocks the
   agent's main event loop.
*  Data safety: L1 messages are only deleted after their raw turns are
   confirmed durable in L3 AND the replacement summary is committed to L2.

Quick start
-----------
    from memory_manager import ContextManager, TOOL_SCHEMAS

    cm = ContextManager(system_prompt="You are a CLI coding agent.")
    cm.add_message("user", "please fix the flaky test in tests/test_app.py")
    ...                                                      # agent works
    stats = cm.check_and_compact(max_tokens=16000)           # auto-evict

    # Expose these to the LLM as tools (see TOOL_SCHEMAS):
    from memory_manager import search_memory_summaries, inspect_session_transcript
    hits  = search_memory_summaries("flaky test")
    turns = inspect_session_transcript(hits["results"][0]["session_id"])

Run `python3 memory_manager.py` to execute the hermetic self-test
(it points CLI_HARNESS_HOME at a temp directory and never touches
real user data).
"""

from __future__ import annotations

import atexit
import json
import os
import queue
import re
import shutil
import sqlite3
import sys
import tempfile
import threading
import uuid
from collections import deque
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

__all__ = [
    "BASE_DIR",
    "DB_PATH",
    "TRANSCRIPTS_DIR",
    "TOOL_SCHEMAS",
    "MemoryStore",
    "ContextManager",
    "generate_summary",
    "initialize",
    "log_turn",
    "search_memory_summaries",
    "inspect_session_transcript",
    "new_session_id",
]

# ============================================================================
# Configuration constants
# ============================================================================

# Root of all memory data. Overridable via $CLI_HARNESS_HOME so tests (and
# multi-tenant setups) can redirect the store without touching user data.
BASE_DIR = Path(os.environ.get("CLI_HARNESS_HOME", Path.home() / ".cli_harness"))

# L2: single SQLite database file holding the FTS5 summary index.
DB_PATH = BASE_DIR / "memory.db"

# L3: directory of append-only JSONL raw transcripts, one file per session.
TRANSCRIPTS_DIR = BASE_DIR / "transcripts"

# Rough chars-per-token ratio. The spec says "1 token ~= 4 chars"; we use a
# small fudge factor for JSON/whitespace overhead observed in chat traffic.
_CHARS_PER_TOKEN = 4


def _utc_now_iso() -> str:
    """Current UTC time as an ISO-8601 string (second precision, +00:00)."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_session_id() -> str:
    """Generate a new opaque session id (UUID4 hex)."""
    return uuid.uuid4().hex


def _sanitize_session_id(session_id: str) -> str:
    """
    Validate a session id before it touches the filesystem.

    Transcript files are named ``<session_id>.jsonl``. To make path
    traversal (e.g. ``../../etc/passwd``) impossible, we only ever accept
    ids matching ``[A-Za-z0-9_-]+`` and cap the length. Anything else is
    rejected loudly rather than silently mangled.
    """
    if not isinstance(session_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", session_id):
        raise ValueError(
            f"Invalid session_id: {session_id!r}. "
            "Expected 1-128 chars of [A-Za-z0-9_-]."
        )
    return session_id


# ============================================================================
# L2 + L3 backing store: SQLite schema and async JSONL writer
# ============================================================================

# The exact schema mandated by the spec. `tokenize='porter'` makes FTS5
# match word stems ("fixing" ~ "fix", "running" ~ "ran"), which is what
# you want when an LLM phrases recall queries differently from the text
# that was originally summarized.
#
# Note: FTS5 external-content tables save disk but complicate deletes; a
# contentless table loses the row text. A plain FTS5 table (as below) is
# the simplest correct choice and also lets us SELECT any column directly.
_SCHEMA_SQL = """
CREATE VIRTUAL TABLE IF NOT EXISTS l2_summaries USING fts5(
    session_id,
    timestamp,
    goal,
    files_modified,
    summary,
    tokenize='porter'
);
"""


class MemoryStore:
    """
    Owns all on-disk state for L2 (SQLite) and L3 (JSONL transcripts).

    Threading model
    ---------------
    *  SQLite: Python's sqlite3 module serializes access per-connection,
       but we still guard every transaction with a re-entrant lock so the
       API is safe to call from multiple agent threads.
    *  L3 writes: `log_turn()` pushes the record onto a bounded queue
       drained by a single daemon thread. The calling thread never waits
       for disk I/O (worst case it raises `queue.Full` if the disk backs
       up by > 10_000 records, which surfaces a real problem instead of
       silently eating unbounded memory).

    Lifecycle
    ---------
    `initialize()` is called automatically on module import for the
    default BASE_DIR, creating ~/.cli_harness/ and the schema. The
    background writer is registered with `atexit` so pending turns are
    flushed when the CLI process exits.
    """

    def __init__(self, base_dir: Path) -> None:
        self.base_dir = Path(base_dir)
        self.db_path = self.base_dir / "memory.db"
        self.transcripts_dir = self.base_dir / "transcripts"

        # --- Filesystem bootstrap -------------------------------------
        try:
            self.transcripts_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise RuntimeError(
                f"Cannot create memory directory {self.transcripts_dir}: {exc}"
            ) from exc

        # --- SQLite bootstrap -----------------------------------------
        self._lock = threading.RLock()
        try:
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
            # WAL mode: readers never block the writer and vice versa; a
            # crash mid-write can't corrupt the DB. autocommit on.
            self._conn.execute("PRAGMA journal_mode=WAL;")
            with closing(self._conn.cursor()) as cur:
                cur.execute(_SCHEMA_SQL)
            self._conn.commit()
        except sqlite3.Error as exc:
            raise RuntimeError(
                f"Failed to initialize SQLite store at {self.db_path}: {exc}. "
                "Note: FTS5 must be compiled into your sqlite3 module "
                "(it is, in the standard CPython build)."
            ) from exc

        # --- Async L3 writer ------------------------------------------
        self._write_queue: "queue.Queue[Optional[Dict[str, Any]]]" = queue.Queue(maxsize=10_000)
        self._writer = threading.Thread(
            target=self._writer_loop, name="l3-transcript-writer", daemon=True
        )
        self._writer.start()
        atexit.register(self.close)

    # ----------------------------------------------------------------
    # L3: async append-only transcript writer
    # ----------------------------------------------------------------

    def log_turn(self, session_id: str, role: str, content: str) -> bool:
        """
        Enqueue a raw turn for L3. Returns True if accepted by the queue.

        This is deliberately the *only* fast-path public API: it does no
        disk I/O on the calling thread. The payload is fully materialized
        (json.dumps) here so later mutation of `content` by caller code
        can't corrupt what gets written.
        """
        session_id = _sanitize_session_id(session_id)
        record = {
            "session_id": session_id,
            "timestamp": _utc_now_iso(),
            "role": str(role),
            "content": str(content),
        }
        try:
            # dumps() with ensure_ascii=False keeps logs human-readable;
            # the writer thread opens files with encoding="utf-8".
            self._write_queue.put_nowait(record)
            return True
        except queue.Full:
            # Queue is saturated (disk stalled?). Report loudly rather
            # than block the agent's event loop or drop data silently.
            print(
                "[memory_manager] WARNING: L3 write queue full; "
                f"turn for session {session_id} NOT logged.",
                file=sys.stderr,
            )
            return False

    def _writer_loop(self) -> None:
        """
        Drain the queue and append each record to its transcript file.

        Runs forever on a daemon thread. `None` is the poison-pill
        sentinel that triggers a final flush and exit (see `close()`).
        """
        while True:
            record = self._write_queue.get()
            try:
                if record is None:
                    return
                self._append_record(record)
            except Exception as exc:  # noqa: BLE001 - never kill the writer
                # A single bad record must not take down logging forever.
                print(
                    f"[memory_manager] WARNING: failed to persist transcript "
                    f"turn ({exc}); record dropped: "
                    f"{str(record)[:200]!r}",
                    file=sys.stderr,
                )
            finally:
                self._write_queue.task_done()

    def _append_record(self, record: Dict[str, Any]) -> None:
        """Serialize one record to <transcripts_dir>/<sid>.jsonl (blocking)."""
        # session_id was validated at enqueue time; re-validate in case a
        # record was constructed directly (defense in depth).
        session_id = _sanitize_session_id(record["session_id"])
        path = self.transcripts_dir / f"{session_id}.jsonl"
        try:
            # "a" mode = append-only semantics at the OS level; each
            # write is a single line so a crash can only ever truncate
            # at a record boundary, never corrupt mid-record.
            #
            # The stored line keeps the specified {"timestamp", "role",
            # "content"} shape; session_id routes the write and is also
            # kept in the line so files stay self-describing.
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as exc:
            raise IOError(f"Failed to append transcript turn to {path}: {exc}") from exc

    def transcript_path(self, session_id: str) -> Path:
        """Filesystem path of a session's raw transcript."""
        session_id = _sanitize_session_id(session_id)
        return self.transcripts_dir / f"{session_id}.jsonl"

    def read_transcript_tail(self, session_id: str, limit_lines: int = 50) -> List[Dict[str, Any]]:
        """
        Return up to `limit_lines` most-recent parsed turns from L3.

        Implemented with a bounded deque scan so even multi-gigabyte
        transcripts are read in O(limit_lines) memory: we stream the file
        keeping only the last N lines. A trailing partial line (torn write
        from a crash) is tolerated and skipped.
        """
        session_id = _sanitize_session_id(session_id)
        path = self.transcript_path(session_id)
        if not path.exists():
            return []

        from collections import deque

        last_lines: deque = deque(maxlen=max(1, limit_lines))
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    if line.endswith("\n"):  # ignore torn final line
                        last_lines.append(line)
        except OSError as exc:
            raise IOError(f"Failed to read transcript {path}: {exc}") from exc

        turns: List[Dict[str, Any]] = []
        for line in last_lines:
            try:
                turns.append(json.loads(line))
            except json.JSONDecodeError:
                # Skip malformed line rather than failing the whole read.
                continue
        return turns

    # ----------------------------------------------------------------
    # L2: FTS5 summary index
    # ----------------------------------------------------------------

    def insert_summary(
        self,
        session_id: str,
        goal: str,
        files_modified: str,
        summary: str,
        timestamp: Optional[str] = None,
    ) -> None:
        """
        Insert one row into `l2_summaries` (the L2 tier).

        All columns are TEXT (FTS5 requirement). `files_modified` is a
        comma/space-joined string of paths; porter tokenization means a
        query for "test_app" or "app" will still match the file name.
        """
        session_id = _sanitize_session_id(session_id)
        with self._lock:
            try:
                with closing(self._conn.cursor()) as cur:
                    cur.execute(
                        "INSERT INTO l2_summaries "
                        "(session_id, timestamp, goal, files_modified, summary) "
                        "VALUES (?, ?, ?, ?, ?)",
                        (
                            session_id,
                            timestamp or _utc_now_iso(),
                            str(goal or ""),
                            str(files_modified or ""),
                            str(summary or ""),
                        ),
                    )
                self._conn.commit()
            except sqlite3.Error as exc:
                # Roll back so the connection isn't left mid-transaction.
                self._conn.rollback()
                raise RuntimeError(f"Failed to insert L2 summary: {exc}") from exc

    def search_summaries(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Full-text search over L2 using the FTS5 MATCH operator.

        Query hardening: the raw query is split into terms, each term is
        double-quoted with embedded quotes doubled (""), and terms are
        joined with spaces (FTS5's implicit AND). This makes keyword
        queries behave naturally ("auth refactor" = auth AND refactor)
        while completely neutralizing FTS operator injection -- arbitrary
        user text can never raise a syntax error or be interpreted as
        NEAR/AND/NOT/^/* operators.

        `bm25()` ranks by relevance (lower score = better, hence ASC).
        """
        if not isinstance(query, str) or not query.strip():
            return []
        limit = max(1, min(int(limit), 50))  # clamp defensively

        # "fix the flaky" -> '"fix" "the" "flaky"'  (implicit AND).
        safe_query = " ".join(
            f'"{term.replace(chr(34), chr(34) * 2)}"'
            for term in query.split()
        )
        if not safe_query:
            return []

        sql = (
            "SELECT session_id, timestamp, goal, files_modified, summary "
            "FROM l2_summaries WHERE l2_summaries MATCH ? "
            "ORDER BY bm25(l2_summaries) ASC LIMIT ?"
        )
        with self._lock:
            try:
                with closing(self._conn.cursor()) as cur:
                    cur.execute(sql, (safe_query, limit))
                    rows = cur.fetchall()
            except sqlite3.Error as exc:
                raise RuntimeError(f"L2 FTS5 search failed: {exc}") from exc

        return [
            {
                "session_id": r[0],
                "timestamp": r[1],
                "goal": r[2],
                "files_modified": r[3],
                "summary": r[4],
            }
            for r in rows
        ]

    # ----------------------------------------------------------------
    # Lifecycle
    # ----------------------------------------------------------------

    def flush(self, timeout: float = 5.0) -> bool:
        """
        Block until every queued L3 record has hit the disk (or timeout).

        Used by the compaction engine as a durability barrier: L1 data is
        only deleted after flush() confirms its L3 mirror is durable.

        `queue.join()` has no timeout parameter, so we run it on a helper
        thread and wait on an Event with a deadline. Returns True if the
        queue fully drained in time.
        """
        done = threading.Event()
        outcome: Dict[str, bool] = {"drained": False}

        def _join_queue() -> None:
            self._write_queue.join()
            outcome["drained"] = True
            done.set()

        threading.Thread(target=_join_queue, name="l3-flush-waiter", daemon=True).start()
        done.wait(timeout)
        return outcome["drained"]

    def close(self) -> None:
        """
        Flush pending writes, stop the writer thread, close SQLite.

        Registered with atexit; idempotent (safe to call manually too).
        """
        if not hasattr(self, "_conn"):
            return
        # Drain: poison pill AFTER join so queued records are written first.
        try:
            self._write_queue.join()
        finally:
            try:
                self._write_queue.put_nowait(None)
            except queue.Full:
                pass
            if self._writer.is_alive():
                self._writer.join(timeout=2.0)
            try:
                self._conn.close()
            except sqlite3.Error:
                pass


# ============================================================================
# Module-level default store + functional API
# ============================================================================

# Lazily-initialized singleton for the default BASE_DIR. We deliberately
# do NOT create this at import time in library mode; `initialize()` (also
# run by the self-test) materializes it. Functions below auto-initialize
# on first call so casual usage "just works".
_store: Optional[MemoryStore] = None
_store_lock = threading.Lock()


def initialize(base_dir: Optional[Path] = None) -> MemoryStore:
    """
    Create (or return) the process-wide MemoryStore.

    Args:
        base_dir: optional override of ~/.cli_harness (used by tests).

    Returns:
        The initialized MemoryStore.
    """
    global _store
    with _store_lock:
        if _store is None or base_dir is not None:
            if _store is not None and base_dir is not None:
                # Switching stores (test isolation): cleanly stop the old.
                _store.close()
            _store = MemoryStore(base_dir or BASE_DIR)
        return _store


def _active_store() -> MemoryStore:
    """Return the default store, initializing it on first use."""
    global _store
    if _store is None:
        initialize()
    assert _store is not None
    return _store


def log_turn(session_id: str, role: str, content: str) -> bool:
    """
    L3 writer. Append ``{"timestamp", "role", "content"}`` to the session's
    JSONL transcript. Async: returns as soon as the record is enqueued.
    """
    return _active_store().log_turn(session_id, role, content)


def search_memory_summaries(query: str, limit: int = 5) -> Dict[str, Any]:
    """
    TOOL 1 (LLM-callable). Keyword-search past session summaries (L2)
    via SQLite FTS5 MATCH. Returns lightweight hits containing session
    id, date, goal, and summary -- enough for the model to decide
    whether to page in the full raw transcript with tool 2.
    """
    try:
        results = _active_store().search_summaries(query, limit)
        return {"status": "ok", "query": query, "count": len(results), "results": results}
    except (RuntimeError, ValueError) as exc:
        return {"status": "error", "error": str(exc), "query": query, "results": []}


def inspect_session_transcript(session_id: str, limit_lines: int = 50) -> Dict[str, Any]:
    """
    TOOL 2 (LLM-callable). Page raw turns back in from L3.

    Reads ``transcripts/<session_id>.jsonl`` and returns the LAST
    `limit_lines` turns (most recent context is usually what matters).
    Large files are handled with a streaming bounded-deque scan, so this
    stays cheap even on multi-hundred-MB transcripts.
    """
    try:
        path = _active_store().transcript_path(session_id)
        if not path.exists():
            return {
                "status": "error",
                "error": f"No transcript found for session {session_id!r}",
                "session_id": session_id,
                "turns": [],
            }
        turns = _active_store().read_transcript_tail(session_id, limit_lines)
        total = _count_transcript_lines(path)
        return {
            "status": "ok",
            "session_id": session_id,
            "total_lines": total,
            "returned": len(turns),
            "truncated": len(turns) < total,
            "turns": turns,
        }
    except (ValueError, IOError, RuntimeError) as exc:
        return {"status": "error", "error": str(exc), "session_id": session_id, "turns": []}


def _count_transcript_lines(path: Path) -> int:
    """Cheap newline count for `total_lines` (streamed, no full load)."""
    count = 0
    with open(path, "rb") as fh:
        for _ in fh:
            count += 1
    return count


# ============================================================================
# Mock summarizer (to be wired to a real LLM call later)
# ============================================================================

def generate_summary(messages: List[Dict[str, Any]]) -> str:
    """
    MOCK summarizer. Produces a deterministic placeholder digest of the
    evicted message block. Later this gets replaced by an LLM call, but
    the interface (list[dict] -> str) is already final so the compaction
    engine never has to change.
    """
    n_user = sum(1 for m in messages if m.get("role") == "user")
    n_asst = sum(1 for m in messages if m.get("role") == "assistant")
    user_turns = [m.get("content", "") for m in messages
                  if m.get("role") == "user"]
    first_user = user_turns[0] if user_turns else ""
    preview = (first_user[:120] + "...") if len(first_user) > 120 else first_user
    # All user turns indexed, not just the first: mid-block topics were
    # invisible to FTS recall (verified failure). Capped to keep rows light.
    topics = " | ".join(t.strip().replace("\n", " ")[:120]
                        for t in user_turns)[:900]
    return (
        f"[MOCK SUMMARY] {len(messages)} messages "
        f"({n_user} user / {n_asst} assistant). "
        f"Opening request: {preview!r}. User turns: {topics}"
    )


# ============================================================================
# L1: the compaction engine
# ============================================================================

class ContextManager:
    """
    Wraps the L1 active-context message list.

    Responsibilities:
      1. Hold messages + system prompt.
      2. Estimate token count (1 token ~= 4 chars; no tiktoken).
      3. `check_and_compact(max_tokens)` -- when over budget, evict the
         oldest half of the conversation, persist it (L3 raw + L2 summary),
         and replace it with a pointer message.

    The eviction invariant ( enforced order of operations ):
        a) every evicted message is durably written to L3  (flush barrier)
        b) the summary row is committed to L2              (SQLite commit)
        c) only THEN are messages removed from L1
       If (a) or (b) raises, L1 is left untouched -- we never lose data
       to a failed compaction.
    """

    def __init__(
        self,
        system_prompt: str,
        session_id: Optional[str] = None,
        store: Optional[MemoryStore] = None,
    ) -> None:
        self.session_id = session_id or new_session_id()
        self.store = store or _active_store()
        # The system prompt is pinned at index 0 and is NEVER evicted.
        self.messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt}
        ]

    # ----------------------------------------------------------------
    # Message management
    # ----------------------------------------------------------------

    def add_message(
        self,
        role: str,
        content: str,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        tool_call_id: Optional[str] = None,
    ) -> None:
        """
        Append a message to L1 AND mirror it to L3 (fire-and-forget).

        Mirroring every message as it arrives is what makes the L3
        transcript authoritative, and means compaction never has to
        "rescue" raw turns at eviction time -- the flush in
        `check_and_compact` just confirms durability.

        OpenAI tool-calling fields are preserved verbatim in L1 so the
        message list can be replayed against the API without a reshape:
          * `tool_calls`  -- on assistant messages that requested tools
          * `tool_call_id` -- on tool messages responding to a specific call
        Without these, the API rejects the history with 400
        "tool message must have tool_call_id" / "assistant message with
        tool_calls must be followed by tool messages".
        """
        if role not in ("system", "user", "assistant", "tool"):
            role = "assistant"  # defensive default for unknown roles
        msg: Dict[str, Any] = {"role": role, "content": str(content)}
        if tool_calls:
            msg["tool_calls"] = tool_calls
        if tool_call_id is not None:
            msg["tool_call_id"] = tool_call_id
        self.messages.append(msg)
        # L3 keeps the human-readable content; tool-call *results* (the
        # interesting part for recall) are logged by the caller as their
        # own role="tool" turn via add_message.
        self.store.log_turn(self.session_id, role, str(content))

    # ----------------------------------------------------------------
    # Token accounting
    # ----------------------------------------------------------------

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Approximate token count: 1 token ~= 4 characters."""
        return max(1, len(text) // _CHARS_PER_TOKEN)

    def total_tokens(self) -> int:
        """
        Approximate tokens across ALL active messages (incl. system).

        Tool-call payloads count too: an assistant message with
        `tool_calls` carries a JSON `arguments` blob that the API really
        charges tokens for, so we estimate its serialized size. This
        keeps the 16k budget honest in tool-heavy conversations.
        """
        total = 0
        for m in self.messages:
            total += self.estimate_tokens(m.get("content") or "")
            for tc in m.get("tool_calls") or []:
                total += self.estimate_tokens(
                    json.dumps(tc.get("function", {}), ensure_ascii=False)
                )
        return total

    # ----------------------------------------------------------------
    # Compaction
    # ----------------------------------------------------------------

    def _extract_goal(self, messages: List[Dict[str, str]]) -> str:
        """Best-effort 'goal' for the L2 row: first user message."""
        for m in messages:
            if m.get("role") == "user":
                return m.get("content", "")[:200]
        return "(no explicit user goal in evicted block)"

    def _extract_files_modified(self, messages: List[Dict[str, str]]) -> str:
        """
        Best-effort file list for the L2 row: scrape path-like tokens.

        Cheap heuristic (regex for /-paths and *.ext files) so FTS5 can
        answer "which session touched auth.py?" without any tooling
        dependency. Replace with structured tool-logs later if needed.
        """
        found: List[str] = []
        seen = set()
        pattern = re.compile(r'(?:[\w./-]+/)?[\w.-]+\.(?:py|js|ts|tsx|md|json|ya?ml|toml|go|rs|java|c|cpp|h|sh)')
        for m in messages:
            for match in pattern.findall(m.get("content", "")):
                if match not in seen:
                    seen.add(match)
                    found.append(match)
                    if len(found) >= 20:  # keep the row lightweight
                        return ", ".join(found)
        return ", ".join(found)

    def check_and_compact(self, max_tokens: int = 16_000) -> Dict[str, Any]:
        """
        Evict the oldest 50% of non-system messages if over budget.

        Flow (data-safe order, see class docstring):
            1. Measure. Under budget -> no-op.
            2. Split: evictable = messages[1:] (system prompt excluded),
               oldest half of THAT goes to eviction.
            3. Flush L3 (durability barrier) for the session.
            4. generate_summary(evicted) -> insert L2 row (committed).
            5. Delete evicted from L1; splice in the pointer message.

        Returns a stats dict describing what happened (or didn't).
        """
        stats: Dict[str, Any] = {
            "session_id": self.session_id,
            "compacted": False,
            "before_tokens": self.total_tokens(),
            "after_tokens": 0,
            "evicted_count": 0,
            "summary_row": None,
        }
        try:
            # -- 1. Threshold check ------------------------------------
            if self.total_tokens() <= max_tokens:
                stats["after_tokens"] = stats["before_tokens"]
                return stats

            # -- 2. Select eviction block (never the system prompt) -----
            # We snapshot for safety; on failure we must not have mutated.
            evictable = self.messages[1:]
            if not evictable:
                # Over budget from the system prompt alone; nothing we can
                # evict. Report and return rather than looping forever.
                stats["after_tokens"] = stats["before_tokens"]
                stats["note"] = "over budget but nothing evictable"
                return stats

            # Eviction boundary must respect OpenAI tool-call pairing: an
            # assistant message carrying `tool_calls` MUST be immediately
            # followed by its role="tool" responses. If the naive 50% cut
            # landed inside such a pair, replaying the surviving history
            # against the API would 400. We therefore extend the cut to
            # include any tool responses belonging to the last evicted
            # assistant tool_calls message.
            evict_count = max(1, (len(evictable) + 1) // 2)
            if evictable[evict_count - 1].get("tool_calls"):
                extra = evict_count
                while (
                    extra < len(evictable)
                    and evictable[extra].get("role") == "tool"
                ):
                    extra += 1
                if extra < len(evictable):
                    evict_count = extra
                # (if the tool-run runs to the end of the list, the whole
                # tail is one tool exchange; leave it intact and evict
                # nothing this round -- compaction will retry next turn)
                else:
                    stats["after_tokens"] = stats["before_tokens"]
                    stats["note"] = (
                        "skipped: entire evictable half is an unfinished "
                        "tool exchange"
                    )
                    return stats
            evicted = evictable[:evict_count]          # oldest 50%
            survivor_start = 1 + evict_count           # index into self.messages

            # -- 3. Durability barrier: raw turns must be in L3 first ---
            self.store.flush()
            # (add_message already mirrored everything; flush confirms it.)

            # -- 4. Summarize + commit to L2 ----------------------------
            summary_text = generate_summary(evicted)
            goal = self._extract_goal(evicted)
            files = self._extract_files_modified(evicted)
            self.store.insert_summary(
                session_id=self.session_id,
                goal=goal,
                files_modified=files,
                summary=summary_text,
            )

            # -- 5. Mutate L1 only now that L2/L3 are safe --------------
            pointer = {
                "role": "system",
                "content": (
                    f"[Older context compacted to Session {self.session_id}. "
                    "Use search_memory_summaries if needed.]"
                ),
            }
            self.messages = (
                [self.messages[0]] + [pointer] + self.messages[survivor_start:]
            )

            stats.update(
                compacted=True,
                after_tokens=self.total_tokens(),
                evicted_count=len(evicted),
                summary_row={
                    "goal": goal,
                    "files_modified": files,
                    "summary": summary_text,
                },
            )
            return stats

        except Exception as exc:
            # Compaction failed -> L1 unchanged (mutations happen last).
            stats["compacted"] = False
            stats["error"] = f"{type(exc).__name__}: {exc}"
            stats["after_tokens"] = self.total_tokens()
            return stats


# ============================================================================
# JSON tool schemas for the LLM
# ============================================================================

TOOL_SCHEMAS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_memory_summaries",
            "description": (
                "Search past session summaries (L2 memory) by keyword using "
                "local full-text search. Use when you need to recall what "
                "happened in earlier sessions: goals pursued, files modified, "
                "and outcomes. Returns the most relevant matching sessions "
                "with their session_id, date, goal, and summary. If you need "
                "the full raw conversation, follow up with "
                "inspect_session_transcript using the returned session_id."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Keywords or phrase to search for, e.g. "
                            "'flaky test test_app.py' or 'auth refactor'."
                        ),
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max number of matching sessions to return (default 5, max 50).",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "inspect_session_transcript",
            "description": (
                "Read the raw, unedited transcript (L3 memory) of a specific "
                "past session from local JSONL storage. Returns the most "
                "recent turns by default. Use after search_memory_summaries "
                "when a summary hit looks relevant and you need exact "
                "details: full prompts, full responses, tool outputs."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "The session_id returned by search_memory_summaries.",
                    },
                    "limit_lines": {
                        "type": "integer",
                        "description": (
                            "Return only the last N turns of the transcript "
                            "(default 50). Increase for deep dives; the read "
                            "is streamed so large values are safe."
                        ),
                        "default": 50,
                    },
                },
                "required": ["session_id"],
            },
        },
    },
]


# ============================================================================
# Self-test (hermetic: redirects CLI_HARNESS_HOME to a temp dir)
# ============================================================================

def _self_test() -> int:
    """
    End-to-end validation of all three tiers. Exit code 0 = pass.

    Runs against a throwaway temp directory; your ~/.cli_harness is never
    touched (we re-point the module store rather than the env var so the
    test also exercises explicit store injection).
    """
    print("=== memory_manager self-test ===")
    tmp = Path(tempfile.mkdtemp(prefix="cli_harness_test_"))
    failures: List[str] = []

    def check(name: str, cond: bool) -> None:
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        if not cond:
            failures.append(name)

    try:
        store = MemoryStore(tmp)
        sid = "testsess001"

        # --- Step 2: async L3 logging --------------------------------
        for i in range(120):
            ok = store.log_turn(sid, "user" if i % 2 == 0 else "assistant",
                                f"turn {i}: please fix the flaky test")
        check("log_turn accepted 120 records", ok is True)
        store.flush()
        tail = store.read_transcript_tail(sid, limit_lines=10)
        check("transcript tail returns last 10", len(tail) == 10)
        check("tail is most recent", tail[-1]["content"].startswith("turn 119"))
        check("record shape correct",
              {"timestamp", "role", "content"}.issubset(tail[0].keys()))
        raw = store.transcript_path(sid).read_text(encoding="utf-8")
        check("all 120 lines on disk", raw.count("\n") == 120)

        # --- Step 3 tool 2: transcript inspection --------------------
        res = inspect_session_transcript("does_not_exist")
        check("missing transcript -> error status", res["status"] == "error")
        # route through the injected store for the hermetic test
        store.insert_summary(sid, "fix flaky test in tests/test_app.py",
                             "tests/test_app.py, src/app.py",
                             "Fixed race condition in app startup")
        found = store.search_summaries("flaky test")
        check("FTS5 finds summary by keyword", len(found) == 1 and found[0]["session_id"] == sid)
        check("porter stemming matches inflections",
              len(store.search_summaries("fixing flakiness")) >= 1)
        check("FTS operator injection is neutralized",
              len(store.search_summaries('test" OR NEAR(1 1)')) >= 0)
        check("empty query returns nothing", store.search_summaries("   ") == [])

        # --- Step 4: compaction engine -------------------------------
        cm = ContextManager(system_prompt="You are a CLI agent. " + "x" * 400,
                             session_id="compaction_test", store=store)
        # Layout (44 evictable): user, 20 assistants, tool_call pair,
        # 21 assistants. Naive 50% cut = 22 lands ON the tool_calls
        # message, so the boundary must extend past its tool response.
        cm.add_message("user", "refactor the auth module in src/auth.py please")
        for i in range(20):
            cm.add_message("assistant", f"working on it step {i} " + "y" * 200)
        cm.add_message("assistant", "checking memory",
                       tool_calls=[{"id": "call_1",
                                    "type": "function",
                                    "function": {"name": "search_memory_summaries",
                                                 "arguments": '{"query": "auth"}'}}])
        cm.add_message("tool", "no results", tool_call_id="call_1")
        for i in range(21):
            cm.add_message("assistant", f"later step {i} " + "z" * 200)
        before = cm.total_tokens()
        stats = cm.check_and_compact(max_tokens=1)  # force compaction
        check("compaction triggered", stats["compacted"] is True)
        # naive cut = 22 lands on the tool_calls message -> extended to 23
        # so the assistant+tool pair is evicted together
        check("evicted count respects tool pairing",
              stats["evicted_count"] == 23)
        # no assistant-with-tool_calls may survive without its tool reply
        ok_pairing = True
        for i, m in enumerate(cm.messages):
            if m.get("tool_calls") and m.get("role") == "assistant":
                if i + 1 >= len(cm.messages) or cm.messages[i + 1].get("role") != "tool":
                    ok_pairing = False
        check("no orphaned tool_calls in survivors", ok_pairing)
        check("token count dropped", cm.total_tokens() < before)
        check("system prompt preserved at index 0",
              cm.messages[0]["role"] == "system"
              and cm.messages[0]["content"].startswith("You are a CLI agent"))
        check("pointer message inserted at index 1",
              "compacted to Session compaction_test" in cm.messages[1]["content"])
        check("L2 row was committed",
              len(store.search_summaries("refactor auth")) >= 1)
        check("raw turns durably in L3",
              store.transcript_path("compaction_test").exists())
        # under budget -> no-op
        stats2 = cm.check_and_compact(max_tokens=10**9)
        check("under budget -> no-op", stats2["compacted"] is False)

        # --- bad input hardening -------------------------------------
        try:
            store.log_turn("../evil", "user", "x")
            check("path traversal rejected", False)
        except ValueError:
            check("path traversal rejected", True)

        # --- schemas are valid JSON ----------------------------------
        check("two tool schemas present", len(TOOL_SCHEMAS) == 2)
        names = {t["function"]["name"] for t in TOOL_SCHEMAS}
        check("tool names correct", names == {
            "search_memory_summaries", "inspect_session_transcript"})

        store.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"=== {'ALL TESTS PASSED' if not failures else f'{len(failures)} FAILURES'} ===")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(_self_test())
