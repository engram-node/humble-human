"""
main.py — "Humble Human" CLI coding agent (REPL execution loop)
===============================================================

Connects the 3-tier memory engine (`memory_manager.ContextManager`) to any
OpenAI-compatible LLM endpoint, giving the agent persistent, searchable
long-term memory via local tool-calling.

Architecture
------------
    terminal (you)
        |
        v
    [User]: input ──> ContextManager.add_message() ──> L1
        |
        v
    LLM API call (messages=cm.messages, tools=TOOL_SCHEMAS)
        |
        ├── finish_reason == "stop"          -> print answer, done
        |
        └── finish_reason == "tool_calls"    -> execute tool LOCALLY
                |                               (search_memory_summaries /
                |                                inspect_session_transcript)
                v
            append role="tool" result to ContextManager
                |
                v
            auto-reprompt LLM with updated context  ──┐
                |                                     |
                └────── loop (bounded, see MAX_TOOL_ROUNDS) ──┘
        |
        v
    cm.check_and_compact(max_tokens=16000)  at end of every interaction
        |
        └── if stats["compacted"]: notify the terminal

How to run
----------
1. Create the venv and install the one dependency:

       python3 -m venv .venv
       .venv/bin/pip install openai

2. Export credentials for any OpenAI-compatible endpoint (BYOK).
   Examples:

       # DeepSeek (official API)
       export OPENAI_API_KEY="sk-..."
       export LLM_MODEL="deepseek-chat"
       unset OPENAI_BASE_URL                    # use the SDK default

       # Moonshot / Kimi
       export OPENAI_API_KEY="sk-..."
       export LLM_MODEL="moonshot-v1-32k"

       # Self-hosted vLLM / Ollama / LM Studio (any /v1 endpoint)
       export OPENAI_API_KEY="token-abc123"     # or "dummy" if unused
       export OPENAI_BASE_URL="http://localhost:8000/v1"
       export LLM_MODEL="Qwen/Qwen2.5-Coder-32B-Instruct"

   (Your shell may already have OPENAI_API_KEY/OPENAI_BASE_URL set;
   the SDK reads both automatically -- see openai.OpenAI defaults.)

3. Launch the agent:

       .venv/bin/python main.py

   Optional flags:
       --model NAME        override LLM_MODEL
       --base-url URL      override OPENAI_BASE_URL
       --api-key KEY       override OPENAI_API_KEY (prefer env vars!)
       --max-tokens N      L1 context budget (default 16000)
       --compact-notify    always print compaction stats (default: quiet)
       --self-test         run the offline mock-LLM test (no network,
                           no API key required) and exit

4. In the REPL:
       [User]: what did I work on in previous sessions?
       [Humble Human]: (calls search_memory_summaries -> answers)

   Type 'exit', 'quit', or press Ctrl-D to leave. Ctrl-C interrupts the
   current generation and returns to the prompt without killing the
   agent or losing memory.

Design notes
------------
*  The ONLY third-party import is `openai`. Everything memory-related
   stays stdlib (see memory_manager.py).
*  Tool loop is bounded (MAX_TOOL_ROUNDS). A misbehaving model that
   endless-loops on tool calls degrades to an error message instead of
   a stuck terminal.
*  System prompt tells the model about both memory tools and when to
   use them, so recall is model-initiated, not harness-forced.
*  If the `openai` package is missing, main.py still runs --self-test
   (it only imports memory_manager), so the memory engine can be
   validated on a box without the SDK.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

from memory_manager import (
    ContextManager,
    TOOL_SCHEMAS,
    inspect_session_transcript,
    search_memory_summaries,
)

# ============================================================================
# Configuration
# ============================================================================

#: Default model. Override with $LLM_MODEL or --model.
DEFAULT_MODEL = os.environ.get("LLM_MODEL", "deepseek-chat")

#: Hard cap on tool-call rounds per user turn. Prevents infinite
#: tool-looping when a model repeatedly re-issues the same call.
MAX_TOOL_ROUNDS = 8

#: L1 context budget for check_and_compact().
DEFAULT_MAX_TOKENS = 16_000

#: Strong identity + memory instructions for the model.
SYSTEM_PROMPT = """\
You are Humble Human, a CLI coding agent with perfect long-term memory.

MEMORY CAPABILITY:
You have two local tools that give you access to conversation history
across sessions. Your context is compacted automatically when it grows
large, but nothing is ever lost -- raw transcripts live on local disk.

- search_memory_summaries(query, limit): Search past session summaries by
  keyword. Use this FIRST when the user refers to earlier work ("what did
  I do last time", "continue the auth refactor", "that bug I mentioned").
- inspect_session_transcript(session_id, limit_lines): Read the raw
  transcript of a specific past session. Use this when a summary hit is
  relevant but you need exact details (full prompts, full code, exact
  error messages).

WHEN TO USE MEMORY TOOLS:
- The user references prior sessions or past work -> search summaries.
- A summary hit needs detail -> inspect its transcript.
- Otherwise, answer directly from active context.

CONDUCT:
- Be direct and technically precise. Admit uncertainty; never invent
  prior work you cannot find in memory.
- When you use a memory tool, briefly note what you looked up.
- You are humble: you do not claim skills or history you lack evidence
  for, and you ask when requirements are ambiguous.
"""


# ============================================================================
# LLM client
# ============================================================================

def build_client(api_key: Optional[str], base_url: Optional[str]):
    """
    Construct the OpenAI SDK client.

    Precedence per argument: explicit flag > environment variable.
    The SDK natively reads OPENAI_API_KEY and OPENAI_BASE_URL, so in the
    common case both args are None and we just let it use the env.
    """
    import openai  # deferred: lets --self-test run without the SDK

    kwargs: Dict[str, Any] = {}
    if api_key:
        kwargs["api_key"] = api_key
    if base_url:
        kwargs["base_url"] = base_url
    try:
        return openai.OpenAI(**kwargs)
    except openai.OpenAIError as exc:
        print(f"[system]: cannot configure LLM client: {exc}", file=sys.stderr)
        print(
            "           Set OPENAI_API_KEY (and optionally OPENAI_BASE_URL, "
            "LLM_MODEL). See the docstring at the top of main.py.",
            file=sys.stderr,
        )
        raise SystemExit(2)


def call_llm(
    client,
    model: str,
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    One LLM round-trip. Returns a normalized dict:

        {"content": str | None,
         "tool_calls": [ {id, type, function: {name, arguments}} ] | None,
         "finish_reason": "stop" | "tool_calls" | ...}

    Normalization matters: different OpenAI-compatible servers (vLLM,
    DeepSeek, Moonshot, LM Studio) vary in null/absent field handling,
    so we coerce everything to plain Python before the REPL touches it.
    """
    kwargs: Dict[str, Any] = {"model": model, "messages": messages}
    if tools:
        kwargs["tools"] = tools
    response = client.chat.completions.create(**kwargs)
    choice = response.choices[0]
    message = choice.message

    # SDK message objects: use model_dump() when available (pydantic v2),
    # otherwise fall back to manual extraction for exotic SDK builds.
    if hasattr(message, "model_dump"):
        msg = message.model_dump()
    else:
        msg = {
            "role": message.role,
            "content": message.content,
            "tool_calls": getattr(message, "tool_calls", None),
        }

    tool_calls = msg.get("tool_calls") or None
    # Ensure arguments are a STRING (some servers emit dicts). The L1
    # history must match what the API expects on the next round.
    if tool_calls:
        for tc in tool_calls:
            fn = tc.get("function") or {}
            args = fn.get("arguments")
            if not isinstance(args, str):
                fn["arguments"] = json.dumps(args or {})
            tc["function"] = fn
            tc.setdefault("type", "function")
            tc.setdefault("id", "")

    return {
        "content": msg.get("content") or None,
        "tool_calls": tool_calls,
        "finish_reason": getattr(choice, "finish_reason", None),
    }


# ============================================================================
# Tool dispatch
# ============================================================================

#: Tool name -> local Python callable. This is the entire "tool server".
TOOL_FUNCTIONS = {
    "search_memory_summaries": search_memory_summaries,
    "inspect_session_transcript": inspect_session_transcript,
}


def execute_tool(name: str, arguments_json: str) -> str:
    """
    Run one memory tool locally and return its result as a JSON string.

    Never raises: any failure becomes an error payload the model can
    read and react to. This is important for robustness -- a malformed
    tool call from the model should produce a corrective message, not
    crash the REPL.
    """
    func = TOOL_FUNCTIONS.get(name)
    if func is None:
        return json.dumps(
            {"status": "error", "error": f"unknown tool: {name!r}"}
        )
    try:
        args = json.loads(arguments_json) if arguments_json.strip() else {}
    except json.JSONDecodeError as exc:
        return json.dumps(
            {"status": "error",
             "error": f"invalid tool arguments JSON: {exc}"}
        )
    try:
        result = func(**args)
    except TypeError as exc:
        # e.g. model passed an unexpected kwarg
        return json.dumps(
            {"status": "error",
             "error": f"bad arguments for {name}: {exc}"}
        )
    return json.dumps(result, ensure_ascii=False)


# ============================================================================
# The REPL
# ============================================================================

class HumbleHumanAgent:
    """Wires the LLM client, the ContextManager, and the tool loop."""

    def __init__(self, client, model: str, cm: ContextManager,
                 max_tokens: int = DEFAULT_MAX_TOKENS,
                 echo: Any = print) -> None:
        self.client = client
        self.model = model
        self.cm = cm
        self.max_tokens = max_tokens
        self.echo = echo  # injectable for tests

    # ----------------------------------------------------------------

    def handle_user_turn(self, user_input: str) -> None:
        """
        Full interaction loop for one user message.

        1. Append user msg to L1.
        2. Call the LLM.
        3. If it asked for tools: execute locally, append role="tool"
           results, and auto-reprompt (bounded by MAX_TOOL_ROUNDS).
        4. Compaction check at the end; notify on compaction.
        """
        self.cm.add_message("user", user_input)

        for round_no in range(1, MAX_TOOL_ROUNDS + 1):
            reply = call_llm(self.client, self.model, self.cm.messages, TOOL_SCHEMAS)

            if reply["tool_calls"]:
                # ---- assistant requested tool(s) --------------------
                # L1 must retain the assistant's tool_calls verbatim
                # (with empty/None content) or the next API call 400s.
                self.cm.add_message(
                    "assistant", reply["content"] or "",
                    tool_calls=reply["tool_calls"],
                )
                for tc in reply["tool_calls"]:
                    fn = tc.get("function", {})
                    name = fn.get("name", "")
                    args_json = fn.get("arguments", "{}")
                    self.echo(f"[system]: calling tool {name}({args_json})")
                    result = execute_tool(name, args_json)
                    self.cm.add_message(
                        "tool", result, tool_call_id=tc.get("id", "")
                    )
                # auto-reprompt: loop continues with updated context
                continue

            # ---- plain assistant answer ----------------------------
            self.cm.add_message("assistant", reply["content"] or "")
            self.echo(f"[Humble Human]: {reply['content']}")
            break
        else:
            # for/else: loop exhausted without a final answer.
            self.echo(
                "[system]: stopped after max tool rounds "
                f"({MAX_TOOL_ROUNDS}); the model kept requesting tools."
            )

        # ---- token management: compact if over budget ---------------
        stats = self.cm.check_and_compact(max_tokens=self.max_tokens)
        if stats.get("compacted"):
            self.echo(
                f"[system]: context compacted "
                f"({stats.get('before_tokens', '?')} -> "
                f"{stats.get('after_tokens', '?')} tokens, "
                f"{stats.get('evicted_count', '?')} messages archived "
                "to long-term memory)"
            )
            if stats.get("error"):
                self.echo(f"[system]: compaction warning: {stats['error']}")


# ============================================================================
# CLI entry point
# ============================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="humble-human",
        description="CLI coding agent with 3-tier long-term memory.",
    )
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help=f"model name (default: {DEFAULT_MODEL}; env: LLM_MODEL)")
    p.add_argument("--base-url", default=None,
                   help="OpenAI-compatible base URL (env: OPENAI_BASE_URL)")
    p.add_argument("--api-key", default=None,
                   help="API key (env: OPENAI_API_KEY; prefer the env var)")
    p.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
                   help=f"L1 context budget (default: {DEFAULT_MAX_TOKENS})")
    p.add_argument("--self-test", action="store_true",
                   help="run the offline mock-LLM test (no network needed)")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if args.self_test:
        return run_self_test()

    client = build_client(args.api_key, args.base_url)

    cm = ContextManager(system_prompt=SYSTEM_PROMPT)
    agent = HumbleHumanAgent(client, args.model, cm,
                             max_tokens=args.max_tokens)

    print("Humble Human — CLI coding agent (type 'exit' or Ctrl-D to quit)")
    print(f"model: {args.model} | memory: ~/.cli_harness/ | "
          f"session: {cm.session_id[:8]}…")
    print()

    while True:
        try:
            user_input = input("[User]: ").strip()
        except EOFError:
            print()
            break
        except KeyboardInterrupt:
            print("\n[system]: interrupted — type 'exit' to quit")
            continue
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            break

        try:
            agent.handle_user_turn(user_input)
        except KeyboardInterrupt:
            print("\n[system]: generation interrupted")
        except Exception as exc:  # noqa: BLE001 — REPL must survive anything
            print(f"[system]: error: {type(exc).__name__}: {exc}",
                  file=sys.stderr)

    print("Goodbye. Your session is archived in ~/.cli_harness/")
    return 0


# ============================================================================
# Offline self-test: mock LLM, no network, no API key
# ============================================================================

class _MockLLM:
    """
    Scripted stand-in for the OpenAI client.

    Implements just enough of the chat.completions surface for
    call_llm(): a queue of scripted responses consumed one per call.
    Each scripted response is a dict as returned by call_llm's
    normalization, except tool_calls use the raw SDK-ish shape that
    call_llm must handle (including the dict-arguments curveball some
    servers throw).
    """

    def __init__(self, script: List[Dict[str, Any]]) -> None:
        self.script = list(script)
        self.calls: List[Dict[str, Any]] = []  # (messages snapshot log)

    class _Msg:
        def __init__(self, m: Dict[str, Any]) -> None:
            self._m = m

        def model_dump(self) -> Dict[str, Any]:
            return self._m

    class _Choice:
        def __init__(self, m: Dict[str, Any], finish_reason: Optional[str]) -> None:
            self.message = _MockLLM._Msg(m)
            self.finish_reason = finish_reason

    class _Resp:
        def __init__(self, choices: List[Any]) -> None:
            self.choices = choices

    class _Completions:
        def __init__(self, outer: "_MockLLM") -> None:
            self._outer = outer

        def create(self, **kwargs: Any) -> "_MockLLM._Resp":
            self._outer.calls.append(kwargs)
            item = self._outer.script.pop(0)
            return _MockLLM._Resp([_MockLLM._Choice(
                {"role": "assistant",
                 "content": item.get("content"),
                 "tool_calls": item.get("tool_calls")},
                item.get("finish_reason"),
            )])

    class _Chat:
        def __init__(self, outer: "_MockLLM") -> None:
            self.completions = _MockLLM._Completions(outer)

    class _Client:
        def __init__(self, outer: "_MockLLM") -> None:
            self.chat = _MockLLM._Chat(outer)

    def client(self) -> "_MockLLM._Client":
        return _MockLLM._Client(self)


def run_self_test() -> int:
    """
    Offline validation of the full main.py flow minus the network:

    1. user turn -> tool_call round -> tool executed locally ->
       auto-reprompt -> final answer
    2. tool result lands in L1 with role="tool" + tool_call_id
    3. assistant tool_calls survive in L1 verbatim
    4. compaction runs at turn end when over budget
    5. unknown-tool and bad-JSON tool calls produce error payloads,
       not crashes

    Uses a temp CLI_HARNESS_HOME so ~/.cli_harness is never touched.
    """
    import tempfile
    from pathlib import Path
    import shutil

    print("=== main.py offline self-test ===")
    tmp = Path(tempfile.mkdtemp(prefix="humble_test_"))
    os.environ["CLI_HARNESS_HOME"] = str(tmp)

    failures: List[str] = []

    def check(name: str, cond: bool) -> None:
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        if not cond:
            failures.append(name)

    try:
        from memory_manager import initialize
        initialize(Path(tmp))

        # --- scenario 1: tool round-trip ------------------------------
        script = [
            {   # round 1: model asks for a memory search
                "content": None,
                "finish_reason": "tool_calls",
                "tool_calls": [{
                    "id": "call_abc", "type": "function",
                    "function": {
                        "name": "search_memory_summaries",
                        # dict arguments (not str) on purpose: some
                        # servers do this; call_llm must normalize.
                        "arguments": {"query": "auth refactor"},
                    },
                }],
            },
            {   # round 2: final answer
                "content": "Found 0 prior sessions about auth refactor.",
                "finish_reason": "stop",
            },
        ]
        mock = _MockLLM(script)
        cm = ContextManager(system_prompt=SYSTEM_PROMPT)
        out: List[str] = []
        agent = HumbleHumanAgent(mock.client(), "mock-model", cm,
                                 max_tokens=16_000, echo=out.append)
        agent.handle_user_turn("what did I do about the auth refactor?")

        check("final answer echoed", any("Found 0 prior sessions" in s for s in out))
        check("tool round executed exactly once",
              len(mock.calls) == 2)
        # L1 shape: sys, user, asst(tool_calls), tool, asst(final)
        roles = [m["role"] for m in cm.messages]
        check("L1 role sequence correct",
              roles == ["system", "user", "assistant", "tool", "assistant"])
        asst_tc = cm.messages[2]
        check("assistant tool_calls preserved in L1",
              asst_tc.get("tool_calls")[0]["function"]["name"]
              == "search_memory_summaries")
        check("tool_call arguments normalized to str",
              isinstance(asst_tc["tool_calls"][0]["function"]["arguments"], str))
        tool_msg = cm.messages[3]
        check("tool result message carries tool_call_id",
              tool_msg.get("tool_call_id") == "call_abc")
        parsed = json.loads(tool_msg["content"])
        check("tool result is searchable JSON payload",
              parsed.get("status") == "ok" and "results" in parsed)
        check("compaction did not fire (under budget)",
              not any("compacted" in s for s in out))

        # --- scenario 2: compaction fires and is reported ------------
        script2 = [
            {"content": "ok", "finish_reason": "stop"},
        ]
        mock2 = _MockLLM(script2)
        cm2 = ContextManager(system_prompt=SYSTEM_PROMPT)
        out2: List[str] = []
        agent2 = HumbleHumanAgent(mock2.client(), "mock-model", cm2,
                                  max_tokens=100, echo=out2.append)
        # oversized user message forces compaction at turn end
        agent2.handle_user_turn("huge " * 2000)
        check("compaction notice printed",
              any("compacted" in s for s in out2))
        check("compaction actually evicted",
              "Older context compacted" in cm2.messages[1]["content"])

        # --- scenario 3: hostile tool usage degrades gracefully ------
        script3 = [
            {   # unknown tool + invalid JSON arguments
                "content": None,
                "finish_reason": "tool_calls",
                "tool_calls": [{
                    "id": "call_x", "type": "function",
                    "function": {"name": "rm_rf_everything",
                                 "arguments": "{not json"},
                }],
            },
            {"content": "recovered", "finish_reason": "stop"},
        ]
        mock3 = _MockLLM(script3)
        cm3 = ContextManager(system_prompt=SYSTEM_PROMPT)
        out3: List[str] = []
        agent3 = HumbleHumanAgent(mock3.client(), "mock-model", cm3,
                                  echo=out3.append)
        agent3.handle_user_turn("do something impossible")
        check("unknown tool produced error payload, no crash",
              any("unknown tool" in s for s in out3) or
              json.loads(cm3.messages[3]["content"]).get("status") == "error")

        # --- scenario 4: tool-loop bound ------------------------------
        script4 = [{"content": None, "finish_reason": "tool_calls",
                    "tool_calls": [{
                        "id": f"c{i}", "type": "function",
                        "function": {"name": "search_memory_summaries",
                                     "arguments": "{\"query\": \"x\"}"},
                    }]} for i in range(20)]
        mock4 = _MockLLM(script4)
        cm4 = ContextManager(system_prompt=SYSTEM_PROMPT)
        out4: List[str] = []
        agent4 = HumbleHumanAgent(mock4.client(), "mock-model", cm4,
                                  echo=out4.append)
        agent4.handle_user_turn("loop forever")
        check("tool loop bounded at MAX_TOOL_ROUNDS",
              len(mock4.calls) == MAX_TOOL_ROUNDS and
              any("max tool rounds" in s for s in out4))

        # --- scenario 5: tool schemas are OpenAI-shaped ---------------
        names = {t["function"]["name"] for t in TOOL_SCHEMAS}
        check("TOOL_SCHEMAS names match dispatch table",
              names == set(TOOL_FUNCTIONS))

    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"=== {'ALL TESTS PASSED' if not failures else f'{len(failures)} FAILURES'} ===")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
