import { useEffect, useRef, useState } from "react";
import type { FormEvent, KeyboardEvent } from "react";
import {
  Brain,
  Check,
  ChevronDown,
  ChevronRight,
  Database,
  Loader2,
  MessageSquare,
  Search,
  Send,
  Square,
  Terminal,
  TriangleAlert,
} from "lucide-react";
import type { ChatMessage, MemoryToolCall, SlashCommand } from "../types";
import { MODEL_OPTIONS } from "../api/client";

// ---------------------------------------------------------------------------
// Slash-command autocomplete (file-based prompt templates via GET /commands)
// ---------------------------------------------------------------------------

/** Prefix matches first, then substring matches on name + description. */
function filterCommands(cmds: SlashCommand[], query: string): SlashCommand[] {
  const needle = query.toLowerCase().trim();
  if (!needle) return cmds.slice(0, 8);
  const prefix: SlashCommand[] = [];
  const rest: SlashCommand[] = [];
  for (const cmd of cmds) {
    if (cmd.name.toLowerCase().startsWith(needle)) prefix.push(cmd);
    else if (
      cmd.name.toLowerCase().includes(needle) ||
      cmd.description.toLowerCase().includes(needle)
    ) {
      rest.push(cmd);
    }
  }
  return [...prefix, ...rest].slice(0, 8);
}

// ---------------------------------------------------------------------------
// Memory tool indicator card
// ---------------------------------------------------------------------------

function ToolCallCard({ call }: { call: MemoryToolCall }) {
  const [open, setOpen] = useState(true);
  const isSearch = call.name === "search_memory_summaries";
  const Icon = isSearch ? Search : Database;

  const queryLine = isSearch
    ? `query: "${String(call.args.query ?? "…")}" · limit: ${String(call.args.limit ?? 5)}`
    : `session: ${String(call.args.session_id ?? "…")} · lines: ${String(call.args.limit_lines ?? 200)}`;

  return (
    <div className="my-2 rounded-xl border border-accent/30 bg-accent/5 text-left">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center gap-2 px-3 py-2 text-xs"
      >
        <span className="grid h-6 w-6 shrink-0 place-items-center rounded-md border border-accent/30 bg-accent/10">
          <Icon size={13} className="text-accent" />
        </span>
        <span className="min-w-0 flex-1">
          <span className="block truncate font-medium text-accent">{call.name || "memory tool"}</span>
          <span className="block truncate font-mono text-[10px] text-zinc-500">{queryLine}</span>
        </span>
        {call.status === "running" ? (
          <span className="flex shrink-0 items-center gap-1 text-[10px] text-accent">
            <Loader2 size={11} className="animate-spin" />
            running
          </span>
        ) : call.status === "error" ? (
          <span className="shrink-0 text-[10px] font-semibold text-zinc-100">
            <TriangleAlert size={10} className="mr-0.5 inline" />error
          </span>
        ) : (
          <span className="flex shrink-0 items-center gap-1 text-[10px] text-accent">
            <Check size={11} />
            done
          </span>
        )}
        {open ? (
          <ChevronDown size={14} className="shrink-0 text-zinc-500" />
        ) : (
          <ChevronRight size={14} className="shrink-0 text-zinc-500" />
        )}
      </button>

      {open && (
        <div className="border-t border-accent/20 px-3 py-2 text-xs">
          {call.result ? (
            <>
              <div className="mb-1.5 flex flex-wrap items-center gap-1">
                <span className="text-[10px] uppercase tracking-wide text-zinc-500">
                  Matched sessions
                </span>
                {call.result.sessionIds.length > 0 ? (
                  call.result.sessionIds.map((id) => (
                    <span
                      key={id}
                      className="rounded-md border border-zinc-700 bg-zinc-800/80 px-1.5 py-0.5 font-mono text-[10px] text-accent"
                    >
                      {id}
                    </span>
                  ))
                ) : (
                  <span className="text-[10px] italic text-zinc-600">none</span>
                )}
              </div>
              {call.result.snippets.length > 0 && (
                <ul className="space-y-1">
                  {call.result.snippets.map((snippet, i) => (
                    <li
                      key={i}
                      className="border-l-2 border-accent/40 pl-2 text-[11px] leading-relaxed text-zinc-300"
                    >
                      {snippet}
                    </li>
                  ))}
                </ul>
              )}
            </>
          ) : call.status === "running" ? (
            <p className="animate-pulse text-[11px] text-zinc-500">
              Querying the memory engine…
            </p>
          ) : (
            <p className="text-[11px] text-zinc-500">No result payload returned.</p>
          )}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Transcript bubbles
// ---------------------------------------------------------------------------

function MessageBubble({ msg }: { msg: ChatMessage }) {
  if (msg.role === "system") {
    return (
      <div className="mx-auto my-3 max-w-[85%] rounded-full border border-zinc-800 bg-zinc-900/60 px-3 py-1 text-center text-[11px] text-zinc-500">
        {msg.content}
      </div>
    );
  }

  const isUser = msg.role === "user";
  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"}`}>
      <div
        className={`max-w-[88%] whitespace-pre-wrap break-words rounded-2xl border px-3 py-2 text-sm leading-relaxed ${
          isUser
            ? "rounded-br-md border-accent/30 bg-accent/10 text-zinc-100"
            : "rounded-bl-md border-zinc-800 bg-zinc-900 text-zinc-200"
        }`}
      >
        {msg.toolCalls?.map((call) => <ToolCallCard key={call.id} call={call} />)}
        {msg.content || (msg.toolCalls?.length ? "" : <span className="text-zinc-600">…</span>)}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Chat panel
// ---------------------------------------------------------------------------

export interface ChatPanelProps {
  messages: ChatMessage[];
  isStreaming: boolean;
  error: string | null;
  /** File-based slash commands discovered by the proxy. */
  commands: SlashCommand[];
  model: string;
  /** Roster override (e.g. active provider's verified models); defaults to built-ins. */
  modelOptions?: string[];
  onModelChange: (model: string) => void;
  onSend: (text: string) => void;
  onStop: () => void;
}

export default function ChatPanel({
  messages,
  isStreaming,
  error,
  commands,
  model,
  modelOptions: modelOptionsProp,
  onModelChange,
  onSend,
  onStop,
}: ChatPanelProps) {
  const [draft, setDraft] = useState("");
  const [slashIndex, setSlashIndex] = useState(0);
  const [slashDismissed, setSlashDismissed] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  // Menu is active only while the whole draft is still a single '/token'.
  const slashQuery = /^\/(\S*)$/.exec(draft)?.[1];
  const slashResults =
    slashQuery !== undefined && !slashDismissed
      ? filterCommands(commands, slashQuery)
      : [];
  const showSlashMenu = slashQuery !== undefined && slashResults.length > 0;

  useEffect(() => setSlashIndex(0), [slashQuery]);

  // Keep the transcript pinned to the latest output.
  useEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [messages]);

  const submit = (e?: FormEvent) => {
    e?.preventDefault();
    const text = draft.trim();
    if (!text || isStreaming) return;
    onSend(text);
    setDraft("");
  };

  const pickCommand = (cmd: SlashCommand) => {
    setDraft(`/${cmd.name} `);
    setSlashDismissed(false);
  };

  const onKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (showSlashMenu) {
      if (e.key === "ArrowDown") {
        e.preventDefault();
        setSlashIndex((i) => (i + 1) % slashResults.length);
        return;
      }
      if (e.key === "ArrowUp") {
        e.preventDefault();
        setSlashIndex((i) => (i - 1 + slashResults.length) % slashResults.length);
        return;
      }
      if (e.key === "Enter" || e.key === "Tab") {
        if (slashIndex >= 0) e.preventDefault();
        pickCommand(slashResults[slashIndex]);
        return;
      }
      if (e.key === "Escape") {
        e.preventDefault();
        setSlashDismissed(true);
        return;
      }
    }
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  };

  const baseModelOptions = modelOptionsProp ?? MODEL_OPTIONS;
  const modelOptions = baseModelOptions.includes(model)
    ? baseModelOptions
    : [model, ...baseModelOptions];
  const rows = Math.min(5, Math.max(1, draft.split("\n").length));

  return (
    <div className="flex h-full min-h-0 flex-col bg-zinc-950">
      {/* Panel header */}
      <div className="flex h-10 shrink-0 items-center gap-2 border-b border-zinc-800 px-3">
        <MessageSquare size={14} className="text-accent" />
        <span className="text-xs font-medium text-zinc-300">Chat &amp; Memory Stream</span>
        <span className="ml-auto flex items-center gap-1.5 text-[10px] text-zinc-500">
          <Brain size={11} className="text-accent" />
          {isStreaming ? "streaming…" : "memory tools armed"}
        </span>
      </div>

      {/* Transcript */}
      <div ref={scrollRef} className="min-h-0 flex-1 space-y-3 overflow-y-auto px-3 py-3">
        {messages.map((msg) => (
          <MessageBubble key={msg.id} msg={msg} />
        ))}
      </div>

      {/* Error banner */}
      {error && (
        <div className="mx-3 mb-2 flex shrink-0 items-start gap-2 rounded-lg border border-zinc-500/50 bg-zinc-500/10 px-3 py-2 text-xs text-zinc-200">
          <TriangleAlert size={13} className="mt-0.5 shrink-0" />
          <span className="break-words">{error}</span>
        </div>
      )}

      {/* Composer */}
      <form onSubmit={submit} className="shrink-0 border-t border-zinc-800 p-3">
        <div className="rounded-xl border border-zinc-700 bg-zinc-900 focus-within:border-accent/50">
          {showSlashMenu && (
            <div className="mx-3 mb-2 max-h-48 overflow-y-auto rounded-xl border border-zinc-700 bg-zinc-900 shadow-xl shadow-black/50">
              {slashResults.map((cmd, idx) => (
                <button
                  key={`${cmd.source}:${cmd.name}`}
                  type="button"
                  onMouseDown={(e) => {
                    e.preventDefault(); // keep textarea focus
                    pickCommand(cmd);
                  }}
                  onMouseEnter={() => setSlashIndex(idx)}
                  className={`flex w-full items-center gap-2.5 px-3 py-2 text-left ${
                    idx === slashIndex ? "bg-accent/10" : ""
                  }`}
                >
                  <Terminal
                    size={12}
                    className={idx === slashIndex ? "text-accent" : "text-zinc-600"}
                  />
                  <span className="shrink-0 font-mono text-xs text-zinc-200">
                    /{cmd.name}
                  </span>
                  {cmd.argument_hint && (
                    <span className="shrink-0 font-mono text-[10px] text-zinc-600">
                      {cmd.argument_hint}
                    </span>
                  )}
                  <span className="min-w-0 flex-1 truncate text-[11px] text-zinc-500">
                    {cmd.description}
                  </span>
                  {cmd.source === "project" && (
                    <span className="shrink-0 rounded bg-zinc-800 px-1 py-0.5 text-[9px] uppercase tracking-wide text-zinc-500">
                      project
                    </span>
                  )}
                </button>
              ))}
            </div>
          )}
          <textarea
            value={draft}
            onChange={(e) => {
              setDraft(e.target.value);
              setSlashDismissed(false);
            }}
            onKeyDown={onKeyDown}
            rows={rows}
            placeholder="Ask anything — past sessions are searchable…"
            className="max-h-40 w-full resize-none bg-transparent px-3 pt-2.5 text-sm text-zinc-100 outline-none placeholder:text-zinc-600"
          />
          <div className="flex items-center gap-2 px-2 pb-2">
            <select
              value={model}
              onChange={(e) => onModelChange(e.target.value)}
              title="Model"
              className="max-w-[55%] cursor-pointer rounded-lg border border-zinc-700 bg-zinc-800 px-2 py-1 text-[11px] text-zinc-300 outline-none hover:border-zinc-600"
            >
              {modelOptions.map((m) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))}
            </select>
            <span className="hidden text-[10px] text-zinc-600 sm:block">⏎ send · ⇧⏎ newline</span>
            {isStreaming ? (
              <button
                type="button"
                onClick={onStop}
                title="Stop generating"
                className="ml-auto grid h-8 w-8 place-items-center rounded-lg border border-zinc-500/50 bg-zinc-500/10 text-zinc-200 transition hover:bg-zinc-500/25"
              >
                <Square size={13} />
              </button>
            ) : (
              <button
                type="submit"
                disabled={!draft.trim()}
                title="Send"
                className="ml-auto grid h-8 w-8 place-items-center rounded-lg bg-accent text-zinc-950 transition hover:brightness-110 disabled:cursor-not-allowed disabled:opacity-40"
              >
                <Send size={13} />
              </button>
            )}
          </div>
        </div>
      </form>
    </div>
  );
}
