import { useEffect, useRef, useState } from "react";
import { Brain, Code2, MessageSquare, Settings } from "lucide-react";
import ChatPanel from "./components/ChatPanel";
import EditorPanel from "./components/EditorPanel";
import { languageFromFilename } from "./components/EditorPanel";
import SettingsPanel from "./components/SettingsPanel";
import { syncMonacoAccentFromDom } from "./monaco-setup";
import {
  getSessionId,
  getSessionTranscript,
  listCommands,
  listSessions,
  loadSettings,
  newSessionId,
  runCommand,
  saveSettings,
  setSessionId,
  streamChatCompletion,
  type ApiChatMessage,
  type ClientSettings,
} from "./api/client";
import {
  parseMemoryResult,
  type ChatMessage,
  type MemoryToolCall,
  type OpenFile,
  type SessionInfo,
  type SlashCommand,
} from "./types";

const SYSTEM_PROMPT =
  "You are Humble Human, an AI pair-programmer with access to a long-term memory engine. " +
  "Use search_memory_summaries to recall relevant past work before answering questions about history, " +
  "and inspect_session_transcript to dig into a specific session.";

const uid = () => Math.random().toString(36).slice(2, 10);

// ---------------------------------------------------------------------------
// Seed state — demo content so the layout (and the memory-tool card) renders
// on first launch, before the proxy is reachable.
// ---------------------------------------------------------------------------

const seedMessages: ChatMessage[] = [
  {
    id: uid(),
    role: "system",
    content:
      "Workspace connected. Memory tools (search_memory_summaries, inspect_session_transcript) are active.",
    demo: true,
    createdAt: Date.now(),
  },
  {
    id: uid(),
    role: "assistant",
    content:
      "Welcome back! Last time we scaffolded this GUI and wired up SSE streaming. I can recall any past session — just ask.",
    demo: true,
    toolCalls: [
      {
        id: "tc-demo",
        name: "search_memory_summaries",
        argsText: '{"query":"vite gui scaffold streaming","limit":3}',
        args: { query: "vite gui scaffold streaming", limit: 3 },
        status: "done",
        result: {
          raw: "demo",
          sessionIds: ["ses_2026-09-28_a41f", "ses_2026-09-15_9c02"],
          snippets: [
            "Scaffolded the Vite + React + Tailwind GUI with a dark split-pane layout…",
            "Debugged the SSE parser for /v1/chat/completions — CRLF normalization bug…",
          ],
        },
      },
    ],
    createdAt: Date.now(),
  },
];

const seedFiles: OpenFile[] = [
  {
    id: uid(),
    name: "main.py",
    language: languageFromFilename("main.py"),
    content: [
      '"""Humble Human — REPL entry point."""',
      "from memory_manager import ContextManager",
      "",
      "cm = ContextManager()",
      "",
      "def main() -> None:",
      '    print("[Humble Human] memory engine online")',
      "",
      'if __name__ == "__main__":',
      "    main()",
      "",
    ].join("\n"),
  },
  {
    id: uid(),
    name: "App.tsx",
    language: languageFromFilename("App.tsx"),
    originalContent: [
      "export default function App() {",
      "  return (",
      '    <div className="p-4">',
      "      <h1>Humble Human</h1>",
      "      <p>Single-pane layout</p>",
      "    </div>",
      "  );",
      "}",
      "",
    ].join("\n"),
    content: [
      "export default function App() {",
      "  return (",
      '    <div className="lg:grid lg:grid-cols-[35%_65%]">',
      "      <h1>Humble Human</h1>",
      "      <p>Responsive split / tabbed layout</p>",
      "    </div>",
      "  );",
      "}",
      "",
    ].join("\n"),
  },
];

// ---------------------------------------------------------------------------
// App shell
// ---------------------------------------------------------------------------

export default function App() {
  const [messages, setMessages] = useState<ChatMessage[]>(seedMessages);
  const [files, setFiles] = useState<OpenFile[]>(seedFiles);
  const [activeFileId, setActiveFileId] = useState<string | null>(seedFiles[0]?.id ?? null);
  const [mobileView, setMobileView] = useState<"chat" | "editor">("chat");
  const [settings, setSettings] = useState<ClientSettings>(loadSettings);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [sessionId, setSessionIdState] = useState<string>(getSessionId);
  const [commands, setCommands] = useState<SlashCommand[]>([]);
  const [sessions, setSessions] = useState<SessionInfo[]>([]);
  const [sessionsLoading, setSessionsLoading] = useState(false);

  // Slash-command menu content comes from the proxy's prompt dirs.
  useEffect(() => {
    listCommands(settings)
      .then(setCommands)
      .catch(() => setCommands([]));
  }, [settings.baseUrl]);

  // Refresh the session list whenever the settings panel opens.
  useEffect(() => {
    if (!settingsOpen) return;
    setSessionsLoading(true);
    listSessions(settings)
      .then(setSessions)
      .catch(() => setSessions([]))
      .finally(() => setSessionsLoading(false));
  }, [settingsOpen, settings]);

  // Monochrome + accent: the single accent colour is one CSS variable;
  // Monaco's hh-dark theme re-derives itself from the same value.
  useEffect(() => {
    document.documentElement.style.setProperty("--accent", settings.accent);
    syncMonacoAccentFromDom();
  }, [settings.accent]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const updateMessage = (id: string, fn: (m: ChatMessage) => ChatMessage) =>
    setMessages((prev) => prev.map((m) => (m.id === id ? fn(m) : m)));

  const appendSystemMessage = (content: string) =>
    setMessages((prev) => [...prev, { id: uid(), role: "system", content, createdAt: Date.now() }]);

  // ---- chat streaming ------------------------------------------------------

  const handleSend = (text: string) => {
    const history: ApiChatMessage[] = [
      { role: "system", content: SYSTEM_PROMPT },
      ...messages
        .filter((m) => !m.demo && m.content.trim().length > 0 && m.role !== "system")
        .map((m) => ({ role: m.role, content: m.content })),
      { role: "user", content: text },
    ];

    const asstId = uid();
    setMessages((prev) => [
      ...prev,
      { id: uid(), role: "user", content: text, createdAt: Date.now() },
      { id: asstId, role: "assistant", content: "", createdAt: Date.now() },
    ]);
    setIsStreaming(true);
    setError(null);

    const controller = new AbortController();
    abortRef.current = controller;

    // Accumulate streamed tool-call fragments by index (OpenAI format).
    const tcBuffer = new Map<number, MemoryToolCall>();
    const syncToolCalls = () => updateMessage(asstId, (m) => ({ ...m, toolCalls: [...tcBuffer.values()] }));

    streamChatCompletion({
      settings,
      messages: history,
      signal: controller.signal,
      handlers: {
        onToken: (t) => updateMessage(asstId, (m) => ({ ...m, content: m.content + t })),
        onToolCallDelta: (d) => {
          const existing =
            tcBuffer.get(d.index) ??
            ({ id: d.id ?? `tc-${d.index}-${uid()}`, name: "", argsText: "", args: {}, status: "running" } satisfies MemoryToolCall);
          if (d.id) existing.id = d.id;
          if (d.function?.name) existing.name += d.function.name;
          if (d.function?.arguments) existing.argsText += d.function.arguments;
          try {
            existing.args = JSON.parse(existing.argsText) as Record<string, unknown>;
          } catch {
            /* arguments still streaming — keep partial text */
          }
          tcBuffer.set(d.index, existing);
          syncToolCalls();
        },
        onToolResult: (toolCallId, content) => {
          for (const tc of tcBuffer.values()) {
            if (tc.id === toolCallId) {
              tc.result = parseMemoryResult(content);
              tc.status = "done";
            }
          }
          syncToolCalls();
        },
        onSystemNotice: (notice) => appendSystemMessage(notice),
        onFinish: () =>
          updateMessage(asstId, (m) => ({
            ...m,
            toolCalls: m.toolCalls?.map((tc) =>
              tc.status === "running" ? { ...tc, status: "done" } : tc,
            ),
          })),
      },
    })
      .catch((err: unknown) => {
        if (err instanceof DOMException && err.name === "AbortError") {
          updateMessage(asstId, (m) => ({ ...m, content: m.content + "\n\n— stopped —" }));
          return;
        }
        const message = err instanceof Error ? err.message : String(err);
        setError(message);
        updateMessage(asstId, (m) => ({
          ...m,
          content: m.content || `⚠️ ${message}`,
          toolCalls: m.toolCalls?.map((tc) =>
            tc.status === "running" ? { ...tc, status: "error" } : tc,
          ),
        }));
      })
      .finally(() => setIsStreaming(false));
  };

  const handleStop = () => abortRef.current?.abort();

  // ---- session switching ------------------------------------------------------

  const freshSessionSeed = (): ChatMessage[] => [
    {
      id: uid(),
      role: "system",
      content: "New memory session — your past sessions remain searchable.",
      demo: true,
      createdAt: Date.now(),
    },
  ];

  const handleNewSession = () => {
    if (isStreaming) return;
    setSessionIdState(newSessionId());
    setMessages(freshSessionSeed());
    setError(null);
  };

  const handleSwitchSession = (id: string) => {
    if (isStreaming || id === sessionId) return;
    setSessionId(id);
    setSessionIdState(id);
    setError(null);
    setMessages([
      {
        id: uid(),
        role: "system",
        content: `Loading session ${id}…`,
        demo: true,
        createdAt: Date.now(),
      },
    ]);
    getSessionTranscript(settings, id)
      .then((entries) => {
        const hydrated: ChatMessage[] = entries
          .filter((e) => typeof e.content === "string")
          .map((e): ChatMessage => {
            // Rebuild completed memory-tool cards from stored tool_calls.
            const toolCalls = Array.isArray(e.tool_calls)
              ? e.tool_calls.map((tc): MemoryToolCall => {
                  const t = tc as {
                    id?: string;
                    function?: { name?: string; arguments?: string };
                  };
                  const argsText = t.function?.arguments ?? "";
                  let args: Record<string, unknown> = {};
                  try {
                    args = JSON.parse(argsText || "{}") as Record<string, unknown>;
                  } catch {
                    /* keep empty args */
                  }
                  return {
                    id: t.id ?? uid(),
                    name: t.function?.name ?? "",
                    argsText,
                    args,
                    status: "done",
                  };
                })
              : undefined;
            return {
              id: uid(),
              role: e.role === "assistant" ? "assistant" : "user",
              content: e.content,
              toolCalls,
              createdAt: Date.now(),
            };
          })
          // Drop empty assistant turns that only carried tool calls already
          // represented by the cards; keeps user turns untouched.
          .filter(
            (m) =>
              m.role === "user" ||
              m.content.length > 0 ||
              (m.toolCalls?.length ?? 0) > 0,
          );
        setMessages(
          hydrated.length
            ? hydrated
            : [
                {
                  id: uid(),
                  role: "system",
                  content: "That session has no visible turns yet.",
                  demo: true,
                  createdAt: Date.now(),
                },
              ],
        );
      })
      .catch((err: unknown) =>
        setMessages([
          {
            id: uid(),
            role: "system",
            content: `Could not load session: ${err instanceof Error ? err.message : String(err)}`,
            demo: true,
            createdAt: Date.now(),
          },
        ]),
      );
  };

  // ---- editor actions --------------------------------------------------------

  const handleCloseFile = (id: string) =>
    setFiles((prev) => {
      const next = prev.filter((f) => f.id !== id);
      if (id === activeFileId) setActiveFileId(next[0]?.id ?? null);
      return next;
    });

  const handleApplyDiff = (id: string) => {
    setFiles((prev) => prev.map((f) => (f.id === id ? { ...f, originalContent: undefined } : f)));
    const file = files.find((f) => f.id === id);
    appendSystemMessage(`Diffs applied to ${file?.name ?? id}.`);
  };

  const handleRunCommand = (file: OpenFile) => {
    const cmd = file.name.endsWith(".py") ? `.venv/bin/python ${file.name}` : `cat ${file.name}`;
    appendSystemMessage(`▶ Running: ${cmd}`);
    if (mobileView === "editor") setMobileView("chat");
    runCommand(settings, cmd)
      .then((r) => {
        const out = (r.stdout || r.stderr || "(no output)").trim();
        const cut = out.length > 500 ? `${out.slice(0, 500)}…` : out;
        appendSystemMessage(
          r.status === "ok" ? `exit ${r.exit_code} — ${cut}` : `${r.status}: ${r.stderr || cut}`,
        );
      })
      .catch((err: unknown) =>
        appendSystemMessage(`⚠️ Run failed: ${err instanceof Error ? err.message : String(err)}`),
      );
  };

  // ---- render ----------------------------------------------------------------

  let host = settings.baseUrl;
  try {
    host = new URL(settings.baseUrl).host;
  } catch {
    /* keep raw string */
  }
  const dotClass = error
    ? "bg-zinc-200"
    : isStreaming
      ? "bg-accent animate-pulse"
      : "bg-zinc-600";

  // Verified provider rosters feed the model picker (xynex pattern).
  const activeProfile = settings.providers.find(
    (p) => p.id === settings.activeProvider,
  );
  const activeModelOptions =
    activeProfile && activeProfile.models.length > 0
      ? activeProfile.models
      : undefined;

  return (
    <div className="flex h-dvh flex-col overflow-hidden bg-zinc-950 text-zinc-100">
      {/* Header */}
      <header className="flex h-12 shrink-0 items-center justify-between gap-3 border-b border-zinc-800 px-3 lg:px-4">
        <div className="flex min-w-0 items-center gap-2">
          <div className="grid h-7 w-7 shrink-0 place-items-center rounded-lg border border-accent/30 bg-accent/15">
            <Brain size={15} className="text-accent" />
          </div>
          <div className="min-w-0">
            <h1 className="truncate text-sm font-semibold leading-tight">Humble Human</h1>
            <p className="hidden truncate text-[10px] leading-tight text-zinc-500 sm:block">
              infinite-memory coding workspace
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <span className="hidden items-center gap-1.5 text-[11px] text-zinc-500 md:flex">
            <span className={`h-1.5 w-1.5 rounded-full ${dotClass}`} />
            {host}
          </span>
          <button
            type="button"
            aria-label="Settings"
            onClick={() => setSettingsOpen((v) => !v)}
            className="rounded-lg border border-zinc-800 p-2 text-zinc-400 transition hover:bg-zinc-800 hover:text-zinc-200"
          >
            <Settings size={15} />
          </button>
        </div>
      </header>

      {/* Split (desktop) / tabbed (mobile) panes */}
      <main className="flex min-h-0 flex-1 flex-col lg:grid lg:grid-cols-[minmax(0,35fr)_minmax(0,65fr)]">
        <section
          className={`min-h-0 min-w-0 flex-1 flex-col border-zinc-800 lg:flex lg:border-r ${
            mobileView === "chat" ? "flex" : "hidden"
          }`}
        >
          <ChatPanel
            messages={messages}
            isStreaming={isStreaming}
            error={error}
            commands={commands}
            model={settings.model}
            modelOptions={activeModelOptions}
            onModelChange={(model) => {
              const next = { ...settings, model };
              setSettings(next);
              saveSettings(next);
            }}
            onSend={handleSend}
            onStop={handleStop}
          />
        </section>
        <section
          className={`min-h-0 min-w-0 flex-1 flex-col lg:flex ${
            mobileView === "editor" ? "flex" : "hidden"
          }`}
        >
          <EditorPanel
            files={files}
            activeFileId={activeFileId}
            onSelectFile={setActiveFileId}
            onCloseFile={handleCloseFile}
            onChangeFile={(id, content) =>
              setFiles((prev) => prev.map((f) => (f.id === id ? { ...f, content } : f)))
            }
            onApplyDiff={handleApplyDiff}
            onRunCommand={handleRunCommand}
          />
        </section>
      </main>

      {/* Mobile bottom nav */}
      <nav className="grid h-14 shrink-0 grid-cols-2 border-t border-zinc-800 pb-[env(safe-area-inset-bottom)] lg:hidden">
        {(
          [
            { key: "chat", label: "Chat", Icon: MessageSquare },
            { key: "editor", label: "Editor", Icon: Code2 },
          ] as const
        ).map(({ key, label, Icon }) => (
          <button
            key={key}
            type="button"
            onClick={() => setMobileView(key)}
            className={`flex flex-col items-center justify-center gap-0.5 text-[10px] transition ${
              mobileView === key ? "text-accent" : "text-zinc-500 hover:text-zinc-300"
            }`}
          >
            <Icon size={17} />
            {label}
          </button>
        ))}
      </nav>

      {/* Settings popover */}
      {settingsOpen && (
        <SettingsPanel
          settings={settings}
          sessions={sessions}
          sessionsLoading={sessionsLoading}
          currentSessionId={sessionId}
          onClose={() => setSettingsOpen(false)}
          onSave={(s) => {
            setSettings(s);
            saveSettings(s);
            setSettingsOpen(false);
          }}
          onSwitchSession={handleSwitchSession}
          onNewSession={handleNewSession}
        />
      )}
    </div>
  );
}
