/**
 * API client for the Humble Human proxy server.
 *
 * Talks to an OpenAI-compatible endpoint (default: proxy_server.py at
 * http://localhost:8000/v1/chat/completions) with streaming (SSE) support so
 * assistant output renders token-by-token and memory tool calls
 * (search_memory_summaries / inspect_session_transcript) surface in the chat
 * as they happen.
 */

import type {
  ChatRole,
  ProviderProfile,
  SessionInfo,
  SlashCommand,
} from "../types";

// ---------------------------------------------------------------------------
// Settings (persisted to localStorage; configurable for remote deployment)
// ---------------------------------------------------------------------------

export interface ClientSettings {
  /** OpenAI-compatible base URL, e.g. "http://localhost:8000/v1". */
  baseUrl: string;
  /** Bearer token. Leave empty for local proxies that ignore auth. */
  apiKey: string;
  /** Default model, e.g. "deepseek-chat", "moonshot-v1-32k". */
  model: string;
  /** The monochrome UI's single accent colour (#rrggbb). */
  accent: string;
  /** UI-managed provider profiles (BYOK). Empty = use proxy env config. */
  providers: ProviderProfile[];
  /** Active profile id; "" = the proxy's env-configured upstream. */
  activeProvider: string;
}

export const DEFAULT_SETTINGS: ClientSettings = {
  baseUrl: "http://localhost:8000/v1",
  apiKey: "",
  model: "deepseek-chat",
  accent: "#34d399",
  providers: [],
  activeProvider: "",
};

export const MODEL_OPTIONS: string[] = [
  "deepseek-chat",
  "deepseek-reasoner",
  "moonshot-v1-32k",
  "moonshot-v1-128k",
  "kimi-k2-0905-preview",
];

const SETTINGS_KEY = "humble-human:settings:v1";

export function loadSettings(): ClientSettings {
  try {
    const raw = localStorage.getItem(SETTINGS_KEY);
    if (!raw) return DEFAULT_SETTINGS;
    const merged = {
      ...DEFAULT_SETTINGS,
      ...(JSON.parse(raw) as Partial<ClientSettings>),
    };
    // Type-repair: corrupt/old entries must never crash callers.
    if (!Array.isArray(merged.providers)) merged.providers = [];
    if (typeof merged.activeProvider !== "string") merged.activeProvider = "";
    return merged;
  } catch {
    return DEFAULT_SETTINGS;
  }
}

export function saveSettings(settings: ClientSettings): void {
  try {
    localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings));
  } catch {
    // Storage unavailable (private mode etc.) — settings stay in memory only.
  }
}

export function chatCompletionsUrl(baseUrl: string): string {
  return `${baseUrl.replace(/\/+$/, "")}/chat/completions`;
}

/** Server root (baseUrl minus the /v1 suffix) for non-OpenAI endpoints like /run. */
export function serverRootUrl(baseUrl: string): string {
  return baseUrl.replace(/\/+$/, "").replace(/\/v1$/, "");
}

// ---------------------------------------------------------------------------
// Session id — pins the GUI to one server-side ContextManager so the proxy's
// L1/L2/L3 memory accumulates across requests (and page reloads).
// ---------------------------------------------------------------------------

const SESSION_KEY = "humble-human:session:v1";

export function getSessionId(): string {
  try {
    let id = localStorage.getItem(SESSION_KEY);
    if (!id) {
      id = mintSessionId();
    }
    return id;
  } catch {
    return "gui_ephemeral";
  }
}

function mintSessionId(): string {
  const id = `gui_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 10)}`;
  try {
    localStorage.setItem(SESSION_KEY, id);
  } catch {
    /* storage unavailable */
  }
  return id;
}

/** Switch the pinned session (existing id from GET /sessions). */
export function setSessionId(id: string): void {
  try {
    localStorage.setItem(SESSION_KEY, id);
  } catch {
    /* storage unavailable */
  }
}

/** Start a brand-new memory session. */
export function newSessionId(): string {
  return mintSessionId();
}

/** BYOK headers derived from the active provider profile. */
export function upstreamHeaders(settings: ClientSettings): Record<string, string> {
  const p = settings.providers.find((pr) => pr.id === settings.activeProvider);
  const h: Record<string, string> = {};
  if (p) {
    if (p.baseUrl) h["X-HH-Base-URL"] = p.baseUrl;
    if (p.apiKey) h["X-HH-API-Key"] = p.apiKey;
  }
  return h;
}

// ---------------------------------------------------------------------------
// Tool schemas — mirror TOOL_SCHEMAS in memory_manager.py
// ---------------------------------------------------------------------------

export const MEMORY_TOOL_SCHEMAS = [
  {
    type: "function",
    function: {
      name: "search_memory_summaries",
      description:
        "Search past session summaries by natural-language query. Returns the most relevant session summaries (tier L2) with their session ids.",
      parameters: {
        type: "object",
        properties: {
          query: {
            type: "string",
            description: "What to look for, e.g. 'vite gui scaffold' or 'refactor auth'.",
          },
          limit: {
            type: "integer",
            description: "Max number of summaries to return (default 5).",
            default: 5,
          },
        },
        required: ["query"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "inspect_session_transcript",
      description:
        "Read the raw message transcript (tier L3) of one past session, given its session id from search_memory_summaries.",
      parameters: {
        type: "object",
        properties: {
          session_id: {
            type: "string",
            description: "The session id returned by search_memory_summaries, e.g. 'ses_2026-09-28_a41f'.",
          },
          limit_lines: {
            type: "integer",
            description: "Max number of transcript lines to return (default 200).",
            default: 200,
          },
        },
        required: ["session_id"],
      },
    },
  },
];

// ---------------------------------------------------------------------------
// Streaming chat completions
// ---------------------------------------------------------------------------

export interface ApiChatMessage {
  role: ChatRole;
  content: string;
}

/** One streamed tool-call fragment (OpenAI streaming format). */
export interface ToolCallDelta {
  index: number;
  id?: string;
  function?: {
    name?: string;
    arguments?: string;
  };
}

export interface StreamHandlers {
  /** Incremental assistant text. */
  onToken?: (text: string) => void;
  /** Incremental tool-call fragments (accumulate by `index`). */
  onToolCallDelta?: (delta: ToolCallDelta) => void;
  /**
   * Tool result echoed by the proxy (if the proxy executes memory tools
   * server-side and streams results back as role="tool" deltas).
   */
  onToolResult?: (toolCallId: string, content: string) => void;
  /** Operational notice from the proxy (compaction, finish quirks, …). */
  onSystemNotice?: (message: string) => void;
  /** Stream finished. `reason` is the finish_reason, or null if unknown. */
  onFinish?: (reason: string | null) => void;
}

export class ApiError extends Error {
  status?: number;
  constructor(message: string, status?: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

export interface StreamOptions {
  settings: ClientSettings;
  messages: ApiChatMessage[];
  signal?: AbortSignal;
  handlers: StreamHandlers;
}

/**
 * POST /v1/chat/completions with stream=true and dispatch SSE chunks to the
 * given handlers. Throws ApiError on HTTP/transport failures; an aborted
 * signal rejects with DOMException("AbortError").
 */
export async function streamChatCompletion(options: StreamOptions): Promise<void> {
  const { settings, messages, signal, handlers } = options;
  const { baseUrl, apiKey, model } = settings;

  const res = await fetch(chatCompletionsUrl(baseUrl), {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...(apiKey ? { Authorization: `Bearer ${apiKey}` } : {}),
      ...upstreamHeaders(settings), // BYOK: per-request provider override
    },
    body: JSON.stringify({
      model,
      messages,
      stream: true,
      tools: MEMORY_TOOL_SCHEMAS,
      tool_choice: "auto",
      session_id: getSessionId(), // proxy extension: persistent memory session
    }),
    signal,
  });

  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new ApiError(
      `Request failed (${res.status} ${res.statusText})${body ? `: ${body.slice(0, 300)}` : ""}`,
      res.status,
    );
  }
  if (!res.body) {
    throw new ApiError("The endpoint returned no body — is streaming enabled on the proxy?");
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";
  let sawDone = false;

  const handleEvent = (rawEvent: string): void => {
    const data = rawEvent
      .split("\n")
      .filter((line) => line.startsWith("data:"))
      .map((line) => line.slice(5).trimStart())
      .join("\n")
      .trim();
    if (!data) return;
    if (data === "[DONE]") {
      sawDone = true;
      handlers.onFinish?.("stop");
      return;
    }

    let chunk: unknown;
    try {
      chunk = JSON.parse(data);
    } catch {
      return; // skip malformed/keep-alive events
    }

    // Proxy error protocol: terminal error chunk mid-stream.
    const error = (chunk as { error?: { message?: string } }).error;
    if (error) {
      throw new ApiError(error.message ?? "Upstream stream error");
    }

    const choice = (chunk as { choices?: Array<Record<string, unknown>> }).choices?.[0];
    if (!choice) return;
    const delta = (choice.delta ?? {}) as Record<string, unknown>;

    // Operational notices (compaction, loop bounds) from the proxy.
    if (delta.role === "hh_system" && typeof delta.content === "string" && delta.content) {
      handlers.onSystemNotice?.(delta.content);
      return;
    }

    // Tool results streamed back by a server-side tool loop.
    if (delta.role === "tool" && typeof delta.content === "string" && delta.content) {
      handlers.onToolResult?.(String(delta.tool_call_id ?? ""), delta.content);
    }

    if (typeof delta.content === "string" && delta.content.length > 0 && delta.role !== "tool") {
      handlers.onToken?.(delta.content);
    }

    if (Array.isArray(delta.tool_calls)) {
      for (const tc of delta.tool_calls as Array<Record<string, unknown>>) {
        handlers.onToolCallDelta?.({
          index: typeof tc.index === "number" ? tc.index : 0,
          id: typeof tc.id === "string" ? tc.id : undefined,
          function:
            typeof tc.function === "object" && tc.function !== null
              ? {
                  name: (tc.function as Record<string, unknown>).name as string | undefined,
                  arguments: (tc.function as Record<string, unknown>).arguments as
                    | string
                    | undefined,
                }
              : undefined,
        });
      }
    }

    if (typeof choice.finish_reason === "string" && choice.finish_reason) {
      sawDone = true;
      handlers.onFinish?.(choice.finish_reason);
    }
  };

  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, "\n");
      const events = buffer.split("\n\n");
      buffer = events.pop() ?? "";
      for (const evt of events) handleEvent(evt);
    }
    buffer += decoder.decode(); // flush
    if (buffer.trim()) handleEvent(buffer);
  } finally {
    reader.releaseLock();
  }

  if (!sawDone) handlers.onFinish?.(null);
}

// ---------------------------------------------------------------------------
// /run — local command execution on the proxy host
// ---------------------------------------------------------------------------

export interface RunResult {
  status: "ok" | "disabled" | "error" | "timeout";
  exit_code: number | null;
  stdout: string;
  stderr: string;
}

export async function runCommand(
  settings: ClientSettings,
  command: string,
): Promise<RunResult> {
  const res = await fetch(`${serverRootUrl(settings.baseUrl)}/run`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ command }),
  });
  if (!res.ok) {
    throw new ApiError(`/run failed (${res.status})`, res.status);
  }
  return (await res.json()) as RunResult;
}

// ---------------------------------------------------------------------------
// Proxy metadata endpoints: slash commands, model roster, session browsing
// ---------------------------------------------------------------------------

async function getJson<T>(settings: ClientSettings, path: string): Promise<T> {
  const res = await fetch(`${serverRootUrl(settings.baseUrl)}${path}`, {
    headers: { ...upstreamHeaders(settings) },
  });
  const body = (await res.json().catch(() => ({}))) as {
    error?: { message?: string };
  };
  if (!res.ok) {
    throw new ApiError(
      body.error?.message ?? `${path} failed (${res.status})`,
      res.status,
    );
  }
  return body as T;
}

/** File-based slash commands (project `.hh/prompts/` + global prompts dir). */
export async function listCommands(settings: ClientSettings): Promise<SlashCommand[]> {
  const body = await getJson<{ commands?: SlashCommand[] }>(settings, "/commands");
  return body.commands ?? [];
}

/**
 * Live model roster — used to verify a provider profile before saving and
 * to populate the model picker. `byok` overrides the active profile once
 * (used while editing a profile that isn't saved yet).
 */
export async function listModels(
  settings: ClientSettings,
  byok?: { baseUrl: string; apiKey: string },
): Promise<string[]> {
  const effective = byok
    ? {
        ...settings,
        providers: [
          { id: "__probe__", name: "probe", models: [], ...byok },
        ],
        activeProvider: "__probe__",
      }
    : settings;
  const body = await getJson<{ models?: Array<{ id: string }> }>(effective, "/models");
  return (body.models ?? []).map((m) => m.id);
}

/** Recent memory sessions, newest first. */
export async function listSessions(settings: ClientSettings): Promise<SessionInfo[]> {
  const body = await getJson<{ sessions?: SessionInfo[] }>(settings, "/sessions");
  return body.sessions ?? [];
}

/** Visible turns (user/assistant) of a past session, for UI hydration. */
export async function getSessionTranscript(
  settings: ClientSettings,
  sessionId: string,
  limit = 200,
): Promise<Array<{ role: string; content: string; tool_calls?: unknown[] }>> {
  const body = await getJson<{
    entries?: Array<{ role: string; content: string; tool_calls?: unknown[] }>;
  }>(settings, `/sessions/${encodeURIComponent(sessionId)}/transcript?limit=${limit}`);
  return body.entries ?? [];
}
