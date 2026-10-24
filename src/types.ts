/**
 * Shared domain types for the Humble Human GUI.
 */

export type ChatRole = "user" | "assistant" | "system";

export type ToolStatus = "running" | "done" | "error";

/** The two long-term-memory tools exposed by memory_manager.py. */
export const MEMORY_TOOL_NAMES = [
  "search_memory_summaries",
  "inspect_session_transcript",
] as const;

export function isMemoryTool(name: string): boolean {
  return (MEMORY_TOOL_NAMES as readonly string[]).includes(name);
}

export interface MemoryToolResult {
  /** Raw payload returned by the tool, as text. */
  raw: string;
  /** Session ids surfaced from the result (parsing is best-effort). */
  sessionIds: string[];
  /** Human-readable context snippets retrieved from memory. */
  snippets: string[];
}

export interface MemoryToolCall {
  id: string;
  /** "search_memory_summaries" | "inspect_session_transcript" | any future tool */
  name: string;
  /** Raw JSON argument string as streamed by the API. */
  argsText: string;
  /** Best-effort parsed arguments (may lag behind argsText while streaming). */
  args: Record<string, unknown>;
  result?: MemoryToolResult;
  status: ToolStatus;
}

export interface ChatMessage {
  id: string;
  role: ChatRole;
  content: string;
  /** Memory tool calls attached to assistant messages. */
  toolCalls?: MemoryToolCall[];
  /** Seed/demo messages render in the UI but are never sent to the API. */
  demo?: boolean;
  createdAt: number;
}

/** A file-based slash command discovered by the proxy (GET /commands). */
export interface SlashCommand {
  name: string;
  description: string;
  argument_hint: string;
  source: "global" | "project";
}

/** A session as listed by GET /sessions. */
export interface SessionInfo {
  id: string;
  name: string;
  message_count: number;
  first_message: string;
  modified_at: number;
}

/** A UI-managed provider profile (BYOK), sent per request to the proxy. */
export interface ProviderProfile {
  id: string;
  name: string;
  baseUrl: string;
  /** Empty = keep using the proxy's env-configured key. */
  apiKey: string;
  models: string[];
}

export interface OpenFile {
  id: string;
  name: string;
  language: string;
  content: string;
  /**
   * When set, the file has un-applied AI edits. The diff viewer compares
   * originalContent (before) against content (after).
   */
  originalContent?: string;
}

const SESSION_ID_RE =
  /\bses_[A-Za-z0-9_-]+|\bsession[_-][A-Za-z0-9_-]+|\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b/g;

/**
 * Best-effort extraction of session ids + context snippets from a memory
 * tool result. The proxy may return JSON (`{ results: [...] }`, `[...]`,
 * `{ sessions: [...] }`) or plain text; both are handled.
 */
export function parseMemoryResult(raw: string): MemoryToolResult {
  const sessionIds = Array.from(new Set(raw.match(SESSION_ID_RE) ?? []));

  let snippets: string[] = [];
  try {
    const parsed: unknown = JSON.parse(raw);
    const items: unknown[] = Array.isArray(parsed)
      ? parsed
      : typeof parsed === "object" && parsed !== null
        ? (((parsed as Record<string, unknown>).results ??
            (parsed as Record<string, unknown>).sessions ??
            (parsed as Record<string, unknown>).matches ??
            []) as unknown[])
        : [];
    snippets = items
      .map((item) => {
        if (typeof item === "string") return item;
        if (typeof item === "object" && item !== null) {
          const rec = item as Record<string, unknown>;
          return String(rec.summary ?? rec.text ?? rec.content ?? rec.snippet ?? "");
        }
        return "";
      })
      .filter((s) => s.trim().length > 0);
  } catch {
    // Not JSON — fall back to the most useful raw lines.
    snippets = raw
      .split("\n")
      .map((l) => l.trim())
      .filter((l) => l.length > 0)
      .slice(0, 5);
  }

  return { raw, sessionIds, snippets: snippets.slice(0, 5) };
}
