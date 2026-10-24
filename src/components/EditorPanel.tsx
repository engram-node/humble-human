import { useEffect, useState } from "react";
import Editor, { DiffEditor } from "@monaco-editor/react";
import {
  Check,
  Copy,
  FileCode,
  GitPullRequest,
  Play,
  X,
} from "lucide-react";
import { HH_THEME } from "../monaco-setup";
import type { OpenFile } from "../types";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const EXT_TO_LANG: Record<string, string> = {
  ts: "typescript",
  tsx: "typescript",
  js: "javascript",
  jsx: "javascript",
  py: "python",
  json: "json",
  md: "markdown",
  css: "css",
  scss: "scss",
  html: "html",
  yml: "yaml",
  yaml: "yaml",
  toml: "ini",
  sh: "shell",
  rs: "rust",
  go: "go",
  java: "java",
  c: "c",
  cpp: "cpp",
};

export function languageFromFilename(name: string): string {
  const ext = name.split(".").pop()?.toLowerCase() ?? "";
  return EXT_TO_LANG[ext] ?? "plaintext";
}

const EDITOR_OPTIONS = {
  fontSize: 13,
  fontFamily: "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace",
  lineNumbers: "on" as const,
  minimap: { enabled: false },
  scrollBeyondLastLine: false,
  automaticLayout: true,
  padding: { top: 12 },
  smoothScrolling: true,
  renderWhitespace: "selection" as const,
  tabSize: 2,
};

const DIFF_OPTIONS = {
  ...EDITOR_OPTIONS,
  renderSideBySide: true,
  originalEditable: false,
  readOnly: false,
  automaticLayout: true,
};

// ---------------------------------------------------------------------------
// Editor panel
// ---------------------------------------------------------------------------

export interface EditorPanelProps {
  files: OpenFile[];
  activeFileId: string | null;
  onSelectFile: (id: string) => void;
  onCloseFile: (id: string) => void;
  onChangeFile: (id: string, content: string) => void;
  /** Accept the pending AI diff for a file (discard originalContent). */
  onApplyDiff: (id: string) => void;
  onRunCommand: (file: OpenFile) => void;
}

export default function EditorPanel({
  files,
  activeFileId,
  onSelectFile,
  onCloseFile,
  onChangeFile,
  onApplyDiff,
  onRunCommand,
}: EditorPanelProps) {
  const active = files.find((f) => f.id === activeFileId) ?? null;
  const [view, setView] = useState<"code" | "diff">("code");
  const [copied, setCopied] = useState(false);

  // Reset to code view whenever the active tab changes.
  useEffect(() => {
    setView("code");
    setCopied(false);
  }, [activeFileId]);

  const hasDiff = active?.originalContent != null;

  const copyActive = async () => {
    if (!active) return;
    try {
      await navigator.clipboard.writeText(active.content);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1500);
    } catch {
      // Clipboard API unavailable (non-secure context) — silently ignore.
    }
  };

  return (
    <div className="flex h-full min-h-0 flex-col bg-zinc-950">
      {/* Tab bar */}
      <div className="flex h-10 shrink-0 items-stretch overflow-x-auto border-b border-zinc-800">
        {files.length === 0 && (
          <span className="flex items-center px-3 text-[11px] italic text-zinc-600">
            No open files
          </span>
        )}
        {files.map((file) => {
          const isActive = file.id === activeFileId;
          return (
            <div
              key={file.id}
              className={`group flex shrink-0 cursor-pointer items-center gap-1.5 border-r border-zinc-800 px-3 text-xs ${
                isActive
                  ? "bg-zinc-900 text-zinc-100 shadow-[inset_0_2px_0_0_var(--accent)]"
                  : "text-zinc-500 hover:bg-zinc-900/60 hover:text-zinc-300"
              }`}
              onClick={() => onSelectFile(file.id)}
              role="tab"
              aria-selected={isActive}
            >
              <FileCode size={12} className={file.originalContent != null ? "text-accent" : "text-zinc-500"} />
              <span className="max-w-32 truncate">{file.name}</span>
              <button
                type="button"
                title="Close file"
                onClick={(e) => {
                  e.stopPropagation();
                  onCloseFile(file.id);
                }}
                className="rounded p-0.5 opacity-0 transition hover:bg-zinc-700/60 group-hover:opacity-100"
              >
                <X size={11} />
              </button>
            </div>
          );
        })}
      </div>

      {/* Toolbar */}
      <div className="flex h-11 shrink-0 items-center gap-2 border-b border-zinc-800 px-3">
        {hasDiff && (
          <div className="flex overflow-hidden rounded-lg border border-zinc-700 text-[11px]">
            <button
              type="button"
              onClick={() => setView("code")}
              className={`px-2.5 py-1 transition ${
                view === "code" ? "bg-zinc-700 text-zinc-100" : "text-zinc-400 hover:bg-zinc-800"
              }`}
            >
              Code
            </button>
            <button
              type="button"
              onClick={() => setView("diff")}
              className={`px-2.5 py-1 transition ${
                view === "diff" ? "bg-zinc-700 text-zinc-100" : "text-zinc-400 hover:bg-zinc-800"
              }`}
            >
              Diff
            </button>
          </div>
        )}
        <div className="ml-auto flex items-center gap-1.5">
          <button
            type="button"
            onClick={copyActive}
            disabled={!active}
            className="flex items-center gap-1.5 rounded-lg border border-zinc-700 px-2.5 py-1.5 text-[11px] text-zinc-300 transition hover:bg-zinc-800 disabled:cursor-not-allowed disabled:opacity-40"
          >
            {copied ? <Check size={12} className="text-accent" /> : <Copy size={12} />}
            {copied ? "Copied" : "Copy Code"}
          </button>
          <button
            type="button"
            onClick={() => active && onApplyDiff(active.id)}
            disabled={!hasDiff}
            title={hasDiff ? "Accept pending AI edits" : "No pending diff"}
            className="flex items-center gap-1.5 rounded-lg border border-accent/40 bg-accent/10 px-2.5 py-1.5 text-[11px] text-accent transition hover:bg-accent/20 disabled:cursor-not-allowed disabled:border-zinc-700 disabled:bg-transparent disabled:text-zinc-500 disabled:opacity-40"
          >
            <GitPullRequest size={12} />
            Apply Diffs
          </button>
          <button
            type="button"
            onClick={() => active && onRunCommand(active)}
            disabled={!active}
            className="flex items-center gap-1.5 rounded-lg bg-accent px-2.5 py-1.5 text-[11px] font-medium text-zinc-950 transition hover:brightness-110 disabled:cursor-not-allowed disabled:opacity-40"
          >
            <Play size={12} />
            Run Command
          </button>
        </div>
      </div>

      {/* Editor / diff viewer */}
      <div className="min-h-0 flex-1">
        {active ? (
          view === "diff" && hasDiff ? (
            <DiffEditor
              height="100%"
              theme={HH_THEME}
              language={active.language}
              original={active.originalContent}
              modified={active.content}
              options={DIFF_OPTIONS}
            />
          ) : (
            <Editor
              height="100%"
              theme={HH_THEME}
              language={active.language}
              value={active.content}
              onChange={(value) => onChangeFile(active.id, value ?? "")}
              options={EDITOR_OPTIONS}
            />
          )
        ) : (
          <div className="grid h-full place-items-center p-6 text-center">
            <div>
              <FileCode size={28} className="mx-auto mb-2 text-zinc-700" />
              <p className="text-sm text-zinc-500">No file open</p>
              <p className="mt-1 text-xs text-zinc-600">
                Ask the agent to open or create a file to start editing.
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
