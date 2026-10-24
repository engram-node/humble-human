/**
 * Bundle Monaco from node_modules instead of the default jsDelivr CDN,
 * so the editor works offline and in air-gapped deployments.
 * Vite's `?worker` imports produce the required web workers.
 */
import * as monaco from "monaco-editor";
import { loader } from "@monaco-editor/react";
import editorWorker from "monaco-editor/esm/vs/editor/editor.worker?worker";
import jsonWorker from "monaco-editor/esm/vs/language/json/json.worker?worker";
import cssWorker from "monaco-editor/esm/vs/language/css/css.worker?worker";
import htmlWorker from "monaco-editor/esm/vs/language/html/html.worker?worker";
import tsWorker from "monaco-editor/esm/vs/language/typescript/ts.worker?worker";

self.MonacoEnvironment = {
  getWorker(_workerId: string, label: string): Worker {
    switch (label) {
      case "json":
        return new jsonWorker();
      case "css":
      case "scss":
      case "less":
        return new cssWorker();
      case "html":
      case "handlebars":
      case "razor":
        return new htmlWorker();
      case "typescript":
      case "javascript":
        return new tsWorker();
      default:
        return new editorWorker();
    }
  },
};

loader.config({ monaco });

// ---------------------------------------------------------------------------
// hh-dark theme: monochrome + one accent color (re-definable at runtime)
// ---------------------------------------------------------------------------

export const HH_THEME = "hh-dark";

/** Append an alpha byte to a #rrggbb hex color. */
function hexA(hex: string, alpha: string): string {
  return hex.length === 7 ? `${hex}${alpha}` : hex;
}

/**
 * (Re)define the hh-dark theme from an accent color, then switch to it.
 * Called on startup and whenever the user picks a different accent.
 */
export function applyMonacoAccent(accent: string): void {
  monaco.editor.defineTheme(HH_THEME, {
    base: "vs-dark",
    inherit: true,
    rules: [
      // Monochrome tokens — everything is a shade of gray.
      { token: "", foreground: "d4d4d8" },
      { token: "comment", foreground: "3f3f46", fontStyle: "italic" },
      { token: "string", foreground: "a1a1aa" },
      { token: "string.escape", foreground: "71717a" },
      { token: "number", foreground: "e4e4e7" },
      { token: "keyword", foreground: "fafafa", fontStyle: "bold" },
      { token: "operator", foreground: "71717a" },
      { token: "type", foreground: "e4e4e7" },
      { token: "identifier", foreground: "d4d4d8" },
      { token: "delimiter", foreground: "52525b" },
      { token: "attribute.name", foreground: "e4e4e7" },
      { token: "attribute.value", foreground: "a1a1aa" },
      { token: "tag", foreground: "fafafa", fontStyle: "bold" },
      { token: "regexp", foreground: "a1a1aa" },
    ],
    colors: {
      "editor.background": "#09090b",
      "editor.foreground": "#d4d4d8",
      "editor.lineHighlightBackground": "#ffffff08",
      "editorLineNumber.foreground": "#3f3f46",
      "editorLineNumber.activeForeground": "#a1a1aa",
      "editorCursor.foreground": accent,
      "editor.selectionBackground": hexA(accent, "2e"),
      "editor.inactiveSelectionBackground": hexA(accent, "1a"),
      "editorSuggestWidget.selectedBackground": hexA(accent, "26"),
      "editorSuggestWidget.highlightForeground": accent,
      "editorBracketHighlight.foreground1": accent,
      // Diff viewer: inserted = accent tint, removed = dim gray hatch.
      "diffEditor.insertedTextBackground": hexA(accent, "26"),
      "diffEditor.insertedLineBackground": hexA(accent, "14"),
      "diffEditor.removedTextBackground": "#52525b30",
      "diffEditor.removedLineBackground": "#27272a66",
      "diffEditor.diagonalFill": "#52525b80",
      "editorGutter.addedBackground": accent,
      "editorGutter.deletedBackground": "#52525b",
      "editorGutter.modifiedBackground": "#71717a",
      "scrollbarSlider.background": "#3f3f4680",
      "scrollbarSlider.hoverBackground": "#52525b80",
      "minimap.background": "#09090b",
    },
  });
  monaco.editor.setTheme(HH_THEME);
}

/** Read --accent from the DOM and apply it to Monaco (best effort). */
export function syncMonacoAccentFromDom(): void {
  const v = getComputedStyle(document.documentElement)
    .getPropertyValue("--accent")
    .trim();
  if (/^#[0-9a-fA-F]{6}$/.test(v)) applyMonacoAccent(v);
}
