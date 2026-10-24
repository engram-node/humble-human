/**
 * Settings popover: connection, BYOK provider profiles, accent colour, and
 * session management — everything configurable from the UI itself.
 *
 * Patterns borrowed from the xyne desktop app (xynex): provider profiles with
 * verify-before-save against the live /models roster, masked API keys with
 * reveal, and a session switcher listed newest-first.
 */

import { useEffect, useRef, useState } from "react";
import {
  Check,
  ChevronDown,
  ChevronRight,
  Eye,
  EyeOff,
  History,
  KeyRound,
  Loader2,
  Palette,
  Plus,
  Trash2,
  X,
} from "lucide-react";
import type { ClientSettings } from "../api/client";
import { listModels } from "../api/client";
import type { ProviderProfile, SessionInfo } from "../types";

const ACCENT_PRESETS = [
  "#34d399", // emerald
  "#38bdf8", // sky
  "#a78bfa", // violet
  "#fbbf24", // amber
  "#fb7185", // rose
  "#fb923c", // orange
  "#f4f4f5", // neutral (fully monochrome)
];
const ACCENT_RE = /^#[0-9a-fA-F]{6}$/;

interface SettingsPanelProps {
  settings: ClientSettings;
  sessions: SessionInfo[];
  sessionsLoading: boolean;
  currentSessionId: string;
  onSave: (s: ClientSettings) => void;
  onClose: () => void;
  onSwitchSession: (id: string) => void;
  onNewSession: () => void;
}

type VerifyState =
  | { status: "idle" }
  | { status: "busy" }
  | { status: "ok"; msg: string }
  | { status: "err"; msg: string };

function newProfile(): ProviderProfile {
  return {
    id: `p_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 7)}`,
    name: "",
    baseUrl: "",
    apiKey: "",
    models: [],
  };
}

function ProviderEditor({
  profile,
  draft,
  onChange,
  onRemove,
}: {
  profile: ProviderProfile;
  draft: ClientSettings;
  onChange: (p: ProviderProfile) => void;
  onRemove: () => void;
}) {
  const [open, setOpen] = useState(!profile.name && !profile.baseUrl);
  const [reveal, setReveal] = useState(false);
  const [verify, setVerify] = useState<VerifyState>({ status: "idle" });
  const busy = verify.status === "busy";

  const patch = (p: Partial<ProviderProfile>) =>
    onChange({ ...profile, ...p });

  const verifyNow = async () => {
    setVerify({ status: "busy" });
    try {
      const ids = await listModels(draft, {
        baseUrl: profile.baseUrl,
        apiKey: profile.apiKey,
      });
      patch({ models: ids.slice(0, 100) });
      setVerify({ status: "ok", msg: `✓ ${ids.length} model${ids.length === 1 ? "" : "s"}` });
    } catch (e) {
      setVerify({
        status: "err",
        msg: e instanceof Error ? e.message : "Verification failed",
      });
    }
  };

  return (
    <div className="rounded-lg border border-zinc-700/80 bg-zinc-900/60">
      <button
        type="button"
        onClick={() => setOpen(!open)}
        className="flex w-full items-center gap-2 px-2.5 py-2 text-left"
      >
        {open ? <ChevronDown size={13} /> : <ChevronRight size={13} />}
        <span className="truncate text-xs font-medium text-zinc-200">
          {profile.name || "Untitled provider"}
        </span>
        <span className="ml-auto max-w-32 truncate text-[10px] text-zinc-500">
          {profile.baseUrl || "env default"}
        </span>
      </button>
      {open && (
        <div className="space-y-2 px-2.5 pb-2.5">
          <input
            value={profile.name}
            onChange={(e) => patch({ name: e.target.value })}
            placeholder="Provider name (e.g. Kimi)"
            className="w-full rounded-lg border border-zinc-700 bg-zinc-950 px-2.5 py-1.5 text-xs text-zinc-200 outline-none focus:border-accent/50"
          />
          <input
            value={profile.baseUrl}
            onChange={(e) => patch({ baseUrl: e.target.value })}
            placeholder="Base URL, e.g. https://api.moonshot.ai/v1"
            className="w-full rounded-lg border border-zinc-700 bg-zinc-950 px-2.5 py-1.5 font-mono text-xs text-zinc-200 outline-none focus:border-accent/50"
          />
          <div className="relative">
            <input
              type={reveal ? "text" : "password"}
              value={profile.apiKey}
              onChange={(e) => patch({ apiKey: e.target.value })}
              placeholder={
                profile.apiKey
                  ? ""
                  : "API key — leave empty to use proxy env"
              }
              className="w-full rounded-lg border border-zinc-700 bg-zinc-950 px-2.5 py-1.5 pr-8 font-mono text-xs text-zinc-200 outline-none focus:border-accent/50"
            />
            <button
              type="button"
              title={reveal ? "Hide key" : "Reveal key"}
              onClick={() => setReveal(!reveal)}
              className="absolute right-1.5 top-1/2 -translate-y-1/2 rounded p-1 text-zinc-500 hover:text-zinc-300"
            >
              {reveal ? <EyeOff size={12} /> : <Eye size={12} />}
            </button>
          </div>
          <textarea
            value={profile.models.join(", ")}
            onChange={(e) =>
              patch({
                models: e.target.value
                  .split(/[\n,]+/)
                  .map((m) => m.trim())
                  .filter(Boolean),
              })
            }
            placeholder="Models, comma-separated (filled automatically by Verify)"
            rows={2}
            className="w-full resize-none rounded-lg border border-zinc-700 bg-zinc-950 px-2.5 py-1.5 font-mono text-xs text-zinc-200 outline-none focus:border-accent/50"
          />
          {verify.status !== "idle" && (
            <p
              className={`break-words text-[10px] ${
                verify.status === "ok" ? "text-accent" : "text-zinc-400"
              }`}
            >
              {busy ? "Checking…" : (
                verify.status === "ok" || verify.status === "err" ? verify.msg : ""
              )}
            </p>
          )}
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={verifyNow}
              disabled={busy || !profile.baseUrl}
              title="Verify key by fetching the live model roster"
              className="flex items-center gap-1 rounded-lg border border-zinc-700 px-2.5 py-1 text-[11px] text-zinc-300 hover:border-accent/50 disabled:cursor-not-allowed disabled:opacity-40"
            >
              {busy ? (
                <Loader2 size={11} className="animate-spin" />
              ) : (
                <Check size={11} />
              )}
              Verify
            </button>
            <button
              type="button"
              onClick={onRemove}
              title="Delete profile"
              className="ml-auto rounded-lg p-1.5 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200"
            >
              <Trash2 size={12} />
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

export default function SettingsPanel(props: SettingsPanelProps) {
  const {
    settings,
    sessions,
    sessionsLoading,
    currentSessionId,
    onSave,
    onClose,
    onSwitchSession,
    onNewSession,
  } = props;

  const [draft, setDraft] = useState<ClientSettings>(settings);
  const [customAccent, setCustomAccent] = useState(
    ACCENT_PRESETS.includes(settings.accent) ? "" : settings.accent,
  );
  const sessionsRef = useRef<HTMLDivElement>(null);

  useEffect(() => setDraft(settings), [settings]);

  const saveAccent = (hex: string) => {
    if (ACCENT_RE.test(hex)) setDraft({ ...draft, accent: hex.toLowerCase() });
  };

  const upsertProfile = (p: ProviderProfile) =>
    setDraft({
      ...draft,
      providers: draft.providers.some((q) => q.id === p.id)
        ? draft.providers.map((q) => (q.id === p.id ? p : q))
        : [...draft.providers, p],
    });

  const removeProfile = (id: string) =>
    setDraft({
      ...draft,
      providers: draft.providers.filter((p) => p.id !== id),
      activeProvider: draft.activeProvider === id ? "" : draft.activeProvider,
    });

  const sessionTitle = (s: SessionInfo) =>
    s.name || s.first_message || "Untitled chat";

  return (
    <>
      <div className="fixed inset-0 z-40" onClick={onClose} />
      <div className="absolute right-2 top-12 z-50 w-[26rem] max-w-[calc(100vw-1rem)] rounded-xl border border-zinc-700 bg-zinc-900 shadow-2xl shadow-black/60">
        <div className="flex max-h-[80vh] flex-col">
          {/* header */}
          <div className="flex shrink-0 items-center justify-between border-b border-zinc-800 px-4 py-3">
            <h2 className="text-sm font-semibold text-zinc-200">Settings</h2>
            <button
              type="button"
              onClick={onClose}
              className="rounded p-1 text-zinc-500 hover:bg-zinc-800"
            >
              <X size={14} />
            </button>
          </div>

          <div className="min-h-0 flex-1 space-y-5 overflow-y-auto px-4 py-3">
            {/* ---- Connection ---- */}
            <section>
              <h3 className="mb-2 flex items-center gap-1.5 text-[11px] font-medium uppercase tracking-wide text-zinc-500">
                Connection
              </h3>
              <input
                value={draft.baseUrl}
                onChange={(e) => setDraft({ ...draft, baseUrl: e.target.value })}
                placeholder="http://localhost:8000/v1"
                className="mb-2 w-full rounded-lg border border-zinc-700 bg-zinc-950 px-2.5 py-1.5 font-mono text-xs text-zinc-200 outline-none focus:border-accent/50"
              />
              <input
                type="password"
                value={draft.apiKey}
                onChange={(e) => setDraft({ ...draft, apiKey: e.target.value })}
                placeholder="Proxy bearer key (optional for local)"
                className="w-full rounded-lg border border-zinc-700 bg-zinc-950 px-2.5 py-1.5 font-mono text-xs text-zinc-200 outline-none focus:border-accent/50"
              />
              <p className="mt-1.5 text-[10px] leading-relaxed text-zinc-600">
                Point at proxy_server.py (local) or a remote deployment exposing
                an OpenAI-compatible /chat/completions.
              </p>
            </section>

            {/* ---- Providers (BYOK) ---- */}
            <section>
              <h3 className="mb-2 flex items-center gap-1.5 text-[11px] font-medium uppercase tracking-wide text-zinc-500">
                <KeyRound size={11} /> Providers
              </h3>
              <select
                value={draft.activeProvider}
                onChange={(e) =>
                  setDraft({ ...draft, activeProvider: e.target.value })
                }
                title="Active provider"
                className="mb-2 w-full cursor-pointer rounded-lg border border-zinc-700 bg-zinc-800 px-2.5 py-1.5 text-xs text-zinc-200 outline-none hover:border-zinc-600"
              >
                <option value="">Proxy env config (default)</option>
                {draft.providers.map((p) => (
                  <option key={p.id} value={p.id}>
                    {p.name || p.baseUrl || "Untitled provider"}
                  </option>
                ))}
              </select>
              <div className="space-y-2">
                {draft.providers.map((p) => (
                  <ProviderEditor
                    key={p.id}
                    profile={p}
                    draft={draft}
                    onChange={upsertProfile}
                    onRemove={() => removeProfile(p.id)}
                  />
                ))}
                <button
                  type="button"
                  onClick={() => {
                    const p = newProfile();
                    setDraft({
                      ...draft,
                      providers: [...draft.providers, p],
                      activeProvider: draft.activeProvider || p.id,
                    });
                  }}
                  className="flex items-center gap-1.5 rounded-lg border border-dashed border-zinc-700 px-2.5 py-1.5 text-[11px] text-zinc-400 hover:border-zinc-500 hover:text-zinc-200"
                >
                  <Plus size={12} /> Add provider
                </button>
              </div>
              <p className="mt-1.5 text-[10px] leading-relaxed text-zinc-600">
                Keys stay in this browser and are sent per request to the proxy
                as override headers. “Verify” fetches the provider's live model
                list and pre-fills the models field.
              </p>
            </section>

            {/* ---- Appearance ---- */}
            <section>
              <h3 className="mb-2 flex items-center gap-1.5 text-[11px] font-medium uppercase tracking-wide text-zinc-500">
                <Palette size={11} /> Accent colour
              </h3>
              <div className="flex flex-wrap items-center gap-2">
                {ACCENT_PRESETS.map((hex) => (
                  <button
                    key={hex}
                    type="button"
                    title={hex}
                    onClick={() => {
                      setDraft({ ...draft, accent: hex });
                      setCustomAccent("");
                    }}
                    className={`grid h-7 w-7 place-items-center rounded-full border transition ${
                      draft.accent === hex && !customAccent
                        ? "border-zinc-200 ring-2 ring-zinc-200/40"
                        : "border-zinc-700 hover:border-zinc-500"
                    }`}
                    style={{ backgroundColor: hex }}
                  >
                    {draft.accent === hex && !customAccent && (
                      <Check size={12} className="text-zinc-950" />
                    )}
                  </button>
                ))}
                <div className="flex items-center gap-1.5">
                  <input
                    value={customAccent}
                    onChange={(e) => {
                      setCustomAccent(e.target.value);
                      saveAccent(e.target.value);
                    }}
                    placeholder="#rrggbb"
                    className="w-24 rounded-lg border border-zinc-700 bg-zinc-950 px-2.5 py-1.5 font-mono text-xs text-zinc-200 outline-none focus:border-accent/50"
                  />
                  <span
                    className="h-7 w-7 rounded-full border border-zinc-700"
                    style={{
                      backgroundColor: ACCENT_RE.test(customAccent)
                        ? customAccent
                        : "transparent",
                    }}
                    title="Preview"
                  />
                </div>
              </div>
              <p className="mt-1.5 text-[10px] leading-relaxed text-zinc-600">
                One accent for everything interactive — the rest stays
                monochrome. Applies after Save.
              </p>
            </section>

            {/* ---- Sessions ---- */}
            <section ref={sessionsRef}>
              <h3 className="mb-2 flex items-center gap-1.5 text-[11px] font-medium uppercase tracking-wide text-zinc-500">
                <History size={11} /> Sessions
              </h3>
              <button
                type="button"
                onClick={() => {
                  onNewSession();
                  onClose();
                }}
                className="mb-2 flex items-center gap-1.5 rounded-lg border border-dashed border-zinc-700 px-2.5 py-1.5 text-[11px] text-zinc-400 hover:border-accent/50 hover:text-zinc-200"
              >
                <Plus size={12} /> New session
              </button>
              <div className="max-h-48 space-y-1 overflow-y-auto">
                {sessionsLoading && (
                  <p className="flex items-center gap-1.5 px-1 py-2 text-[11px] text-zinc-500">
                    <Loader2 size={11} className="animate-spin" /> Loading…
                  </p>
                )}
                {!sessionsLoading && sessions.length === 0 && (
                  <p className="px-1 py-2 text-[11px] text-zinc-600">
                    No sessions yet — chat one message and it'll appear here.
                  </p>
                )}
                {sessions.map((s) => {
                  const active = s.id === currentSessionId;
                  return (
                    <button
                      key={s.id}
                      type="button"
                      disabled={active}
                      onClick={() => {
                        onSwitchSession(s.id);
                        onClose();
                      }}
                      title={s.id}
                      className={`w-full rounded-lg border px-2.5 py-2 text-left transition ${
                        active
                          ? "border-accent/40 bg-accent/5"
                          : "border-transparent hover:border-zinc-700 hover:bg-zinc-800/60"
                      }`}
                    >
                      <div className="flex items-baseline gap-2">
                        <span className="min-w-0 truncate text-xs text-zinc-200">
                          {sessionTitle(s)}
                        </span>
                        <span className="ml-auto shrink-0 text-[10px] text-zinc-600">
                          {s.message_count} msgs
                        </span>
                      </div>
                      <div className="mt-0.5 text-[10px] text-zinc-600">
                        {new Date(s.modified_at * 1000).toLocaleString()}
                      </div>
                    </button>
                  );
                })}
              </div>
            </section>
          </div>

          {/* footer */}
          <div className="flex shrink-0 justify-end gap-2 border-t border-zinc-800 px-4 py-3">
            <button
              type="button"
              onClick={onClose}
              className="rounded-lg border border-zinc-700 px-3 py-1.5 text-xs text-zinc-400 hover:bg-zinc-800"
            >
              Cancel
            </button>
            <button
              type="button"
              onClick={() => onSave(draft)}
              className="rounded-lg bg-accent px-3 py-1.5 text-xs font-medium text-zinc-950 hover:brightness-110"
            >
              Save
            </button>
          </div>
        </div>
      </div>
    </>
  );
}
