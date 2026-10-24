# Humble Human — GUI

Cross-platform (desktop + mobile) web GUI for **Humble Human**, the AI coding
workspace with infinite long-term memory backed by `memory_manager.py`.

## Stack

Vite · React 19 · TypeScript · Tailwind CSS 4 · `@monaco-editor/react` · `lucide-react`

## Setup

```bash
npm install
npm run dev        # http://localhost:5173
```

Production build: `npm run build` (type-checks first), preview with `npm run preview`.

## Layout

- **Desktop (≥1024px):** split view — left pane (35%) chat + memory-activity
  stream, right pane (65%) Monaco editor / diff viewer with file tabs.
- **Mobile (<1024px):** single pane with a bottom tab bar switching between
  Chat and Editor.

## Backend: `proxy_server.py`

The GUI talks to `proxy_server.py` — a FastAPI wrapper that runs the same
LLM + memory tool loop as `main.py`, over HTTP:

```bash
.venv/bin/pip install -r requirements.txt
export OPENAI_API_KEY="sk-..."            # or use mock mode (below)
export LLM_MODEL="deepseek-chat"         # default upstream model
.venv/bin/python proxy_server.py          # http://127.0.0.1:8000
HH_MOCK_LLM=1 .venv/bin/python proxy_server.py   # offline demo mode
```

**Security note:** `/run` executes local commands, so CORS is restricted to
the GUI dev origins (localhost:5173/5174) — without this, any website you
visit could drive your machine through the proxy. Add origins via
`HH_CORS_ORIGINS="https://gui.example.com,..."` when deploying remotely.

Endpoints:
- `POST /v1/chat/completions` — OpenAI-compatible; `stream:true` → SSE,
  `stream:false` → aggregated JSON. Extra field `session_id` pins the request
  to a persistent memory session (the GUI generates and reuses one).
- `GET /health` — liveness + config.
- `POST /run` — local command execution for the GUI's **Run Command** button
  (timeout 20s, disable with `HH_DISABLE_RUN=1`).
- `GET /commands` — file-based slash commands (see below).
- `GET /models` — live model roster of the active upstream (drives the GUI's
  provider **Verify** button and model dropdown).
- `GET /sessions` — recent memory sessions, newest first.
- `GET /sessions/{id}/transcript?limit=100` — visible turns for UI hydration
  when switching sessions.

### BYOK: provider profiles in the GUI

The GUI stores provider profiles (base URL + key + models) in the browser and
sends the active one **per request** as `X-HH-Base-URL` / `X-HH-API-Key`
headers. The proxy then talks to that upstream instead of the env-configured
default — manage keys entirely from the Settings gear, no env vars needed.
The **Verify** button fetches the provider's live `/models` roster and
pre-fills the model dropdown.

### Slash commands (file-based prompt templates)

`.md` files in either of these dirs become chat commands (project shadows
global):
- `<project>/.hh/prompts/`
- `~/.cli_harness/prompts/` (harness home)

Optional front matter: `---\ndescription: …\nargument-hint: "[args]"\n---`.
Arguments: `$1…$9`, `$ARGUMENTS` / `$@` (all), `${2:-default}`. Example in
`.hh/prompts/review.md`. Type `/` in the composer for autocomplete — the proxy
expands the template server-side before ingest; unknown commands return 400.

Protocol extensions handled by the GUI (standard clients can ignore them):
tool results echo as `delta.role="tool"` chunks, and proxy notices
(compaction, tool-loop bound) arrive as `delta.role="hh_system"` chunks.

With `HH_MOCK_LLM=1` no API key is needed: a heuristic mock streams answers
and exercises the real memory tool loop — ideal for GUI development. Both
`main.py --self-test` and `proxy_server.py --self-test` run fully offline.

## Backend contract

The UI streams from `http://localhost:8000/v1/chat/completions` by default.
Click the gear icon (top right) to point it at a remote deployment or set an
API key; settings persist in `localStorage`.

Requests are sent with `stream: true` (SSE) and the two memory tool schemas:
- `search_memory_summaries(query, limit)`
- `inspect_session_transcript(session_id, limit_lines)`

Streamed `tool_calls` deltas render live as collapsible memory-activity cards
in the chat; echoed tool results fill in matched session ids and snippets.

## Structure

```
src/
  api/client.ts         streaming SSE client + settings persistence
  components/ChatPanel.tsx    transcript, memory tool cards, composer
  components/EditorPanel.tsx  file tabs, Monaco editor/diff, actions
  monaco-setup.ts       bundles monaco from node_modules (no CDN)
  types.ts              domain types + memory-result parsing
  App.tsx               responsive shell + streaming orchestration
```
