# TradingAgents Web (local MVP)

A local dashboard around the existing TradingAgents LangGraph engine. It adds run configuration, a persistent run queue/history, stage progress, and report viewing. It does not place trades. Credentials typed for a single test or run remain transient and are never stored. Saved custom endpoint profile keys are the exception: they are stored in the operating-system credential manager (macOS Keychain on macOS), while profile metadata is kept in SQLite; keys are never returned to the browser/API or stored in SQLite.

## Start

From the repository root:

```bash
python -m pip install -e '.[web]'
uvicorn webapp.app:app --host 127.0.0.1 --port 8765
```

Open `http://127.0.0.1:8765`. Keep the host bound to `127.0.0.1`; this MVP has no user authentication and is intended for local use only.

Set provider credentials in the repository `.env` (or environment) before starting, or enter a credential in the dashboard's password-masked field. A credential entered directly is used for endpoint checks and the queued run, applied to the worker's environment only while processing, then discarded; a successful run submission clears the form field. It is not stored in the run record, SQLite, browser storage, or API responses. To retain a custom OpenAI-compatible endpoint and API key, use **Saved custom OpenAI-compatible endpoint**: profile details live in SQLite and the key is stored using the OS keyring (macOS Keychain on this machine). macOS may prompt you to authorize the dashboard's Python interpreter on first access. The key is never returned to the browser; leave the key field blank to keep a saved key, or check the clear option when updating to remove it. Environment-configured credentials remain a fallback when the field is blank. Keep the app bound to `127.0.0.1` and do not expose it to a network: this MVP has no user authentication. Select a cloud provider and its model IDs, or choose a local endpoint preset: Ollama (`127.0.0.1:11434`), LM Studio (`127.0.0.1:1234`), or vLLM (`127.0.0.1:8000`). Custom OpenAI-compatible URLs are also supported; enter model IDs served by that endpoint. The endpoint must be reachable from the machine running TradingAgents. Runs are processed serially because the core engine uses process-level configuration and per-ticker/date checkpoints. Ollama is keyless; use OpenAI-compatible for a local server requiring authentication. Bedrock run credentials are supported, but the non-generative endpoint probe does not support Bedrock.

## What is included

- Ticker/date validation, analyst selection, model/provider selection, language and debate settings.
- A “Test endpoint” check that queries the selected provider’s model-list endpoint without sending a model prompt or consuming generation tokens; local/custom URLs are checked through the OpenAI-compatible API.
- Saved custom OpenAI-compatible endpoint profiles with create/update/delete controls; profile metadata is stored in SQLite and API keys use the operating-system keyring, with a fail-closed check against insecure keyring backends.
- SQLite-backed run history at `~/.tradingagents/web/runs.sqlite3`.
- One background worker and polling progress updates.
- LangGraph checkpointing by default; the engine clears a checkpoint after a successful run.
- Incremental Markdown report output at `~/.tradingagents/logs/web_runs/<run-id>/`.
- Existing engine artifacts and decision memory continue to use the normal `~/.tradingagents` locations.

Server restarts mark queued/running jobs as interrupted. For checkpoint-enabled failed or interrupted runs, use **Resume** in the history row; it creates a new run with the original settings and continues from the engine checkpoint if one is available (otherwise it starts from the beginning). Re-enter the provider credential if needed because credentials are transient. **Delete** removes only the history record; report files and checkpoints are left on disk. Active queued/running jobs cannot be deleted from history.

## Verification

Run the web API/dashboard tests without making an LLM request:

```bash
.venv/bin/python -m unittest webapp.tests.test_webapp -v
```

The web stack is an optional extra; the core CLI does not require FastAPI or Uvicorn.
