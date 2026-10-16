# TradingAgents Web (local MVP)

A local dashboard around the existing TradingAgents LangGraph engine. It adds run configuration, a persistent run queue/history, stage progress, and report viewing. It does not place trades. Provider credentials can be entered in a password-masked field and are sent to the local server for endpoint checks or run processing; they are never returned in dashboard/API responses or saved in SQLite/browser storage.

## Start

From the repository root:

```bash
python -m pip install -e '.[web]'
uvicorn webapp.app:app --host 127.0.0.1 --port 8765
```

Open `http://127.0.0.1:8765`. Keep the host bound to `127.0.0.1`; this MVP has no user authentication and is intended for local use only.

Set provider credentials in the repository `.env` (or environment) before starting, or enter the credential in the dashboard's password-masked field. A dashboard-entered credential is used for endpoint checks and the queued run, applied to the worker's environment only while processing, then discarded; a successful run submission clears the form field. It is not stored in the run record, SQLite, browser storage, or API responses. The web app does not intentionally log the value and redacts it from persisted run errors. Environment-configured credentials remain a fallback when the field is blank. Keep the app bound to `127.0.0.1` and do not expose it to a network: this MVP has no user authentication. Select a cloud provider and its model IDs, or choose a local endpoint preset: Ollama (`127.0.0.1:11434`), LM Studio (`127.0.0.1:1234`), or vLLM (`127.0.0.1:8000`). Custom OpenAI-compatible URLs are also supported; enter model IDs served by that endpoint. The endpoint must be reachable from the machine running TradingAgents. Runs are processed serially because the core engine uses process-level configuration and per-ticker/date checkpoints. Ollama is keyless; use OpenAI-compatible for a local server requiring authentication. Bedrock run credentials are supported, but the non-generative endpoint probe does not support Bedrock.

## What is included

- Ticker/date validation, analyst selection, model/provider selection, language and debate settings.
- A “Test endpoint” check that queries the selected provider’s model-list endpoint without sending a model prompt or consuming generation tokens; local/custom URLs are checked through the OpenAI-compatible API.
- SQLite-backed run history at `~/.tradingagents/web/runs.sqlite3`.
- One background worker and polling progress updates.
- LangGraph checkpointing by default; the engine clears a checkpoint after a successful run.
- Incremental Markdown report output at `~/.tradingagents/logs/web_runs/<run-id>/`.
- Existing engine artifacts and decision memory continue to use the normal `~/.tradingagents` locations.

A server restart marks queued/running jobs as interrupted. Checkpointed engine state is retained, but this MVP does not yet provide an explicit “resume this run” button; resubmit the same ticker/date and graph settings to use the engine checkpoint.

## Verification

Run the web API/dashboard tests without making an LLM request:

```bash
.venv/bin/python -m unittest webapp.tests.test_webapp -v
```

The web stack is an optional extra; the core CLI does not require FastAPI or Uvicorn.
