# TradingAgents Web (local MVP)

A local dashboard around the existing TradingAgents LangGraph engine. It adds run configuration, a persistent run queue/history, stage progress, and report viewing. It does not place trades or send credentials to the browser.

## Start

From the repository root:

```bash
python -m pip install -e '.[web]'
uvicorn webapp.app:app --host 127.0.0.1 --port 8765
```

Open `http://127.0.0.1:8765`. Keep the host bound to `127.0.0.1`; this MVP has no user authentication and is intended for local use only.

Set your provider credentials in the repository `.env` (or the environment) before starting. The dashboard never returns credential values. Select a cloud provider and its model IDs, or choose a local endpoint preset: Ollama (`127.0.0.1:11434`), LM Studio (`127.0.0.1:1234`), or vLLM (`127.0.0.1:8000`). Custom OpenAI-compatible URLs are also supported; enter model IDs served by that endpoint. The endpoint must be reachable from the machine running TradingAgents. Runs are processed serially because the core engine uses process-level configuration and per-ticker/date checkpoints.

## What is included

- Ticker/date validation, analyst selection, model/provider selection, language and debate settings.
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
