"""Local web UI job orchestration for TradingAgents."""
from __future__ import annotations

import json
import os
import queue
import sqlite3
import threading
import uuid
from datetime import date, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from pydantic import BaseModel, Field, SecretStr

from cli.models import AnalystType
from cli.prompts import detect_asset_type, normalize_ticker_symbol, is_valid_ticker_input
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.llm_clients.api_key_env import PROVIDER_API_KEY_ENV
from tradingagents.dataflows.config import run_config

from .secure_store import KeyringSecretStore, SecretStoreError


def configured_default_provider() -> str:
    """Prefer the configured default, falling back to OpenRouter when keyed."""
    provider = str(DEFAULT_CONFIG["llm_provider"]).lower()
    env_name = PROVIDER_API_KEY_ENV.get(provider)
    if env_name and not os.environ.get(env_name) and os.environ.get("OPENROUTER_API_KEY"):
        return "openrouter"
    return provider


def configured_default_models() -> tuple[str, str]:
    """Use models from the active config, with known-working OpenRouter defaults."""
    quick = DEFAULT_CONFIG["quick_think_llm"]
    deep = DEFAULT_CONFIG["deep_think_llm"]
    if configured_default_provider() == "openrouter":
        if not os.environ.get("TRADINGAGENTS_QUICK_THINK_LLM"):
            quick = "gpt-4o-mini"
        if not os.environ.get("TRADINGAGENTS_DEEP_THINK_LLM"):
            deep = "gpt-4o"
    return quick, deep


class RunRequest(BaseModel):
    ticker: str
    analysis_date: str = Field(default_factory=lambda: date.today().isoformat())
    analysts: list[str] = Field(default_factory=lambda: ["market", "social", "news", "fundamentals"])
    provider: str = Field(default_factory=lambda: configured_default_provider())
    quick_model: str = Field(default_factory=lambda: configured_default_models()[0])
    deep_model: str = Field(default_factory=lambda: configured_default_models()[1])
    backend_url: str | None = DEFAULT_CONFIG.get("backend_url")
    credential: SecretStr | None = Field(default=None, exclude=True, repr=False)
    saved_endpoint_id: str | None = None
    debate_rounds: int = Field(default=DEFAULT_CONFIG["max_debate_rounds"], ge=1, le=10)
    risk_rounds: int = Field(default=DEFAULT_CONFIG["max_risk_discuss_rounds"], ge=1, le=10)
    max_tokens: int | None = Field(default=DEFAULT_CONFIG.get("max_tokens"), ge=256, le=65536)
    output_language: str = DEFAULT_CONFIG["output_language"]
    checkpoint: bool = True


class SavedEndpointRequest(BaseModel):
    name: str
    base_url: str
    quick_model: str
    deep_model: str
    api_key: SecretStr | None = Field(default=None, exclude=True, repr=False)
    clear_api_key: bool = False


def _validate_base_url(value: str) -> str:
    base_url = value.strip().rstrip("/")
    try:
        parsed = urlsplit(base_url)
        parsed.port
    except ValueError as exc:
        raise ValueError("Endpoint URL has an invalid port.") from exc
    if (parsed.scheme not in {"http", "https"} or not parsed.hostname
            or parsed.username or parsed.password or parsed.query or parsed.fragment):
        raise ValueError("Endpoint URL must be an http:// or https:// base URL without credentials, query, or fragment.")
    return base_url


def validate_saved_endpoint(request: SavedEndpointRequest) -> SavedEndpointRequest:
    name = request.name.strip()
    quick_model = request.quick_model.strip()
    deep_model = request.deep_model.strip()
    base_url = _validate_base_url(request.base_url)
    if not name or len(name) > 80:
        raise ValueError("Endpoint profile name must contain 1–80 characters.")
    if not quick_model or not deep_model:
        raise ValueError("Quick and deep model IDs are required.")
    return request.model_copy(update={
        "name": name, "base_url": base_url,
        "quick_model": quick_model, "deep_model": deep_model,
    })


def validate_run_request(request: RunRequest) -> RunRequest:
    """Canonicalize and validate user-controlled run inputs before queueing."""
    ticker = request.ticker.strip()
    if not is_valid_ticker_input(ticker):
        raise ValueError("Enter a valid ticker symbol (letters, digits, . _ - ^ = only).")
    ticker = normalize_ticker_symbol(ticker)
    if not ticker or len(ticker) > 32 or not all(c.isalnum() or c in "._-^=" for c in ticker):
        raise ValueError("Ticker is not a safe supported symbol.")

    try:
        parsed = datetime.strptime(request.analysis_date, "%Y-%m-%d").date()
    except (TypeError, ValueError) as exc:
        raise ValueError("Analysis date must use YYYY-MM-DD format.") from exc
    if parsed.isoformat() != request.analysis_date:
        raise ValueError("Analysis date must use YYYY-MM-DD format.")
    if parsed > date.today():
        raise ValueError("Analysis date cannot be in the future.")

    analyst_keys = {item.value for item in AnalystType}
    if not request.analysts or len(set(request.analysts)) != len(request.analysts):
        raise ValueError("Select at least one analyst; duplicates are not allowed.")
    invalid = sorted(set(request.analysts) - analyst_keys)
    if invalid:
        raise ValueError(f"Unsupported analyst(s): {', '.join(invalid)}")

    providers = {key.lower() for key in PROVIDER_API_KEY_ENV}
    provider = request.provider.strip().lower()
    if provider not in providers:
        raise ValueError(f"Unsupported LLM provider: {provider}")
    if request.credential is not None and request.credential.get_secret_value() and provider == "ollama":
        raise ValueError("Ollama's built-in endpoint is keyless. For an authenticated local server, use OpenAI-compatible.")
    if not request.quick_model.strip() or not request.deep_model.strip():
        raise ValueError("Both quick and deep model IDs are required.")
    backend_url = _validate_base_url(request.backend_url) if request.backend_url else None

    return request.model_copy(update={"ticker": ticker, "provider": provider, "backend_url": backend_url})


class JobService:
    """SQLite-backed run history with one serialized worker for safe local use."""

    def __init__(
        self, start_worker: bool = True, db_path: str | Path | None = None,
        secret_store: Any | None = None,
    ):
        default_path = Path.home() / ".tradingagents" / "web" / "runs.sqlite3"
        self.db_path = str(db_path or default_path)
        self._secret_store = secret_store or KeyringSecretStore()
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db_lock = threading.RLock()
        self._credential_lock = threading.Lock()
        self._credentials: dict[str, tuple[str, str]] = {}
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("""CREATE TABLE IF NOT EXISTS runs (
            id TEXT PRIMARY KEY, ticker TEXT NOT NULL, analysis_date TEXT NOT NULL,
            status TEXT NOT NULL, request_json TEXT NOT NULL, progress_json TEXT NOT NULL,
            report_path TEXT, error TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
        )""")
        self._conn.execute("""CREATE TABLE IF NOT EXISTS saved_endpoints (
            id TEXT PRIMARY KEY, name TEXT NOT NULL, base_url TEXT NOT NULL,
            quick_model TEXT NOT NULL, deep_model TEXT NOT NULL,
            has_api_key INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL, updated_at TEXT NOT NULL
        )""")
        self._conn.execute(
            "UPDATE runs SET status='interrupted', error='Server restarted before this run completed', updated_at=? WHERE status IN ('queued','running')",
            (datetime.now().isoformat(timespec="seconds"),),
        )
        self._conn.commit()
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._closed = False
        self._worker: threading.Thread | None = None
        if start_worker:
            self._worker = threading.Thread(target=self._worker_loop, name="ta-web-worker", daemon=True)
            self._worker.start()

    def submit(self, request: RunRequest) -> dict[str, Any]:
        if request.saved_endpoint_id:
            if request.provider.strip().lower() != "openai_compatible":
                raise ValueError("Saved custom endpoint profiles require the OpenAI-compatible provider.")
            profile = self.get_saved_endpoint(request.saved_endpoint_id)
            if profile is None:
                raise ValueError("Saved endpoint profile was not found; refresh the endpoint list.")
            request = request.model_copy(update={"backend_url": profile["base_url"]})
            provided_secret = (
                request.credential.get_secret_value()
                if request.credential is not None else ""
            )
            if not provided_secret and profile["has_api_key"]:
                saved_secret = self.get_saved_endpoint_secret(request.saved_endpoint_id)
                if saved_secret:
                    request = request.model_copy(update={"credential": SecretStr(saved_secret)})
        request = validate_run_request(request)
        secret = request.credential.get_secret_value() if request.credential is not None else ""
        credential_env = "AWS_BEARER_TOKEN_BEDROCK" if request.provider == "bedrock" else PROVIDER_API_KEY_ENV.get(request.provider)
        if secret and not credential_env:
            raise ValueError(f"Provider '{request.provider}' does not accept a credential in this field.")
        job_id = uuid.uuid4().hex
        now = datetime.now().isoformat(timespec="seconds")
        initial_progress = {"stage": "Queued", "percent": 0, "completed_analysts": []}
        with self._db_lock:
            self._conn.execute(
                "INSERT INTO runs (id,ticker,analysis_date,status,request_json,progress_json,created_at,updated_at) VALUES (?,?,?,?,?,?,?,?)",
                (job_id, request.ticker, request.analysis_date, "queued", request.model_dump_json(), json.dumps(initial_progress), now, now),
            )
            self._conn.commit()
        if secret and credential_env:
            with self._credential_lock:
                self._credentials[job_id] = (credential_env, secret)
        if self._worker:
            self._queue.put(job_id)
        job = self.get(job_id)
        if job is None:
            raise RuntimeError("Run was saved but could not be read back.")
        return job

    def create_saved_endpoint(self, request: SavedEndpointRequest) -> dict[str, Any]:
        request = validate_saved_endpoint(request)
        if request.clear_api_key:
            raise ValueError("clear_api_key is only valid when updating a saved endpoint.")
        api_key = request.api_key.get_secret_value() if request.api_key is not None else ""
        endpoint_id = uuid.uuid4().hex
        now = datetime.now().isoformat(timespec="seconds")
        if api_key:
            self._secret_store.set_secret(endpoint_id, api_key)
        try:
            with self._db_lock:
                self._conn.execute(
                    "INSERT INTO saved_endpoints (id,name,base_url,quick_model,deep_model,has_api_key,created_at,updated_at) VALUES (?,?,?,?,?,?,?,?)",
                    (endpoint_id, request.name, request.base_url, request.quick_model,
                     request.deep_model, int(bool(api_key)), now, now),
                )
                self._conn.commit()
        except Exception:
            if api_key:
                self._secret_store.delete_secret(endpoint_id)
            raise
        profile = self.get_saved_endpoint(endpoint_id)
        if profile is None:
            raise RuntimeError("Saved endpoint was inserted but could not be read back.")
        return profile

    def update_saved_endpoint(
        self, endpoint_id: str, request: SavedEndpointRequest
    ) -> dict[str, Any] | None:
        request = validate_saved_endpoint(request)
        api_key = request.api_key.get_secret_value() if request.api_key is not None else ""
        if api_key and request.clear_api_key:
            raise ValueError("Enter a new API key or choose to clear the saved key, not both.")
        with self._db_lock:
            row = self._conn.execute(
                "SELECT has_api_key FROM saved_endpoints WHERE id=?", (endpoint_id,)
            ).fetchone()
        if row is None:
            return None
        has_api_key = bool(row["has_api_key"])
        if api_key:
            self._secret_store.set_secret(endpoint_id, api_key)
            has_api_key = True
        elif request.clear_api_key:
            if has_api_key:
                self._secret_store.delete_secret(endpoint_id)
            has_api_key = False
        now = datetime.now().isoformat(timespec="seconds")
        with self._db_lock:
            self._conn.execute(
                "UPDATE saved_endpoints SET name=?,base_url=?,quick_model=?,deep_model=?,has_api_key=?,updated_at=? WHERE id=?",
                (request.name, request.base_url, request.quick_model,
                 request.deep_model, int(has_api_key), now, endpoint_id),
            )
            self._conn.commit()
        return self.get_saved_endpoint(endpoint_id)

    def delete_saved_endpoint(self, endpoint_id: str) -> dict[str, str] | None:
        with self._db_lock:
            row = self._conn.execute(
                "SELECT has_api_key FROM saved_endpoints WHERE id=?", (endpoint_id,)
            ).fetchone()
        if row is None:
            return None
        if row["has_api_key"]:
            self._secret_store.delete_secret(endpoint_id)
        with self._db_lock:
            self._conn.execute("DELETE FROM saved_endpoints WHERE id=?", (endpoint_id,))
            self._conn.commit()
        return {"deleted": endpoint_id}

    def get_saved_endpoint(self, endpoint_id: str) -> dict[str, Any] | None:
        with self._db_lock:
            row = self._conn.execute(
                "SELECT * FROM saved_endpoints WHERE id=?", (endpoint_id,)
            ).fetchone()
        return self._serialize_saved_endpoint(row) if row else None

    def list_saved_endpoints(self) -> list[dict[str, Any]]:
        with self._db_lock:
            rows = self._conn.execute(
                "SELECT * FROM saved_endpoints ORDER BY name COLLATE NOCASE, created_at"
            ).fetchall()
        return [self._serialize_saved_endpoint(row) for row in rows]

    def get_saved_endpoint_secret(self, endpoint_id: str) -> str | None:
        profile = self.get_saved_endpoint(endpoint_id)
        if profile is None:
            raise KeyError(endpoint_id)
        if not profile["has_api_key"]:
            return None
        secret = self._secret_store.get_secret(endpoint_id)
        if not secret:
            raise SecretStoreError(
                "The saved API key is unavailable. Update the profile or enter the key again."
            )
        return secret

    @staticmethod
    def _serialize_saved_endpoint(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"], "name": row["name"], "base_url": row["base_url"],
            "quick_model": row["quick_model"], "deep_model": row["deep_model"],
            "has_api_key": bool(row["has_api_key"]),
            "created_at": row["created_at"], "updated_at": row["updated_at"],
        }

    def resume(self, job_id: str, credential: SecretStr | None = None) -> dict[str, Any]:
        with self._db_lock:
            row = self._conn.execute(
                "SELECT status,request_json FROM runs WHERE id=?", (job_id,)
            ).fetchone()
        if row is None:
            raise KeyError(job_id)
        if row["status"] not in {"failed", "interrupted"}:
            raise ValueError("Only failed or interrupted runs can be resumed.")
        request = RunRequest.model_validate_json(row["request_json"])
        if not request.checkpoint:
            raise ValueError("This run did not have checkpointing enabled; submit a new run instead.")
        if request.saved_endpoint_id and self.get_saved_endpoint(request.saved_endpoint_id) is None:
            request = request.model_copy(update={"saved_endpoint_id": None})
        return self.submit(request.model_copy(update={"credential": credential}))

    def delete(self, job_id: str) -> dict[str, str]:
        with self._db_lock:
            row = self._conn.execute("SELECT status FROM runs WHERE id=?", (job_id,)).fetchone()
            if row is None:
                raise KeyError(job_id)
            if row["status"] in {"queued", "running"}:
                raise ValueError("Cannot delete a queued or running analysis.")
            self._conn.execute("DELETE FROM runs WHERE id=?", (job_id,))
            self._conn.commit()
        return {"deleted": job_id}

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._db_lock:
            row = self._conn.execute("SELECT * FROM runs WHERE id=?", (job_id,)).fetchone()
        return self._serialize(row) if row else None

    def list(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._db_lock:
            rows = self._conn.execute("SELECT * FROM runs ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        return [self._serialize(row) for row in rows]

    def report(self, job_id: str) -> str | None:
        with self._db_lock:
            row = self._conn.execute("SELECT report_path FROM runs WHERE id=?", (job_id,)).fetchone()
        if not row or not row["report_path"]:
            return None
        path = Path(row["report_path"])
        return path.read_text(encoding="utf-8") if path.is_file() else None

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._worker:
            self._queue.put(None)
            self._worker.join(timeout=3)
        with self._credential_lock:
            self._credentials.clear()
        with self._db_lock:
            self._conn.close()

    @staticmethod
    def _serialize(row: sqlite3.Row) -> dict[str, Any]:
        request = json.loads(row["request_json"])
        resume_available = (
            row["status"] in {"failed", "interrupted"}
            and bool(request.get("checkpoint", True))
        )
        return {
            "id": row["id"], "ticker": row["ticker"], "analysis_date": row["analysis_date"],
            "status": row["status"], "progress": json.loads(row["progress_json"]),
            "report_available": bool(row["report_path"] and Path(row["report_path"]).is_file()),
            "resume_available": resume_available,
            "error": row["error"], "created_at": row["created_at"], "updated_at": row["updated_at"],
        }

    def _update(self, job_id: str, **fields: Any) -> None:
        fields["updated_at"] = datetime.now().isoformat(timespec="seconds")
        columns = ",".join(f"{name}=?" for name in fields)
        values = [json.dumps(v) if name == "progress_json" else v for name, v in fields.items()]
        with self._db_lock:
            self._conn.execute(f"UPDATE runs SET {columns} WHERE id=?", (*values, job_id))
            self._conn.commit()

    def _worker_loop(self) -> None:
        while True:
            job_id = self._queue.get()
            try:
                if job_id is None:
                    return
                self._execute(job_id)
            finally:
                self._queue.task_done()

    def _execute(self, job_id: str) -> None:
        job = self.get(job_id)
        if not job:
            return
        request = RunRequest.model_validate_json(self._request_json(job_id))
        progress = {"stage": "Starting", "percent": 1, "completed_analysts": []}
        self._update(job_id, status="running", error=None, progress_json=progress)
        with self._credential_lock:
            runtime_credential = self._credentials.pop(job_id, None)
        credential_env = runtime_credential[0] if runtime_credential else None
        credential_value = runtime_credential[1] if runtime_credential else None
        previous_credential = os.environ.get(credential_env) if credential_env else None
        try:
            if credential_env and credential_value:
                os.environ[credential_env] = credential_value
            config = DEFAULT_CONFIG.copy()
            config.update({
                "llm_provider": request.provider,
                "quick_think_llm": request.quick_model.strip(),
                "deep_think_llm": request.deep_model.strip(),
                "backend_url": request.backend_url,
                "max_debate_rounds": request.debate_rounds,
                "max_risk_discuss_rounds": request.risk_rounds,
                "max_tokens": request.max_tokens,
                "output_language": request.output_language,
                "checkpoint_enabled": request.checkpoint,
            })
            asset_type = detect_asset_type(request.ticker).value
            graph = TradingAgentsGraph(request.analysts, config=config)
            run_dir = Path(config["results_dir"]) / "web_runs" / job_id
            run_dir.mkdir(parents=True, exist_ok=True)
            latest_state: dict[str, Any] = {}

            with run_config(config):
                initial_state = graph.create_run_state(request.ticker, request.analysis_date, asset_type)
                thread_id = graph.begin_checkpoint(request.ticker, request.analysis_date, asset_type)
                args = graph.propagator.get_graph_args()
                if thread_id:
                    args.setdefault("config", {}).setdefault("configurable", {})["thread_id"] = thread_id
                try:
                    for snapshot in graph.graph.stream(graph.checkpoint_input(initial_state), **args):
                        latest_state = snapshot
                        progress = self._progress_for(snapshot, request.analysts)
                        self._update(job_id, progress_json=progress, report_path=str(run_dir / "complete_report.md"))
                        graph.save_reports(snapshot, request.ticker, run_dir)
                    if not latest_state.get("final_trade_decision"):
                        raise RuntimeError("Analysis ended without a final decision.")
                    graph._log_state(request.analysis_date, latest_state)
                    graph.record_decision(request.ticker, request.analysis_date, latest_state)
                    graph.clear_checkpoint_on_success(request.ticker, request.analysis_date, asset_type)
                    report_file = graph.save_reports(latest_state, request.ticker, run_dir)
                finally:
                    graph.end_checkpoint()
            progress = self._progress_for(latest_state, request.analysts)
            progress.update({"stage": "Complete", "percent": 100})
            self._update(job_id, status="completed", progress_json=progress, report_path=str(report_file), error=None)
        except Exception as exc:
            # Keep interim reports and never persist a submitted credential in errors.
            current = self.get(job_id)
            progress = current["progress"] if current else {"stage": "Failed", "percent": 0}
            progress["stage"] = "Failed"
            error = f"{type(exc).__name__}: {exc}"
            if credential_value:
                error = error.replace(credential_value, "[redacted]")
            self._update(job_id, status="failed", progress_json=progress, error=error)
        finally:
            if credential_env:
                if previous_credential is None:
                    os.environ.pop(credential_env, None)
                else:
                    os.environ[credential_env] = previous_credential

    def _request_json(self, job_id: str) -> str:
        with self._db_lock:
            row = self._conn.execute("SELECT request_json FROM runs WHERE id=?", (job_id,)).fetchone()
        if not row:
            raise KeyError(job_id)
        return row["request_json"]

    @staticmethod
    def _progress_for(state: dict[str, Any], analysts: list[str]) -> dict[str, Any]:
        report_map = {
            "market": ("market_report", "Market Analyst"),
            "social": ("sentiment_report", "Sentiment Analyst"),
            "news": ("news_report", "News Analyst"),
            "fundamentals": ("fundamentals_report", "Fundamentals Analyst"),
        }
        done = [key for key in analysts if state.get(report_map[key][0])]
        total = len(analysts) + 8
        progress = int(len(done) / total * 100)
        if len(done) < len(analysts):
            stage = f"Analysts · {len(done)}/{len(analysts)}"
        elif state.get("investment_debate_state", {}).get("judge_decision"):
            if state.get("trader_investment_plan") and state.get("risk_debate_state", {}).get("judge_decision"):
                stage = "Portfolio decision"
                progress = max(progress, 88)
            elif state.get("trader_investment_plan"):
                stage = "Risk review"
                progress = max(progress, 68)
            else:
                stage = "Trader plan"
                progress = max(progress, 55)
        else:
            stage = "Bull/bear research debate"
            progress = max(progress, 35)
        return {"stage": stage, "percent": min(progress, 95), "completed_analysts": done}
