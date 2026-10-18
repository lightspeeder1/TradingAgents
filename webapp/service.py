"""Local web UI job orchestration for TradingAgents."""
from __future__ import annotations

import json
import logging
import os
import queue
import re
import sqlite3
import threading
import uuid
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, Field, SecretStr

from cli.models import AnalystType
from cli.prompts import detect_asset_type, normalize_ticker_symbol, is_valid_ticker_input
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.graph.checkpointer import checkpoint_step
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


class ScheduleRequest(BaseModel):
    name: str
    cadence: Literal["daily", "weekly"]
    local_time: str
    timezone: str
    weekday: int | None = None
    enabled: bool = True
    run_request: RunRequest


def next_schedule_time(
    cadence: str, local_time: str, timezone_name: str, weekday: int | None = None,
    after: datetime | None = None,
) -> datetime:
    """Return the next local wall-clock occurrence as an aware UTC datetime."""
    if not re.fullmatch(r"(?:[01]\d|2[0-3]):[0-5]\d", local_time):
        raise ValueError("Schedule time must use 24-hour HH:MM format.")
    try:
        zone = ZoneInfo(timezone_name)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise ValueError("Choose a valid IANA timezone, such as America/New_York.") from exc
    if cadence not in {"daily", "weekly"}:
        raise ValueError("Schedule cadence must be daily or weekly.")
    if cadence == "weekly" and (weekday is None or not 0 <= weekday <= 6):
        raise ValueError("Weekly schedules require a weekday from 0 (Monday) to 6 (Sunday).")
    if cadence == "daily" and weekday is not None:
        raise ValueError("Daily schedules do not use a weekday.")
    now = (after or datetime.now(timezone.utc)).astimezone(zone)
    hour, minute = map(int, local_time.split(":"))
    day = now.date()
    if cadence == "weekly":
        day += timedelta(days=(weekday - day.weekday()) % 7)
    candidate = datetime.combine(day, time(hour, minute), tzinfo=zone)
    if candidate <= now:
        candidate += timedelta(days=7 if cadence == "weekly" else 1)
    # ZoneInfo uses fold=0 for repeated fall-back times; nonexistent spring-forward
    # wall times normalize forward when round-tripped through UTC.
    return candidate.astimezone(timezone.utc)


def validate_schedule_request(request: ScheduleRequest) -> ScheduleRequest:
    name = request.name.strip()
    if not name or len(name) > 80:
        raise ValueError("Schedule name must contain 1–80 characters.")
    next_schedule_time(request.cadence, request.local_time, request.timezone, request.weekday)
    run_request = request.run_request
    if run_request.credential and run_request.credential.get_secret_value():
        raise ValueError("Transient API credentials cannot be saved in a schedule. Use a saved endpoint profile or environment credentials.")
    run_request = validate_run_request(run_request.model_copy(update={"credential": None}))
    return request.model_copy(update={"name": name, "run_request": run_request})


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
            report_path TEXT, error TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
            auto_resume_count INTEGER NOT NULL DEFAULT 0, recovery_of TEXT,
            transient_credential INTEGER NOT NULL DEFAULT 0,
            environment_credential INTEGER NOT NULL DEFAULT 0
        )""")
        existing_run_columns = {row["name"] for row in self._conn.execute("PRAGMA table_info(runs)")}
        # Runs created before environment credential provenance was tracked cannot be
        # safely auto-resumed: a typed override may have been used without persistence.
        credential_history_unknown = "environment_credential" not in existing_run_columns
        for column, declaration in (
            ("auto_resume_count", "INTEGER NOT NULL DEFAULT 0"),
            ("recovery_of", "TEXT"),
            ("transient_credential", "INTEGER NOT NULL DEFAULT 0"),
            ("environment_credential", "INTEGER NOT NULL DEFAULT 0"),
        ):
            if column not in existing_run_columns:
                self._conn.execute(f"ALTER TABLE runs ADD COLUMN {column} {declaration}")
        self._conn.execute("""CREATE TABLE IF NOT EXISTS saved_endpoints (
            id TEXT PRIMARY KEY, name TEXT NOT NULL, base_url TEXT NOT NULL,
            quick_model TEXT NOT NULL, deep_model TEXT NOT NULL,
            has_api_key INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL, updated_at TEXT NOT NULL
        )""")
        self._conn.execute("""CREATE TABLE IF NOT EXISTS schedules (
            id TEXT PRIMARY KEY, name TEXT NOT NULL, cadence TEXT NOT NULL,
            local_time TEXT NOT NULL, timezone TEXT NOT NULL, weekday INTEGER,
            enabled INTEGER NOT NULL, run_request_json TEXT NOT NULL,
            next_run_at TEXT, last_run_id TEXT, last_error TEXT,
            created_at TEXT NOT NULL, updated_at TEXT NOT NULL
        )""")
        self._conn.execute("""CREATE TABLE IF NOT EXISTS schedule_occurrences (
            schedule_id TEXT NOT NULL, scheduled_for TEXT NOT NULL,
            run_id TEXT NOT NULL, created_at TEXT NOT NULL,
            PRIMARY KEY (schedule_id, scheduled_for)
        )""")
        self._conn.execute("""CREATE TABLE IF NOT EXISTS run_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL,
            created_at TEXT NOT NULL, event_type TEXT NOT NULL, message TEXT NOT NULL
        )""")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_run_events_run_id_id ON run_events(run_id,id)")
        if credential_history_unknown:
            self._conn.execute(
                "UPDATE runs SET transient_credential=1 WHERE status IN ('queued','running')"
            )
        self._conn.commit()
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._closed = False
        self._worker: threading.Thread | None = None
        self._dispatch_lock = threading.Lock()
        self._scheduler_stop = threading.Event()
        self._scheduler_thread: threading.Thread | None = None
        pending_recovered_jobs = self._recover_startup_runs()
        if start_worker:
            self._worker = threading.Thread(target=self._worker_loop, name="ta-web-worker", daemon=True)
            self._worker.start()
            for job_id in pending_recovered_jobs:
                self._queue.put(job_id)
            self._scheduler_thread = threading.Thread(target=self._scheduler_loop, name="ta-web-scheduler", daemon=True)
            self._scheduler_thread.start()

    def _recover_startup_runs(self) -> list[str]:
        """Recover queued work and create at most one checkpoint-based retry per run."""
        pending: list[str] = []
        with self._db_lock:
            stale = self._conn.execute(
                "SELECT * FROM runs WHERE status IN ('queued','running') ORDER BY created_at"
            ).fetchall()
        for row in stale:
            request = RunRequest.model_validate_json(row["request_json"])
            credential_env = "AWS_BEARER_TOKEN_BEDROCK" if request.provider == "bedrock" else PROVIDER_API_KEY_ENV.get(request.provider)
            environment_credential_lost = bool(
                row["environment_credential"] and credential_env and not os.environ.get(credential_env)
            )
            credentials_available = not environment_credential_lost and self._recovery_credentials_available(request)
            if row["status"] == "queued" and not row["transient_credential"] and credentials_available:
                pending.append(row["id"])
                continue
            now = datetime.now().isoformat(timespec="seconds")
            error = "Server restarted before this run completed"
            if row["status"] == "queued" and row["transient_credential"]:
                error += "; its transient credential was lost, so enter it again before resuming"
            elif row["status"] == "queued" and not credentials_available:
                error += "; required provider credentials are unavailable, so submit it again after restoring them"
            with self._db_lock:
                self._conn.execute(
                    "UPDATE runs SET status='interrupted',error=?,updated_at=? WHERE id=?",
                    (error, now, row["id"]),
                )
                self._conn.commit()
            if row["status"] != "running" or not request.checkpoint:
                continue
            if row["auto_resume_count"] >= 1 or row["transient_credential"] or not credentials_available:
                continue
            asset_type = detect_asset_type(request.ticker).value
            signature = "|".join([
                "analysts=" + ",".join(request.analysts),
                f"debate={request.debate_rounds}", f"risk={request.risk_rounds}",
                f"asset={asset_type}", "portfolio=none",
            ])
            try:
                step = checkpoint_step(DEFAULT_CONFIG["data_cache_dir"], request.ticker, request.analysis_date, signature)
            except Exception:
                logging.getLogger(__name__).exception("Could not inspect checkpoint for run %s", row["id"])
                continue
            if step is None:
                continue
            job_id = uuid.uuid4().hex
            progress = json.loads(row["progress_json"])
            progress["stage"] = "Auto-resuming from saved checkpoint"
            with self._db_lock:
                self._conn.execute(
                    "INSERT INTO runs (id,ticker,analysis_date,status,request_json,progress_json,created_at,updated_at,auto_resume_count,recovery_of,transient_credential,environment_credential) VALUES (?,?,?,?,?,?,?,?,?,?,0,?)",
                    (job_id, row["ticker"], row["analysis_date"], "queued", row["request_json"],
                     json.dumps(progress), now, now, row["auto_resume_count"] + 1, row["id"], row["environment_credential"]),
                )
                self._conn.commit()
            pending.append(job_id)
        return pending

    def _recovery_credentials_available(self, request: RunRequest) -> bool:
        if request.saved_endpoint_id:
            try:
                profile = self.get_saved_endpoint(request.saved_endpoint_id)
                if profile is None:
                    return False
                if profile["has_api_key"]:
                    return bool(self.get_saved_endpoint_secret(request.saved_endpoint_id))
                return True
            except (KeyError, SecretStoreError):
                return False
        credential_env = "AWS_BEARER_TOKEN_BEDROCK" if request.provider == "bedrock" else PROVIDER_API_KEY_ENV.get(request.provider)
        if credential_env and os.environ.get(credential_env):
            return True
        if credential_env and request.provider not in {"openai_compatible", "bedrock"}:
            return False
        return True

    def _scheduler_loop(self) -> None:
        while not self._scheduler_stop.wait(5):
            try:
                self._restart_worker_if_dead()
                self._enqueue_due_schedules()
            except Exception:
                logging.getLogger(__name__).exception("Schedule/worker supervision failed")

    def _restart_worker_if_dead(self) -> bool:
        if self._closed or self._worker is None or self._worker.is_alive():
            return False
        logging.getLogger(__name__).error("Analysis worker exited unexpectedly; recovering persisted queue")
        while True:
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except queue.Empty:
                break
        pending = self._recover_startup_runs()
        self._worker = threading.Thread(target=self._worker_loop, name="ta-web-worker-restarted", daemon=True)
        self._worker.start()
        for job_id in pending:
            self._queue.put(job_id)
        return True

    def _enqueue_due_schedules(self, now: datetime | None = None) -> list[str]:
        """Enqueue due schedules once; missed occurrences older than 15 minutes are skipped."""
        now = now or datetime.now(timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        now = now.astimezone(timezone.utc)
        now_text = now.isoformat(timespec="seconds")
        created_runs: list[str] = []
        with self._dispatch_lock:
            with self._db_lock:
                rows = self._conn.execute(
                    "SELECT * FROM schedules WHERE enabled=1 AND next_run_at IS NOT NULL AND next_run_at<=? ORDER BY next_run_at",
                    (now_text,),
                ).fetchall()
            for row in rows:
                scheduled_for = row["next_run_at"]
                scheduled_at = datetime.fromisoformat(scheduled_for).astimezone(timezone.utc)
                next_run = next_schedule_time(
                    row["cadence"], row["local_time"], row["timezone"], row["weekday"], now
                )
                if now - scheduled_at > timedelta(minutes=15):
                    with self._db_lock:
                        self._conn.execute(
                            "UPDATE schedules SET next_run_at=?,last_error=?,updated_at=? WHERE id=? AND next_run_at=?",
                            (next_run.isoformat(timespec="seconds"), "Skipped missed occurrence (more than 15 minutes late)", now_text, row["id"], scheduled_for),
                        )
                        self._conn.commit()
                    continue
                try:
                    request = RunRequest.model_validate_json(row["run_request_json"])
                    request = request.model_copy(update={
                        "analysis_date": now.astimezone(ZoneInfo(row["timezone"])).date().isoformat()
                    })
                    run = self.submit(request)
                    with self._db_lock:
                        self._conn.execute(
                            "INSERT OR IGNORE INTO schedule_occurrences (schedule_id,scheduled_for,run_id,created_at) VALUES (?,?,?,?)",
                            (row["id"], scheduled_for, run["id"], now_text),
                        )
                        self._conn.execute(
                            "UPDATE schedules SET next_run_at=?,last_run_id=?,last_error=NULL,updated_at=? WHERE id=? AND next_run_at=? AND enabled=1",
                            (next_run.isoformat(timespec="seconds"), run["id"], now_text, row["id"], scheduled_for),
                        )
                        self._conn.commit()
                    created_runs.append(run["id"])
                except Exception as exc:
                    logging.getLogger(__name__).exception("Could not dispatch schedule %s", row["id"])
                    with self._db_lock:
                        self._conn.execute(
                            "UPDATE schedules SET next_run_at=?,last_error=?,updated_at=? WHERE id=? AND next_run_at=?",
                            (next_run.isoformat(timespec="seconds"), f"{type(exc).__name__}: {exc}"[:500], now_text, row["id"], scheduled_for),
                        )
                        self._conn.commit()
        return created_runs

    def create_schedule(self, request: ScheduleRequest) -> dict[str, Any]:
        request = validate_schedule_request(request)
        schedule_id = uuid.uuid4().hex
        now = datetime.now(timezone.utc)
        next_run = (
            next_schedule_time(request.cadence, request.local_time, request.timezone, request.weekday, now)
            if request.enabled else None
        )
        with self._db_lock:
            self._conn.execute(
                "INSERT INTO schedules (id,name,cadence,local_time,timezone,weekday,enabled,run_request_json,next_run_at,created_at,updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (schedule_id, request.name, request.cadence, request.local_time, request.timezone,
                 request.weekday, int(request.enabled), request.run_request.model_dump_json(),
                 next_run.isoformat(timespec="seconds") if next_run else None,
                 now.isoformat(timespec="seconds"), now.isoformat(timespec="seconds")),
            )
            self._conn.commit()
        return self.get_schedule(schedule_id)

    def list_schedules(self) -> list[dict[str, Any]]:
        with self._db_lock:
            rows = self._conn.execute("SELECT * FROM schedules ORDER BY enabled DESC,next_run_at,name COLLATE NOCASE").fetchall()
        return [self._serialize_schedule(row) for row in rows]

    def get_schedule(self, schedule_id: str) -> dict[str, Any] | None:
        with self._db_lock:
            row = self._conn.execute("SELECT * FROM schedules WHERE id=?", (schedule_id,)).fetchone()
        return self._serialize_schedule(row) if row else None

    def get_schedule_detail(self, schedule_id: str) -> dict[str, Any] | None:
        with self._db_lock:
            row = self._conn.execute("SELECT * FROM schedules WHERE id=?", (schedule_id,)).fetchone()
        if row is None:
            return None
        result = self._serialize_schedule(row)
        result["run_request"] = json.loads(row["run_request_json"])
        return result

    @staticmethod
    def _serialize_schedule(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"], "name": row["name"], "cadence": row["cadence"],
            "local_time": row["local_time"], "timezone": row["timezone"],
            "weekday": row["weekday"], "enabled": bool(row["enabled"]),
            "next_run_at": row["next_run_at"], "last_run_id": row["last_run_id"],
            "last_error": row["last_error"], "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def update_schedule(self, schedule_id: str, request: ScheduleRequest) -> dict[str, Any] | None:
        request = validate_schedule_request(request)
        now = datetime.now(timezone.utc)
        next_run = (
            next_schedule_time(request.cadence, request.local_time, request.timezone, request.weekday, now)
            if request.enabled else None
        )
        with self._db_lock:
            cursor = self._conn.execute(
                "UPDATE schedules SET name=?,cadence=?,local_time=?,timezone=?,weekday=?,enabled=?,run_request_json=?,next_run_at=?,last_error=NULL,updated_at=? WHERE id=?",
                (request.name, request.cadence, request.local_time, request.timezone,
                 request.weekday, int(request.enabled), request.run_request.model_dump_json(),
                 next_run.isoformat(timespec="seconds") if next_run else None,
                 now.isoformat(timespec="seconds"), schedule_id),
            )
            self._conn.commit()
            if cursor.rowcount == 0:
                return None
        return self.get_schedule(schedule_id)

    def set_schedule_enabled(self, schedule_id: str, enabled: bool) -> dict[str, Any] | None:
        now = datetime.now(timezone.utc)
        with self._db_lock:
            row = self._conn.execute("SELECT * FROM schedules WHERE id=?", (schedule_id,)).fetchone()
            if row is None:
                return None
            next_run = None
            if enabled:
                next_run = next_schedule_time(row["cadence"], row["local_time"], row["timezone"], row["weekday"], now)
            self._conn.execute(
                "UPDATE schedules SET enabled=?,next_run_at=?,last_error=NULL,updated_at=? WHERE id=?",
                (int(enabled), next_run.isoformat(timespec="seconds") if next_run else None,
                 now.isoformat(timespec="seconds"), schedule_id),
            )
            self._conn.commit()
        return self.get_schedule(schedule_id)

    def delete_schedule(self, schedule_id: str) -> bool:
        with self._db_lock:
            cursor = self._conn.execute("DELETE FROM schedules WHERE id=?", (schedule_id,))
            self._conn.execute("DELETE FROM schedule_occurrences WHERE schedule_id=?", (schedule_id,))
            self._conn.commit()
        return cursor.rowcount > 0

    def run_schedule_now(self, schedule_id: str) -> dict[str, Any] | None:
        with self._db_lock:
            row = self._conn.execute("SELECT * FROM schedules WHERE id=?", (schedule_id,)).fetchone()
        if row is None:
            return None
        request = RunRequest.model_validate_json(row["run_request_json"])
        request = request.model_copy(update={
            "analysis_date": datetime.now(ZoneInfo(row["timezone"])).date().isoformat()
        })
        run = self.submit(request)
        with self._db_lock:
            self._conn.execute(
                "UPDATE schedules SET last_run_id=?,last_error=NULL,updated_at=? WHERE id=?",
                (run["id"], datetime.now().isoformat(timespec="seconds"), schedule_id),
            )
            self._conn.commit()
        return run

    def submit(self, request: RunRequest) -> dict[str, Any]:
        user_supplied_secret = bool(request.credential and request.credential.get_secret_value())
        saved_profile_has_key = False
        if request.saved_endpoint_id:
            if request.provider.strip().lower() != "openai_compatible":
                raise ValueError("Saved custom endpoint profiles require the OpenAI-compatible provider.")
            profile = self.get_saved_endpoint(request.saved_endpoint_id)
            if profile is None:
                raise ValueError("Saved endpoint profile was not found; refresh the endpoint list.")
            saved_profile_has_key = bool(profile["has_api_key"])
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
        transient_credential = user_supplied_secret
        environment_credential = bool(
            credential_env and os.environ.get(credential_env)
            and not user_supplied_secret and not saved_profile_has_key
        )
        job_id = uuid.uuid4().hex
        now = datetime.now().isoformat(timespec="seconds")
        initial_progress = {"stage": "Queued", "percent": 0, "completed_analysts": []}
        with self._db_lock:
            self._conn.execute(
                "INSERT INTO runs (id,ticker,analysis_date,status,request_json,progress_json,created_at,updated_at,transient_credential,environment_credential) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (job_id, request.ticker, request.analysis_date, "queued", request.model_dump_json(), json.dumps(initial_progress), now, now, int(transient_credential), int(environment_credential)),
            )
            self._conn.commit()
        if secret and credential_env:
            with self._credential_lock:
                self._credentials[job_id] = (credential_env, secret)
        self.record_activity(job_id, "Queued; waiting for the serial analysis worker.", "queued")
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
            self._conn.execute("DELETE FROM run_events WHERE run_id=?", (job_id,))
            self._conn.execute("DELETE FROM runs WHERE id=?", (job_id,))
            self._conn.commit()
        return {"deleted": job_id}

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._db_lock:
            row = self._conn.execute("SELECT * FROM runs WHERE id=?", (job_id,)).fetchone()
        return self._serialize(row) if row else None

    def record_activity(self, job_id: str, message: str, event_type: str = "info") -> int | None:
        message = " ".join(str(message).split())[:500]
        event_type = "".join(character for character in str(event_type) if character.isalnum() or character in "_-")[:32] or "info"
        with self._db_lock:
            exists = self._conn.execute("SELECT 1 FROM runs WHERE id=?", (job_id,)).fetchone()
            if not exists:
                return None
            cursor = self._conn.execute(
                "INSERT INTO run_events (run_id,created_at,event_type,message) VALUES (?,?,?,?)",
                (job_id, datetime.now().isoformat(timespec="seconds"), event_type, message),
            )
            self._conn.commit()
            return int(cursor.lastrowid)

    def list_activity(self, job_id: str, after_id: int = 0, limit: int = 100) -> dict[str, Any] | None:
        with self._db_lock:
            exists = self._conn.execute("SELECT 1 FROM runs WHERE id=?", (job_id,)).fetchone()
            if not exists:
                return None
            rows = self._conn.execute(
                "SELECT id,created_at,event_type,message FROM run_events WHERE run_id=? AND id>? ORDER BY id LIMIT ?",
                (job_id, after_id, limit),
            ).fetchall()
        items = [dict(row) for row in rows]
        return {"items": items, "latest_id": items[-1]["id"] if items else after_id}

    def list(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._db_lock:
            rows = self._conn.execute("SELECT * FROM runs ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        return [self._serialize(row) for row in rows]

    def list_reports(
        self, limit: int = 50, offset: int = 0,
        ticker: str | None = None, status: str | None = None,
    ) -> dict[str, Any]:
        """List report artifacts associated with run history, without exposing paths."""
        clauses = ["report_path IS NOT NULL"]
        params: list[Any] = []
        if ticker and ticker.strip():
            clauses.append("ticker = ? COLLATE NOCASE")
            params.append(ticker.strip())
        if status and status.strip():
            clauses.append("status = ?")
            params.append(status.strip().lower())
        where = " AND ".join(clauses)
        with self._db_lock:
            total = self._conn.execute(
                f"SELECT COUNT(*) FROM runs WHERE {where}", params
            ).fetchone()[0]
            rows = self._conn.execute(
                f"SELECT id,ticker,analysis_date,status,progress_json,report_path,updated_at "
                f"FROM runs WHERE {where} ORDER BY updated_at DESC LIMIT ? OFFSET ?",
                (*params, limit, offset),
            ).fetchall()
        items = []
        for row in rows:
            report_path = Path(row["report_path"])
            items.append({
                "id": row["id"], "ticker": row["ticker"],
                "analysis_date": row["analysis_date"], "status": row["status"],
                "progress": json.loads(row["progress_json"]),
                "available": report_path.is_file(), "updated_at": row["updated_at"],
            })
        return {"items": items, "total": total, "limit": limit, "offset": offset}

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
        self._scheduler_stop.set()
        if self._scheduler_thread:
            self._scheduler_thread.join(timeout=3)
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
            "auto_resume_count": row["auto_resume_count"], "recovery_of": row["recovery_of"],
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
        self.record_activity(job_id, "Analysis worker started.", "started")
        with self._credential_lock:
            runtime_credential = self._credentials.pop(job_id, None)
        credential_env = runtime_credential[0] if runtime_credential else (
            "AWS_BEARER_TOKEN_BEDROCK" if request.provider == "bedrock" else PROVIDER_API_KEY_ENV.get(request.provider)
        )
        credential_value = runtime_credential[1] if runtime_credential else None
        previous_credential = None
        environment_overridden = False
        try:
            if not credential_value and request.saved_endpoint_id:
                saved_secret = self.get_saved_endpoint_secret(request.saved_endpoint_id)
                if saved_secret:
                    credential_value = saved_secret
            if credential_env and credential_value:
                previous_credential = os.environ.get(credential_env)
                os.environ[credential_env] = credential_value
                environment_overridden = True
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
                args["stream_mode"] = ["values", "debug"]
                if thread_id:
                    args.setdefault("config", {}).setdefault("configurable", {})["thread_id"] = thread_id
                try:
                    for event in graph.graph.stream(graph.checkpoint_input(initial_state), **args):
                        if isinstance(event, tuple) and len(event) == 2:
                            stream_mode, chunk = event
                        else:
                            stream_mode, chunk = "values", event
                        if stream_mode == "debug":
                            self._record_graph_debug_event(job_id, chunk)
                            continue
                        if stream_mode != "values" or not isinstance(chunk, dict):
                            continue
                        latest_state = chunk
                        progress = self._progress_for(chunk, request.analysts)
                        self._update(job_id, progress_json=progress, report_path=str(run_dir / "complete_report.md"))
                        graph.save_reports(chunk, request.ticker, run_dir)
                        self.record_activity(job_id, "Interim report draft refreshed after a graph step.", "report_update")
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
            self.record_activity(job_id, "Analysis completed and report saved.", "completed")
        except Exception as exc:
            # Keep interim reports and never persist a submitted credential in errors.
            current = self.get(job_id)
            progress = current["progress"] if current else {"stage": "Failed", "percent": 0}
            progress["stage"] = "Failed"
            error = f"{type(exc).__name__}: {exc}"
            if credential_value:
                error = error.replace(credential_value, "[redacted]")
            self._update(job_id, status="failed", progress_json=progress, error=error)
            self.record_activity(job_id, "Analysis failed; see run details for the error.", "failed")
        finally:
            if environment_overridden and credential_env:
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

    def _record_graph_debug_event(self, job_id: str, event: Any) -> None:
        if not isinstance(event, dict):
            return
        kind = event.get("type")
        payload = event.get("payload") or event.get("data") or {}
        if not isinstance(payload, dict):
            return
        name = " ".join(str(payload.get("name") or "graph").replace("_", " ").split())[:100]
        if name == "start":
            name = "analysis setup"
        elif name == "end":
            name = "finalization"
        if kind == "task":
            self.record_activity(job_id, f"Running graph step: {name}.", "node_start")
        elif kind == "task_result":
            if payload.get("error"):
                self.record_activity(job_id, f"Graph step failed: {name}.", "node_failed")
            else:
                self.record_activity(job_id, f"Finished graph step: {name}.", "node_complete")

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
