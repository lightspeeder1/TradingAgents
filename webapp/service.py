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

from pydantic import BaseModel, Field

from cli.models import AnalystType
from cli.prompts import detect_asset_type, normalize_ticker_symbol, is_valid_ticker_input
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.llm_clients.api_key_env import PROVIDER_API_KEY_ENV
from tradingagents.dataflows.config import run_config


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
    debate_rounds: int = Field(default=DEFAULT_CONFIG["max_debate_rounds"], ge=1, le=10)
    risk_rounds: int = Field(default=DEFAULT_CONFIG["max_risk_discuss_rounds"], ge=1, le=10)
    max_tokens: int | None = Field(default=DEFAULT_CONFIG.get("max_tokens"), ge=256, le=65536)
    output_language: str = DEFAULT_CONFIG["output_language"]
    checkpoint: bool = True


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
    if not request.quick_model.strip() or not request.deep_model.strip():
        raise ValueError("Both quick and deep model IDs are required.")
    if request.backend_url and not request.backend_url.startswith(("http://", "https://")):
        raise ValueError("Backend URL must start with http:// or https://.")

    return request.model_copy(update={"ticker": ticker, "provider": provider})


class JobService:
    """SQLite-backed run history with one serialized worker for safe local use."""

    def __init__(self, start_worker: bool = True, db_path: str | Path | None = None):
        default_path = Path.home() / ".tradingagents" / "web" / "runs.sqlite3"
        self.db_path = str(db_path or default_path)
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db_lock = threading.RLock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("""CREATE TABLE IF NOT EXISTS runs (
            id TEXT PRIMARY KEY, ticker TEXT NOT NULL, analysis_date TEXT NOT NULL,
            status TEXT NOT NULL, request_json TEXT NOT NULL, progress_json TEXT NOT NULL,
            report_path TEXT, error TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
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
        request = validate_run_request(request)
        job_id = uuid.uuid4().hex
        now = datetime.now().isoformat(timespec="seconds")
        initial_progress = {"stage": "Queued", "percent": 0, "completed_analysts": []}
        with self._db_lock:
            self._conn.execute(
                "INSERT INTO runs (id,ticker,analysis_date,status,request_json,progress_json,created_at,updated_at) VALUES (?,?,?,?,?,?,?,?)",
                (job_id, request.ticker, request.analysis_date, "queued", request.model_dump_json(), json.dumps(initial_progress), now, now),
            )
            self._conn.commit()
        if self._worker:
            self._queue.put(job_id)
        job = self.get(job_id)
        if job is None:
            raise RuntimeError("Run was saved but could not be read back.")
        return job

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
        with self._db_lock:
            self._conn.close()

    @staticmethod
    def _serialize(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"], "ticker": row["ticker"], "analysis_date": row["analysis_date"],
            "status": row["status"], "progress": json.loads(row["progress_json"]),
            "report_available": bool(row["report_path"] and Path(row["report_path"]).is_file()),
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
        try:
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
            # Keep any interim report and surface a concise error in the run record.
            current = self.get(job_id)
            progress = current["progress"] if current else {"stage": "Failed", "percent": 0}
            progress["stage"] = "Failed"
            self._update(job_id, status="failed", progress_json=progress, error=f"{type(exc).__name__}: {exc}")

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
