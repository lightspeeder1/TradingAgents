"""FastAPI application for the local TradingAgents dashboard."""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse

from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.llm_clients.api_key_env import PROVIDER_API_KEY_ENV

from .service import JobService, RunRequest, configured_default_models, configured_default_provider


ANALYSTS = [
    {"id": "market", "label": "Market", "detail": "Price action & technicals"},
    {"id": "social", "label": "Sentiment", "detail": "Social mood & positioning"},
    {"id": "news", "label": "News", "detail": "Company & macro events"},
    {"id": "fundamentals", "label": "Fundamentals", "detail": "Financials & valuation"},
]


def create_app(service: JobService | None = None) -> FastAPI:
    owns_service = service is None
    active_service = service

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        nonlocal active_service
        if active_service is None:
            active_service = JobService()
            app.state.jobs = active_service
        yield
        if owns_service and active_service is not None:
            active_service.close()

    app = FastAPI(title="TradingAgents Web", version="0.1.0", lifespan=lifespan)
    app.state.jobs = active_service

    def get_jobs() -> JobService:
        if app.state.jobs is None:
            raise RuntimeError("Run service has not started.")
        return app.state.jobs

    @app.get("/", include_in_schema=False)
    def dashboard():
        return FileResponse(Path(__file__).parent / "static" / "index.html")

    @app.get("/api/health")
    def health():
        return {"status": "ok"}

    @app.get("/api/options")
    def options():
        quick_model, deep_model = configured_default_models()
        return {
            "analysts": ANALYSTS,
            "providers": sorted(PROVIDER_API_KEY_ENV),
            "defaults": {
                "provider": configured_default_provider(),
                "quick_model": quick_model,
                "deep_model": deep_model,
                "debate_rounds": DEFAULT_CONFIG["max_debate_rounds"],
                "risk_rounds": DEFAULT_CONFIG["max_risk_discuss_rounds"],
                "max_tokens": DEFAULT_CONFIG.get("max_tokens"),
                "output_language": DEFAULT_CONFIG["output_language"],
                "backend_url": DEFAULT_CONFIG.get("backend_url"),
            },
        }

    @app.get("/api/runs")
    def list_runs():
        return get_jobs().list()

    @app.post("/api/runs", status_code=202)
    def create_run(request: RunRequest):
        try:
            return get_jobs().submit(request)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/runs/{job_id}")
    def get_run(job_id: str):
        run = get_jobs().get(job_id)
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")
        return run

    @app.get("/api/runs/{job_id}/report", response_class=PlainTextResponse)
    def get_report(job_id: str):
        content = get_jobs().report(job_id)
        if content is None:
            raise HTTPException(status_code=404, detail="Report is not available yet")
        return content

    return app


app = create_app()
