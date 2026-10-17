"""FastAPI application for the local TradingAgents dashboard."""
from __future__ import annotations

from contextlib import asynccontextmanager
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request as UrlRequest, urlopen

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel, Field, SecretStr

from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.llm_clients.api_key_env import PROVIDER_API_KEY_ENV
from tradingagents.llm_clients.openai_client import OPENAI_COMPATIBLE_PROVIDERS

from .secure_store import SecretStoreError
from .service import (
    JobService, RunRequest, SavedEndpointRequest,
    configured_default_models, configured_default_provider,
)


ANALYSTS = [
    {"id": "market", "label": "Market", "detail": "Price action & technicals"},
    {"id": "social", "label": "Sentiment", "detail": "Social mood & positioning"},
    {"id": "news", "label": "News", "detail": "Company & macro events"},
    {"id": "fundamentals", "label": "Fundamentals", "detail": "Financials & valuation"},
]


class EndpointTestRequest(BaseModel):
    provider: str
    backend_url: str | None = None
    saved_endpoint_id: str | None = None
    credential: SecretStr | None = Field(default=None, exclude=True, repr=False)


class ResumeRunRequest(BaseModel):
    credential: SecretStr | None = Field(default=None, exclude=True, repr=False)


def _endpoint_test_target(
    provider: str, backend_url: str | None, credential: SecretStr | None = None
) -> tuple[str, dict[str, str]]:
    """Resolve a provider's non-generative model-list endpoint and auth headers."""
    provider = provider.strip().lower()
    if provider not in PROVIDER_API_KEY_ENV:
        raise ValueError(f"Unsupported LLM provider: {provider}")
    credential_value = credential.get_secret_value() if credential is not None else None
    if provider == "ollama" and credential_value:
        raise ValueError("The built-in Ollama endpoint does not use credentials. Use OpenAI-compatible for an authenticated proxy.")
    if backend_url and provider not in {"ollama", "openai_compatible"}:
        raise ValueError("Use the OpenAI-compatible provider for a custom endpoint; hosted provider credentials will not be sent to custom URLs.")

    headers = {"Accept": "application/json"}
    if provider in OPENAI_COMPATIBLE_PROVIDERS:
        spec = OPENAI_COMPATIBLE_PROVIDERS[provider]
        base_url = backend_url or (
            os.environ.get(spec.base_url_env) if spec.base_url_env else None
        ) or spec.base_url
        if not base_url and provider == "openai":
            base_url = "https://api.openai.com/v1"
        if not base_url:
            raise ValueError("Enter a base URL for this OpenAI-compatible endpoint.")
        api_key_env = PROVIDER_API_KEY_ENV.get(provider)
        api_key = credential_value or (os.environ.get(api_key_env) if api_key_env else None)
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return f"{base_url.rstrip('/')}/models", headers

    if provider == "anthropic":
        base_url = backend_url or "https://api.anthropic.com"
        api_key = credential_value or os.environ.get("ANTHROPIC_API_KEY")
        if api_key:
            headers["x-api-key"] = api_key
        headers["anthropic-version"] = "2023-06-01"
        suffix = "/models" if base_url.rstrip("/").endswith("/v1") else "/v1/models"
        return f"{base_url.rstrip('/')}{suffix}", headers

    if provider == "google":
        base_url = backend_url or "https://generativelanguage.googleapis.com"
        api_key = credential_value or os.environ.get("GOOGLE_API_KEY")
        if api_key:
            headers["x-goog-api-key"] = api_key
        return f"{base_url.rstrip('/')}/v1beta/models", headers

    if provider == "azure":
        base_url = backend_url or os.environ.get("AZURE_OPENAI_ENDPOINT")
        if not base_url:
            raise ValueError("Set AZURE_OPENAI_ENDPOINT or enter an endpoint URL first.")
        api_key = credential_value or os.environ.get("AZURE_OPENAI_API_KEY")
        if api_key:
            headers["api-key"] = api_key
        version = os.environ.get("OPENAI_API_VERSION", "2024-10-21")
        return f"{base_url.rstrip('/')}/openai/models?api-version={version}", headers

    if provider == "bedrock":
        raise ValueError("Bedrock endpoint testing uses AWS credentials; this probe supports HTTP endpoints only.")

    raise ValueError(f"Endpoint testing is not configured for provider: {provider}")


def probe_endpoint(
    provider: str, backend_url: str | None, credential: SecretStr | None = None
) -> dict[str, object]:
    """Check endpoint reachability without sending a model prompt or spending tokens."""
    if backend_url:
        supplied = urlsplit(backend_url)
        if (supplied.scheme not in ("http", "https") or not supplied.hostname
                or supplied.username or supplied.password or supplied.query or supplied.fragment):
            raise ValueError("Endpoint must be an http:// or https:// base URL without credentials, query, or fragment.")
        try:
            supplied.port
        except ValueError as exc:
            raise ValueError("Endpoint URL has an invalid port.") from exc
    target, headers = _endpoint_test_target(provider, backend_url, credential)
    parsed = urlsplit(target)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError("Resolved provider endpoint is not a valid http:// or https:// URL.")

    request = UrlRequest(target, headers=headers, method="GET")
    try:
        with urlopen(request, timeout=4) as response:
            status = response.status
    except HTTPError as exc:
        status = exc.code
    except (URLError, TimeoutError, OSError):
        return {"ok": False, "reachable": False, "status_code": None,
                "message": "Could not connect. Confirm the endpoint URL and that its service is running."}

    if 200 <= status < 300:
        return {"ok": True, "reachable": True, "status_code": status,
                "message": f"Endpoint is responding (HTTP {status})."}
    if status in (401, 403):
        message = f"Endpoint is reachable but rejected authentication (HTTP {status}); check provider credentials."
    elif status == 404:
        message = "Endpoint is reachable, but its models route returned HTTP 404; check the base URL and /v1 path."
    else:
        message = f"Endpoint is reachable but returned HTTP {status}."
    return {"ok": False, "reachable": True, "status_code": status, "message": message}


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

    @app.get("/api/endpoints")
    def list_saved_endpoints():
        return get_jobs().list_saved_endpoints()

    @app.post("/api/endpoints", status_code=201)
    def create_saved_endpoint(request: SavedEndpointRequest):
        try:
            return get_jobs().create_saved_endpoint(request)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except SecretStoreError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.put("/api/endpoints/{endpoint_id}")
    def update_saved_endpoint(endpoint_id: str, request: SavedEndpointRequest):
        try:
            result = get_jobs().update_saved_endpoint(endpoint_id, request)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except SecretStoreError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        if result is None:
            raise HTTPException(status_code=404, detail="Saved endpoint profile not found")
        return result

    @app.delete("/api/endpoints/{endpoint_id}")
    def delete_saved_endpoint(endpoint_id: str):
        try:
            result = get_jobs().delete_saved_endpoint(endpoint_id)
        except SecretStoreError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        if result is None:
            raise HTTPException(status_code=404, detail="Saved endpoint profile not found")
        return result

    @app.post("/api/test-endpoint")
    def test_endpoint(request: EndpointTestRequest):
        try:
            backend_url = request.backend_url
            credential = request.credential
            if request.saved_endpoint_id:
                profile = get_jobs().get_saved_endpoint(request.saved_endpoint_id)
                if profile is None:
                    raise HTTPException(status_code=404, detail="Saved endpoint profile not found")
                if request.provider.strip().lower() != "openai_compatible":
                    raise ValueError("Saved custom endpoint profiles require the OpenAI-compatible provider.")
                backend_url = profile["base_url"]
                if credential is None or not credential.get_secret_value():
                    saved_secret = get_jobs().get_saved_endpoint_secret(request.saved_endpoint_id)
                    if saved_secret:
                        credential = SecretStr(saved_secret)
            return probe_endpoint(request.provider, backend_url, credential)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except SecretStoreError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.get("/api/runs")
    def list_runs():
        return get_jobs().list()

    @app.post("/api/runs", status_code=202)
    def create_run(request: RunRequest):
        try:
            return get_jobs().submit(request)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except SecretStoreError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.post("/api/runs/{job_id}/resume", status_code=202)
    def resume_run(job_id: str, request: ResumeRunRequest):
        try:
            return get_jobs().resume(job_id, request.credential)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Run not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except SecretStoreError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.delete("/api/runs/{job_id}")
    def delete_run(job_id: str):
        try:
            return get_jobs().delete(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Run not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

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
