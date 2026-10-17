import json
import os
import unittest
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
from fastapi.testclient import TestClient
from pydantic import SecretStr

from webapp.app import create_app
from webapp.secure_store import KeyringSecretStore, SecretStoreError
from webapp.service import JobService, RunRequest, SavedEndpointRequest, validate_run_request


class MemorySecretStore:
    def __init__(self):
        self.values = {}

    def set_secret(self, key, value):
        self.values[key] = value

    def get_secret(self, key):
        return self.values.get(key)

    def delete_secret(self, key):
        self.values.pop(key, None)


class RequestValidationTests(unittest.TestCase):
    def test_rejects_future_analysis_date(self):
        with self.assertRaises(ValueError):
            validate_run_request(RunRequest(ticker="MRVL", analysis_date="2999-01-01"))

    def test_rejects_invalid_ticker_path_component(self):
        with self.assertRaises(ValueError):
            validate_run_request(RunRequest(ticker="../escape", analysis_date="2026-10-15"))

    def test_accepts_supported_analyst_selection_and_date(self):
        run = validate_run_request(RunRequest(
            ticker="MRVL", analysis_date=date.today().isoformat(),
            analysts=["market", "news"]
        ))
        self.assertEqual(run.ticker, "MRVL")
        self.assertEqual(run.analysts, ["market", "news"])

    def test_accepts_local_openai_compatible_endpoint(self):
        run = validate_run_request(RunRequest(
            ticker="MRVL", analysis_date=date.today().isoformat(),
            provider="openai_compatible", backend_url="http://127.0.0.1:1234/v1",
            quick_model="local/model", deep_model="local/model", analysts=["market"]
        ))
        self.assertEqual(run.backend_url, "http://127.0.0.1:1234/v1")

    def test_rejects_embedded_credentials_in_backend_url(self):
        with self.assertRaises(ValueError):
            validate_run_request(RunRequest(
                ticker="MRVL", analysis_date=date.today().isoformat(),
                provider="openai_compatible", backend_url="https://user:secret@example.com/v1",
            ))

    def test_accepts_ollama_local_endpoint(self):
        run = validate_run_request(RunRequest(
            ticker="MRVL", analysis_date=date.today().isoformat(),
            provider="ollama", backend_url="http://127.0.0.1:11434/v1",
            quick_model="llama3.1", deep_model="llama3.1", analysts=["market"]
        ))
        self.assertEqual(run.provider, "ollama")
        self.assertEqual(run.backend_url, "http://127.0.0.1:11434/v1")

    def test_credential_is_available_in_memory_but_excluded_from_serialization(self):
        request = RunRequest(
            ticker="MRVL", analysis_date=date.today().isoformat(),
            credential="never-persist-this-value"
        )
        self.assertEqual(request.credential.get_secret_value(), "never-persist-this-value")
        self.assertNotIn("never-persist-this-value", request.model_dump_json())

    def test_rejects_empty_analysts(self):
        with self.assertRaises(ValueError):
            validate_run_request(RunRequest(ticker="MRVL", analysis_date="2026-10-15", analysts=[]))


class SecureStoreTests(unittest.TestCase):
    def test_refuses_unrecommended_keyring_backend(self):
        import keyring

        with patch.object(keyring, "get_keyring", return_value=SimpleNamespace(priority=0)):
            with self.assertRaises(SecretStoreError):
                KeyringSecretStore()._keyring()


class SavedEndpointPersistenceTests(unittest.TestCase):
    def test_profile_metadata_and_key_reference_survive_service_restart(self):
        with TemporaryDirectory() as directory:
            database = Path(directory) / "runs.sqlite3"
            secrets = MemorySecretStore()
            first = JobService(start_worker=False, db_path=database, secret_store=secrets)
            profile = first.create_saved_endpoint(SavedEndpointRequest(
                name="Persistent", base_url="https://llm.example/v1",
                quick_model="fast", deep_model="deep", api_key="persisted-secret",
            ))
            first.close()

            second = JobService(start_worker=False, db_path=database, secret_store=secrets)
            try:
                loaded = second.get_saved_endpoint(profile["id"])
                self.assertEqual(loaded["name"], "Persistent")
                self.assertEqual(second.get_saved_endpoint_secret(profile["id"]), "persisted-secret")
            finally:
                second.close()


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.service = JobService(
            start_worker=False, db_path=":memory:", secret_store=MemorySecretStore()
        )
        self.client = TestClient(create_app(self.service))

    def tearDown(self):
        self.client.close()
        self.service.close()

    def test_health_and_options(self):
        self.assertEqual(self.client.get("/api/health").json(), {"status": "ok"})
        options = self.client.get("/api/options").json()
        self.assertIn("market", [item["id"] for item in options["analysts"]])
        self.assertIn("openrouter", options["providers"])

    def test_submit_returns_queued_job_without_running_llm(self):
        response = self.client.post("/api/runs", json={
            "ticker": "MRVL", "analysis_date": "2026-10-15",
            "analysts": ["market", "news"], "provider": "openrouter",
            "quick_model": "example/quick", "deep_model": "example/deep"
        })
        self.assertEqual(response.status_code, 202)
        body = response.json()
        self.assertEqual(body["status"], "queued")
        self.assertEqual(body["ticker"], "MRVL")

    def test_invalid_submission_returns_422(self):
        response = self.client.post("/api/runs", json={
            "ticker": "../bad", "analysis_date": "2026-10-15"
        })
        self.assertEqual(response.status_code, 422)

    def test_resume_interrupted_run_copies_settings_and_accepts_fresh_credential(self):
        original = self.service.submit(RunRequest(
            ticker="MRVL", analysis_date=date.today().isoformat(),
            provider="openrouter", quick_model="quick-example", deep_model="deep-example",
            analysts=["market", "news"], debate_rounds=3, risk_rounds=2, checkpoint=True
        ))
        self.service._update(original["id"], status="interrupted", error="test interruption")
        response = self.client.post(f"/api/runs/{original['id']}/resume", json={
            "credential": "resume-only-token"
        })
        self.assertEqual(response.status_code, 202)
        self.assertNotIn("resume-only-token", response.text)
        resumed_id = response.json()["id"]
        self.assertNotEqual(resumed_id, original["id"])
        resumed_request = json.loads(self.service._request_json(resumed_id))
        self.assertEqual(resumed_request["ticker"], "MRVL")
        self.assertEqual(resumed_request["analysis_date"], date.today().isoformat())
        self.assertEqual(resumed_request["provider"], "openrouter")
        self.assertEqual(resumed_request["quick_model"], "quick-example")
        self.assertEqual(resumed_request["deep_model"], "deep-example")
        self.assertEqual(resumed_request["analysts"], ["market", "news"])
        self.assertEqual(resumed_request["debate_rounds"], 3)
        self.assertEqual(resumed_request["risk_rounds"], 2)
        self.assertTrue(resumed_request["checkpoint"])
        self.assertNotIn("resume-only-token", self.service._request_json(resumed_id))
        self.assertEqual(self.service._credentials[resumed_id][1], "resume-only-token")
        self.assertTrue(self.service.get(original["id"])["resume_available"])

    def test_resume_rejects_completed_and_uncheckpointed_runs(self):
        completed = self.service.submit(RunRequest(
            ticker="MRVL", analysis_date=date.today().isoformat(), checkpoint=True
        ))
        self.service._update(completed["id"], status="completed")
        completed_response = self.client.post(
            f"/api/runs/{completed['id']}/resume", json={}
        )
        self.assertEqual(completed_response.status_code, 409)
        self.assertFalse(self.service.get(completed["id"])["resume_available"])

        uncheckpointed = self.service.submit(RunRequest(
            ticker="NVDA", analysis_date=date.today().isoformat(), checkpoint=False
        ))
        self.service._update(uncheckpointed["id"], status="interrupted")
        resume_response = self.client.post(
            f"/api/runs/{uncheckpointed['id']}/resume", json={}
        )
        self.assertEqual(resume_response.status_code, 409)
        self.assertFalse(self.service.get(uncheckpointed["id"])["resume_available"])

    def test_delete_removes_terminal_run_record_only(self):
        job = self.service.submit(RunRequest(
            ticker="MRVL", analysis_date=date.today().isoformat()
        ))
        self.service._update(job["id"], status="failed", error="test failure")
        response = self.client.delete(f"/api/runs/{job['id']}")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"deleted": job["id"]})
        self.assertIsNone(self.service.get(job["id"]))

    def test_delete_rejects_queued_run(self):
        job = self.service.submit(RunRequest(
            ticker="MRVL", analysis_date=date.today().isoformat()
        ))
        response = self.client.delete(f"/api/runs/{job['id']}")
        self.assertEqual(response.status_code, 409)
        self.assertIsNotNone(self.service.get(job["id"]))

    def test_root_serves_dashboard(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("TradingAgents", response.text)
        self.assertIn("color-scheme:dark", response.text)
        self.assertIn("ollama_local", response.text)
        self.assertIn("lm_studio_local", response.text)
        self.assertIn("vllm_local", response.text)
        self.assertIn("testEndpointButton", response.text)
        self.assertIn('id="apiCredential"', response.text)
        self.assertIn('type="password"', response.text)
        self.assertIn("Held transiently in memory only", response.text)
        self.assertIn("data-resume", response.text)
        self.assertIn("data-delete", response.text)
        self.assertIn('id="savedEndpointSelect"', response.text)
        self.assertIn('id="saveEndpointButton"', response.text)
        self.assertIn('id="updateEndpointButton"', response.text)
        self.assertIn('id="deleteEndpointButton"', response.text)

    def test_saved_endpoint_rejects_url_credentials(self):
        response = self.client.post("/api/endpoints", json={
            "name": "Bad URL", "base_url": "https://user:secret@example.com/v1",
            "quick_model": "fast", "deep_model": "deep", "api_key": "not-stored",
        })
        self.assertEqual(response.status_code, 422)
        self.assertNotIn("secret", response.text)
        self.assertEqual(self.client.get("/api/endpoints").json(), [])
        self.assertEqual(self.service._secret_store.values, {})

    def test_saved_endpoint_crud_stores_key_outside_sqlite_and_feeds_runs(self):
        secret = "keychain-only-secret"
        payload = {
            "name": "Local gateway", "base_url": "http://127.0.0.1:9123/v1",
            "quick_model": "fast-model", "deep_model": "deep-model", "api_key": secret,
        }
        created = self.client.post("/api/endpoints", json=payload)
        self.assertEqual(created.status_code, 201)
        profile = created.json()
        self.assertTrue(profile["has_api_key"])
        self.assertNotIn("api_key", profile)
        self.assertNotIn(secret, created.text)
        profile_id = profile["id"]

        listed = self.client.get("/api/endpoints")
        self.assertEqual(listed.status_code, 200)
        self.assertNotIn(secret, listed.text)
        row = self.service._conn.execute(
            "SELECT * FROM saved_endpoints WHERE id=?", (profile_id,)
        ).fetchone()
        self.assertNotIn(secret, str(tuple(row)))

        run_response = self.client.post("/api/runs", json={
            "ticker": "MRVL", "analysis_date": date.today().isoformat(),
            "provider": "openai_compatible", "backend_url": "http://other.example/v1",
            "quick_model": payload["quick_model"], "deep_model": payload["deep_model"],
            "saved_endpoint_id": profile_id,
        })
        self.assertEqual(run_response.status_code, 202)
        run_id = run_response.json()["id"]
        self.assertNotIn(secret, run_response.text)
        run_request = self.service._request_json(run_id)
        self.assertNotIn(secret, run_request)
        self.assertEqual(json.loads(run_request)["backend_url"], payload["base_url"])
        self.assertEqual(self.service._credentials[run_id][1], secret)

        update = self.client.put(f"/api/endpoints/{profile_id}", json={
            "name": "Renamed gateway", "base_url": "http://127.0.0.1:9124/v1",
            "quick_model": "new-fast", "deep_model": "new-deep",
        })
        self.assertEqual(update.status_code, 200)
        self.assertTrue(update.json()["has_api_key"])
        self.assertEqual(self.service._secret_store.get_secret(profile_id), secret)
        self.assertNotIn(secret, update.text)

        replaced = self.client.put(f"/api/endpoints/{profile_id}", json={
            "name": "Renamed gateway", "base_url": "http://127.0.0.1:9124/v1",
            "quick_model": "new-fast", "deep_model": "new-deep", "api_key": "replacement-key",
        })
        self.assertTrue(replaced.json()["has_api_key"])
        self.assertEqual(self.service._secret_store.get_secret(profile_id), "replacement-key")
        self.assertNotIn("replacement-key", replaced.text)

        cleared = self.client.put(f"/api/endpoints/{profile_id}", json={
            "name": "Renamed gateway", "base_url": "http://127.0.0.1:9124/v1",
            "quick_model": "new-fast", "deep_model": "new-deep", "clear_api_key": True,
        })
        self.assertFalse(cleared.json()["has_api_key"])
        self.assertIsNone(self.service._secret_store.get_secret(profile_id))

        deleted = self.client.delete(f"/api/endpoints/{profile_id}")
        self.assertEqual(deleted.status_code, 200)
        self.assertEqual(self.client.get("/api/endpoints").json(), [])

    def test_saved_endpoint_probe_uses_keychain_key_without_returning_it(self):
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        import threading

        observed_auth = []

        class ModelsHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                observed_auth.append(self.headers.get("Authorization"))
                self.send_response(200 if self.path == "/v1/models" else 404)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"data":[]}')

            def log_message(self, format, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), ModelsHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            created = self.client.post("/api/endpoints", json={
                "name": "Probe profile",
                "base_url": f"http://127.0.0.1:{server.server_port}/v1",
                "quick_model": "fast", "deep_model": "deep", "api_key": "probe-profile-secret",
            })
            self.assertEqual(created.status_code, 201)
            response = self.client.post("/api/test-endpoint", json={
                "provider": "openai_compatible", "saved_endpoint_id": created.json()["id"],
                "backend_url": "http://127.0.0.1:1/credential-capture",
            })
        finally:
            server.shutdown()
            thread.join(timeout=2)
            server.server_close()
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["ok"])
        self.assertEqual(observed_auth, ["Bearer probe-profile-secret"])
        self.assertNotIn("probe-profile-secret", response.text)

    def test_endpoint_test_probes_selected_local_compatible_server(self):
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        import threading

        observed_auth = []

        class ModelsHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                observed_auth.append(self.headers.get("Authorization"))
                if self.path != "/v1/models":
                    self.send_response(404)
                    self.end_headers()
                    return
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"data":[]}')

            def log_message(self, format, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), ModelsHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            response = self.client.post("/api/test-endpoint", json={
                "provider": "openai_compatible",
                "backend_url": f"http://127.0.0.1:{server.server_port}/v1",
                "credential": "endpoint-test-secret",
            })
        finally:
            server.shutdown()
            thread.join(timeout=2)
            server.server_close()
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["ok"])
        self.assertIn("HTTP 200", response.json()["message"])
        self.assertEqual(observed_auth, ["Bearer endpoint-test-secret"])
        self.assertNotIn("endpoint-test-secret", response.text)

    def test_endpoint_probe_rejects_custom_url_for_hosted_provider(self):
        with patch("webapp.app.urlopen") as mocked_open:
            response = self.client.post("/api/test-endpoint", json={
                "provider": "openrouter",
                "backend_url": "http://127.0.0.1:9000/v1",
            })
        self.assertEqual(response.status_code, 422)
        mocked_open.assert_not_called()

    def test_endpoint_probe_rejects_malformed_port(self):
        response = self.client.post("/api/test-endpoint", json={
            "provider": "openai_compatible",
            "backend_url": "http://127.0.0.1:notaport/v1",
        })
        self.assertEqual(response.status_code, 422)

    def test_api_submit_does_not_return_or_persist_credential(self):
        secret = "api-submit-only-secret"
        response = self.client.post("/api/runs", json={
            "ticker": "MRVL", "analysis_date": date.today().isoformat(),
            "provider": "openrouter", "credential": secret,
        })
        self.assertEqual(response.status_code, 202)
        self.assertNotIn(secret, response.text)
        job_id = response.json()["id"]
        self.assertNotIn(secret, self.service._request_json(job_id))
        self.assertEqual(self.service._credentials[job_id][1], secret)

    def test_submit_keeps_credential_in_memory_without_persisting_or_returning_it(self):
        secret = "transient-run-credential"
        job = self.service.submit(RunRequest(
            ticker="MRVL", analysis_date=date.today().isoformat(),
            provider="openrouter", credential=SecretStr(secret)
        ))
        self.assertNotIn(secret, self.service._request_json(job["id"]))
        self.assertNotIn(secret, str(job))
        self.assertEqual(self.service._credentials[job["id"]][1], secret)

    def test_worker_redacts_credential_from_persisted_error(self):
        secret = "failure-path-private-key"

        class FailingGraph:
            def __init__(self, *_args, **_kwargs):
                raise RuntimeError(f"simulated error containing {secret}")

        with TemporaryDirectory() as directory:
            service = JobService(start_worker=True, db_path=Path(directory) / "runs.db")
            try:
                with patch.dict(os.environ, {"OPENROUTER_API_KEY": "ambient-original"}):
                    with patch("webapp.service.TradingAgentsGraph", FailingGraph):
                        submitted = service.submit(RunRequest(
                            ticker="MRVL", analysis_date="2026-10-15",
                            provider="openrouter", credential=SecretStr(secret)
                        ))
                        service._queue.join()
                    failed = service.get(submitted["id"])
                    if failed is None:
                        self.fail("worker did not persist the failed run")
                    self.assertEqual(os.environ["OPENROUTER_API_KEY"], "ambient-original")
                self.assertEqual(failed["status"], "failed")
                self.assertIn("[redacted]", failed["error"])
                self.assertNotIn(secret, failed["error"])
                self.assertNotIn(submitted["id"], service._credentials)
            finally:
                service.close()

    def test_worker_finishes_run_and_persists_report(self):
        class FakeGraph:
            observed_api_key = None

            def __init__(self, *_args, **_kwargs):
                FakeGraph.observed_api_key = os.environ.get("OPENROUTER_API_KEY")
                self.graph = self
                self.propagator = self

            def create_run_state(self, *_args):
                return {}

            def begin_checkpoint(self, *_args):
                return None

            def get_graph_args(self):
                return {}

            def checkpoint_input(self, state):
                return state

            def stream(self, *_args, **_kwargs):
                yield {
                    "market_report": "market notes",
                    "sentiment_report": "sentiment notes",
                    "news_report": "news notes",
                    "fundamentals_report": "fundamentals notes",
                    "investment_debate_state": {"judge_decision": "research decision"},
                    "trader_investment_plan": "trade plan",
                    "risk_debate_state": {"judge_decision": "portfolio decision"},
                    "final_trade_decision": "Buy",
                    "company_of_interest": "MRVL",
                    "trade_date": "2026-10-15",
                }

            def save_reports(self, _state, _ticker, save_path):
                path = Path(save_path) / "complete_report.md"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("test report", encoding="utf-8")
                return path

            def _log_state(self, *_args):
                pass

            def record_decision(self, *_args):
                pass

            def clear_checkpoint_on_success(self, *_args):
                pass

            def end_checkpoint(self):
                pass

        with TemporaryDirectory() as directory:
            config = {
                **__import__("tradingagents.default_config", fromlist=["DEFAULT_CONFIG"]).DEFAULT_CONFIG,
                "results_dir": directory,
                "data_cache_dir": str(Path(directory) / "cache"),
                "memory_log_path": str(Path(directory) / "memory.md"),
            }
            service = JobService(start_worker=True, db_path=Path(directory) / "runs.db")
            try:
                secret = "run-only-openrouter-key"
                with patch.dict(os.environ, {"OPENROUTER_API_KEY": "ambient-original"}):
                    with patch("webapp.service.DEFAULT_CONFIG", config), patch("webapp.service.TradingAgentsGraph", FakeGraph):
                        submitted = service.submit(RunRequest(
                            ticker="MRVL", analysis_date="2026-10-15",
                            analysts=["market", "social", "news", "fundamentals"],
                            provider="openrouter", credential=SecretStr(secret)
                        ))
                        service._queue.join()
                    finished = service.get(submitted["id"])
                    self.assertEqual(FakeGraph.observed_api_key, secret)
                    self.assertEqual(os.environ["OPENROUTER_API_KEY"], "ambient-original")
                self.assertEqual(finished["status"], "completed")
                self.assertTrue(finished["report_available"])
                self.assertEqual(service.report(submitted["id"]), "test report")
                self.assertNotIn(secret, service._request_json(submitted["id"]))
                self.assertNotIn(submitted["id"], service._credentials)
            finally:
                service.close()


if __name__ == "__main__":
    unittest.main()
