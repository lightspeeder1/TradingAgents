import json
import os
import unittest
from datetime import date, datetime, timedelta, timezone
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
        self.assertIn(".secondary:disabled{opacity:.55;cursor:not-allowed}", response.text)
        self.assertIn('id="reportsNav"', response.text)
        self.assertIn('id="reportsOverlay"', response.text)
        self.assertIn('id="activityItems"', response.text)
        self.assertIn("Live activity", response.text)
        self.assertIn("after_id=", response.text)
        self.assertIn("live report draft", response.text)
        self.assertIn('id="schedulesOverlay"', response.text)

    def test_run_activity_endpoint_returns_ordered_events_without_payloads(self):
        run = self.service.submit(RunRequest(
            ticker="AMD", analysis_date=date.today().isoformat(), provider="ollama",
            quick_model="local", deep_model="local", analysts=["market"],
        ))
        self.service.record_activity(run["id"], "Running node: Market Analyst", "node_start")
        self.service.record_activity(run["id"], "Market Analyst finished", "node_complete")
        response = self.client.get(f"/api/runs/{run['id']}/activity?after_id=0")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual([item["message"] for item in body["items"]], [
            "Queued; waiting for the serial analysis worker.",
            "Running node: Market Analyst", "Market Analyst finished",
        ])
        self.assertEqual(body["latest_id"], body["items"][-1]["id"])
        self.assertNotIn("payload", body["items"][0])
        self.assertEqual(self.client.get("/api/runs/missing/activity").status_code, 404)

    def test_schedule_crud_and_run_now_preserve_credentials_and_recurrence(self):
        run_request = {
            "ticker": "AMD", "analysis_date": date.today().isoformat(),
            "analysts": ["market"], "provider": "ollama",
            "quick_model": "local", "deep_model": "local",
            "backend_url": "http://127.0.0.1:11434/v1",
        }
        payload = {
            "name": "Weekday open", "cadence": "daily", "local_time": "09:30",
            "timezone": "America/Los_Angeles", "weekday": None, "enabled": True,
            "run_request": run_request,
        }
        created = self.client.post("/api/schedules", json=payload)
        self.assertEqual(created.status_code, 201, created.text)
        schedule = created.json()
        self.assertEqual(schedule["name"], "Weekday open")
        self.assertTrue(schedule["enabled"])
        self.assertNotIn("run_request", schedule)
        self.assertNotIn("backend_url", schedule)
        details = self.client.get(f"/api/schedules/{schedule['id']}").json()
        self.assertEqual(details["run_request"]["ticker"], "AMD")
        self.assertNotIn("credential", details["run_request"])
        self.assertTrue(schedule["next_run_at"])
        self.assertEqual(self.client.get("/api/schedules").json()[0]["id"], schedule["id"])

        payload.update({"name": "Weekly open", "cadence": "weekly", "weekday": 0, "local_time": "10:15"})
        updated = self.client.put(f"/api/schedules/{schedule['id']}", json=payload)
        self.assertEqual(updated.status_code, 200, updated.text)
        self.assertEqual(updated.json()["cadence"], "weekly")
        self.assertEqual(updated.json()["weekday"], 0)

        paused = self.client.post(f"/api/schedules/{schedule['id']}/pause")
        self.assertFalse(paused.json()["enabled"])
        resumed = self.client.post(f"/api/schedules/{schedule['id']}/resume")
        self.assertTrue(resumed.json()["enabled"])

        queued = self.client.post(f"/api/schedules/{schedule['id']}/run-now")
        self.assertEqual(queued.status_code, 202, queued.text)
        self.assertEqual(queued.json()["ticker"], "AMD")
        self.assertEqual(self.client.get("/api/schedules").json()[0]["last_run_id"], queued.json()["id"])
        self.assertEqual(self.client.delete(f"/api/schedules/{schedule['id']}").status_code, 200)
        self.assertEqual(self.client.get("/api/schedules").json(), [])

    def test_startup_auto_resumes_checkpointed_crash_once(self):
        with TemporaryDirectory() as directory:
            database = Path(directory) / "runs.sqlite3"
            first = JobService(start_worker=False, db_path=database, secret_store=MemorySecretStore())
            original = first.submit(RunRequest(
                ticker="AMD", analysis_date=date.today().isoformat(), provider="ollama",
                backend_url="http://127.0.0.1:11434/v1", quick_model="local", deep_model="local",
                analysts=["market"], checkpoint=True,
            ))
            first._update(original["id"], status="running")
            first.close()

            with patch("webapp.service.checkpoint_step", return_value=4):
                recovered = JobService(start_worker=False, db_path=database, secret_store=MemorySecretStore())
            try:
                old = recovered.get(original["id"])
                self.assertEqual(old["status"], "interrupted")
                runs = recovered.list()
                candidates = [run for run in runs if run["id"] != original["id"]]
                self.assertEqual(len(candidates), 1)
                self.assertEqual(candidates[0]["status"], "queued")
                self.assertEqual(candidates[0]["recovery_of"], original["id"])
                self.assertEqual(candidates[0]["auto_resume_count"], 1)
            finally:
                recovered.close()

    def test_auto_resume_chain_stops_after_one_automatic_recovery(self):
        with TemporaryDirectory() as directory:
            database = Path(directory) / "runs.sqlite3"
            first = JobService(start_worker=False, db_path=database, secret_store=MemorySecretStore())
            original = first.submit(RunRequest(
                ticker="AMD", analysis_date=date.today().isoformat(), provider="ollama",
                backend_url="http://127.0.0.1:11434/v1", quick_model="local", deep_model="local",
                analysts=["market"], checkpoint=True,
            ))
            first._update(original["id"], status="running")
            first.close()
            with patch("webapp.service.checkpoint_step", return_value=4):
                second = JobService(start_worker=False, db_path=database, secret_store=MemorySecretStore())
            recovered_id = next(run["id"] for run in second.list() if run["id"] != original["id"])
            second._update(recovered_id, status="running")
            second.close()
            with patch("webapp.service.checkpoint_step", return_value=8):
                third = JobService(start_worker=False, db_path=database, secret_store=MemorySecretStore())
            try:
                self.assertEqual(len(third.list()), 2)
                self.assertEqual(third.get(recovered_id)["status"], "interrupted")
            finally:
                third.close()

    def test_startup_skips_auto_resume_if_environment_key_was_lost(self):
        with TemporaryDirectory() as directory:
            database = Path(directory) / "runs.sqlite3"
            with patch.dict(os.environ, {"OPENROUTER_API_KEY": "restart-key"}):
                first = JobService(start_worker=False, db_path=database, secret_store=MemorySecretStore())
                original = first.submit(RunRequest(
                    ticker="AMD", analysis_date=date.today().isoformat(), provider="openrouter",
                    quick_model="fast", deep_model="deep", analysts=["market"], checkpoint=True,
                ))
                first._update(original["id"], status="running")
                first.close()
            with patch.dict(os.environ, {"OPENROUTER_API_KEY": ""}):
                with patch("webapp.service.checkpoint_step", return_value=4):
                    recovered = JobService(start_worker=False, db_path=database, secret_store=MemorySecretStore())
            try:
                self.assertEqual(len(recovered.list()), 1)
                self.assertEqual(recovered.get(original["id"])["status"], "interrupted")
                with recovered._db_lock:
                    row = recovered._conn.execute(
                        "SELECT environment_credential FROM runs WHERE id=?", (original["id"],)
                    ).fetchone()
                self.assertEqual(row["environment_credential"], 1)
            finally:
                recovered.close()

    def test_startup_does_not_persist_or_auto_resume_transient_credentials(self):
        with TemporaryDirectory() as directory:
            database = Path(directory) / "runs.sqlite3"
            first = JobService(start_worker=False, db_path=database, secret_store=MemorySecretStore())
            original = first.submit(RunRequest(
                ticker="AMD", analysis_date=date.today().isoformat(), provider="openai",
                quick_model="fast", deep_model="deep", analysts=["market"],
                credential="temporary-only", checkpoint=True,
            ))
            first._update(original["id"], status="running")
            first.close()
            with patch("webapp.service.checkpoint_step", return_value=4):
                recovered = JobService(start_worker=False, db_path=database, secret_store=MemorySecretStore())
            try:
                self.assertEqual(recovered.list(), [recovered.get(original["id"])])
                self.assertEqual(recovered.get(original["id"])["status"], "interrupted")
                with recovered._db_lock:
                    stored = recovered._conn.execute("SELECT request_json FROM runs WHERE id=?", (original["id"],)).fetchone()[0]
                self.assertNotIn("temporary-only", stored)
            finally:
                recovered.close()

    def test_scheduler_dispatches_due_occurrence_once_and_advances_it(self):
        payload = {
            "name": "Dispatch test", "cadence": "daily", "local_time": "09:30",
            "timezone": "America/Los_Angeles", "weekday": None, "enabled": True,
            "run_request": {
                "ticker": "AMD", "analysis_date": date.today().isoformat(),
                "analysts": ["market"], "provider": "ollama", "quick_model": "local",
                "deep_model": "local", "backend_url": "http://127.0.0.1:11434/v1",
            },
        }
        schedule = self.client.post("/api/schedules", json=payload).json()
        now = datetime.now(timezone.utc).replace(microsecond=0)
        due = (now - timedelta(minutes=5)).isoformat(timespec="seconds")
        with self.service._db_lock:
            self.service._conn.execute("UPDATE schedules SET next_run_at=? WHERE id=?", (due, schedule["id"]))
            self.service._conn.commit()

        first = self.service._enqueue_due_schedules(now)
        second = self.service._enqueue_due_schedules(now)
        self.assertEqual(len(first), 1)
        self.assertEqual(second, [])
        updated = self.service.get_schedule(schedule["id"])
        self.assertEqual(updated["last_run_id"], first[0])
        self.assertGreater(datetime.fromisoformat(updated["next_run_at"]), now)
        with self.service._db_lock:
            occurrences = self.service._conn.execute(
                "SELECT COUNT(*) FROM schedule_occurrences WHERE schedule_id=?", (schedule["id"],)
            ).fetchone()[0]
        self.assertEqual(occurrences, 1)

    def test_schedule_rejects_transient_credentials_and_invalid_timezone(self):
        payload = {
            "name": "Unsafe", "cadence": "daily", "local_time": "09:30",
            "timezone": "Mars/Olympus", "enabled": True,
            "run_request": {"ticker": "AMD", "credential": "must-not-persist"},
        }
        response = self.client.post("/api/schedules", json=payload)
        self.assertEqual(response.status_code, 422)

    def test_reports_catalog_lists_report_state_and_safe_download(self):
        run = self.service.submit(RunRequest(
            ticker="MRVL", analysis_date=date.today().isoformat(), provider="openrouter"
        ))
        with TemporaryDirectory() as directory:
            report_path = Path(directory) / "draft.md"
            report_path.write_text("# Interim report\n\nWorking notes", encoding="utf-8")
            self.service._update(run["id"], status="failed", report_path=str(report_path))

            catalog = self.client.get("/api/reports?limit=10&offset=0")
            self.assertEqual(catalog.status_code, 200)
            body = catalog.json()
            self.assertEqual(body["total"], 1)
            self.assertEqual(body["items"][0]["id"], run["id"])
            self.assertTrue(body["items"][0]["available"])
            self.assertNotIn("report_path", body["items"][0])

            download = self.client.get(f"/api/reports/{run['id']}/download")
            self.assertEqual(download.status_code, 200)
            self.assertEqual(download.text, "# Interim report\n\nWorking notes")
            self.assertIn("attachment; filename=", download.headers["content-disposition"])

    def test_reports_catalog_marks_missing_artifact_unavailable(self):
        run = self.service.submit(RunRequest(
            ticker="AMD", analysis_date=date.today().isoformat(), provider="openrouter"
        ))
        self.service._update(run["id"], status="failed", report_path="/missing/report.md")
        response = self.client.get("/api/reports")
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()["items"][0]["available"])
        self.assertEqual(self.client.get(f"/api/reports/{run['id']}/download").status_code, 404)

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
                return {"stream_mode": "values"}

            def checkpoint_input(self, state):
                return state

            def stream(self, *_args, **kwargs):
                FakeGraph.requested_stream_mode = kwargs.get("stream_mode")
                yield ("debug", {"type": "task", "payload": {"name": "research_node", "input": {"private": "must-not-log"}}})
                yield ("values", {
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
                })
                yield ("debug", {"type": "task_result", "payload": {"name": "research_node", "result": {"private": "must-not-log"}}})

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
                self.assertEqual(set(FakeGraph.requested_stream_mode), {"values", "debug"})
                activity = service.list_activity(submitted["id"])["items"]
                self.assertIn("Running graph step: research node.", [item["message"] for item in activity])
                self.assertIn("Finished graph step: research node.", [item["message"] for item in activity])
                self.assertNotIn("must-not-log", str(activity))
                self.assertNotIn(secret, service._request_json(submitted["id"]))
                self.assertNotIn(submitted["id"], service._credentials)
            finally:
                service.close()


if __name__ == "__main__":
    unittest.main()
