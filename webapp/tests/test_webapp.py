import unittest
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from fastapi.testclient import TestClient

from webapp.app import create_app
from webapp.service import JobService, RunRequest, validate_run_request


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

    def test_accepts_ollama_local_endpoint(self):
        run = validate_run_request(RunRequest(
            ticker="MRVL", analysis_date=date.today().isoformat(),
            provider="ollama", backend_url="http://127.0.0.1:11434/v1",
            quick_model="llama3.1", deep_model="llama3.1", analysts=["market"]
        ))
        self.assertEqual(run.provider, "ollama")
        self.assertEqual(run.backend_url, "http://127.0.0.1:11434/v1")

    def test_rejects_empty_analysts(self):
        with self.assertRaises(ValueError):
            validate_run_request(RunRequest(ticker="MRVL", analysis_date="2026-10-15", analysts=[]))


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.service = JobService(start_worker=False, db_path=":memory:")
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

    def test_root_serves_dashboard(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("TradingAgents", response.text)
        self.assertIn("color-scheme:dark", response.text)
        self.assertIn("ollama_local", response.text)
        self.assertIn("lm_studio_local", response.text)
        self.assertIn("vllm_local", response.text)

    def test_worker_finishes_run_and_persists_report(self):
        class FakeGraph:
            def __init__(self, *_args, **_kwargs):
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
                with patch("webapp.service.DEFAULT_CONFIG", config), patch("webapp.service.TradingAgentsGraph", FakeGraph):
                    submitted = service.submit(RunRequest(
                        ticker="MRVL", analysis_date="2026-10-15",
                        analysts=["market", "social", "news", "fundamentals"]
                    ))
                    service._queue.join()
                finished = service.get(submitted["id"])
                self.assertEqual(finished["status"], "completed")
                self.assertTrue(finished["report_available"])
                self.assertEqual(service.report(submitted["id"]), "test report")
            finally:
                service.close()


if __name__ == "__main__":
    unittest.main()
