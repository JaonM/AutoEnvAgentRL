"""Transient review connection failures must not become repair defects."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/sandbox/run_readonly_review.py"
SPEC = importlib.util.spec_from_file_location("run_readonly_review", SCRIPT)
assert SPEC and SPEC.loader
review = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(review)


class ReadonlyReviewRetryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.response = self.root / "response.json"
        self.stdout = self.root / "stdout.log"
        self.stderr = self.root / "stderr.log"

    def run_review(self) -> int:
        return review.run_review(
            cwd=self.root, prompt="check sandbox", response=self.response,
            stdout=self.stdout, stderr=self.stderr, retry_delay=0,
        )

    def test_transient_failure_retries_then_succeeds(self) -> None:
        calls = 0

        def invoke(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
            nonlocal calls
            calls += 1
            if calls == 1:
                return subprocess.CompletedProcess([], 1, "", "HTTP 503")
            self.response.write_text('{"status":"pass"}', encoding="utf-8")
            return subprocess.CompletedProcess([], 0, "reviewed", "")

        with patch.object(review.subprocess, "run", side_effect=invoke):
            self.assertEqual(self.run_review(), 0)
        self.assertEqual(calls, 2)
        self.assertIn("attempt 2/3", self.stdout.read_text(encoding="utf-8"))

    def test_exhausted_connection_retries_are_infrastructure(self) -> None:
        failed = subprocess.CompletedProcess([], 1, "", "connection reset")
        with patch.object(review.subprocess, "run", return_value=failed) as run:
            self.assertEqual(self.run_review(), review.INFRA_EXIT)
        self.assertEqual(run.call_count, 3)

    def test_product_or_command_error_does_not_retry(self) -> None:
        failed = subprocess.CompletedProcess([], 2, "", "invalid model name")
        with patch.object(review.subprocess, "run", return_value=failed) as run:
            self.assertEqual(self.run_review(), 2)
        self.assertEqual(run.call_count, 1)


if __name__ == "__main__":
    unittest.main()
