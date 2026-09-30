import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).parents[2] / "scripts" / "sandbox" / "defect_attempt_budget.py"
SPEC = importlib.util.spec_from_file_location("defect_attempt_budget", SCRIPT)
budget = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(budget)


class DefectAttemptBudgetTest(unittest.TestCase):
    def test_each_distinct_defect_gets_its_own_persistent_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory) / "attempts.json"
            first = {
                "category": "semantic correctness", "tool_name": "audit_material_labels",
                "file": "task_impl.py", "contract_reference": "label specification",
                "fix_required": "Check the percentage tolerance",
                "evidence": "Textile totals of 99% are accepted despite a 0.5% tolerance",
            }
            second = {**first,
                "fix_required": "Separate missing evidence from confirmed violations",
                "evidence": "Missing lab reports are returned as confirmed clause 5.2 violations",
            }
            self.assertEqual(budget.reserve_attempt(state, first, 3), ("DEFECT-0001", 1))
            self.assertEqual(budget.reserve_attempt(state, first, 3), ("DEFECT-0001", 2))
            self.assertEqual(budget.reserve_attempt(state, second, 3), ("DEFECT-0002", 1))
            self.assertEqual(budget.reserve_attempt(state, first, 3), ("DEFECT-0001", 3))
            with self.assertRaisesRegex(RuntimeError, "DEFECT-0001 exhausted"):
                budget.reserve_attempt(state, first, 3)
            self.assertEqual(budget.reserve_attempt(state, second, 3), ("DEFECT-0002", 2))
            persisted = json.loads(state.read_text())
            self.assertEqual([item["attempts"] for item in persisted["defects"]], [3, 2])

    def test_delivery_failure_uses_failure_gate_instead_of_full_log(self):
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory) / "attempts.json"
            first = {"category": "delivery_failure", "contract_reference": "outer workflow validation",
                     "evidence": '{"failures":[{"gate":"counterfactual_execution","message":"corrupted_arguments_1: execution did not produce a client rejection"}]}'}
            same = {**first, "evidence": first["evidence"] + "\nrequest_id=req-123456abcdef"}
            other = {**first, "evidence": '{"failure":{"type":"SandboxError","message":"step 3 expected HTTP 200, got 500"}}'}
            self.assertEqual(budget.reserve_attempt(state, first, 3), ("DEFECT-0001", 1))
            self.assertEqual(budget.reserve_attempt(state, same, 3), ("DEFECT-0001", 2))
            self.assertEqual(budget.reserve_attempt(state, other, 3), ("DEFECT-0002", 1))

    def test_similar_findings_for_distinct_fields_have_independent_budgets(self):
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory) / "attempts.json"
            first = {
                "category": "semantic correctness", "tool_name": "audit_material_labels",
                "file": "task_impl.py",
                "fix_required": "Validate material_code in the item lookup result",
                "evidence": "The material_code field is missing from lookup output",
            }
            second = {**first,
                "fix_required": "Validate supplier_code in the item lookup result",
                "evidence": "The supplier_code field is missing from lookup output",
            }
            for attempt in range(1, 4):
                self.assertEqual(budget.reserve_attempt(state, first, 3), ("DEFECT-0001", attempt))
            self.assertEqual(budget.reserve_attempt(state, second, 3), ("DEFECT-0002", 1))
            rephrased = {**second,
                "evidence": "The supplier_code field is missing from lookup output.",
            }
            self.assertEqual(budget.reserve_attempt(state, rephrased, 3), ("DEFECT-0002", 2))

    def test_cli_moves_to_next_defect_when_first_budget_is_exhausted(self):
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory) / "attempts.json"
            defects_file = Path(directory) / "defects.json"
            defects = [
                {"category": "missing_rule", "evidence": "alpha"},
                {"category": "wrong_calculation", "evidence": "beta"},
            ]
            defects_file.write_text(json.dumps(defects))
            arguments = [sys.executable, str(SCRIPT), "--state", str(state),
                         "--defects", str(defects_file), "--max-attempts", "1"]
            first = subprocess.run(arguments, capture_output=True, text=True, check=True)
            subprocess.run([sys.executable, str(SCRIPT), "--state", str(state),
                            "--commit-id", "DEFECT-0001"], check=True)
            second = subprocess.run(arguments, capture_output=True, text=True, check=True)
            subprocess.run([sys.executable, str(SCRIPT), "--state", str(state),
                            "--commit-id", "DEFECT-0002"], check=True)
            exhausted = subprocess.run(arguments, capture_output=True, text=True)
            self.assertEqual(first.stdout.strip(), "0 DEFECT-0001 1")
            self.assertEqual(second.stdout.strip(), "1 DEFECT-0002 1")
            self.assertNotEqual(exhausted.returncode, 0)
            self.assertIn("all currently reported defects exhausted", exhausted.stderr)

    def test_interrupted_or_released_attempt_does_not_spend_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory) / "attempts.json"
            defect = {"category": "business", "evidence": "wrong total"}
            self.assertEqual(budget.reserve_attempt(state, defect, 1, pending=True),
                             ("DEFECT-0001", 1))
            self.assertEqual(budget.reserve_attempt(state, defect, 1, pending=True),
                             ("DEFECT-0001", 1))
            budget.finish_pending(state, "DEFECT-0001", release=True)
            self.assertEqual(budget.reserve_attempt(state, defect, 1, pending=True),
                             ("DEFECT-0001", 1))
            budget.finish_pending(state, "DEFECT-0001", release=False)
            with self.assertRaisesRegex(RuntimeError, "exhausted"):
                budget.reserve_attempt(state, defect, 1, pending=True)

    def test_total_budget_spans_distinct_defects_and_excludes_released_attempts(self):
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory) / "attempts.json"
            first = {"category": "business", "evidence": "wrong total"}
            second = {"category": "business", "evidence": "missing item"}
            third = {"category": "business", "evidence": "wrong status"}
            self.assertEqual(budget.reserve_attempt(
                state, first, 3, maximum_total=2, pending=True,
            ), ("DEFECT-0001", 1))
            budget.finish_pending(state, "DEFECT-0001", release=True)
            self.assertEqual(budget.reserve_attempt(
                state, first, 3, maximum_total=2,
            ), ("DEFECT-0001", 1))
            self.assertEqual(budget.reserve_attempt(
                state, second, 3, maximum_total=2,
            ), ("DEFECT-0002", 1))
            with self.assertRaisesRegex(RuntimeError, "total repair attempts"):
                budget.reserve_attempt(state, third, 3, maximum_total=2)
            self.assertEqual([entry["attempts"] for entry in json.loads(state.read_text())["defects"]], [1, 1])

    def test_reward_defect_rephrasing_keeps_same_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory) / "attempts.json"
            first = {
                "category": "reward_semantics", "tool_name": "parse_inventory_sales_text",
                "file": "BUILD_CONTRACT.json", "contract_reference": "metric_implementations",
                "evidence": "The process metric requires raw_text to equal one canonical CSV fixture.",
            }
            rephrased = {**first,
                "contract_reference": "metrics and metric_implementations",
                "evidence": "A fixed expected-call map for raw_text denies credit to other valid CSV rows.",
            }
            self.assertEqual(budget.reserve_attempt(state, first, 3), ("DEFECT-0001", 1))
            self.assertEqual(budget.reserve_attempt(state, rephrased, 3), ("DEFECT-0001", 2))


if __name__ == "__main__":
    unittest.main()
