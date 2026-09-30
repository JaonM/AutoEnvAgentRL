import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]

class CompiledHookOwnershipTest(unittest.TestCase):
    def test_real_snapshot_restores_compiled_hook_but_allows_declared_extension(self):
        spec = importlib.util.spec_from_file_location("ownership_planner", ROOT / "scripts/sandbox/generate_development_plan.py")
        planner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(planner)
        workflow = (ROOT / "scripts/develop_sandbox_with_agent.sh").read_text()
        functions = workflow[workflow.index("platform_owned_files=("):workflow.index("validate_delivery() {")]
        for custom in (False, True):
            contract = {"artifacts": {"generation_pipeline": {"backend": "code_agent"}}}
            if custom:
                contract["tools"] = [{"function": {"name": "custom_business_operation"}}]
            plan = planner.build_plan(contract)
            self.assertEqual(plan["task_implementation_editable"], custom)
            integration = next(node for node in plan["nodes"] if node["id"] == "business_integration")
            self.assertEqual(integration["outputs"], ["tests/business_integration"])
            with self.subTest(custom=custom), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "task_impl.py").write_text("original implementation")
                (root / "development_plan.json").write_text(json.dumps(plan))
                script = functions + '\noutput_path="$1"\nbackup=$(snapshot_platform_assets)\nprintf "modified implementation" > "$output_path/task_impl.py"\nrestore_and_reject_platform_changes "$backup"\n'
                result = subprocess.run(["bash", "-c", script, "ownership-test", str(root)], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0 if custom else 1, result.stderr + result.stdout)
                self.assertEqual((root / "task_impl.py").read_text(), "modified implementation" if custom else "original implementation")
