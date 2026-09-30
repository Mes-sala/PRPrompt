"""Integration checks: run -> seal -> score -> resume -> complete summary."""
import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from prprompt.model import TwoLayerGCN
from prprompt.protocol import sha_file, save_json


def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "experiments" / filename)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


runner = module("v2_runner", "run_benchmark.py")
summary = module("v2_summary", "summarize.py")


class PipelineTests(unittest.TestCase):
    def test_complete_resume_and_incomplete_rejection(self):
        torch.manual_seed(32)
        x = torch.randn(24, 7)
        n = torch.arange(24)
        edges = torch.stack([torch.cat([n, (n+1)%24]), torch.cat([(n+1)%24, n])])
        labels = n % 3
        pool, query = n[:12], n[12:]
        data = (x, edges, labels, pool, query, 3)
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            checkpoint = temp / "encoder.pth"
            torch.save(TwoLayerGCN(7, 8, 8).state_dict(), checkpoint)
            config = {"fit": {"steps": 2, "lr": .01, "weight_decay": .0001},
                      "selection": {"learning_rates": [.01], "weight_decays": [.0001]},
                      "methods": ["FrozenLinear", "PRPrompt"], "shots": [1, 3],
                      "split_seeds": [31, 32], "train_seeds": [0]}
            save_json(temp / "config.json", config)
            save_json(temp / "manifest.json", {"jobs": [{"id": "fixture", "dataset": "fixture", "family": "fixture",
                        "checkpoint": str(checkpoint), "checkpoint_sha256": sha_file(checkpoint), "input_dim": 7,
                        "hidden_dim": 8, "output_dim": 8}], "data_root": "unused", "unavailable": []})
            args = SimpleNamespace(config=temp / "config.json", manifest=temp / "manifest.json", checkpoint_id=None,
                                   methods=None, smoke=False, output=temp / "outputs", dry_run=False, device="cpu", threads=2)
            with patch.object(runner, "load_dataset", return_value=data), contextlib.redirect_stdout(io.StringIO()):
                runner.run(args)
                output = next((temp / "outputs/paper").iterdir())
                files = list(output.rglob("metrics.json"))
                self.assertEqual(len(files), 8)
                original_mtimes = {p: p.stat().st_mtime_ns for p in files}
                with patch.object(runner, "fit_predict", side_effect=AssertionError("Resume unexpectedly retrained")):
                    runner.run(args)
                self.assertEqual(original_mtimes, {p: p.stat().st_mtime_ns for p in files})
                summary.summarize(output)
                self.assertTrue((output / "method_comparison.csv").is_file())
                p = files[0]
                original = p.read_bytes()
                p.unlink()
                with self.assertRaisesRegex(ValueError, "runs missing"):
                    summary.summarize(output)
                p.write_bytes(original)
                state = p.parent / "state.pt"
                with state.open("ab") as f:
                    f.write(b"tamper")
                with self.assertRaisesRegex(ValueError, "Resume state/selection changed"):
                    runner.run(args)
                with self.assertRaisesRegex(ValueError, "Changed state"):
                    summary.summarize(output)


if __name__ == "__main__":
    unittest.main(verbosity=2)
