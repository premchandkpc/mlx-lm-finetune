"""Regression checks for sweep isolation and dataset structure."""
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import sweep
import prepare_data


class ReviewRegressionTests(unittest.TestCase):
    def test_structured_exact_dedup_keeps_role_boundaries(self):
        a = {"messages": [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "world"}]}
        b = {"messages": [{"role": "system", "content": "hello"}, {"role": "assistant", "content": "world"}]}
        self.assertNotEqual(prepare_data.exact_key(a), prepare_data.exact_key(b))

    def test_sweep_requires_validation_even_if_test_exists(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            data = root / "data"
            data.mkdir()
            (data / "test.jsonl").write_text('{"text":"test"}\n')
            config = root / "config.yaml"
            config.write_text("data: " + str(data) + "\n")
            with self.assertRaisesRegex(SystemExit, "valid.jsonl"):
                sweep.main(["--config", str(config), "--out-root", str(root / "out")])

    def test_split_rejects_invalid_ratios(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "input.jsonl"
            source.write_text('{"text":"hello"}\n')
            args = prepare_data.build_parser().parse_args(
                ["--input", str(source), "--valid-split", "0.8", "--test-split", "0.3"])
            self.assertEqual(prepare_data.run(args), 1)


if __name__ == "__main__":
    unittest.main()
