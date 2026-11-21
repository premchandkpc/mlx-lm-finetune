"""Regression checks for sweep isolation and dataset structure."""
import math
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import sweep
import prepare_data
import compare


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

    def test_group_split_never_exceeds_holdout_target(self):
        examples = [
            {"text": f"item-{i}", "_group": "large" if i < 15 else str(i)}
            for i in range(20)
        ]
        train, valid, test, notes = prepare_data.split_examples(
            examples, 0.2, 0.2, 1, True, 42)
        self.assertLessEqual(len(valid), 4)
        self.assertLessEqual(len(test), 4)
        self.assertEqual(len(train) + len(valid) + len(test), 20)
        self.assertFalse(any(prepare_data.audit_leakage(train, valid, test).values()))

    def test_perplexity_does_not_hide_divergence(self):
        self.assertTrue(math.isinf(compare.perplexity(1000.0)))
        self.assertAlmostEqual(compare.perplexity(0.0), 1.0)

    def test_multiturn_evaluation_uses_only_prior_messages(self):
        row = {"messages": [
            {"role": "system", "content": "Be helpful"},
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "answer one"},
            {"role": "user", "content": "second"},
            {"role": "assistant", "content": "answer two"},
        ]}
        pairs = compare.split_prompt_references(row)
        self.assertEqual(len(pairs), 2)
        self.assertEqual(pairs[0][1], "answer one")
        self.assertEqual(len(pairs[0][0]), 2)
        self.assertEqual(pairs[1][1], "answer two")
        self.assertEqual(len(pairs[1][0]), 4)

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
