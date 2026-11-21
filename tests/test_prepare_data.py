import json
from pathlib import Path

import pytest

import prepare_data as pd

ROOT = Path(__file__).resolve().parent.parent


def chat(user, assistant, system=None):
    msgs = [{"role": "system", "content": system}] if system else []
    msgs += [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]
    return {"messages": msgs}


def write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def read_jsonl(path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


# ---------- validation / conversion ----------

@pytest.mark.parametrize("ex,reason", [
    ({"messages": []}, "bad_messages"),
    ({"messages": [{"role": "user", "content": "hi"}]}, "chat_not_ending_in_assistant"),
    ({"messages": [{"role": "robot", "content": "x"}]}, "bad_role"),
    ({"messages": [{"role": "user", "content": "  "}, {"role": "assistant", "content": "a"}]},
     "empty_content"),
    ({"messages": [{"role": "assistant", "content": "a"}]}, "chat_without_user_turn"),
    ({"prompt": "", "completion": "x"}, "empty_prompt"),
    ({"prompt": "x", "completion": ""}, "empty_completion"),
    ({"text": " "}, "empty_text"),
])
def test_validate_rejects(ex, reason):
    assert pd.validate(ex) == reason


def test_validate_accepts_good_rows():
    assert pd.validate(chat("q", "a", system="s")) is None
    assert pd.validate({"prompt": "p", "completion": "c"}) is None
    assert pd.validate({"text": "hello"}) is None


def test_convert_roundtrips():
    ex = chat("q", "a")
    c = pd.convert(ex, "completions")
    assert c == {"prompt": "user: q", "completion": "a"}
    back = pd.convert({"prompt": "p", "completion": "c"}, "chat")
    assert [m["role"] for m in back["messages"]] == ["user", "assistant"]
    assert pd.convert({"text": "t"}, "chat")["messages"][1]["content"] == "t"
    assert pd.convert({"text": "t"}, "completions") is None  # nothing to complete


def test_convert_survives_malformed_messages():
    assert pd.convert({"messages": [{"nope": 1}]}, "text") is None
    assert pd.convert({"messages": [{"nope": 1}, {"role": "assistant", "content": "x"}]},
                      "completions") is None


# ---------- dedup ----------

def test_exact_dedup_ignores_case_and_whitespace():
    rows = [chat("What is  MLX?", "An array framework."),
            chat("what is mlx?", "an array   framework."),
            chat("different", "row")]
    out, dropped = pd.dedup_exact(rows)
    assert dropped == 1 and len(out) == 2


def test_near_dedup_catches_light_edits_but_keeps_distinct():
    pytest.importorskip("numpy")
    base = ("Gradient checkpointing trades extra compute for lower memory by recomputing "
            "activations during the backward pass instead of storing them all in memory")
    edited = base + " today"
    other = ("A tokenizer splits raw text into subword units so that the model can map "
             "each unit to an embedding vector and process sequences of integers")
    rows = [chat("explain A", base), chat("explain A", edited), chat("explain B", other)]
    out, dropped = pd.dedup_near(rows, threshold=0.8)
    assert dropped == 1
    assert out[0] is rows[0] and out[1] is rows[2]  # first occurrence kept


def test_near_dedup_threshold_is_respected():
    pytest.importorskip("numpy")
    a = chat("q", " ".join(f"w{i}" for i in range(40)))
    b = chat("q", " ".join(f"w{i}" for i in range(20)) + " " + " ".join(f"z{i}" for i in range(20)))
    out, dropped = pd.dedup_near([a, b], threshold=0.95)
    assert dropped == 0


# ---------- splitting / leakage ----------

def make_rows(n):
    return [chat(f"question number {i}", f"answer number {i}") for i in range(n)]


def test_split_sizes_and_disjoint():
    rows = make_rows(100)
    tr, va, te, notes = pd.split_examples(rows, 0.1, 0.1, 4, True, 0)
    assert (len(tr), len(va), len(te)) == (80, 10, 10)
    assert not any(pd.audit_leakage(tr, va, te).values())


def test_same_prompt_never_straddles_splits():
    rows = []
    for i in range(30):  # every prompt appears 3x with different answers
        for j in range(3):
            rows.append(chat(f"prompt {i}", f"answer {i}-{j}"))
    tr, va, te, _ = pd.split_examples(rows, 0.2, 0.2, 4, True, 1)
    keys = lambda s: {pd.prompt_key(e) for e in s}
    assert not (keys(tr) & keys(va)) and not (keys(tr) & keys(te)) and not (keys(va) & keys(te))
    assert len(va) > 0 and len(te) > 0


def test_explicit_group_key_keeps_documents_together():
    rows = []
    for d in range(20):
        for j in range(5):
            ex = chat(f"doc {d} question {j}", f"doc {d} answer {j}")
            ex[pd.GROUP_FIELD] = f"doc{d}"
            rows.append(ex)
    tr, va, te, _ = pd.split_examples(rows, 0.2, 0.0, 4, True, 3)
    assert {e[pd.GROUP_FIELD] for e in tr}.isdisjoint({e[pd.GROUP_FIELD] for e in va})


def test_tiny_splits_are_dropped_not_written_and_train_is_protected():
    rows = make_rows(6)
    tr, va, te, notes = pd.split_examples(rows, 0.5, 0.5, 4, True, 0)
    assert len(tr) >= 4 and not va and not te  # train > valid > test priority
    assert notes


def test_min_split_enforced_on_small_data():
    rows = make_rows(18)
    tr, va, te, _ = pd.split_examples(rows, 0.05, 0.05, 4, True, 0)
    assert (len(tr), len(va), len(te)) == (10, 4, 4)


def test_audit_detects_leakage():
    a, b = chat("same", "x"), chat("same", "y")
    rep = pd.audit_leakage([a], [b], [])
    assert rep["prompt_train_valid"] == 1


def test_split_is_deterministic_per_seed():
    rows = make_rows(50)
    a = pd.split_examples(rows, 0.1, 0.1, 4, True, 7)
    b = pd.split_examples(rows, 0.1, 0.1, 4, True, 7)
    c = pd.split_examples(rows, 0.1, 0.1, 4, True, 8)
    assert a[0] == b[0] and a[0] != c[0]


def test_suggest_iters():
    assert pd.suggest_iters(1000, 4, 3) == 750
    assert pd.suggest_iters(10, 4, 3) == 8
    assert pd.suggest_iters(10, 4, 3, grad_accum=2) == 4
    assert pd.suggest_iters(0, 4, 3) == 1


def test_percentiles():
    assert pd.percentiles([]) == {}
    p = pd.percentiles(list(range(1, 101)))
    assert (p["min"], p["p50"], p["p90"], p["max"]) == (1, 50, 90, 100)


# ---------- CLI end to end ----------

def test_cli_end_to_end(tmp_path):
    src, out = tmp_path / "raw.jsonl", tmp_path / "out"
    rows = make_rows(60) + [chat("question number 3", "answer number 3")]  # exact dup
    rows.append({"messages": [{"role": "user", "content": "dangling"}]})   # invalid
    src.write_text("\n".join(json.dumps(r) for r in rows) + "\nnot json\n", encoding="utf-8")

    rc = pd.main(["--input", str(src), "--output", str(out), "--format", "chat",
                  "--valid-split", "0.1", "--test-split", "0.1", "--min-split", "4"])
    assert rc == 0
    stats = json.loads((out / "stats.json").read_text())
    assert stats["loaded"] == 62
    assert stats["dropped"]["duplicate_exact"] == 1
    assert stats["dropped"]["chat_not_ending_in_assistant"] == 1
    assert stats["dropped"]["invalid_json"] == 1
    assert sum(stats["splits"].values()) == 60
    assert not any(stats["leakage_audit"].values())
    # internal bookkeeping must never leak into the training files
    for name in ("train", "valid", "test"):
        for r in read_jsonl(out / f"{name}.jsonl"):
            assert set(r) == {"messages"}


def test_cli_removes_stale_splits(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "test.jsonl").write_text('{"text": "stale"}\n')
    src = tmp_path / "a.jsonl"
    write_jsonl(src, make_rows(20))
    assert pd.main(["--input", str(src), "--output", str(out), "--format", "chat"]) == 0
    assert not (out / "test.jsonl").exists()  # no --test-split => stale file deleted


def test_cli_errors(tmp_path):
    assert pd.main(["--input", str(tmp_path / "missing"), "--output", str(tmp_path / "o")]) == 1
    empty = tmp_path / "empty"
    empty.mkdir()
    assert pd.main(["--input", str(empty), "--output", str(tmp_path / "o")]) == 1
    f = tmp_path / "x.jsonl"
    write_jsonl(f, make_rows(5))
    assert pd.main(["--input", str(f), "--output", str(tmp_path / "o"),
                    "--max-tokens", "10"]) == 1  # needs --tokenizer


def test_cli_group_key_and_length_filter(tmp_path):
    rows = []
    for d in range(12):
        for j in range(3):
            r = chat(f"doc {d} q{j}", f"doc {d} a{j}")
            r["doc"] = f"d{d}"
            rows.append(r)
    rows.append(chat("x" * 500, "y" * 500))
    src, out = tmp_path / "g.jsonl", tmp_path / "out"
    write_jsonl(src, rows)
    rc = pd.main(["--input", str(src), "--output", str(out), "--format", "chat",
                  "--group-key", "doc", "--max-chars", "300",
                  "--valid-split", "0.2", "--min-split", "4"])
    assert rc == 0
    stats = json.loads((out / "stats.json").read_text())
    assert stats["dropped"]["too_long_chars"] == 1
    # the row-level `doc` field is dropped on conversion; grouping used the raw value
    assert not any(stats["leakage_audit"].values())


def test_sample_dataset_is_clean():
    rows = [json.loads(l) for l in (ROOT / "data/sample/chat_sample.jsonl").read_text().splitlines()]
    assert all(pd.validate(r) is None for r in rows)
    _, dropped = pd.dedup_exact(rows)
    assert dropped == 0


# ---------- mixed-format guard (auto mode) ----------

def test_shape_of():
    assert pd.shape_of(chat("q", "a")) == "chat"
    assert pd.shape_of({"text": "t"}) == "text"
    assert pd.shape_of({"prompt": "p", "completion": "c"}) == "completions"
    assert pd.shape_of({"foo": 1}) == "unknown"


def test_mixed_formats_rejected_in_auto_mode(tmp_path, capsys):
    src = tmp_path / "mixed.jsonl"
    write_jsonl(src, make_rows(10) + [{"text": f"plain text row {i}"} for i in range(10)])
    rc = pd.main(["--input", str(src), "--output", str(tmp_path / "o")])
    err = capsys.readouterr().err
    assert rc == 1 and "mixed example formats" in err and "chat=10" in err and "text=10" in err
    assert not (tmp_path / "o" / "train.jsonl").exists()


def test_forced_format_converts_mixed_input(tmp_path):
    src = tmp_path / "mixed.jsonl"
    write_jsonl(src, make_rows(10) + [{"text": f"plain text row {i}"} for i in range(10)])
    assert pd.main(["--input", str(src), "--output", str(tmp_path / "o"), "--format", "chat"]) == 0
    rows = read_jsonl(tmp_path / "o" / "train.jsonl")
    assert rows and all(set(r) == {"messages"} for r in rows)


def test_homogeneous_auto_mode_still_passes(tmp_path):
    src = tmp_path / "a.jsonl"
    write_jsonl(src, make_rows(20))
    assert pd.main(["--input", str(src), "--output", str(tmp_path / "o")]) == 0


def test_undersized_holdout_rows_return_to_training():
    rows = make_rows(20)
    train, valid, test, notes = pd.split_examples(rows, 0.1, 0.1, 4, True, 17)
    assert len(train) == 12 and len(valid) == 4 and len(test) == 4
    # A large indivisible group can leave a holdout below the minimum.
    grouped = []
    for i in range(11):
        row = chat(f"question {i}", f"answer {i}")
        row[pd.GROUP_FIELD] = "big" if i < 7 else str(i)
        grouped.append(row)
    train, valid, test, notes = pd.split_examples(grouped, 0.25, 0, 3, True, 17)
    assert len(train) + len(valid) + len(test) == len(grouped)
    assert not any(pd.audit_leakage(train, valid, test).values())
