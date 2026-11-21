import json

import pytest

import compare


def test_normalize_and_em():
    assert compare.normalize_answer("  The Cat, sat!! ") == "the cat sat"
    assert compare.exact_match("Hello, World", "hello world") == 1.0
    assert compare.exact_match("a", "b") == 0.0


def test_token_f1_known_values():
    assert compare.token_f1("the cat sat", "the cat sat") == 1.0
    assert compare.token_f1("", "") == 1.0
    assert compare.token_f1("a", "") == 0.0
    assert compare.token_f1("a b c d", "a b x y") == pytest.approx(0.5)
    # Multiset overlap: repeated tokens count as many times as BOTH sides have them.
    # "a a b" vs "a a c" shares two 'a's -> P=R=2/3 -> F1=2/3. A set-based overlap
    # would give 1/3, so this case distinguishes the two implementations.
    assert compare.token_f1("a a b", "a a c") == pytest.approx(2 / 3)
    # Over-repeating a correct token must not be rewarded: only one 'a' is in the reference.
    assert compare.token_f1("a a a", "a b c") == pytest.approx(1 / 3)


def test_perplexity_reports_overflow():
    assert compare.perplexity(0.0) == 1.0
    assert compare.perplexity(1e9) == float("inf")


def test_split_prompt_reference():
    p, r = compare.split_prompt_reference({"messages": [
        {"role": "system", "content": "s"}, {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a"}]})
    assert [m["role"] for m in p] == ["system", "user"] and r == "a"
    p, r = compare.split_prompt_reference({"prompt": "q", "completion": "a"})
    assert p == [{"role": "user", "content": "q"}] and r == "a"
    assert compare.split_prompt_reference({"text": "t"}) is None
    assert compare.split_prompt_reference({"messages": [{"role": "user", "content": "q"}]}) is None


def test_summarize_generations():
    s = compare.summarize_generations([{"pred": "a b", "ref": "a b"}, {"pred": "x", "ref": "a b c d"}])
    assert s["n"] == 2 and s["exact_match"] == 0.5
    assert s["avg_pred_words"] == 1.5 and s["avg_ref_words"] == 3.0
    assert compare.summarize_generations([]) == {"n": 0}


def _res(bl, tl, bf=None, tf=None, ratio=1.0):
    gen = lambda f: {"n": 5, "token_f1": f, "exact_match": 0, "avg_pred_words": 5,
                     "avg_ref_words": 5, "length_ratio": ratio} if f is not None else {"n": 0}
    return ({"loss": bl, "ppl": compare.perplexity(bl), "gen": gen(bf)},
            {"loss": tl, "ppl": compare.perplexity(tl), "gen": gen(tf)})


def test_verdict_improvement():
    v = compare.verdict(*_res(2.0, 1.5, 0.2, 0.5))
    assert "improved" in v[0] and "25.0%" in v[0]
    assert not any("generations did not" in x for x in v)


def test_verdict_regression_is_loud():
    v = compare.verdict(*_res(1.5, 1.8))
    assert "did NOT improve" in v[0]


def test_verdict_flags_loss_up_but_generations_flat():
    v = compare.verdict(*_res(2.0, 1.0, 0.4, 0.3))
    assert any("generations did not" in x for x in v)


def test_verdict_flags_length_blowup():
    v = compare.verdict(*_res(2.0, 1.0, 0.1, 0.5, ratio=3.0))
    assert any("3.0x" in x for x in v)


def test_render_report_contains_table_and_clips_samples():
    base, tuned = _res(2.0, 1.5, 0.2, 0.5)
    res = {"model": "m", "adapter": "a", "split": "test", "n_examples": 10, "mask_prompt": True,
           "base": base, "tuned": tuned, "verdict": compare.verdict(base, tuned),
           "samples": [{"prompt": "p", "ref": "r", "base": "b" * 5000, "tuned": "t"}]}
    md = compare.render_report(res)
    assert "| loss | 2.0000 | 1.5000 | -0.5000 |" in md
    assert "assistant tokens only" in md
    assert len(md) < 2000 and "…" in md  # 5000-char sample was clipped


def test_render_report_loss_only():
    base, tuned = _res(2.0, 1.5)
    res = {"model": "m", "adapter": "a", "split": "valid", "n_examples": 4, "mask_prompt": False,
           "base": base, "tuned": tuned, "verdict": compare.verdict(base, tuned), "samples": []}
    md = compare.render_report(res)
    assert "token F1" not in md and "all tokens" in md


def test_pick_split(tmp_path):
    (tmp_path / "valid.jsonl").write_text("{}\n")
    assert compare.pick_split(tmp_path, None) == "valid"
    (tmp_path / "test.jsonl").write_text("{}\n")
    assert compare.pick_split(tmp_path, None) == "test"  # test preferred when present
    assert compare.pick_split(tmp_path, "valid") == "valid"
    with pytest.raises(SystemExit):
        compare.pick_split(tmp_path, "nope")


def test_pick_split_nothing(tmp_path):
    with pytest.raises(SystemExit):
        compare.pick_split(tmp_path, None)


def test_check_adapter(tmp_path):
    with pytest.raises(SystemExit, match="make train"):
        compare.check_adapter(str(tmp_path))
    (tmp_path / "adapter_config.json").write_text("{}")
    (tmp_path / "adapters.safetensors").write_text("x")
    compare.check_adapter(str(tmp_path))  # real mlx-lm filename is accepted
