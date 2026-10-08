import math

from training import log_format as L


def test_step_line_shows_loss_ppl_and_bpb():
    s = L.format_step_line(1200, 3.14e9, 2.4310, 2.40, 0.842, 3e-4, 52300.0, 0.312)
    assert "TRAIN" in s and "loss  2.4310" in s and f"ppl {math.exp(2.40):7.2f}" in s and "bpb  0.842" in s and "31.2%" in s and "3.14B" in s


def test_eval_block_has_train_val_rows_with_ppl_and_bpb_and_per_language_lines():
    losses = {"train": 2.4, "train_ce": 2.39, "train_bpb": 0.83, "val": 2.55, "val_ce": 2.54, "val_bpb": 0.889, "val_old": 2.31,
              "val_lang_macro": 2.9, "val_by_lang": {"zul": 3.1, "afr": 2.2}, "val_bpb_by_lang": {"zul": 1.02, "afr": 0.7}}
    b = L.format_eval_block(2000, 4.1e9, losses)
    lines = b.splitlines()
    assert lines[0].startswith("=") and "EVAL  step 2000" in lines[0] and lines[-1].startswith("=")
    assert any(l.strip().startswith("train") and f"{math.exp(2.39):.2f}" in l for l in lines)
    assert any(l.strip().startswith("val ") or l.strip().startswith("val") and "bpb" not in l for l in lines)
    assert b.index("afr") < b.index("zul")                    # best language first
    assert "0.8890" in b and "per language" in b


def test_missing_values_print_a_dash_not_a_crash():
    b = L.format_eval_block(1, 1e6, {"train": 2.0, "val": float("nan")})
    assert "-" in b and "M tokens" in b
