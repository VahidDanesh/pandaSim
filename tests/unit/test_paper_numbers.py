"""experiments/paper_numbers.py: runs on the committed results and catches a wrong number in the paper."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments"))
import paper_numbers as pn  # noqa: E402


def fake_tex(rows):
    return "\n".join(t.replace("{}", "%s") % tuple(pn.fmt(v) for v in vals) for *_, checks in rows for t, vals in checks)


def test_prints_the_committed_results(capsys):
    assert pn.main(["--no-check"]) == 0
    out = capsys.readouterr().out
    assert "Random pivots: n = 300" in out and "free rotation" in out and "Random tasks: n = 100" in out


def test_check_accepts_the_paper_spelling_and_catches_a_wrong_value(tmp_path):
    rows = pn.parameters()
    tex = fake_tex(rows)
    assert pn.check(rows, tex) == []
    assert pn.check(rows, tex.replace("$0.3$~rad/s", "$0.30$~rad/s")) == []          # same value, other spelling
    wrong = tex.replace("grasp force & $70$~N", "grasp force & $25$~N")
    assert [sym for sym, _ in pn.check(rows, wrong)] == ["f_g"]
    assert [sym for sym, _ in pn.check(rows, tex.replace("~m and $5$~kg", "~m and $1$~kg"))] == ["m"]
    path = tmp_path / "main.tex"
    path.write_text(wrong)
    assert pn.main(["--tex", str(path)]) == 1


def test_wilson_interval():
    lo, hi = pn.wilson(297, 300)
    assert 0.97 < lo < 0.972 and 0.996 < hi < 0.998
