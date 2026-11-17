"""experiments/ablation.py report: summaries of the committed results, success by distance from the base."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments"))
import ablation as ab  # noqa: E402


def test_by_distance_counts_every_trial_once():
    d = json.loads((ab.RESULTS / "ablation_pivots.json").read_text())
    rows = ab.by_distance(d["trials"])
    assert sum(n for _, _, n, _ in rows) == d["n"]
    for m in ab.PIVOT_MODES:
        assert sum(k[m] for *_, k in rows) == d["summary"]["all"][m]["success"]


def test_report_runs_on_the_committed_results(capsys):
    assert ab.main(["report", str(ab.RESULTS / "ablation_pivots.json"), str(ab.RESULTS / "ablation_tasks.json")]) == 0
    out = capsys.readouterr().out
    assert "by distance" in out and "free" in out
