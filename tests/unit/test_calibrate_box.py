"""experiments/calibrate_box.py: recovers a known box pose from touched points and never commands the robot."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments"))
import calibrate_box as cb  # noqa: E402
import run_box as rb  # noqa: E402
import run_plan as rp  # noqa: E402

SIZE = (0.15, 0.06, 0.25)


def top_corners(pose_text):
    """Three top corners of the box at a run_box.py pose: c1-c2 along the longest top edge, c3 next to c2."""
    box = rp.make_box(dict(box=dict(size=list(SIZE), mass=1.0, mu=0.5)))
    T = rp.box_pose(box, rb.parse_pose(pose_text))
    h = np.array(SIZE) / 2
    corners = [T[:3, :3] @ (h * np.array(s)) + T[:3, 3] for s in np.array(np.meshgrid(*[[-1, 1]] * 3)).T.reshape(-1, 3)]
    top = sorted(corners, key=lambda c: -c[2])[:4]
    edges = sorted(((i, j) for i in range(4) for j in range(i + 1, 4)), key=lambda e: -np.linalg.norm(top[e[0]] - top[e[1]]))
    i, j = edges[2]                                                # edges[0:2] are the diagonals
    k = min((m for m in range(4) if m not in (i, j)), key=lambda m: np.linalg.norm(top[m] - top[j]))
    return top[i], top[j], top[k]


@pytest.mark.parametrize("pose", ["0.55,0.10,30,lying", "0.45,-0.20,-70,standing", "0.60,0.00,90,lying"])
def test_box_pose_from_corners(pose):
    x, y, yaw, face = pose.split(",")
    r = cb.box_from_top_corners(*top_corners(pose), 0.0, SIZE)
    assert r["face"] == face and abs(r["height_error"]) < 1e-9
    assert np.allclose([r["x"], r["y"]], [float(x), float(y)], atol=1e-9)
    assert abs(((r["yaw_deg"] - float(yaw)) + 90) % 180 - 90) < 1e-6     # yaw up to the box symmetry


def test_table_plane():
    z, tilt = cb.plane([[0.4, 0.0, 0.001], [0.6, 0.2, 0.001], [0.5, -0.3, 0.001]])
    assert z == pytest.approx(0.001) and tilt == pytest.approx(0.0, abs=1e-9)


class ReadOnlyPanda:
    """Only get_state exists; any other call (a controller, a motion) fails the test."""

    def __init__(self, qs):
        self.qs, self.k = list(qs), 0

    def get_state(self):
        class S:
            pass
        s = S()
        s.q = self.qs[min(self.k // 50, len(self.qs) - 1)]
        self.k += 1
        return s

    def __getattr__(self, name):
        raise AssertionError(f"calibrate_box called panda.{name}: it must only read the state")


def test_session_only_reads_the_robot():
    from pandaSim.geometry.utils import create_virtual_panda

    robot = create_virtual_panda()
    qs = [robot.qr[:7] + 0.05 * k for k in range(6)]
    out = []
    r = cb.session(ReadOnlyPanda(qs), robot, SIZE, ask=lambda p: "", out=out.append)
    assert "table_z" in r and len(out) >= 2
