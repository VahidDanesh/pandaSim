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
    """The three corners touched at a run_box.py pose, found from the geometry alone: c1 the top corner of the
    vertical edge that carries the mark, c2 the other end of the longest top edge from c1, c3 next to c2; and
    whether the mark is at the top."""
    box = rp.make_box(dict(box=dict(size=list(SIZE), mass=1.0, mu=0.5)))
    T = rp.box_pose(box, rb.parse_pose(pose_text))
    h = np.array(SIZE) / 2
    world = lambda s: T[:3, :3] @ (h * np.array(s)) + T[:3, 3]
    mark = world(rb.MARK)
    corners = [world(s) for s in np.array(np.meshgrid(*[[-1, 1]] * 3)).T.reshape(-1, 3)]
    top = [c for c in corners if c[2] > T[2, 3]]
    c1 = min(top, key=lambda c: np.linalg.norm((c - mark)[:2]))
    edge = lambda a, b: np.linalg.norm((a - b)[:2])
    near = sorted((c for c in top if c is not c1), key=lambda c: edge(c, c1))[:2]     # the two neighbours of c1
    c2 = near[1]                                                                     # along the longest edge
    c3 = min((c for c in top if c is not c1 and c is not c2), key=lambda c: edge(c, c2))
    return (c1, c2, c3), bool(mark[2] > T[2, 3])


POSES = [f"{x},{y},{yaw},{face}" for (x, y), yaw, face in zip(
    [(0.55, 0.10), (0.45, -0.20), (0.60, 0.00), (0.50, 0.05), (0.40, 0.20), (0.65, -0.10), (0.52, 0.0), (0.58, 0.1)],
    [30, -70, 90, 180, -150, 120, -90, 0],
    ["lying", "standing", "lying_flip", "standing_flip", "lying", "lying_flip", "standing", "standing_flip"])]


@pytest.mark.parametrize("pose", POSES)
def test_box_pose_from_corners(pose):
    """The marked box: every face and the yaw over 360 deg come back exactly."""
    x, y, yaw, face = pose.split(",")
    corners, top = top_corners(pose)
    r = cb.box_from_top_corners(*corners, 0.0, SIZE, mark_on_top=top)
    assert r["face"] == face and abs(r["height_error"]) < 1e-9 and r["fit_mm"] < 1e-6
    assert np.allclose([r["x"], r["y"]], [float(x), float(y)], atol=1e-9)
    assert abs((r["yaw_deg"] - float(yaw) + 180) % 360 - 180) < 1e-6
    assert rb.pose_text(rp.box_pose(rp.make_box(dict(box=dict(size=list(SIZE), mass=1.0, mu=0.5))),
                                    rb.parse_pose(pose))).split(",")[3] == face


def test_a_wrong_answer_about_the_mark_is_seen_in_the_fit():
    corners, top = top_corners("0.55,0.10,30,lying")
    r = cb.box_from_top_corners(*corners, 0.0, SIZE, mark_on_top=not top)
    assert r["face"] == "lying_flip" and r["fit_mm"] > 50


def test_measured_error_against_the_goal_of_a_run(tmp_path):
    box = rp.make_box(dict(box=dict(size=list(SIZE), mass=1.0, mu=0.5)))
    Tg = rp.box_pose(box, rb.parse_pose("0.55,0.10,30,lying"))
    np.savez(tmp_path / "expected.npz", Tg=Tg)
    corners, top = top_corners("0.553,0.096,32,lying")
    m = cb.measured_error(cb.box_from_top_corners(*corners, 0.0, SIZE, mark_on_top=top), tmp_path, SIZE)
    assert m["face_ok"] and m["error_mm"] == pytest.approx(5.0, abs=1e-6) and m["yaw_error_deg"] == pytest.approx(2.0)
    corners, top = top_corners("0.55,0.10,210,lying")          # the right place, turned by 180 deg: a failure
    m = cb.measured_error(cb.box_from_top_corners(*corners, 0.0, SIZE, mark_on_top=top), tmp_path, SIZE)
    assert m["error_deg"] == pytest.approx(180.0) and m["error_mm"] < 1e-6


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
    r = cb.session(ReadOnlyPanda(qs), robot, SIZE, ask=lambda p: "y", out=out.append, table=False)
    assert r["table_z"] == 0.0
