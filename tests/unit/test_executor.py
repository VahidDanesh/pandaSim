"""Primitive executor, backends, viewer plumbing and the read-only connection check (no robot)."""
import time

import numpy as np
import pytest
import spatialgeometry as sg
import spatialmath as sm
from pytransform3d import rotations as pr

from pandaSim.control.backends import PandaBackend, SimBackend
from pandaSim.control.contact_qp import FER_QD_MAX, ContactQP
from pandaSim.control.primitive_executor import (
    Log, approach_reference, check_profile, contact_pose, run_pivot, run_segment)
from pandaSim.planning.primitives import lifting_angle, planned_pose, pose_error_twist

BASE_Z = 0.013                                  # lab: robot base above the object table (tests/rtb.ipynb)
BOX = [0.27, 0.065, 0.207]


@pytest.fixture(scope="module")
def scene():
    from pandaSim.geometry.rtb_adapter import RoboticsToolboxAdapter
    from pandaSim.geometry.utils import create_virtual_panda
    from pandaSim.planning.screw_motion_planner import ScrewMotionPlanner

    robot = create_virtual_panda()
    robot.base = sm.SE3(0, 0, BASE_Z)
    adapter = RoboticsToolboxAdapter({"viewer": None})
    box = sg.Cuboid(scale=BOX, pose=sm.SE3(0.5, 0.0, BOX[2] / 2))
    T_wd, qs, s_axes = ScrewMotionPlanner(adapter).compute_grasp(
        box, adapter, grasp_height="top", prefer_closer_grasp=True, gripper_offset=sm.SE3.Rx(np.pi).A)
    q_axis, s_axis = qs.reshape(-1, 3)[0], s_axes.reshape(-1, 3)[0]
    pivot = dict(q=q_axis, s_axis=s_axis, theta=lifting_angle(q_axis, s_axis, np.pi / 2, box.T[:3, 3]), Tf=5.0)
    return dict(robot=robot, T_wd=T_wd, pivot=pivot, T_w_obj=box.T.copy())


def pose_err(T_a, T_b):
    E = np.linalg.inv(T_a) @ T_b
    return np.linalg.norm(E[:3, 3]), np.degrees(pr.axis_angle_from_matrix(E[:3, :3])[3])


def test_contact_pose_includes_robot_base(scene):
    r = scene["robot"]
    assert np.allclose(contact_pose(r, r.qr), r.fkine(r.qr, end="panda_finger_virtual").A)
    assert np.allclose(contact_pose(r, r.qr[:7]), contact_pose(r, np.r_[r.qr[:7], 0.0]))


def test_approach_reference_is_smooth_bounded_and_converges(scene):
    r, T_wd = scene["robot"], scene["T_wd"]
    fk = lambda q: contact_pose(r, q)
    q0 = r.qr.copy()
    v_max, w_max = 0.1, 0.3
    ref, Tf = approach_reference(fk, fk(q0), T_wd, K_e=1.0, v_max=v_max, w_max=w_max)
    assert np.allclose(ref(0.0, q0), 0.0)                                   # no jump at the start
    V_ff = np.array([approach_reference(fk, fk(q0), T_wd, K_e=0.0)[0](t, q0) for t in np.linspace(0, Tf, 201)])
    assert np.linalg.norm(V_ff[:, 3:], axis=1).max() <= v_max + 1e-9
    assert np.linalg.norm(V_ff[:, :3], axis=1).max() <= w_max + 1e-9
    assert np.allclose(V_ff[-1], 0.0)

    backend, log = SimBackend(q0[:7]), Log()
    ok, phi = run_segment(backend, ContactQP(r), ref, Tf, log, "approach",
                          done=lambda q: np.linalg.norm(pose_error_twist(fk(q), T_wd)) < 1e-3)
    assert ok
    pos, rot = pose_err(fk(np.r_[backend.theta(), phi]), T_wd)
    assert pos < 1e-3 and rot < 0.1


def test_free_rotation_run(scene):
    r = scene["robot"]
    backend = SimBackend(r.qr[:7])
    log = run_pivot(backend, r, scene["T_wd"], scene["pivot"], free_rotation=True, grasp=dict(width=0.065, force=20))
    assert log.ok, log.segments
    assert [s["name"] for s in log.segments] == ["approach", "pivot"]
    assert [e[0] for e in log.events] == ["open", "grasp", "release"]

    a = log.arrays()
    piv = a["segment"] == 1
    q_start = np.r_[a["theta"][piv][0], a["phi"][piv][0]]
    q_end = np.r_[a["theta"][piv][-1], a["phi"][piv][-1]]
    T_plan = planned_pose(scene["pivot"]["q"], scene["pivot"]["s_axis"], scene["pivot"]["theta"],
                          contact_pose(r, q_start), 1.0)
    pos, rot = pose_err(T_plan, contact_pose(r, q_end))
    print(f"free: {pos * 1e3:.3f} mm, {rot:.4f} deg; phi {a['phi'][0]:.3f} -> {a['phi'][-1]:.3f} rad")
    assert pos < 1e-3 and rot < 0.1

    prof = check_profile(log)
    print("velocity ratio", np.round(prof["vel_ratio"], 3), "acceleration ratio", np.round(prof["acc_ratio"], 3))
    assert prof["vel_ok"]
    assert np.all(np.abs(a["qd"][:, :7]) <= 0.8 * FER_QD_MAX + 1e-9)


def test_rigid_run_stops_and_reports(scene):
    r = scene["robot"]
    log = run_pivot(SimBackend(r.qr[:7]), r, scene["T_wd"], scene["pivot"], free_rotation=False,
                    grasp=dict(width=0.065, force=20))
    pivot = log.segments[-1]
    s = (pivot["t_end"] - pivot["t0"]) / pivot["Tf"]
    print(f"rigid: {pivot}")
    assert pivot["name"] == "pivot"
    if not pivot["ok"]:                                       # stopped safely, not raised
        assert pivot["cause"] == "infeasible QP" and "release" not in [e[0] for e in log.events]
        assert 0 < s < 1


def test_slack_applies_to_the_pivot_only(scene):
    r = scene["robot"]
    log = run_pivot(SimBackend(r.qr[:7]), r, scene["T_wd"], scene["pivot"], free_rotation=False,
                    qp=dict(slack=100.0), grasp=dict(width=0.065, force=20))
    assert [s["name"] for s in log.segments] == ["approach", "pivot"]
    assert log.segments[0]["ok"]                           # shared approach unaffected by slack
    assert log.segments[1]["ok"]                           # slack never makes the pivot infeasible


class WrenchBackend(SimBackend):
    def wrench(self):
        return (np.r_[0, 0, 50.0, 0, 0, 0] if self.time() > 0.2 else np.zeros(6)), np.zeros(7)


def test_external_wrench_stops_with_zero_velocity(scene):
    r = scene["robot"]
    backend = WrenchBackend(r.qr[:7])
    log = run_pivot(backend, r, scene["T_wd"], scene["pivot"], grasp=dict(width=0.065, force=20))
    assert log.segments[0]["cause"] == "external wrench"
    assert np.all(backend._qd == 0)


# --- PandaBackend wiring against fakes of panda-py ---

class FakeCtrl:
    def __init__(self, **kw):
        self.commands, self.t = [], 0.0

    def set_control(self, qd):
        assert np.shape(qd) == (7,)
        self.commands.append(np.array(qd))

    def get_time(self):
        return self.t


class FakeCtx:
    def __init__(self, panda, fail_after):
        self.panda, self.fail_after, self.n = panda, fail_after, 0

    def ok(self):
        self.n += 1
        self.panda.ctrl.t = self.n * 1e-3
        if self.n > self.fail_after:
            raise RuntimeError("communication_constraints_violation")
        return True


class FakePanda:
    def __init__(self, q, fail_after=100):
        self.q, self.fail_after, self.calls, self.ctrl = np.asarray(q), fail_after, [], None

    def start_controller(self, ctrl):
        self.ctrl = ctrl
        self.calls.append("start")

    def stop_controller(self):
        self.calls.append("stop")

    def create_context(self, frequency):
        return FakeCtx(self, self.fail_after)

    def get_state(self):
        self.calls.append("get_state") if not self.calls or self.calls[-1] != "get_state" else None
        class S:
            q, O_F_ext_hat_K, tau_ext_hat_filtered = self.q, np.zeros(6), np.zeros(7)
        return S()


class FakeGripper:
    def __init__(self, grasp_ok=True):
        self.grasp_ok, self.calls = grasp_ok, []

    def move(self, width, speed):
        self.calls.append("move")
        return True

    def grasp(self, width, speed, force, eps_in, eps_out):
        self.calls.append("grasp")
        return self.grasp_ok


@pytest.fixture
def fake_controllers(monkeypatch):
    from panda_py import controllers

    monkeypatch.setattr(controllers, "IntegratedVelocity", FakeCtrl)


def test_panda_backend_stops_controller_on_robot_error(scene, fake_controllers):
    r = scene["robot"]
    panda = FakePanda(r.qr[:7], fail_after=50)
    log = run_pivot(PandaBackend(panda, FakeGripper()), r, scene["T_wd"], scene["pivot"], grasp=dict(width=0.065))
    assert log.segments[0]["cause"].startswith("robot error")
    assert [c for c in panda.calls if c != "get_state"] == ["start", "stop"]
    assert len(panda.ctrl.commands) > 40 and np.all(panda.ctrl.commands[-1] == 0)


def test_failed_grasp_aborts_before_pivot(scene, fake_controllers, monkeypatch):
    r = scene["robot"]
    panda, gripper = FakePanda(r.qr[:7], fail_after=10**9), FakeGripper(grasp_ok=False)
    # the fake arm does not move: pretend the approach arrived immediately
    monkeypatch.setattr("pandaSim.control.primitive_executor.approach_reference",
                        lambda fk, T0, Td, **kw: ((lambda t, q: np.zeros(6)), 0.01))
    monkeypatch.setattr("pandaSim.control.primitive_executor.pose_error_twist", lambda a, b: np.zeros(6))
    log = run_pivot(PandaBackend(panda, gripper), r, scene["T_wd"], scene["pivot"], grasp=dict(width=0.065))
    assert [s["name"] for s in log.segments] == ["approach"]
    assert gripper.calls == ["move", "grasp"] and [c for c in panda.calls if c != "get_state"] == ["start", "stop"]


# --- viewer plumbing (no Swift: a stand-in target in the child process) ---

def echo_target(shared, base, objects, rate):
    a = np.frombuffer(shared, dtype=float)
    deadline = time.time() + 20
    while a[0] == 0 and time.time() < deadline:
        time.sleep(0.01)
    a[24] = a[:8].sum()                                       # report what the child saw


def test_live_viewer_shares_q_cheaply():
    from pandaSim.control.viewer import LiveViewer

    viewer = LiveViewer(base=np.eye(4), target=echo_target)
    q = np.arange(1.0, 9.0)
    n = 10000
    t0 = time.perf_counter()
    for _ in range(n):
        viewer.update(q)
    per_call_us = (time.perf_counter() - t0) / n * 1e6
    viewer._proc.join(20)
    assert viewer._a[24] == q.sum()
    assert per_call_us < 5, per_call_us
    viewer.close()


def test_connection_report_is_read_only(scene):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parents[2] / "experiments"))
    from check_connection import report

    class ReadOnlyPanda:
        def get_state(self):
            class S:
                q, robot_mode, current_errors = scene["robot"].qr[:7], "kIdle", "none"
                O_F_ext_hat_K, tau_ext_hat_filtered = np.zeros(6), np.zeros(7)
            return S()

        def __getattr__(self, name):                          # any other call would be a command
            raise AssertionError(f"report() must not call panda.{name}")

    class G:
        def read_once(self):
            class S:
                width, max_width, is_grasped = 0.08, 0.08, False
            return S()

    out = report(ReadOnlyPanda(), G(), scene["robot"])
    assert out["q_inside_limits"] and out["gripper_width"] == 0.08


def test_connection_check_locks_and_releases_desk_control(tmp_path, monkeypatch):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parents[2] / "experiments"))
    from check_connection import desk_credentials, lock_brakes

    env = tmp_path / ".env"
    env.write_text("# comment\nPANDA_HOST=1.2.3.4\nPANDA_USER=u\nPANDA_PASS=p=w\n")
    monkeypatch.delenv("PANDA_USER", raising=False)
    monkeypatch.delenv("PANDA_PASS", raising=False)
    assert desk_credentials(env) == ("u", "p=w")
    assert desk_credentials(tmp_path / "missing") is None

    class FakeDesk:
        calls = []
        def take_control(self): self.calls.append("take")
        def lock(self): self.calls.append("lock")
        def release_control(self): self.calls.append("release")

    lock_brakes(FakeDesk())
    assert FakeDesk.calls == ["take", "lock", "release"]
