"""experiments/run_box.py: poses, pre-flight, confirmation and the execution on a fake panda-py (no robot)."""
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments"))
import run_box as rb  # noqa: E402
import run_plan as rp  # noqa: E402

from pandaSim.control.backends import PANDA_LAG  # noqa: E402

START, GOAL = "0.50,0.00,0,lying", "0.60,0.10,20,lying_flip"      # a slide: about 13 s of motion


class Ctrl:
    """panda-py IntegratedVelocity: integrates the last velocity command into the position target q_d."""

    def __init__(self, **kw):
        self.qd, self.t, self.q_d = np.zeros(7), 0.0, None
        self.commands = []                                  # (wall time, velocity) of every set_control

    def set_control(self, qd):
        assert np.shape(qd) == (7,)
        self.qd = np.array(qd, float)
        self.commands.append((time.monotonic(), self.qd.copy()))

    def get_time(self):
        return self.t

    def get_qd(self):
        return self.q_d.copy()


class Ctx:
    """PandaContext.ok() and, inside it, one period of the robot. `inject` of the panda makes things go wrong at
    tick n: error_at (robot reflex), crash_at (a bug: ValueError), interrupt_at (Ctrl-C), stall_at + stall (the
    Python loop hangs for `stall` s of real time while the controller keeps integrating), block_from (the joints
    stop following: a blocked arm), vmax_from + vmax (the joints saturate), wrench_at + wrench (external wrench)."""

    def __init__(self, panda):
        self.panda = panda

    def ok(self):
        p, c = self.panda, self.panda.ctrl
        p.n += 1
        f, dt = p.inject, 1e-3
        if f.get("error_at") == p.n:
            raise RuntimeError("libfranka: Move command aborted: motion aborted by reflex! [cartesian_reflex]")
        if f.get("crash_at") == p.n:
            raise ValueError("a bug in the loop")
        if f.get("interrupt_at") == p.n:
            raise KeyboardInterrupt
        if f.get("stall_at") == p.n:                        # the robot's own 1 kHz thread goes on during the stall
            t_start, qd = time.monotonic(), c.qd.copy()
            time.sleep(f["stall"])
            zeroed = [t for t, v in c.commands if t > t_start and not v.any()]
            p.stall = dict(start=t_start, end=time.monotonic(), zeroed_at=zeroed[0] if zeroed else None)
            dt = (zeroed[0] - t_start) if zeroed else f["stall"]
            c.q_d += qd * dt                                # moved with the last command until someone zeroed it
            c.t += f["stall"]
        else:
            c.q_d += c.qd * dt
            c.t += dt
        # the joints follow the target with the controller's lag Kd / Kp (tau = Kp (q_d - q) - Kd dq)
        q_new = p.q + (c.q_d - p.q) * (1 - np.exp(-dt / p.lag))
        if "block_from" in f and p.n >= f["block_from"]:
            q_new = p.q.copy()
        if "vmax_from" in f and p.n >= f["vmax_from"]:
            q_new = p.q + np.clip(q_new - p.q, -f["vmax"] * dt, f["vmax"] * dt)
        p.dq, p.q = (q_new - p.q) / dt, q_new
        return True


class State:
    def __init__(self, panda):
        self.q, self.dq = panda.q.copy(), panda.dq.copy()
        f = panda.inject
        self.O_F_ext_hat_K, self.tau_ext_hat_filtered = np.zeros(6), np.zeros(7)
        if panda.gripper is not None and panda.gripper.holding:          # the box hangs on the fingers
            self.O_F_ext_hat_K[2] -= f.get("held_weight", 0.0)
        if "wrench_at" in f and panda.n >= f["wrench_at"]:
            self.O_F_ext_hat_K = self.O_F_ext_hat_K + np.asarray(f["wrench"], float)
        self.current_errors = "[]"


class Robot:
    def __init__(self):
        self.collision = None

    def set_collision_behavior(self, *thresholds):
        self.collision = thresholds


class Panda:
    def __init__(self, q, lag=PANDA_LAG, gripper=None, **inject):
        self.q, self.dq, self.ctrl, self.moves = np.array(q, float), np.zeros(7), None, []
        self.lag, self.gripper, self.inject, self.n = lag, gripper, inject, 0
        self.calls, self.robot, self.stall, self.move_ok = [], Robot(), None, True

    def start_controller(self, ctrl):
        self.ctrl = ctrl
        ctrl.q_d = self.q.copy()
        self.calls.append("start")

    def stop_controller(self):
        self.calls.append("stop")
        self.last_command = None if self.ctrl is None else self.ctrl.qd.copy()

    def create_context(self, frequency):
        return Ctx(self)

    def get_state(self):
        return State(self)

    def get_robot(self):
        return self.robot

    def move_to_joint_position(self, q, speed_factor):
        self.moves.append(speed_factor)
        if self.move_ok:
            self.q = np.array(q, float)
        return self.move_ok


class Gripper:
    """grasp_ok=False: the grasp reports failure; fail="grasp" | "release": that call raises (libfranka error)."""

    def __init__(self, grasp_ok=True, fail=None):
        self.calls, self.grasp_ok, self.fail, self.holding = [], grasp_ok, fail, False

    def move(self, width, speed):
        self.calls.append(("move", width))
        if self.fail == "release" and self.holding:
            raise RuntimeError("libfranka gripper: command failed")
        self.holding = False
        return True

    def grasp(self, width, speed, force, e1, e2):
        self.calls.append(("grasp", width))
        if self.fail == "grasp":
            raise RuntimeError("libfranka gripper: command failed")
        self.holding = self.grasp_ok
        return self.grasp_ok

    def read_once(self):
        class G:
            width, max_width, is_grasped = 0.08, 0.08, False
        return G()


def stopped(panda):
    """The arm was commanded to stop: the last velocity sent is zero and the controller was stopped after it."""
    return panda.calls[-1] == "stop" and panda.last_command is not None and not panda.last_command.any()


@pytest.fixture(scope="module")
def planned():
    cfg = rp.load_config(rb.REAL_CONFIG)
    return cfg, rb.plan_and_simulate(cfg, rb.parse_pose(START), rb.parse_pose(GOAL), speed=1.0)


def test_parse_pose():
    assert rb.parse_pose("0.5,-0.1,30,standing") == dict(xy=[0.5, -0.1], rpy=[0.0, 0.0, 30.0])
    assert rb.parse_pose("0.5,0,0,lying")["rpy"][1] == -90.0
    with pytest.raises(SystemExit):
        rb.parse_pose("0.5,0,0,flat")
    with pytest.raises(SystemExit):
        rb.parse_pose("0.5,0,0")


def test_preflight_go_and_the_checks_that_block(planned):
    cfg, res = planned
    items = rb.preflight(res, cfg)
    assert rb.go(items), [c for c in items if not c["ok"]]
    heavy = copy.deepcopy(cfg)
    heavy["box"]["mass"] = 4.0
    res_heavy = dict(res, box=type(res["box"])(res["box"].size, 4.0, res["box"].mu))
    assert not rb.go(rb.preflight(res_heavy, heavy))                    # over the payload
    outside = copy.deepcopy(cfg)
    outside["tasks"]["custom"]["goal"]["xy"] = [0.95, 0.0]
    assert not rb.go(rb.preflight(res, outside))


def test_no_plan_is_no_go():
    cfg = rp.load_config(rb.REAL_CONFIG)
    res = rb.plan_and_simulate(cfg, rb.parse_pose("0.5,0.0,0,lying"), rb.parse_pose("0.5,0.0,0,lying"))
    assert not rb.go(rb.preflight(res, cfg))                              # identical start and goal: nothing to do


def test_confirm_needs_the_exact_phrase():
    assert rb.confirm("RUN air x", lambda prompt: "RUN air x")
    assert not rb.confirm("RUN air x", lambda prompt: "yes")
    assert not rb.confirm("RUN air x", lambda prompt: "")


def test_execute_on_a_fake_panda_matches_the_simulation(planned, tmp_path, monkeypatch):
    from panda_py import controllers

    monkeypatch.setattr(controllers, "IntegratedVelocity", Ctrl)
    cfg, res = planned
    q_ready = rp.exec_kwargs(cfg)["q_ready"]
    run_dir = tmp_path / "r1"
    run_dir.mkdir()
    panda, gripper = Panda(q_ready + 0.01), Gripper()
    assert rb.execute(res, cfg, run_dir, None, "low", 1.0, ask=lambda p: "wrong", panda=panda, gripper=gripper) is None
    assert panda.moves == [] and gripper.calls == []                    # not confirmed: nothing moved
    c = rb.execute(res, cfg, run_dir, None, "low", 1.0, ask=lambda p: f"RUN low {run_dir.name}", panda=panda, gripper=gripper,
                   lead=True)
    assert panda.moves == [0.1] and c["ok"] and c["n_segments"] == c["expected_segments"]
    assert max(s["max_joint_error"] for s in c["segments"]) < 5e-3     # with lag compensation the joints follow the plan
    assert c["object_error_mm"] < 3.0
    assert [g[0] for g in gripper.calls] == ["move", "grasp", "move"]
    assert json.loads((run_dir / "compare.json").read_text())["ok"]
    assert stopped(panda) and c["abort"] is None
    assert panda.robot.collision is not None and panda.robot.collision[1][0] == 25.0     # reflex thresholds were set


def test_robot_checks_refuse_a_bad_start():
    q_ready = np.array(rp.load_config(rb.REAL_CONFIG)["robot"]["q_ready"])
    lim = (np.full(7, -2.8), np.full(7, 2.8))
    panda = Panda(q_ready)
    assert rb.go(rb.robot_checks(panda.get_state(), q_ready, lim))
    far = Panda(q_ready + 1.5)
    assert not rb.go(rb.robot_checks(far.get_state(), q_ready, lim))
    err = panda.get_state()
    err.current_errors = "[communication_constraints_violation]"
    assert not rb.go(rb.robot_checks(err, q_ready, lim))


# ---- failures on the robot: every one must end with the arm commanded to stop and the records saved ----

def run_fake(planned, tmp_path, monkeypatch, panda_kw=None, gripper=None, level="low", cfg_edit=None, **kw):
    from panda_py import controllers

    monkeypatch.setattr(controllers, "IntegratedVelocity", Ctrl)
    cfg, res = planned
    cfg = copy.deepcopy(cfg)
    if cfg_edit:
        cfg_edit(cfg)
    gripper = gripper or Gripper()
    panda = Panda(rp.exec_kwargs(cfg)["q_ready"], gripper=gripper, **(panda_kw or {}))
    run_dir = tmp_path / "run"
    run_dir.mkdir(exist_ok=True)
    c = rb.execute(res, cfg, run_dir, None, level, 1.0, ask=lambda p: f"RUN {level} run", panda=panda, gripper=gripper, **kw)
    return c, panda, gripper, run_dir


def test_a_stalled_loop_is_stopped_by_the_watchdog(planned, tmp_path, monkeypatch):
    c, panda, _, _ = run_fake(planned, tmp_path, monkeypatch, dict(stall_at=3000, stall=0.25))
    assert not c["ok"] and c["abort"]["cause"].startswith("loop stall")
    st = panda.stall                                   # the velocity was zeroed DURING the stall, by the watchdog thread
    assert st["zeroed_at"] is not None and st["zeroed_at"] - st["start"] < 0.15 < st["end"] - st["start"]
    assert stopped(panda)


def test_a_blocked_or_saturated_arm_is_stopped_by_the_tracking_monitor(planned, tmp_path, monkeypatch):
    for inject in (dict(block_from=3000), dict(vmax_from=3000, vmax=0.001)):
        c, panda, _, _ = run_fake(planned, tmp_path, monkeypatch, inject)
        assert not c["ok"] and c["abort"]["cause"].startswith("tracking error")
        assert panda.n - 3000 < 1000                   # within 1 s: 20 Nm of tracking torque at the speed of this move
        assert stopped(panda)


def test_a_wrench_spike_stops_the_arm(planned, tmp_path, monkeypatch):
    c, panda, _, _ = run_fake(planned, tmp_path, monkeypatch, dict(wrench_at=3000, wrench=[0, 0, 50.0, 0, 0, 0]))
    assert c["abort"]["cause"] == "external wrench" and panda.n - 3000 < 5 and stopped(panda)


def test_the_weight_of_the_held_box_is_not_a_collision_but_more_than_it_is(planned, tmp_path, monkeypatch):
    def tight(cfg):                                    # 1 kg box: its weight alone is above this force threshold
        cfg["execution"]["safety"]["f_max"] = 8.0

    c, panda, gripper, _ = run_fake(planned, tmp_path, monkeypatch, dict(held_weight=9.81), cfg_edit=tight, lead=True)
    assert c["ok"] and c["abort"] is None and max(s["max_force"] for s in c["segments"]) > 9.0
    # level air has no box: the same force on the hand stops the run
    c, panda, _, _ = run_fake(planned, tmp_path, monkeypatch, dict(wrench_at=3000, wrench=[0, 0, -9.81, 0, 0, 0]),
                              cfg_edit=tight, level="air", lead=True)
    assert c["abort"]["cause"] == "external wrench" and stopped(panda)


def test_wrench_excess_removes_only_what_the_weight_explains():
    from pandaSim.control.primitive_executor import wrench_excess

    w, J = np.array([0.0, 0.0, -9.81]), np.linspace(-0.5, 0.5, 21).reshape(3, 7)
    load = (w, J.T @ w)
    ex = lambda F, load=load: wrench_excess(np.r_[F, 0, 0, 0], J.T @ np.asarray(F, float), 15.0, 15.0, load)
    assert ex(w) < 1e-9 and ex(0.4 * w) < 1e-9                         # any part of the weight
    assert ex(w, None) == pytest.approx(9.81 / 15.0)                    # no object: it counts
    assert ex(-w) >= 9.81 / 15.0                                        # the same force upwards is not the weight
    assert ex(2 * w) >= 9.81 / 15.0 - 1e-9                              # twice the weight: only one is removed
    assert ex(w + [14.0, 0, 0]) < 1.0 < ex(w + [16.0, 0, 0])            # a sideways force counts in full
    assert wrench_excess(np.r_[w + [12.0, 0, 0], 0, 0, 0], np.zeros(7), 15.0, 15.0, None) > 1.0


def test_gripper_failures_stop_the_run_and_never_open_the_hand(planned, tmp_path, monkeypatch):
    for gripper, what in ((Gripper(grasp_ok=False), "grasp"), (Gripper(fail="grasp"), "grasp"), (Gripper(fail="release"), "release")):
        c, panda, gripper, _ = run_fake(planned, tmp_path, monkeypatch, gripper=gripper, lead=True)
        assert not c["ok"] and c["abort"]["cause"].startswith(f"gripper {what} failed") and c["abort"]["holding"]
        assert stopped(panda)
        names = [g[0] for g in gripper.calls]
        assert names == (["move", "grasp"] if what == "grasp" else ["move", "grasp", "move"])   # nothing after the failure
        if what == "grasp":
            assert "slide" not in [s["name"] for s in c["segments"]]


def test_robot_error_exception_and_ctrl_c_stop_the_arm_and_keep_the_log(planned, tmp_path, monkeypatch):
    c, panda, _, _ = run_fake(planned, tmp_path, monkeypatch, dict(error_at=3000))
    assert c["abort"]["cause"].startswith("robot error") and "reflex" in c["abort"]["cause"] and stopped(panda)

    c, panda, _, run_dir = run_fake(planned, tmp_path, monkeypatch, dict(interrupt_at=3000))
    assert c["abort"]["cause"].startswith("KeyboardInterrupt") and stopped(panda)
    assert (run_dir / "real.npz").exists()

    for f in ("real.npz", "compare.json"):
        (tmp_path / "run" / f).unlink()
    with pytest.raises(ValueError):
        run_fake(planned, tmp_path, monkeypatch, dict(crash_at=3000))
    saved = json.loads((tmp_path / "run" / "compare.json").read_text())
    assert saved["abort"]["cause"].startswith("ValueError") and (tmp_path / "run" / "real.npz").exists()


def test_exception_inside_the_loop_zeroes_the_velocity_first(planned, monkeypatch):
    """run_plan itself (no run_box): after an exception in the loop the last command is zero and stop() was called."""
    from panda_py import controllers

    from pandaSim.control.backends import PandaBackend
    from pandaSim.control.primitive_executor import Log, run_plan

    monkeypatch.setattr(controllers, "IntegratedVelocity", Ctrl)
    cfg, res = planned
    kw = rp.exec_kwargs(cfg)
    panda, log = Panda(kw["q_ready"], gripper=Gripper(), crash_at=2000), Log()
    with pytest.raises(ValueError):
        run_plan(PandaBackend(panda, panda.gripper), res["robot"], res["plan"].steps, res["T0"], log=log, **kw)
    assert stopped(panda) and log.abort["cause"].startswith("ValueError") and not log.segments[-1]["ok"]
    assert len(log.arrays()["t"]) > 1500                                  # the records up to the exception are there


def test_the_run_does_not_start_if_the_move_to_ready_fails(planned, tmp_path, monkeypatch):
    from panda_py import controllers

    monkeypatch.setattr(controllers, "IntegratedVelocity", Ctrl)
    cfg, res = planned
    panda = Panda(rp.exec_kwargs(cfg)["q_ready"] + 0.3, gripper=Gripper())
    panda.move_ok = False
    assert rb.execute(res, cfg, tmp_path, None, "low", 1.0, ask=lambda p: f"RUN low {tmp_path.name}", panda=panda,
                      gripper=panda.gripper) is None
    assert panda.calls == [] and panda.gripper.calls == []                # no controller, no gripper motion


def test_backend_that_stops_ticking_or_skips_time_is_a_failed_segment():
    from pandaSim.control.backends import SimBackend
    from pandaSim.control.primitive_executor import JointMove, Log, run_segment

    class Stops(SimBackend):
        def tick(self):
            return super().tick() and self._t < 0.2

    class Skips(SimBackend):
        def tick(self):
            super().tick()
            if 0.2 < self._t < 0.2015:
                self._t += 0.08                                           # 80 ms without a tick of ours
            return True

    q0 = np.zeros(7)
    for backend, cause in ((Stops(q0), "backend stopped"), (Skips(q0), "loop stall")):
        jm, log = JointMove(q0, q0 + 0.3), Log()
        ok, _ = run_segment(backend, jm, jm.ref, jm.Tf, log, "ready", done=jm.done)
        assert not ok and log.segments[0]["cause"].startswith(cause) and not backend._qd.any()


def test_preflight_knows_the_joint_walls_and_the_controller_lag(planned, monkeypatch):
    cfg, res = planned
    slow = copy.deepcopy(cfg)
    res_lag = dict(res, tracking=rb.simulate_tracking(res, slow, lead=False))
    assert res_lag["tracking"]["ok"] and rb.go(rb.preflight(res_lag, slow))
    slow["execution"]["safety"]["settle"] = 0.5                           # too short for the lag: segments time out
    res_lag = dict(res, tracking=rb.simulate_tracking(res, slow, lead=False))
    assert not res_lag["tracking"]["ok"] and not rb.go(rb.preflight(res_lag, slow))
    assert rb.simulate_tracking(res, slow, lead=True)["ok"]               # with lag compensation it arrives in time
    monkeypatch.setattr(rb, "PANDA_WALL", np.full(7, 0.7))                # pretend the joint walls reach far in
    items = rb.preflight(res, cfg)
    assert not rb.go(items) and any("joint walls" in c["name"] and not c["ok"] for c in items)


def test_a_run_without_lag_compensation_reports_the_measured_lag(planned, tmp_path, monkeypatch):
    c, panda, _, _ = run_fake(planned, tmp_path, monkeypatch, lead=False)
    assert c["ok"]
    measured = [(m, PANDA_LAG[i]) for i, m in enumerate(c["lag_measured_s"]) if m is not None]
    assert len(measured) >= 3 and all(abs(m - model) < 0.25 * model for m, model in measured)
    assert max(c["track_torque_max"]) < 5.0                               # free motion: little tracking torque


def test_the_move_to_ready_is_swept_for_collisions(planned):
    from pandaSim.planning.collision import ArmCollision

    cfg, res = planned
    q_ready, lim = rp.exec_kwargs(cfg)["q_ready"], res["robot_limits"]
    arm = ArmCollision(res["robot"], res["box"].size)
    clear = Panda(q_ready + np.array([0.8, 0.3, -0.5, 0.4, 0.3, -0.5, 1.2]))          # far, but high above the table
    items = rb.robot_checks(clear.get_state(), q_ready, lim, arm=arm, T_box=res["T0"])
    assert rb.go(items) and any(not c["ok"] and not c["hard"] for c in items)          # allowed, with a warning
    low = Panda(np.array([0.0, 1.0, 0.0, -1.5, 0.0, 2.5, 0.78]))                       # the hand a few cm above the table
    items = rb.robot_checks(low.get_state(), q_ready, lim, arm=arm, T_box=None)
    assert not rb.go(items) and not next(c for c in items if "clear of the table" in c["name"])["ok"]
    # the box in the way of an otherwise clear path
    T_box = res["T0"].copy()
    T_box[:3, 3] = res["robot"].fkine(np.r_[q_ready, 0.0], end="panda_hand").t - [0, 0, 0.12]
    sw = arm.sweep(q_ready + np.array([0.0, 0.25, 0, 0, 0, 0, 0]), q_ready, T_box)
    assert sw["d_min"] < 0.03 and sw["link"] is not None
    assert arm.sweep(q_ready + 0.2, q_ready, res["T0"])["d_min"] > 0.03                 # the planned start is clear
    # fingers: open they are 10 mm from the box faces at the grasp, and they are part of the test
    st = res["plan"].steps[0]
    q_grasp = res["log"].arrays()["theta"][np.flatnonzero(res["log"].arrays()["segment"] == 1)[-1]]
    d_open, who = arm.clearance(q_grasp, st.T_before, width=0.08, fingers_vs_box=True)
    assert who.endswith("finger") and 0.005 < d_open < 0.012
    assert arm.clearance(q_grasp, st.T_before, width=0.08)[0] > d_open                  # without the fingers: further away


def test_observability_of_the_grasp(planned):
    import dataclasses

    from pandaSim.planning import se3
    from pandaSim.planning.feasibility import grasp_observability

    cfg, res = planned
    slide = res["plan"].steps[0]
    assert slide.kind == "slide" and grasp_observability(slide)[0]
    # fingers along the vertical: the box could spin about them on the table
    Rx = se3.rp(se3.exp3(np.array([np.pi / 2, 0.0, 0.0])), np.zeros(3))
    bad = dataclasses.replace(slide, plan=dataclasses.replace(slide.plan, T_grasp=slide.plan.T_grasp @ Rx))
    assert not grasp_observability(bad)[0]
    # a pivot of the planner: the grasp is far from the edge; moved onto the edge it is not observable
    box = res["box"]
    T = rp.box_pose(box, rb.parse_pose("0.5,0.0,0,lying"))
    d_o, pl = rp.make_planner(cfg, res["robot"])._box_pivots(box, T)[0]
    step = type(slide)("pivot", T, pl.T_goal, pl)
    ok, arm, _ = grasp_observability(step)
    assert ok and arm > 0.10
    S = np.asarray(pl.S)
    on_axis = pl.T_grasp.copy()
    on_axis[:3, 3] = np.cross(S[:3], S[3:]) + S[:3] * (S[:3] @ pl.T_grasp[:3, 3])       # same place along the edge
    assert not grasp_observability(dataclasses.replace(step, plan=dataclasses.replace(pl, T_grasp=on_axis)))[0]
    assert all(c["ok"] for c in rb.preflight(res, cfg) if "finger axis" in c["name"])
