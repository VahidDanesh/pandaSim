"""experiments/run_box.py: poses, pre-flight, confirmation and the execution on a fake panda-py (no robot)."""
import copy
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments"))
import run_box as rb  # noqa: E402
import run_plan as rp  # noqa: E402

from pandaSim.control.backends import PANDA_LAG  # noqa: E402

START, GOAL = "0.50,0.00,0,lying", "0.60,0.10,20,lying"           # a slide: about 13 s of motion


@pytest.fixture(autouse=True)
def keep_hand_model(monkeypatch):
    """The real config selects the hand model (a module setting): put it back after every test."""
    from pandaSim.planning import clearance

    monkeypatch.setattr(clearance, "HAND_MODEL", clearance.HAND_MODEL)



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
    assert panda.robot.collision is not None and panda.robot.collision[1][0] == pytest.approx(0.29 * 87)   # reflex set


def test_reflex_thresholds_scale_with_alpha_and_stay_below_the_maximum_torque():
    reflex = dict(alpha=1.0, torque_fraction=dict(lower=[0.17] * 4 + [0.5] * 3, upper=[0.29] * 4 + [0.83] * 3),
                  force=dict(lower=[20.0] * 3 + [6.0] * 3, upper=[40.0] * 3 + [12.0] * 3))
    th = rb.reflex_thresholds(reflex)
    assert th["torque"]["upper"] == pytest.approx([25.23] * 4 + [9.96] * 3) and th["force"]["upper"][0] == 40.0
    th = rb.reflex_thresholds(dict(reflex, alpha=2.0))
    assert th["torque"]["upper"] == pytest.approx([50.46] * 4 + [12.0] * 3)        # the wrist is capped at 12 Nm
    assert th["force"]["upper"][0] == 80.0 and th["torque"]["lower"][0] == pytest.approx(2 * 0.17 * 87)


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


def test_a_pose_inside_the_base_keep_out_is_named_in_the_preflight():
    cfg = rp.load_config(rb.REAL_CONFIG)
    res = rb.plan_and_simulate(cfg, rb.parse_pose("0.50,0.00,0,lying"), rb.parse_pose("0.38,0.00,0,lying"))
    items = rb.preflight(res, cfg)                     # the 0.25 m side points at the base: a corner 0.26 m from its axis
    assert not res["plan"].found and not rb.go(items)
    assert [c["name"] for c in items if not c["ok"]] == ["goal pose clear of the robot base", "plan found"]


def test_a_rigid_pivot_that_is_predicted_to_stop_runs_only_with_expect_stop(tmp_path, monkeypatch):
    """7.5: the free-rotation plan executed with the rigid grasp stops by an infeasible QP in the pivot. The pre-flight
    refuses it unless the stop is expected; with --expect-stop it is GO, and on the robot the run stops there with the
    fingers still on the box (no release, no further motion)."""
    from panda_py import controllers

    start, goal = rb.parse_pose("0.50,-0.20,90,lying"), rb.parse_pose("0.50,0.00,90,standing")
    cfg = rp.load_config(rb.REAL_CONFIG, False, True)
    res = rb.plan_and_simulate(cfg, start, goal, 0.3, tracking=True)
    assert [s.kind for s in res["plan"].steps] == ["pivot"] and not res["log"].ok
    stop = rb.predicted_stop(res["log"], res["plan"].steps)
    assert stop["step"] == 0 and 30 < stop["angle_deg"] < 85 and stop["of_deg"] == pytest.approx(90.0)
    assert not rb.go(rb.preflight(res, cfg)) and rb.go(rb.preflight(res, cfg, expect_stop=True))
    coast = next(c for c in rb.preflight(res, cfg, expect_stop=True) if c["name"].startswith("hand coasts"))
    assert coast["ok"] and cfg["execution"]["qp"]["qd_scale"] == pytest.approx(0.8 * 0.3)    # the bound is slowed too
    free = rb.plan_and_simulate(rp.load_config(rb.REAL_CONFIG, True, True), start, goal, 0.3)
    assert free["log"].ok and not rb.go(rb.preflight(free, cfg, expect_stop=True))      # it does not stop

    monkeypatch.setattr(controllers, "IntegratedVelocity", Ctrl)
    cfg["execution"]["robot"]["stall_timeout"] = 1.0     # a long run on the fake; the watchdog is tested elsewhere
    run_dir = tmp_path / "fr"
    run_dir.mkdir()
    panda, gripper = Panda(rp.exec_kwargs(cfg)["q_ready"]), Gripper()
    c = rb.execute(res, cfg, run_dir, None, "low", 0.3, ask=lambda p: f"RUN low {run_dir.name}", panda=panda, gripper=gripper)
    assert c["abort"]["cause"] == "infeasible QP" and c["abort"]["holding"] and stopped(panda)
    assert [g[0] for g in gripper.calls] == ["move", "grasp"]                              # the box stays in the hand


def test_theta_after_a_gripper_call_is_the_current_state(monkeypatch):
    """A gripper call blocks without ticks while the arm can move (the fingers centre the box against the compliant
    controller): the next theta() of PandaBackend must be the robot's state after the call, not the last tick's."""
    from panda_py import controllers

    from pandaSim.control.backends import PandaBackend

    monkeypatch.setattr(controllers, "IntegratedVelocity", Ctrl)

    class Centring(Gripper):
        def grasp(self, *args):
            panda.q = panda.q + 0.002                       # 2 mrad of compliance while the fingers close
            return super().grasp(*args)

        def move(self, *args):
            panda.q = panda.q - 0.001
            return super().move(*args)

    q0 = rp.exec_kwargs(rp.load_config(rb.REAL_CONFIG))["q_ready"]
    gripper = Centring()
    panda = Panda(q0, gripper=gripper)
    backend = PandaBackend(panda, gripper, stall_timeout=0)
    backend.start()
    assert backend.tick() and np.array_equal(backend.theta(), q0)
    assert backend.grasp(0.06, 20.0)
    assert np.allclose(backend.theta(), q0 + 0.002)
    assert backend.tick()                                   # the controller pulls the arm back towards q_d
    q1 = backend.theta()
    assert backend.open()
    assert np.allclose(backend.theta(), q1 - 0.001)
    backend.stop()


def test_a_failing_zero_command_keeps_the_segment_record():
    """A robot error, and then the zero command of run_segment raises too (a controller already gone): the segment
    is still recorded with the robot error, and the second exception goes on to run_plan's abort procedure."""
    from pandaSim.control.backends import SimBackend
    from pandaSim.control.primitive_executor import JointMove, Log, run_segment

    class Gone(SimBackend):
        def tick(self):
            if self._t > 0.2:
                raise RuntimeError("communication_constraints_violation")
            return super().tick()

        def command(self, qd7):
            if self._t > 0.2 and not np.any(qd7):
                raise RuntimeError("the controller is not running")
            super().command(qd7)

    q0 = np.zeros(7)
    backend, log = Gone(q0), Log()
    jm = JointMove(q0, q0 + 0.3)
    with pytest.raises(RuntimeError, match="not running"):
        run_segment(backend, jm, jm.ref, jm.Tf, log, "ready", done=jm.done)
    assert len(log.segments) == 1 and log.segments[0]["name"] == "ready"
    assert log.segments[0]["cause"] == "robot error: communication_constraints_violation" and not log.segments[0]["ok"]


def test_predicted_stop_uses_the_configured_time_scaling():
    """The stop angle follows from the time at the stop through the pivot's time scaling (execution.pivot.method)."""
    from pandaSim.control.primitive_executor import Log
    from pandaSim.planning.primitives import time_scaling

    log = Log()
    log.segments = [dict(name="pre", ok=True, step=0), dict(name="pivot", ok=False, cause="infeasible QP", step=0,
                                                            t0=10.0, t_end=13.0, Tf=10.0)]
    steps = [SimpleNamespace(plan=SimpleNamespace(screws=[(None, np.pi / 2)]))]
    cfg = rp.load_config(rb.REAL_CONFIG)
    assert rb.pivot_method(cfg) == "quintic"
    for method in ("quintic", "cubic", "linear"):
        cfg["execution"]["pivot"] = {**(cfg["execution"].get("pivot") or {}), "method": method}
        stop = rb.predicted_stop(log, steps, rb.pivot_method(cfg))
        assert stop["angle_deg"] == pytest.approx(90.0 * time_scaling(3.0, 10.0, method)[0]) and stop["time_frac"] == 0.3


def test_speed_factor_scales_the_configured_nominal_speeds(monkeypatch):
    """scale_speed multiplies the config's own speeds (defaults: those of the executor), and plan_and_simulate plans
    at the config's nominal speed while it simulates at the factor."""
    cfg = rp.load_config(rb.REAL_CONFIG)
    assert cfg["execution"]["pivot"]["w_max"] == 0.3
    e = rp.scale_speed(copy.deepcopy(cfg), 0.5)["execution"]
    assert e["pivot"]["w_max"] == 0.15 and e["approach"] == dict(v_max=0.05, w_max=0.15)
    assert e["ready"] == dict(qd_max=0.25) and e["qp"]["qd_scale"] == pytest.approx(0.4) and e["qp"]["rho_s"] == 0.13
    assert (e["qp"]["K"], e["qp"]["lam"], e["qp"]["eta"]) == (0.1, 0.1, 0.5)   # the rates of eq:qp (defaults 0.2, 0.2, 1)
    cfg["execution"]["pivot"]["w_max"] = 0.2                            # a config value is honoured, not replaced
    cfg["execution"]["approach"] = dict(v_max=0.08)
    cfg["execution"]["qp"] = dict(K=[0.4] * 7, lam=0.3)
    e = rp.scale_speed(copy.deepcopy(cfg), 0.5)["execution"]
    assert e["pivot"]["w_max"] == 0.1 and e["approach"] == dict(v_max=0.04, w_max=0.15)
    assert e["qp"]["K"] == [0.2] * 7 and e["qp"]["lam"] == 0.15 and e["qp"]["eta"] == 0.5

    seen = {}
    real_make_planner = rp.make_planner
    monkeypatch.setattr(rp, "make_planner", lambda c, robot, gait=False: seen.update(plan=c) or real_make_planner(c, robot, gait))
    monkeypatch.setattr(rp, "run_task", lambda c, robot, name, planner: seen.update(run=c) or dict(plan=SimpleNamespace(found=False)))
    rb.plan_and_simulate(cfg, rb.parse_pose(START), rb.parse_pose(GOAL), speed=0.3)
    assert seen["plan"]["execution"]["pivot"]["w_max"] == 0.2 and seen["plan"]["execution"]["approach"]["v_max"] == 0.08
    assert seen["run"]["execution"]["pivot"]["w_max"] == pytest.approx(0.06) and seen["run"] is cfg


def test_speed_factor_keeps_the_joint_path():
    """With the rates of the QP scaled with the speed (scale_speed), one pivot step run at speed
    0.3 follows the joint path of speed 1.0 as a function of the path parameter t / Tf of every segment (the rollouts
    of the planner see the path that the robot runs). Same discretization per unit of path: dt / speed. Without the
    scaled rates the null-space motion of the QP depends on the speed (D4 of the design review: ~0.1 rad)."""
    from pandaSim.control.backends import SimBackend
    from pandaSim.control.primitive_executor import run_plan
    from pandaSim.planning.sequencer import Sequencer, Step

    cfg0 = rp.load_config(rb.REAL_CONFIG)
    robot, box = rp.make_robot(cfg0), rp.make_box(cfg0)
    T = rp.box_pose(box, rb.parse_pose("0.50,0.00,90,lying"))
    _, pl = Sequencer(y_ref=np.array([0.0, 1.0, 0.0]))._box_pivots(box, T)[0]
    step = Step("pivot", T, pl.T_goal, pl)

    def paths(speed, rates=True):
        cfg = rp.scale_speed(copy.deepcopy(cfg0), speed)
        if not rates:                                     # the rates as before the fix: not scaled
            cfg["execution"]["qp"].update(K=0.2, lam=0.2, eta=1.0)
        kw = rp.exec_kwargs(cfg)
        log, _ = run_plan(SimBackend(kw["q_ready"], dt=0.01 / speed), robot, [step], T, **kw)
        assert log.ok
        a = log.arrays()
        return [(sg["name"], (a["t"][a["segment"] == j] - sg["t0"]) / sg["Tf"], a["theta"][a["segment"] == j])
                for j, sg in enumerate(log.segments) if sg["name"] != "ready"]   # JointMove: its own gain, same goal

    def largest_difference(p, q):
        assert [n for n, *_ in p] == [n for n, *_ in q] == ["pre", "approach", "pivot", "retreat"]
        d = 0.0
        for (_, s, th), (_, s2, th2) in zip(p, q):
            m = s2 <= s[-1]
            d = max(d, float(np.abs(np.array([np.interp(s2[m], s, th[:, k]) for k in range(7)]).T - th2[m]).max()))
        return d

    nominal = paths(1.0)
    assert largest_difference(nominal, paths(0.3)) < 1e-3
    assert largest_difference(nominal, paths(0.3, rates=False)) > 0.05


def test_a_gait_off_the_goal_with_the_lag_needs_lag_compensation(planned, monkeypatch):
    """'object at the goal with the lag' is hard for a gait plan without lag compensation (25 screw motions), a warning
    otherwise; the pivot-arc line reports a deviation of the commanded path (a force on the box), not a squeeze."""
    import dataclasses

    cfg, res = planned
    monkeypatch.setattr(rb, "grasp_observability", lambda st: (True, 1.0, ""))     # the slide steps relabelled as gait
    gait = dict(res, plan=dataclasses.replace(res["plan"], steps=[dataclasses.replace(s, kind="gait") for s in res["plan"].steps]))
    tr = dict(ok=True, fails=[], err_mm=13.0, err_deg=0.6, pivot_dev_mm=3.0)

    def goal_item(r, lead):
        items = rb.preflight(dict(r, tracking=dict(tr, lead=lead)), cfg)
        return rb.go(items), next(c for c in items if "object at the goal with the lag" in c["name"]), items

    go, c, items = goal_item(gait, False)
    assert not go and not c["ok"] and c["hard"] and "--lead" in c["detail"]
    go, c, _ = goal_item(gait, True)
    assert go and not c["ok"] and not c["hard"]
    go, c, _ = goal_item(res, False)                                    # not a gait: a warning
    assert go and not c["ok"] and not c["hard"]
    arc = next(c for c in items if "pivot arc" in c["name"])
    assert not arc["ok"] and not arc["hard"] and "commanded" in arc["name"] and "internal force" in arc["detail"]
    assert "squeeze" not in arc["detail"]


def test_the_protocol_runs_the_gait_with_lag_compensation():
    """protocol.yaml: lead stays off globally; the gait experiment sets its own (protocol.check_entry simulates and
    flags each entry with spec["lead"])."""
    import yaml

    import protocol as proto

    protocol = yaml.safe_load(proto.PROTOCOL.read_text())
    assert protocol["lead"] is False
    leads = {(exp, task, grasp): spec["lead"] for exp, task, grasp, spec in proto.entries(protocol)}
    assert all(v == (k[0] == "gait") for k, v in leads.items()) and sum(leads.values()) == 2
