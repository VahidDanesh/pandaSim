"""Object gaiting of a box with one manipulator (manuscript Sec. 5, Case 1).

The box rests on a face; the gripper can only span the box's thin axis (the finger axis), so the contact edges of
the gait are the two short edges of the support face, the ones parallel to the finger axis. Step j turns the box by
alpha_j about the vertical through a vertex v_j of the support face, without sliding, in one grasp at the far end:

    1. tilt by gamma about the short edge that contains v_j      (the box stands on that edge)
    2. tilt by beta about the box edge through v_j                (the other vertex of the short edge lifts: the box
                                                                   stands on v_j, as in Fig. I2 of the manuscript)
    3. turn by alpha_j about the vertical through v_j
    4. back by beta, 5. back by gamma                             (the box lies on its face again)

All five screw axes pass through v_j, so the step is a planar rotation by alpha_j about v_j. The box is released and
grasped again at the other end, and the next step turns it about a vertex there (consecutive vertices at opposite
ends of the support face: larger steps than alternating the two vertices of one edge). The plan is the sequence of
vertices and angles that reaches the goal exactly with the fewest steps and the smallest |alpha| (eq:gait_a-d),
checked like the sequencer's plans: cheap tests, then every step rolled out with the executor (Rollout).
"""
from __future__ import annotations

import itertools
import math
import time
import warnings
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from pandaSim.planning import box_primitives as P
from pandaSim.planning import se3
from pandaSim.planning.sequencer import REACH, Plan, Step, check_steps, snap_stable

Z = np.array([0.0, 0.0, 1.0])


@dataclass
class GaitPlan:
    screws: list           # [(S, theta)] world frame, applied in order in one grasp
    vertex: np.ndarray     # world position of the vertex v_j; every screw axis passes through it
    T_grasp: np.ndarray    # {c} at the start
    T_goal: np.ndarray     # object pose after the step
    T_start: np.ndarray    # object pose before the step
    width: float           # object extent between the fingers [m]
    phi_lim: tuple = None  # (lo, hi) rad, as PivotPlan


def _axes(T, size):
    """Object axes of a resting box: (up, finger, length) indices and the sign of the up axis.
    The finger axis is the horizontal axis that fits in the gripper; None if none does."""
    up = int(np.argmax(np.abs(T[2, :3])))
    horiz = [i for i in range(3) if i != up]
    fit = [i for i in horiz if size[i] < P.GRIPPER_MAX]
    if not fit:
        return None
    f = min(fit, key=lambda i: size[i])
    return up, f, next(i for i in horiz if i != f), float(np.sign(T[2, up]))


def support_vertices(T, size):
    """The four vertices of the support face, object frame: list of (end, side, point); end = +-1 along the length
    axis, side = +-1 along the finger axis. Empty if the box cannot be grasped in this pose."""
    ax = _axes(T, np.asarray(size, float))
    if ax is None:
        return []
    up, f, ln, s_up = ax
    half = np.asarray(size, float) / 2
    out = []
    for end, side in itertools.product((1.0, -1.0), (1.0, -1.0)):
        v = np.zeros(3)
        v[up], v[f], v[ln] = -s_up * half[up], side * half[f], end * half[ln]
        out.append((end, side, v))
    return out


def _screw(u, p):
    u = np.asarray(u, float) / np.linalg.norm(u)
    return np.r_[u, -np.cross(u, p)]


def gait_step(T, size, end, side, alpha, gamma, beta, y_ref=None):
    """One step of the gait from the resting pose T about the support vertex (end, side): GaitPlan."""
    size = np.asarray(size, float)
    up, f, ln, s_up = _axes(T, size)
    v_o = next(v for e, s, v in support_vertices(T, size) if e == end and s == side)
    b_o = v_o.copy()
    b_o[f] *= -1                                           # the other vertex of the short edge
    d_o = np.zeros(3)
    d_o[ln] = end                                          # toward the pivot end
    pv = P.plan_box_pivot(T, size, d_o, gamma, y_ref)      # gamma about the short edge, grasp at the far end
    R, p = T[:3, :3], T[:3, 3]
    a = R @ v_o + p
    e_up = np.zeros(3)
    e_up[up] = s_up
    S1 = pv.S
    E1 = se3.exp6(S1 * gamma) @ T
    u2 = E1[:3, :3] @ e_up                                 # the box edge through v_j, tilted by gamma
    S2 = _screw(u2, a)
    lift = lambda b: (se3.exp6(S2 * b) @ E1)[:3, :3] @ b_o + (se3.exp6(S2 * b) @ E1)[:3, 3]
    b2 = beta if lift(beta)[2] > lift(-beta)[2] else -beta
    Rz = se3.exp3(Z * alpha)
    screws = [(S1, gamma), (S2, b2), (_screw(Z, a), alpha), (_screw(Rz @ u2, a), -b2), (_screw(Rz @ S1[:3], a), -gamma)]
    T_end = T
    for S, th in screws:
        T_end = se3.exp6(S * th) @ T_end
    return GaitPlan(screws, a, pv.T_grasp, T_end, T, pv.width, pv.phi_lim)


def _rot(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s], [s, c]])


def _apply(p, psi, verts, seq, alphas):
    """Planar box pose (centre p, yaw psi relative to the start) after rotations alphas about the vertices seq."""
    for v, a in zip(seq, alphas):
        pv = p + _rot(psi) @ verts[v]
        p, psi = pv + _rot(a) @ (p - pv), psi + a
    return p, psi


def _apply_jac(p, verts, seq, alphas):
    """d p / d alphas of _apply (psi = 0 at the start): the turn by alpha_i about the vertex pv_i turns all that
    follows rigidly about pv_i, so column i is the final centre minus pv_i, turned by 90 deg."""
    pv, psi = [], 0.0
    for v, a in zip(seq, alphas):
        pv.append(p + _rot(psi) @ verts[v])
        p, psi = pv[-1] + _rot(a) @ (p - pv[-1]), psi + a
    d = p - np.array(pv)
    return np.stack([-d[:, 1], d[:, 0]])


def solve_angles(T0, Tg, size, k_max=5, alpha_max=math.radians(45), restarts=4, k_min=1):
    """Vertex sequences and angles that turn the box from T0 to Tg (same face down) by k_min..k_max rotations about
    support vertices, consecutive vertices at opposite ends: list of (seq of (end, side), alphas), fewest steps first,
    then the smallest |alpha|. Each solution reaches Tg exactly (eq:gait_b, eq:gait_c) with |alpha_j| <= alpha_max."""
    verts = support_vertices(T0, size)
    dR = Tg[:3, :3] @ T0[:3, :3].T
    if not verts or abs(dR[2, 2] - 1.0) > 1e-6:
        return []                                          # not graspable, or the goal rests on another face
    pts = [(T0[:3, :3] @ v)[:2] for _, _, v in verts]     # vertex offsets from the centre at the start, world xy
    p0, pg = T0[:2, 3], Tg[:2, 3]
    dpsi = math.atan2(dR[1, 0], dR[0, 0])
    out = []
    for k in range(k_min, k_max + 1):
        for first, sides in itertools.product((1.0, -1.0), itertools.product((1.0, -1.0), repeat=k)):
            seq = [next(i for i, (e, s, _) in enumerate(verts) if e == first * (-1) ** j and s == sides[j]) for j in range(k)]
            res = lambda a, seq=seq: _apply(p0, 0.0, pts, seq, a)[0] - pg
            jac = lambda a, seq=seq: _apply_jac(p0, pts, seq, a)      # analytic: SLSQP need not difference res
            for target in (dpsi, dpsi + 2 * math.pi, dpsi - 2 * math.pi):
                if abs(target) > k * alpha_max:
                    continue
                cons = [dict(type="eq", fun=res, jac=jac),
                        dict(type="eq", fun=lambda a, t=target: np.sum(a) - t, jac=lambda a: np.ones((1, len(a))))]
                best = None
                for r in range(restarts):
                    a0 = np.full(k, target / k) + (np.random.default_rng(r).normal(0, 0.3, k) if r else 0.0)
                    with warnings.catch_warnings():         # SLSQP may step outside the bounds; it clips them
                        warnings.simplefilter("ignore", RuntimeWarning)
                        sol = minimize(lambda a: a @ a, np.clip(a0, -alpha_max, alpha_max), jac=lambda a: 2 * a,
                                       bounds=[(-alpha_max, alpha_max)] * k, constraints=cons, method="SLSQP",
                                       options=dict(maxiter=300, ftol=1e-12))
                    ok = np.abs(res(sol.x)).max() < 1e-7 and abs(np.sum(sol.x) - target) < 1e-7
                    if ok and np.abs(sol.x).max() <= alpha_max + 1e-9 and (best is None or sol.fun < best.fun):
                        best = sol
                if best is not None:
                    out.append(([verts[i][:2] for i in seq], best.x))
    out.sort(key=lambda o: (len(o[1]), float(np.linalg.norm(o[1]))))
    return out


class GaitPlanner:
    """Plans of object gaiting with the interface of sequencer.Sequencer: plan(o, T0, Tg) -> Plan.
    kin: feasibility.Rollout (check(step) -> (feasible, joint margin), T_ready). gamma, beta, alpha_max in rad.
    Candidates (fewest steps, then the smallest |alpha|) pass the cheap tests of the sequencer (base keep-out and reach
    of the grasp points), then the first max_kin_checks of them are rolled out step by step."""

    def __init__(self, kin, gamma=math.radians(15), beta=math.radians(30), alpha_max=math.radians(45), k_max=5,
                 max_kin_checks=8, base_xy=(0.0, 0.0)):
        self.kin, self.gamma, self.beta, self.alpha_max, self.k_max = kin, gamma, beta, alpha_max, k_max
        self.max_kin_checks, self.base_xy = max_kin_checks, np.asarray(base_xy, float)
        self.y_ref = kin.T_ready[:3, 1] if kin is not None else None

    def steps(self, o, T0, seq, alphas):
        """Step list of one candidate (exact poses: each step starts where the last one ended)."""
        out, T = [], T0
        for (end, side), a in zip(seq, alphas):
            pl = gait_step(T, o.size, end, side, float(a), self.gamma, self.beta, self.y_ref)
            out.append(Step("gait", T, pl.T_goal, pl))
            T = pl.T_goal
        return out

    def _cheap_ok(self, o, step):
        pl = step.plan
        if not P.plan_keepout_ok(pl, o.size, self.base_xy):
            return False
        T, pts = pl.T_grasp, [pl.T_grasp[:3, 3]]
        for S, th in pl.screws:
            T = se3.exp6(S * th) @ T
            pts.append(T[:3, 3])
        return all(REACH[0] <= np.linalg.norm(q[:2] - self.base_xy) <= REACH[1] for q in pts)

    def plan(self, o, T0, Tg):
        """The first candidate whose steps all pass the rollouts, with the fewest steps; cost = steps + |alpha| [rad]."""
        t = time.perf_counter()
        T0, Tg = snap_stable(o, T0), snap_stable(o, Tg)
        n_cands = checked = 0
        for k in range(1, self.k_max + 1):                 # more steps only if no plan with fewer passes
            cands = solve_angles(T0, Tg, o.size, k, self.alpha_max, k_min=k)
            n_cands += len(cands)
            for seq, alphas in cands:
                steps = self.steps(o, T0, seq, alphas)
                if not all(self._cheap_ok(o, s) for s in steps):
                    continue
                if checked >= self.max_kin_checks:
                    return Plan([], math.inf, dict(candidates=n_cands, kin_checks=checked, time_s=time.perf_counter() - t))
                checked += 1
                res = check_steps(self.kin, steps)
                if all(r[0] for r in res):
                    return Plan(steps, len(steps) + float(np.linalg.norm(alphas)),
                                dict(candidates=n_cands, kin_checks=checked, time_s=time.perf_counter() - t,
                                     joint_margin=min(r[1] for r in res), alphas_deg=np.degrees(alphas).round(1).tolist()))
        return Plan([], math.inf, dict(candidates=n_cands, kin_checks=checked, time_s=time.perf_counter() - t))
