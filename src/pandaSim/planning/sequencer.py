"""Planner: a sequence of pivots and slides that takes a box from a start pose to a goal pose.

    [start slide] -> pivots A (90 deg each) -> [slide] -> pivots B -> goal

A resting pose of the box is labelled by the object axis that points down. A pivot needs a thin
axis t (extent < gripper opening) horizontal; it becomes the edge direction and the box tips to
the side d = +-(down x t). A pivot is a fixed body-frame displacement D, so the pose X the box
must be slid to before pivots B is found backwards from the goal: X = T_goal D_B^-1.
The box looks the same after 180 deg turns about its axes, so four goal poses are tried.

Checks, cheap first: base keep-out, slide tipping margin, reach of the grasp points; then the
kinematic check of the whole hand path (`kin`, see feasibility.py) on candidates in cost order.
Cost (quasi-static work): slides mu m g (|path| + r_eff |yaw|), pivots m g (rise of the centre).

All poses are in the world (table) frame, z up, table at z = 0.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from pandaSim.planning import box_primitives as P
from pandaSim.planning import se3

G = 9.81
REACH = (0.30, 0.78)               # m, horizontal distance of grasp points from the base axis
SLIDE_MAX_ROT = math.radians(100)  # max object rotation in one slide grasp
REGRASP_COST = 0.3                 # J-equivalent of one extra grasp
Z = np.array([0.0, 0.0, 1.0])


@dataclass(frozen=True)
class Box:
    size: tuple          # extents (x, y, z) [m], frame at the centre
    mass: float = 5.0    # kg   (mass and mu only enter the slide tipping check and the cost)
    mu: float = 0.5      # box-table friction


@dataclass
class Step:
    kind: str            # "pivot" | "slide"
    T_before: np.ndarray
    T_after: np.ndarray
    plan: object         # PivotPlan | SlidePlan


@dataclass
class Plan:
    steps: list
    cost: float
    stats: dict = field(default_factory=dict)

    @property
    def found(self):
        return math.isfinite(self.cost)


def _planar(T, x, y, yaw_delta=0.0):
    return se3.rp(se3.exp3(Z * yaw_delta) @ T[:3, :3], np.array([x, y, T[2, 3]]))


def down_axis(T):
    """Object axis (+-x, +-y, +-z) that points down."""
    d = T[:3, :3].T @ (-Z)
    u = np.zeros(3)
    i = int(np.argmax(np.abs(d)))
    u[i] = np.sign(d[i])
    return u


def slide_cost(o, Ta, Tb):
    r_eff = 0.5 * math.hypot(o.size[0], o.size[1])
    return o.mu * o.mass * G * (float(np.linalg.norm(Tb[:3, 3] - Ta[:3, 3])) + r_eff * se3.angle_between(Ta[:3, :3], Tb[:3, :3]))


def pivot_rise(o, T, d_o):
    half = np.asarray(o.size, float) / 2
    h = float(np.abs(down_axis(T)) @ half)
    w = float(np.abs(d_o) @ half)
    return math.hypot(w, h) - h


def box_symmetric_goals(Tg):
    """The four poses a box cannot be told apart in: identity and 180 deg turns about each object axis."""
    out = [Tg]
    for i in range(3):
        w = np.zeros(3)
        w[i] = math.pi
        out.append(Tg @ se3.rp(se3.exp3(w), np.zeros(3)))
    return out


def snap_stable(o, T):
    """Nearest exact resting pose: the down axis made exactly vertical, the height set from the size."""
    T = np.asarray(T, float)
    R = T[:3, :3]
    v, w = R @ down_axis(T), -Z
    axis = np.cross(v, w)
    s = float(np.linalg.norm(axis))
    if s > 1e-12:
        R = se3.exp3(axis / s * math.atan2(s, v @ w)) @ R
    return se3.rp(R, np.array([T[0, 3], T[1, 3], np.abs(R[2]) @ np.asarray(o.size, float) / 2]))


def pose_error(T, Tg):
    """Position error [m] and rotation error [rad] to the goal, over the four symmetric goal poses."""
    return (float(np.linalg.norm(T[:3, 3] - Tg[:3, 3])),
            min(se3.angle_between(T[:3, :3], Tv[:3, :3]) for Tv in box_symmetric_goals(Tg)))


def drop_noops(steps, tol_p=3e-3, tol_r=math.radians(2.0)):
    """Remove slides that do not move the box (the plan template allows them)."""
    return [s for s in steps if s.kind != "slide"
            or np.linalg.norm(s.T_after[:3, 3] - s.T_before[:3, 3]) > tol_p
            or se3.angle_between(s.T_after[:3, :3], s.T_before[:3, :3]) > tol_r]


class Sequencer:
    """kin: object with T_ready (4x4, {c} at the ready configuration) and
    check(paths) -> (feasible, joint_margin) per path; see feasibility.Kinematics.
    Without kin (y_ref given) only candidates() can be used."""

    def __init__(self, kin=None, y_ref=None, staging=None, max_pivots=3, max_kin_checks=12, base_xy=(0.0, 0.0)):
        self.kin = kin
        self.y_ref = kin.T_ready[:3, 1] if y_ref is None else np.asarray(y_ref, float)
        self.max_pivots, self.max_kin_checks = max_pivots, max_kin_checks
        self.base_xy = np.asarray(base_xy, float)
        xs, ys = (0.45, 0.55, 0.65), (-0.15, 0.0, 0.15)
        self.staging = staging if staging is not None else [(x, y, k * math.pi / 2) for x in xs for y in ys for k in range(4)]
        self._memo_d = {}

    # ---- primitive plans and cheap checks -------------------------------------------------------
    def _reach_ok(self, points):
        return all(REACH[0] <= np.linalg.norm(p[:2] - self.base_xy) <= REACH[1] for p in points)

    def _slide(self, o, Ta, Tb):
        return P.plan_box_slide(Ta, o.size, Tb, self.y_ref, o.mu, o.mass)

    def _slide_ok(self, o, plan):
        if plan is None or not P.plan_keepout_ok(plan, o.size, self.base_xy):
            return False
        if P.slide_tip_margin(o.size, plan, o.mu, o.mass) <= P.SLIDE_MARGIN_MIN:
            return False
        if sum(abs(th) for S, th in plan.screws if np.linalg.norm(S[:3]) > 0.5) > SLIDE_MAX_ROT:
            return False               # large rotations in one grasp swing the hand into the object
        return self._reach_ok([Th[:3, 3] for _, Th in plan.seg_starts()] + [P.grasp_end(plan)[:3, 3]])

    def _memo(self, key, f):
        if key not in self._memo_d:
            self._memo_d[key] = f()
        return self._memo_d[key]

    def _slide_steps(self, o, Ta, Tb):
        """Slide steps from Ta to Tb: one grasp, or two when the rotation is too large for one
        (rotate half about the vertical through the centre, regrasp, then the rest). None if not feasible."""
        return self._memo(("slide", Ta.tobytes(), Tb.tobytes()), lambda: self._slide_steps_(o, Ta, Tb))

    def _slide_steps_(self, o, Ta, Tb):
        pl = self._slide(o, Ta, Tb)
        if self._slide_ok(o, pl):
            return [Step("slide", Ta, Tb, pl)]
        Rr = Tb[:3, :3] @ Ta[:3, :3].T
        Tm = _planar(Ta, Ta[0, 3], Ta[1, 3], math.atan2(Rr[1, 0], Rr[0, 0]) / 2)
        p1, p2 = self._slide(o, Ta, Tm), self._slide(o, Tm, Tb)
        if self._slide_ok(o, p1) and self._slide_ok(o, p2):
            return [Step("slide", Ta, Tm, p1), Step("slide", Tm, Tb, p2)]
        return None

    def _box_pivots(self, o, T):
        """Possible pivots from pose T: list of (d_o, plan)."""
        half = np.asarray(o.size, float) / 2
        u = down_axis(T)
        out = []
        for i in range(3):
            t = np.zeros(3)
            t[i] = 1.0
            if 2 * half[i] >= P.GRIPPER_MAX or abs(u[i]) > 0.5:
                continue               # the thin axis must fit in the gripper and be horizontal
            d = np.cross(u, t)
            for s in (1.0, -1.0):
                out.append((s * d, P.plan_box_pivot(T, o.size, s * d, math.pi / 2, self.y_ref)))
        return out

    def _pivot_ok(self, o, plan):
        if not P.keepout_ok(plan.S, plan.theta, plan.T_start, o.size, self.base_xy):
            return False
        return self._reach_ok([plan.T_grasp[:3, 3], P.grasp_end(plan)[:3, 3]])

    def _pivot_step(self, o, T, d_o, pl):
        return Step("pivot", T, pl.T_goal, pl), o.mass * G * pivot_rise(o, T, d_o)

    # ---- search ---------------------------------------------------------------------------------
    def _pivot_seqs(self, o, T, depth):
        """All feasible pivot sequences from T up to `depth`: list of (steps, T_end, cost)."""
        return self._memo(("pivots", T.tobytes(), depth), lambda: self._pivot_seqs_(o, T, depth))

    def _pivot_seqs_(self, o, T, depth):
        out, frontier = [([], T, 0.0)], [([], T, 0.0)]
        for _ in range(depth):
            nxt = []
            for seq, Tc, c in frontier:
                for d_o, pl in self._box_pivots(o, Tc):
                    if self._pivot_ok(o, pl):
                        step, rise = self._pivot_step(o, Tc, d_o, pl)
                        nxt.append((seq + [step], pl.T_goal, c + rise))
            out += nxt
            frontier = nxt
        return out

    def _pivot_chain_shapes(self, o, T_mode, goal_down, depth):
        """Pivot choices (lists of d_o) from the mode of T_mode to goal_down, with their body-frame
        displacement D (T_end = T_start D). Feasibility is checked later."""
        out, frontier = [], [([], T_mode)]
        for k in range(depth + 1):
            nxt = []
            for ds, Tc in frontier:
                if np.allclose(down_axis(Tc), goal_down):
                    out.append((ds, se3.inv(T_mode) @ Tc))
                if k == depth:
                    continue
                for d_o, pl in self._box_pivots(o, Tc):
                    nxt.append((ds + [d_o], pl.T_goal))
            frontier = nxt
        return out

    def _box_starts(self, o, T0):
        """Start options: stay, rotate in place, or slide to a staging pose."""
        starts = [([], T0, 0.0)]
        here = [(T0[0, 3], T0[1, 3], k * math.pi / 2) for k in (1, 2, 3)]
        for x, y, dyaw in here + self.staging:
            Ts = _planar(T0, x, y, dyaw)
            st = self._slide_steps(o, T0, Ts)
            if st:
                starts.append((st, Ts, slide_cost(o, T0, Ts) + REGRASP_COST * (len(st) - 1)))
        return starts

    def _chain(self, o, X, ds):
        """Pivot steps ds from X and their costs, or None if one is not allowed."""
        T, steps, rises = X, [], []
        for d_o in ds:
            pl = P.plan_box_pivot(T, o.size, d_o, math.pi / 2, self.y_ref)
            if not self._pivot_ok(o, pl):
                return None
            step, rise = self._pivot_step(o, T, d_o, pl)
            steps.append(step)
            rises.append(rise)
            T = pl.T_goal
        return steps, rises

    def _cycle_protos(self, o, starts, Tg, goal_down):
        """Candidates before their middle slide is planned: (cost bound, finish). The bound is the cost
        without the regrasp of a two-grasp slide; finish() plans the slide and returns (cost, steps) or None."""
        shapes = {}
        for pre, Ts, c0 in starts:
            for seqA, Tm, cA in self._pivot_seqs(o, Ts, self.max_pivots if not pre else 2):
                if pre and not seqA:
                    continue                  # a start slide only makes sense before pivots
                key = tuple(down_axis(Tm).tolist())
                if key not in shapes:
                    shapes[key] = self._pivot_chain_shapes(o, Tm, goal_down, self.max_pivots)
                for ds, D in shapes[key]:
                    if len(seqA) + len(ds) > self.max_pivots:
                        continue
                    X = Tg @ se3.inv(D)
                    if not self._reach_ok([X[:3, 3]]):
                        continue
                    chain = self._memo(("chain", X.tobytes(), tuple(d.tobytes() for d in ds)),
                                       lambda: self._chain(o, X, ds))
                    if chain is None:
                        continue
                    slide = float(np.abs(X - Tm).max()) > 1e-6
                    if not (slide or pre or seqA or chain[0]):
                        continue              # already at the goal
                    bound = c0 + cA
                    if slide:
                        bound += slide_cost(o, Tm, X)
                    for r in chain[1]:
                        bound += r
                    yield bound, (lambda pre=pre, seqA=seqA, c0=c0, cA=cA, Tm=Tm, X=X, chain=chain, slide=slide:
                                  self._finish(o, pre, seqA, c0, cA, Tm, X, chain, slide))

    def _finish(self, o, pre, seqA, c0, cA, Tm, X, chain, slide):
        steps, c = pre + seqA, c0 + cA
        if slide:
            st = self._slide_steps(o, Tm, X)
            if not st:
                return None
            steps = steps + st
            c += slide_cost(o, Tm, X) + REGRASP_COST * (len(st) - 1)
        for r in chain[1]:
            c += r
        return c, steps + chain[0]

    def _fallback(self, o, starts, Tg):
        """No candidate for this goal: reach the goal position with another yaw, then rotate in place
        to the goal yaw (a rotation about the vertical through the grasp never tips a tall box)."""
        cands = []
        for dyaw in (math.pi / 4, -math.pi / 4, math.pi / 2, -math.pi / 2, 3 * math.pi / 4, -3 * math.pi / 4):
            Tv = _planar(Tg, Tg[0, 3], Tg[1, 3], dyaw)
            for _, finish in self._cycle_protos(o, starts, Tv, down_axis(Tg)):
                r = finish()
                if not r:
                    continue
                c, steps = r
                st = self._slide_steps(o, steps[-1].T_after, Tg)
                if st:
                    cands.append((c + slide_cost(o, steps[-1].T_after, Tg) + REGRASP_COST * len(st), steps + st))
        return cands

    def candidates(self, o, T0, Tg, k=None):
        """The k cheapest candidate plans [(cost, steps)] before the kinematic check (all if k is None).
        Middle slides are planned only for candidates whose cost bound can still reach the top k."""
        self._memo_d = {}
        T0, Tg = snap_stable(o, T0), snap_stable(o, Tg)
        k = math.inf if k is None else k
        goals = box_symmetric_goals(Tg)
        starts = self._box_starts(o, T0)
        protos = [(b, g, n, f) for g, Tv in enumerate(goals)
                  for n, (b, f) in enumerate(self._cycle_protos(o, starts, Tv, down_axis(Tv)))]
        protos.sort(key=lambda x: x[:3])
        found, done, kth = [], [False] * len(protos), math.inf   # found: (cost, n_steps, goal, index, steps)
        for j, (b, g, n, f) in enumerate(protos):
            if b > kth:
                break                         # cost >= bound > the k-th cost: cannot enter the top k
            done[j] = True
            r = f()
            if r:
                found.append((r[0], len(r[1]), g, n, r[1]))
                if len(found) >= k:
                    kth = sorted(x[0] for x in found)[k - 1]
        for g in range(len(goals)):           # a goal without candidates takes the fallback
            if any(x[2] == g for x in found):
                continue
            for j, (b, g2, n, f) in enumerate(protos):
                if g2 == g and not done[j]:
                    done[j] = True
                    r = f()
                    if r:
                        found.append((r[0], len(r[1]), g, n, r[1]))
            if not any(x[2] == g for x in found):
                found += [(c, len(st), g, 10**9 + m, st) for m, (c, st) in enumerate(self._fallback(o, starts, goals[g]))]
        found.sort(key=lambda x: x[:4])
        return [(x[0], x[4]) for x in found[: (None if k is math.inf else k)]]

    # ---- kinematic check and choice -------------------------------------------------------------
    def _check(self, top, lo, hi):
        """Feasibility and joint margin of candidates top[lo:hi]."""
        feasible, margin = [], []
        for _, steps in top[lo:hi]:
            res = [self.kin.check(s.plan) for s in steps]
            ok = all(r[0] for r in res)
            feasible.append(ok)
            margin.append(min((r[1] for r in res), default=0.0) if ok else 0.0)
        return feasible, margin

    def ranked(self, o, T0, Tg):
        """Feasible plans, best first: cheapest, equal costs by the larger joint margin. Generator:
        candidates are checked one cost group at a time, so taking the first plan stops early."""
        t = time.perf_counter()
        top = self.candidates(o, T0, Tg, self.max_kin_checks)
        lo = 0
        while lo < len(top):
            hi = lo + 1
            while hi < len(top) and top[hi][0] <= top[lo][0] + 1e-9:
                hi += 1
            feasible, margin = self._check(top, lo, hi)
            order = sorted((i for i in range(lo, hi) if feasible[i - lo]),
                           key=lambda i: (-round(margin[i - lo], 6), i))   # margins within 1e-6 rad are equal
            for i in order:
                yield Plan(drop_noops(top[i][1]), top[i][0],
                           dict(candidates=len(top), kin_checks=hi, time_s=time.perf_counter() - t,
                                joint_margin=margin[i - lo]))
            lo = hi

    def plan(self, o, T0, Tg):
        """The best feasible plan, or Plan([], inf) if none of the cheapest candidates is feasible."""
        t = time.perf_counter()
        for p in self.ranked(o, T0, Tg):
            return p
        return Plan([], math.inf, dict(time_s=time.perf_counter() - t))
