"""Two pivots in one grasp: the box rolls by pi about the finger axis n.

The roll is two same-sense pivots by pi/2 about bottom edges parallel to n, onto the opposite face, 0.40 m further on
(W + H of the cross-section). The bearing takes only part of the pi (the arc of phi in which the palm clears the box);
the hand turns by the rest about n. That needs a grasp in the edge band of the face normal to n, and the study
experiments/double_pivot_study.py (pandaSim_handoff/review_2026-11-24/p2_study.md) found one region with margins:
the corner square at the BACK BOTTOM (back: opposite the roll direction), CHAIN_INSET from the back and bottom faces,
with the hand nearly horizontal behind the box at the start (palm direction CHAIN_ALPHA0 from the roll direction,
in the vertical plane through it) and ending above the box. The planner's pivot grasp (top corner, 20/20 mm) has no
end pose clear of the table. The phi limits come from this tilted grasp frame (clearance.phi_limits), as for every
pivot; the executor runs the two screws in one grasp with phi carried over (primitive_executor.run_plan).

The step starts and ends low beside the box with a tilted hand, so the pre pose and the retreat (straight up by
execution.lift) must clear the box: run it with lift >= CHAIN_LIFT (run_box.py --chain sets it).
"""
from __future__ import annotations

import itertools
import math
import time
from dataclasses import dataclass

import numpy as np

from pandaSim.planning import box_primitives as P
from pandaSim.planning import se3
from pandaSim.planning.clearance import phi_limits
from pandaSim.planning.sequencer import REACH, Plan, Step, down_axis, pivot_rise, pose_error

CHAIN_INSET = (0.010, 0.0275)        # m: grasp point from the back face and from the bottom face (study: 5 mm from the
                                     # edge of the region where the free arc collapses; 12 mm of palm above the table)
CHAIN_ALPHA0 = math.radians(158.0)   # palm direction at the start, from the roll direction toward the vertical
                                     # (68 deg from the hand pointing down: the palm behind the box, slightly up)
CHAIN_LIFT = 0.25                    # m: pre pose and retreat above the grasp (0.10 m: the fingers sweep the box)
G = 9.81


@dataclass
class ChainPlan:
    screws: list           # [(S, theta), (S, theta)] world frame, in one grasp
    T_grasp: np.ndarray    # {c} at the start (tilted: z is not vertical)
    T_goal: np.ndarray     # object pose after both pivots
    T_start: np.ndarray    # object pose before
    width: float           # object extent between the fingers [m]
    phi_lim: tuple = None  # (lo, hi) rad, as PivotPlan


def plan_box_roll(T_wo, size, d_o, y_ref=None, inset=CHAIN_INSET, alpha0=CHAIN_ALPHA0, flip=False):
    """Roll toward the object axis d_o (horizontal, normal to the finger axis): pivot about the bottom edge on side
    d_o, then about the next bottom edge in the same world direction. Grasp at the back bottom corner square.
    flip: the hand turned by pi about its own axis (fingers swapped), against the planner's sign rule for y. With
    a tilted hand that rule (y nearest y_ref) does not pick the better wrist configuration; where n is normal to
    y_ref it is a tie."""
    pl1 = P.plan_box_pivot(T_wo, size, d_o, math.pi / 2, y_ref)
    u = T_wo[:3, :3] @ np.asarray(d_o, float)                     # roll direction, world
    pl2 = P.plan_box_pivot(pl1.T_goal, size, np.round(pl1.T_goal[:3, :3].T @ u), math.pi / 2, y_ref)
    half = np.asarray(size, float) / 2
    d, up = np.asarray(d_o, float), -down_axis(T_wo)
    c = T_wo @ np.r_[-d * (np.abs(d) @ half - inset[0]) - up * (np.abs(up) @ half - inset[1]), 1.0]
    p = math.cos(alpha0) * u + math.sin(alpha0) * np.array([0.0, 0.0, 1.0])   # palm direction: -z of {c}
    y = pl1.T_grasp[:3, 1] * (-1.0 if flip else 1.0)               # finger axis
    T_grasp = se3.rp(np.column_stack([np.cross(y, -p), y, -p]), c[:3])
    return ChainPlan([(pl1.S, pl1.theta), (pl2.S, pl2.theta)], T_grasp, pl2.T_goal, T_wo, pl1.width,
                     phi_limits(se3.inv(T_grasp) @ T_wo, size))


class ChainPlanner:
    """Plans a roll by pi in one grasp from T0 to Tg (the goal must be the end of one of the two rolls), checked like
    the sequencer's steps: base keep-out and reach, then the rollout of the executor (kin, feasibility.Rollout)."""

    def __init__(self, kin, base_xy=(0.0, 0.0), symmetric=True):
        self.kin, self.base_xy, self.symmetric = kin, np.asarray(base_xy, float), symmetric
        self.y_ref = kin.T_ready[:3, 1]

    def plan(self, o, T0, Tg):
        """The roll that ends at Tg (with a marked box, one direction): the first candidate whose rollout passes, the
        finger sign of the planner's rule first (as every other grasp), the hand turned by pi about its axis second."""
        t = time.perf_counter()
        horiz = [i for i in range(3) if abs(T0[2, i]) < 0.5 and o.size[i] >= P.GRIPPER_MAX]   # not the finger axis
        checked = 0
        for flip, i, s in itertools.product((False, True), horiz, (1.0, -1.0)):
            d = np.zeros(3)
            d[i] = s
            pl = plan_box_roll(T0, o.size, d, self.y_ref, flip=flip)
            ep, er = pose_error(pl.T_goal, Tg, self.symmetric)
            if ep > 1e-3 or er > 1e-3:
                continue
            T1 = se3.exp6(pl.screws[0][0] * pl.screws[0][1]) @ T0
            pts = [(se3.exp6(S * th * k / 6) @ T)[:3, 3] for (S, th), T in
                   zip(pl.screws, (pl.T_grasp, se3.exp6(pl.screws[0][0] * pl.screws[0][1]) @ pl.T_grasp)) for k in range(7)]
            if not (P.plan_keepout_ok(pl, o.size, self.base_xy)
                    and all(REACH[0] <= np.linalg.norm(p[:2] - self.base_xy) <= REACH[1] for p in pts)):
                continue
            step = Step("pivot", T0, pl.T_goal, pl)
            checked += 1
            if self.kin.check(step)[0]:
                d1 = np.round(T1[:3, :3].T @ (T0[:3, :3] @ d))
                return Plan([step], o.mass * G * (pivot_rise(o, T0, d) + pivot_rise(o, T1, d1)),
                            dict(kin_checks=checked, time_s=time.perf_counter() - t))
        return Plan([], math.inf, dict(kin_checks=checked, time_s=time.perf_counter() - t))
