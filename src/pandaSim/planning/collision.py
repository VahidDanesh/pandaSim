"""Collision of the whole arm and the hand with the box and the table, with robotics-toolbox and coal.

The meshes are the collision meshes of the robot model (links 0-7 and the hand). Fingers are left out of the
test against the box (they hold it) and checked against the table only; they are modelled closed, which
gives the same height at the fingertips. The base link rests on the table and is only tested against the box.
Distances are signed (coal): clearance(q, T_obj) is the smallest distance of any pair, negative if they
penetrate. Needs `pip install coal` (extra `collision`).
"""
from __future__ import annotations

import numpy as np


class ArmCollision:
    """robot: the pandaSim Panda (8 joints: 7 + phi); size: box dimensions; margin [m]: required distance."""

    def __init__(self, robot, size, margin=0.002, table_size=(3.0, 3.0, 0.2)):
        import spatialgeometry as sg
        from spatialmath import SE3

        self.robot, self.margin = robot, margin
        self.box = sg.Cuboid(scale=list(map(float, size)), collision=True)
        self.table = sg.Cuboid(scale=list(table_size), pose=SE3(0, 0, -table_size[2] / 2), collision=True)
        self.arm = robot.links[:8]
        hand, left, right = robot.grippers[0].links[:3]
        self.hand, self.fingers = hand, (left, right)
        self.names = [l.name for l in self.arm] + [hand.name]

    def _pairs(self, T_obj):
        from spatialmath import SE3

        self.box.T = SE3(np.asarray(T_obj, float), check=False)
        self.box.update()
        shapes = [(l, True, l is not self.arm[0]) for l in self.arm] + [(self.hand, True, True)]
        shapes += [(f, False, True) for f in self.fingers]
        return shapes

    def clearance(self, theta, T_obj, phi=0.0):
        """Smallest distance (arm and hand to the box, arm links 1-7, hand and fingers to the table), and the
        name of the closest link. theta: 7 joint angles."""
        r = self.robot
        r._update_link_tf(np.r_[np.asarray(theta, float)[:7], phi])
        r.update()
        self.table.update()
        best, who = np.inf, None
        for link, vs_box, vs_table in self._pairs(T_obj):
            for use, shape in ((vs_box, self.box), (vs_table, self.table)):
                if not use:
                    continue
                d = link.closest_point(shape, max(self.margin, 0.0) + 0.05, skip=True)[0]
                if d is not None and d < best:
                    best, who = d, link.name
        return float(best), who

    def ok(self, theta, T_obj, phi=0.0):
        d, who = self.clearance(theta, T_obj, phi)
        return d >= self.margin, who, d
