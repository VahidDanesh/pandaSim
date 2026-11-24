"""Collision of the whole arm and the hand with the box and the table, with robotics-toolbox and coal.

The meshes are the collision meshes of the robot model (links 0-7, the hand and the two fingers). The fingers are
placed at their real opening (width). While the box is held they touch it, so they are tested against the box
only when asked (fingers_vs_box: every motion in which the box is not in the hand). The base link rests on the
table and is only tested against the box. Distances are signed (coal): clearance(q, T_obj) is the smallest
distance of any pair, negative if they penetrate. sweep() checks a joint-space move. Needs `pip install coal`.
"""
from __future__ import annotations

import numpy as np


def _sphere(mesh):
    """(centre, radius) of a sphere that encloses a spatialgeometry Mesh, in its own frame: from its vertices as
    coal gets them (Mesh._init_coal: y_up, then scale)."""
    import trimesh
    from spatialgeometry.geom.CollisionShape import _Y_UP_TO_Z_UP

    v = trimesh.load(mesh.filename, force="mesh").vertices
    v = (v @ _Y_UP_TO_Z_UP if mesh.y_up else v) * mesh.scale
    c = (v.min(axis=0) + v.max(axis=0)) / 2
    return c, float(np.linalg.norm(v - c, axis=1).max())


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
        self.obstacle = None
        # every collision mesh once: (mesh, link, vs_box, vs_table, finger, centre, radius of its enclosing sphere)
        links = [(l, True, l is not self.arm[0]) for l in self.arm] + [(hand, True, True)]
        links += [(f, False, True) for f in self.fingers]
        self._shapes = [(col, l, vs_box, vs_table, l in self.fingers, *_sphere(col))
                        for l, vs_box, vs_table in links for col in l.collision]
        self._box_r = float(np.linalg.norm(np.asarray(size, float)) / 2)     # sphere about the box centre

    def clearance(self, theta, T_obj, phi=0.0, width=None, fingers_vs_box=False):
        """Smallest distance (arm and hand to the box, arm links 1-7, hand and fingers to the table), and the
        name of the closest link. theta: 7 joint angles. width: finger opening [m] (None: closed, as before).
        fingers_vs_box: also test the fingers against the box (the box is not in the hand)."""
        from spatialmath import SE3

        r = self.robot
        w = 0.0 if width is None else float(np.clip(width, 0.0, 0.08))
        r.grippers[0].q = [w / 2, w / 2]
        r._update_link_tf(np.r_[np.asarray(theta, float)[:7], phi])
        r.update()
        self.table.update()
        T_obj = np.asarray(T_obj, float)
        self.box.T = SE3(T_obj, check=False)
        self.box.update()
        inf = max(self.margin, 0.0) + 0.05              # pairs farther apart do not count (closest_point: None)
        p_box = T_obj[:3, 3]
        best, who, self.obstacle = np.inf, None, None
        for col, link, vs_box, vs_table, finger, c, rad in self._shapes:
            vs_box = vs_box or (fingers_vs_box and finger)
            c = col._wT[:3, :3] @ c + col._wT[:3, 3]     # centre of the enclosing sphere, world (_wT: the pose coal uses)
            # lower bounds of the distance: sphere to sphere (box), sphere to the table top z = 0. Above inf, coal's
            # distance would be above inf too and dropped, so the pair is skipped (1e-6: coal's tolerance)
            for use, shape, name, lower in ((vs_box, self.box, "box", np.linalg.norm(c - p_box) - rad - self._box_r),
                                            (vs_table, self.table, "table", c[2] - rad)):
                if not use or lower > inf + 1e-6:
                    continue
                d = col.closest_point(shape, inf)[0]
                if d is not None and d < best:
                    best, who, self.obstacle = d, link.name, name      # obstacle: what the closest link is close to
        return float(best), who

    def sweep(self, theta_a, theta_b, T_obj, width=0.08, step=0.02, self_collision=True):
        """A joint-space move in a straight line from theta_a to theta_b (panda-py move_to_joint_position, the
        executor's JointMove) with the box at rest at T_obj and the fingers open by `width`: samples at most
        `step` rad apart in the largest joint motion. Returns dict(d_min, link, s, self_collision): the smallest
        distance to the box or the table (capped at margin + 0.05 m), the link and the path parameter where it
        occurs, and whether the arm touches itself anywhere on the path."""
        a, b = np.asarray(theta_a, float)[:7], np.asarray(theta_b, float)[:7]
        n = max(2, int(np.ceil(np.abs(b - a).max() / step)) + 1)
        out = dict(d_min=np.inf, link=None, s=None, self_collision=False)
        for s in np.linspace(0.0, 1.0, n):
            q = a + s * (b - a)
            d, who = self.clearance(q, T_obj, width=width, fingers_vs_box=True)
            if d < out["d_min"]:
                out.update(d_min=d, link=who, s=float(s))
            if self_collision and not out["self_collision"] and self.robot.isselfcollided(np.r_[q, 0.0]):
                out["self_collision"] = True
        out["d_min"] = float(min(out["d_min"], self.margin + 0.05))
        return out

    def ok(self, theta, T_obj, phi=0.0, width=None, fingers_vs_box=False):
        d, who = self.clearance(theta, T_obj, phi, width, fingers_vs_box)
        return d >= self.margin, who, d
