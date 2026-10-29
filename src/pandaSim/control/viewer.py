"""Swift view of the robot, kept out of the control loop.

LiveViewer runs Swift in a separate process (its own interpreter, so Swift's ~9 ms
step never blocks the 1 kHz loop). The loop only writes the joint vector into shared
memory; the viewer reads it at `rate` Hz. replay() plays a saved run back offline.
The viewer process is started with "spawn": the launching script needs an
`if __name__ == "__main__":` guard.

objects: list of dicts, the first one is the manipulated object, e.g.
    {"type": "cuboid", "scale": [x, y, z], "pose": T (4x4), "color": "blue"}
    {"type": "cylinder", "radius": r, "length": l, "pose": T}
"""
from __future__ import annotations

import multiprocessing as mp
import os

import numpy as np

# shared memory layout: q (8) | attached (1) | T_c_obj (16) | stop (1)
_Q, _ATT, _T, _STOP, _SIZE = slice(0, 8), 8, slice(9, 25), 25, 26


def _make_shape(spec):
    import spatialgeometry as sg
    import spatialmath as sm

    kind = spec["type"].lower()
    kw = {"pose": sm.SE3(np.asarray(spec["pose"]), check=False), "color": spec.get("color", "blue")}
    if kind == "cuboid":
        return sg.Cuboid(scale=spec["scale"], **kw)
    if kind == "cylinder":
        return sg.Cylinder(radius=spec["radius"], length=spec["length"], **kw)
    raise ValueError(f"unknown object type {spec['type']!r}")


def _scene(base, objects, q0):
    import spatialmath as sm
    import swift

    from pandaSim.geometry.utils import create_virtual_panda

    robot = create_virtual_panda()
    robot.base = sm.SE3(np.asarray(base), check=False)
    robot.q = q0
    env = swift.Swift()
    env.launch(realtime=True)
    env.add(robot)
    shapes = [_make_shape(o) for o in objects]
    for s in shapes:
        env.add(s)
    return env, robot, (shapes[0] if shapes else None)


def _swift_main(shared, base, objects, rate):
    from pandaSim.control.primitive_executor import contact_pose

    os.nice(10)                                    # never compete with the control loop
    a = np.frombuffer(shared, dtype=float)
    env, robot, obj = _scene(base, objects, a[_Q].copy())
    while a[_STOP] == 0:
        q = a[_Q].copy()
        robot.q = q
        if obj is not None and a[_ATT] > 0:
            obj.T = contact_pose(robot, q) @ a[_T].reshape(4, 4)
        env.step(1.0 / rate)


class LiveViewer:
    """Live view in a separate process. In the loop: update(q8) (~1 us, no lock; a torn read only
    affects one frame). attach(T_c_obj) makes the first object follow {c}; detach() leaves it."""

    def __init__(self, base, objects=(), q0=None, rate=30.0, target=_swift_main):
        ctx = mp.get_context("spawn")
        self._shared = ctx.RawArray("d", _SIZE)
        self._a = np.frombuffer(self._shared, dtype=float)
        if q0 is not None:
            self._a[_Q] = q0
        self._proc = ctx.Process(target=target, args=(self._shared, np.asarray(base), list(objects), rate),
                                 daemon=True)
        self._proc.start()

    def update(self, q8):
        self._a[_Q] = q8

    def attach(self, T_c_obj):
        self._a[_T] = np.asarray(T_c_obj).ravel()
        self._a[_ATT] = 1.0

    def detach(self):
        self._a[_ATT] = 0.0

    def close(self, timeout=2.0):
        self._a[_STOP] = 1.0
        self._proc.join(timeout)
        if self._proc.is_alive():
            self._proc.terminate()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def replay(path, base, objects=(), speed=1.0, rate=30.0):
    """Play a saved run (Log.save) back in Swift; the first object follows {c} between grasp and release."""
    from pandaSim.control.primitive_executor import contact_pose

    d = np.load(path, allow_pickle=True)
    t, theta, phi = d["t"], d["theta"], d["phi"]
    events = {name: time_ for name, time_, ok in d["events"] if ok}
    q = np.c_[theta, phi]
    env, robot, obj = _scene(base, objects, q[0])
    T_c_obj = None
    for k in np.searchsorted(t, np.arange(t[0], t[-1], speed / rate)):
        robot.q = q[k]
        T_wc = contact_pose(robot, q[k])
        grasped = "grasp" in events and t[k] >= events["grasp"] and t[k] < events.get("release", np.inf)
        if obj is not None and grasped:
            if T_c_obj is None:
                T_c_obj = np.linalg.inv(T_wc) @ np.asarray(objects[0]["pose"])
            obj.T = T_wc @ T_c_obj
        env.step(1.0 / rate)
    return env
