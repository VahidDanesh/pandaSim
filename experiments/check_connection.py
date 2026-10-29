"""Read-only connection check of the FER: connects, reads state and gripper, sends NO commands.

    python experiments/check_connection.py <robot-ip> [--base-z 0.013] [--no-gripper]

Checks: libfranka version match and FCI (connection), realtime setup (kEnforce),
joint positions inside the model limits, robot mode and errors, external wrench
fields used by the safety stop, TCP pose from our model, state read time.
"""
from __future__ import annotations

import argparse
import time

import numpy as np


def report(panda, gripper, robot, n_reads=200):
    """Everything is read-only. Returns a dict and prints it."""
    from pandaSim.control.primitive_executor import contact_pose

    s = panda.get_state()
    q = np.asarray(s.q, float)
    lo, hi = robot.qlim[0, :7], robot.qlim[1, :7]
    t0 = time.perf_counter()
    for _ in range(n_reads):
        panda.get_state()
    out = dict(
        q=q,
        q_inside_limits=bool(np.all((q > lo) & (q < hi))),
        min_margin_to_limits=float(np.minimum(hi - q, q - lo).min()),
        robot_mode=str(s.robot_mode),
        current_errors=str(s.current_errors),
        F_ext=np.asarray(s.O_F_ext_hat_K, float),
        tau_ext=np.asarray(s.tau_ext_hat_filtered, float),
        tcp_pose_model=contact_pose(robot, np.r_[q, 0.0]),
        get_state_ms=(time.perf_counter() - t0) / n_reads * 1e3,
    )
    if gripper is not None:
        g = gripper.read_once()
        out.update(gripper_width=g.width, gripper_max_width=g.max_width, gripper_is_grasped=g.is_grasped)
    for k, v in out.items():
        print(f"{k:22s} {np.array2string(v, precision=4) if isinstance(v, np.ndarray) else v}")
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("hostname")
    p.add_argument("--base-z", type=float, default=0.0, help="robot base height above the object table (m)")
    p.add_argument("--no-gripper", action="store_true")
    args = p.parse_args()

    import panda_py
    import spatialmath as sm
    from panda_py import libfranka

    from pandaSim.geometry.utils import create_virtual_panda

    try:
        panda = panda_py.Panda(args.hostname, realtime_config=libfranka.RealtimeConfig.kEnforce)
        print("realtime               enforced (OK)")
    except RuntimeError as e:
        print(f"realtime               NOT available ({e}); connecting with kIgnore")
        panda = panda_py.Panda(args.hostname)
    gripper = None if args.no_gripper else libfranka.Gripper(args.hostname)
    robot = create_virtual_panda()
    robot.base = sm.SE3(0, 0, args.base_z)
    report(panda, gripper, robot)


if __name__ == "__main__":
    main()
