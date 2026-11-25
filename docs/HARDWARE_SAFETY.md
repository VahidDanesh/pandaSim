# Hardware safety review (branch `exp/hardware-safety`, 2026-11-16)

Scope: the path `run_box.py --execute` -> `run_plan` -> `run_segment` -> `PandaBackend` -> panda-py 1.1.1
(`IntegratedVelocity`) -> libfranka 0.8.0. Nothing here was run on the robot: every statement about the robot is
from the panda-py sources (GitHub `main`; the strings of the vendored binary match it) or from tests with a fake.

## What the controller really does (source audit)

| Question | Answer (panda-py source) | Consequence |
|---|---|---|
| `set_control` is not called (loop stall) | `q_d += dt * dq_d` with the last velocity, forever; no timeout | the arm keeps moving until a joint limit or a robot reflex |
| commands saturate / arm is blocked | torque `Kp (q_d - q) - Kd dq`, rate-limited to 1 Nm/ms, clipped to (87, 87, 87, 87, 12, 12, 12) Nm; `q_d` only clamped to the joint limits | `q_d` runs away from `q`, torque winds up, no error is raised |
| `stop_controller` | sets `motion_finished`, joins the control thread | the robot's own joint controller holds the measured position; the wound-up target is dropped |
| `start_controller`, `move_to_joint_position` | call `recover()` first: automatic error recovery | a reflex from the last run is cleared silently at the next start |
| tracking | damping acts on `dq`, not on the velocity error | `q` lags `q_d` by `Kd/Kp * dq`: 0.083 s (joints 1-3), 0.033 s (4), 0.08 s (5), 0.133 s (6), 0.2 s (7) |
| joint limits | virtual walls added to every torque controller: spring within 0.12 rad (joint 1), 0.09 rad (2-4), 0.035 rad (5-7), damper within twice that | the QP's 0.05 rad clearance is inside the spring zone of joints 1-4 |
| collision thresholds | the constructor does NOT set them (`set_default_behavior` would: 10 N / 10 Nm) | the robot's own reflex level is whatever was set last |
| robot error | raised from `PandaContext.ok()` at the next tick as `RuntimeError` | handled by `run_segment` |
| `mux_.try_lock()` result ignored in the 1 kHz step | data race with `set_control`, as noted in PLAN.md | unchanged; watch for erratic commands |

## Risk register

| # | Risk | Covered by (code, test) | Left for the user on hardware |
|---|---|---|---|
| 1 | Python loop stalls, arm keeps its last velocity | `PandaBackend` watchdog thread zeroes the velocity after `stall_timeout` (50 ms) without a new command; `run_segment` stops on a gap > `dt_max` (50 ms) between ticks. Test: stall of 250 ms, velocity zeroed during the stall | the watchdog cannot run if the interpreter is frozen inside a C call (quadprog, numpy); if the whole process dies the control thread dies too and the robot stops itself (not tested). Set rtprio and the `performance` governor |
| 2 | Arm blocked or saturated, torque winds up | tracking monitor: `Kp (q_d - q) - Kd dq` above (20, 20, 20, 20, 6, 6, 4) Nm stops the run; then `stop_controller`. Test: blocked joints, saturated joints | thresholds are my estimate (free motion with 1 kg needs about 8 Nm on joint 2). Read `track_torque_max` in `compare.json` after the air run and adjust `execution.robot.track_tau_max` |
| 3 | Force on the environment | software stop at 30 N / 15 Nm on the external wrench estimate; while the box is held, its predicted load (closed chain of `planning/load.py`, from the measured hand: `primitive_executor.predicted_load`) and a band of 5 N + 20 % around it are removed first (any share of it while the box is partly on a face or in a slide). Before (design review D14) any fraction of the full weight was removed, so at 5 kg a downward push of up to 79 N passed. Tests: the predicted load and an 8 N push pass, 45 N down or sideways stop (executor pivot and fake-robot slide with 5 kg), the same force without a box stops | sign of `O_F_ext_hat_K` for a hanging load (expected about (0, 0, -m g)); if it is the opposite, the load counts twice (early stops). Real mass of the box. A push stops the run once it is more than band + 30 N from the prediction: 35 to 40 N for the 5 kg box |
| 4 | Robot's own collision reflex unknown | `run_box` sets it once before the motion (`execution.robot.reflex`: stop at 40 N / 25 Nm, wrist 10 Nm), raised (`session_reflex`) only if the predicted peak load of the run plus its band needs it, printed before the confirmation; the pre-flight checks it against the predicted load. One setting for the whole run, free-space phases included. Tests: the call is made, a heavier predicted load raises it, level air keeps the config | the values are my choice: confirm or change them; switching them between free and held phases (controller restart with the box in the hand) is your decision |
| 5 | Stopping distance | after any stop the arm coasts by the lag, `Kd/Kp * speed`: about 2 cm at the hand at 0.3 rad/s | measure once in the air (Ctrl-C during a move) |
| 6 | Controller lag: segments time out, pivot leaves its arc | `SimBackend(lag=...)`; pre-flight simulates the plan with the lag; `settle` raised to 5 s; optional lag compensation (`--lead`), off by default. Simulation: without it 3.7 to 8.1 mm off the arc at full speed and 1.2 to 2.0 mm at speed 0.3, with it at most 0.6 mm | the lag model itself: `lag_measured_s` vs `lag_model_s` in `compare.json` after the air run; only then switch `--lead` on |
| 7 | Joint walls of panda-py fight the QP | real config: `rho_s` 0.13 rad, `rho_e` 0.40 rad; pre-flight refuses a plan that enters a wall zone | none |
| 8 | Unchecked joint-space moves | the path from the measured configuration to ready is swept against table, box and the arm itself (3 cm) before the confirmation; `move_to_joint_position` must arrive within 0.02 rad; fingers at their real opening in every clearance check | the table must hold nothing but the box; cables and the camera are not in the model |
| 9 | Failure with the box in the hand | abort procedure: zero velocity, stop the controller, never move the gripper, no motion afterwards; `log.abort` and the printed message say whether the fingers are on the box. Tests: grasp fails, grasp raises, release raises, robot reflex, exception, Ctrl-C. For an expected stop (rigid runs) the pre-flight checks the load held at the stop pose (pads, friction); no automatic recovery motion (D6: your decision) | what the gripper does on a user stop or power loss; whether a tilted box stays put when the arm holds |
| 10 | Records lost on a crash | `real.npz` and `compare.json` are written whatever ends the run | none |
| 11 | Brakes released before the confirmation | `Desk.unlock()` now comes after the typed phrase | none |
| 12 | Box not where the plan assumes | `experiments/placement_tolerance.py` (below) | the jig |
| 13 | phi is not measured | pre-flight checks that the contact determines phi (`grasp_observability`); phi limits from the hand geometry | bearing friction, slip of the edge on the table, drift of the integrated phi over a pivot |
| 14 | Hand too close to the box | one hand model for limits and check; the real config uses the cautious one (`hand_model: box`: mesh at least 11.8 mm away) | anything mounted on the hand that is not in `hand.stl` |
| 15 | The 5 kg box: forces the kinematic check cannot see (design review D8, D17) | pre-flight checks along the simulated run (`run_box.load_checks`, `planning/load.py`, two bounds: minimum norm and closed chain): hand force within 90 % of the payload weight, joint torques within half the limits and below the tracking stop, pads (f_g 70 N, mu_pad 1.2) within half their friction, table friction at the edge and the vertex with +-5 N of internal force within `box.mu`; warnings for a box that may roll onto a corner of its edge or turn on the table, and for the deflection of the hand (5 kg prim pivot: about 6 to 18 mm, not compensated) and the early landing on the face | measure `box.mu`, the pad friction and the deflection (a low run with the weighed box, HARDWARE_DAY section 3) before any full run; the model is quasi-static |

## Placement tolerance of the box (simulation, kinematics only)

Model: the closing fingers centre the box and turn it flat onto the pads, so the error along the finger axis and the
yaw error are removed by the grasp; the error across the fingers stays.

- Open fingers leave 9.9 mm on each side. Along the finger axis the clearance falls by 1 mm per mm of error.
- Yaw: the slide grasp (near the centre) loses 0.2 mm per degree; the pivot grasp (near the end of the box) loses
  2 mm per degree and a finger lands on the box at about 5 degrees.
- Budget for the jig: error along the fingers [mm] + 2 x yaw error [deg] <= 8 keeps 2 mm. Use +-2 mm and +-2 deg.
- Across the fingers nothing collides, but the box ends that far from the goal, the edge slips by 1.4 x the error
  during a 90 deg pivot, and the hand-box clearance falls by about 0.8 mm per mm (9.2 mm at zero, 3.0 mm at 8 mm
  with the cautious hand model). Use +-3 mm.

## Day one on the robot (in this order; stop at the first surprise)

1. Machine: `ulimit -r` > 0, governor `performance`. `python experiments/check_connection.py 172.16.0.2 --base-z 0.013`.
2. Decide the reflex thresholds in `box_real.yaml` (`execution.robot.reflex`), weigh the box (`box.mass`, 5 kg is a
   placeholder) and measure the friction box-table (`box.mu`) and pads-box (mu_pad 1.2 assumed).
3. Gripper empty, arm at rest: read `O_F_ext_hat_K` (should be below 1 N). Hang the box in the closed gripper by hand
   (a box above the payload: support it, or use a lighter test mass):
   the force should read about (0, 0, -m g). If the sign is the other way, tell me.
4. `run_box.py ... --level air` WITHOUT `--execute`: read the pre-flight; replay in Swift.
5. `--level air --execute` at speed 0.3. Hand on the stop button. During the first joint move press Ctrl-C once on
   purpose: the arm must stop within about 2 cm and `compare.json` must say `KeyboardInterrupt`.
6. A full air run. In `compare.json`: `lag_measured_s` against `lag_model_s` (within about 30 %),
   `track_torque_max` (well below the limits), `period_ms_max` (a few ms at most), `max_force` per segment.
7. If the lag matches: the same air run with `--lead`; the joint error against the simulation should drop to a
   few mrad and the segments should take their planned time.
8. Mark the box pose on the table (jig, +-2 mm, +-2 deg). `--level low` with the weighed box, a slide only, then a
   pivot at `--speed 0.15`. Watch the palm at the end of the pivot and the edge on the table (slip, roll onto a
   corner, turn). Measure the deflection of the hand under the load (compare.json: joint error and tracking torque of
   the pivot) against the pre-flight's range, before any full run.
9. Only then `--level full`.
