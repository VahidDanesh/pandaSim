# Hardware day: from the dry run to the robot

Everything below runs from the repo root after `unset PYTHONPATH`, with `.venv/bin/python`. Only `--execute` moves the arm, and only after you type the confirmation phrase. Keep a hand on the user stop. What the safety layer does and what only the robot can confirm: [HARDWARE_SAFETY.md](HARDWARE_SAFETY.md).

## 1. Machine (once)
- Real-time priority for your user: add `vahid - rtprio 99` and `vahid - memlock unlimited` to `/etc/security/limits.d/99-realtime.conf`, log out and in, check `ulimit -r` prints 99.
- CPU governor: `sudo cpupower frequency-set -g performance` (resets at reboot).
- `.venv/bin/python experiments/check_connection.py 172.16.0.2 --base-z 0.013`: read only. It must report realtime available, no errors, joints inside the limits, a plausible TCP pose and the gripper; it locks the brakes at the end.

## 2. Measure before the first run
- Weigh the box and write `box.mass` into `experiments/configs/box_real.yaml` (the 1 kg there is a placeholder; the Panda carries 3 kg with the hand). Check `box.size`.
- Decide the robot's own reflex thresholds (`execution.robot.reflex`, now 40 N / 25 Nm, above the software stops of 30 N / 15 Nm) and the tracking-torque limits (`execution.robot.track_tau_max`); both are estimates.
- Force sign: gripper empty and the arm at rest, `O_F_ext_hat_K` should read below 1 N; with the box hanging in the closed gripper about (0, 0, -m g). The wrench stop removes the box weight assuming this sign.
- Unlock the brakes, close the gripper, and run `.venv/bin/python experiments/calibrate_box.py 172.16.0.2` (read only). Guide the arm by hand and touch three table points (the table height should be within about 2 mm of 0; otherwise fix `robot.base`), then three top corners of the box. It prints the box pose as `--start x,y,yaw,face`.
- Mark the box outline on the table with tape or a jig so that a run can be repeated from the same place. The plan assumes the start pose is exact; nothing measures the box during the run. Tolerance from the placement study: along the fingers +-2 mm and +-2 deg (rule: error along the fingers in mm + 2 x yaw in deg <= 8), across the fingers +-3 mm.
- Keep the box inside x 0.30 to 0.75 m and |y| < 0.35 m (the pre-flight refuses other poses).

## 3. The ladder
1. Dry run and replay: `experiments/run_box.py --start ... --goal ... --replay` (no robot). Read the pre-flight; it must end in GO. Watch the replay in a normal browser (`http://localhost:52000/?53000`).
2. `--execute --level air --speed 0.3`: no box on the table. The fingers close to the box width in the air. Check that the motion matches the replay. Press Ctrl-C once on purpose during the first joint move: the arm must stop within about 2 cm and `compare.json` must say `KeyboardInterrupt`.
   After a full air run read in `compare.json`: `lag_measured_s` against `lag_model_s` (within about 30 %), `track_torque_max` (well below the limits), the largest period and `max_force` per segment. Only if the lag matches, repeat with `--lead` (lag compensation of the joint controller; off by default, `execution.robot.lead`).
3. `--level low --speed 0.3` with the box, a slide-only task first (for example `--start 0.50,0.00,0,lying --goal 0.60,0.10,20,lying`), then one pivot, then a slide and a pivot.
4. `--level full` (speed up to 1.0) once `low` runs are clean.
After every run: `experiments/analyze_run.py experiments/runs/box/<name>` (figure and summary), and measure the final box pose against the goal (tape or ruler), since the logged object pose is forward kinematics, not a measurement.

## 4. Reading the pre-flight
| Line | Means | If it fails |
|---|---|---|
| pose inside the table area | start and goal within the region the planner was tested on | move the pose |
| pose clear of the robot base | no box corner within 0.30 m of the base axis | move the pose (such tasks have no plan) |
| plan found | a sequence exists whose simulation passes every check | other goal or face; with `--rigid` many more fail |
| simulation: ... | the simulated run of the plan: segments finish, QP feasible, speeds, goal reached, back at ready, palm and arm clear | should not fail when a plan is found; report it |
| the box cannot turn about the finger axis on its own | the grasp makes the passive bearing angle follow from the arm (observability) | other grasp, i.e. other start or goal |
| joint-limit margin | smallest distance to a joint limit in the simulation (refused below 0.05 rad, warning below 0.15 rad) | other start or goal |
| joints stay out of the controller's joint walls | panda-py adds repulsive walls within 0.12 rad (joint 1), 0.09 rad (2-4), 0.035 rad (5-7) of the limits | other start or goal |
| simulation with the joint controller's lag | the same plan with the measured-style lag of the robot's joint controller: segments finish, goal reached, how far the hand leaves the pivot arc (warning: the box is squeezed against the table by that much) | lower `--speed`, or `--lead` after it has been checked in the air |
| arm and hand clearance | smallest distance of the arm and hand to the box and table | refused below 5 mm |
| box mass within the payload / weight vs force stop | the mass in the config against 3 kg and against `safety.f_max` | check the mass; raise `f_max` only with care |
| joint accelerations | warning only (the robot integrates the commanded velocities) | lower `--speed` |
| robot: ... (after connecting) | joints inside limits, no error, no load, within 1 rad of the ready pose, gripper answers | move the arm near the ready pose in guiding mode, clear errors in Desk |

## 5. When a run stops
The arm is commanded to zero velocity and the controller is stopped; the gripper is left as it is (it may still hold the box). The log is written even after Ctrl-C or an exception. The cause is printed and saved (`compare.json`, `real.npz`): `external wrench`, `infeasible QP`, `robot error: ...`, `timeout`, `loop stall` (the watchdog zeroed the command after 50 ms without one), `tracking error` (a joint too far from its target), `KeyboardInterrupt`, or a failed grasp; the log also records whether the fingers are on the box. Open the gripper from Desk or with a gripper move, put the box back on its marks, run `analyze_run.py`, and lower `--speed` or change the task before trying again. Do not rerun a task that stopped on a wrench until you know what it touched.

## 6. Keep
The whole run folder `experiments/runs/box/<name>/` (`plan.json`, `preflight.json`, `expected.npz`, `real.npz`, `compare.json`, `analysis.pdf`, `summary.txt`), the measured final pose, a photo or video, and the config used (`git rev-parse HEAD` and `box_real.yaml`). `experiments/runs/` is ignored by git: copy it somewhere safe.
