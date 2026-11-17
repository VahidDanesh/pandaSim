# Hardware day: from the dry run to the robot

Everything below runs from the repo root after `unset PYTHONPATH`, with `.venv/bin/python`. Only `--execute` moves the arm, and only after you type the confirmation phrase. Keep a hand on the user stop. What the safety layer does and what only the robot can confirm: [HARDWARE_SAFETY.md](HARDWARE_SAFETY.md).

## 1. Machine (once)
- Real-time priority for your user: add `vahid - rtprio 99` and `vahid - memlock unlimited` to `/etc/security/limits.d/99-realtime.conf`, log out and in, check `ulimit -r` prints 99.
- CPU governor: `sudo cpupower frequency-set -g performance` (resets at reboot).
- `.venv/bin/python experiments/check_connection.py 172.16.0.2 --base-z 0.013`: read only. It must report realtime available, no errors, joints inside the limits, a plausible TCP pose and the gripper; it locks the brakes at the end.

## 2. Measure before the first run
- Weigh the box and write `box.mass` into `experiments/configs/box_real.yaml` (the 1 kg there is a placeholder; the Panda carries 3 kg with the hand). Check `box.size`.
- The robot's own reflex thresholds (`execution.robot.reflex`): joint torques are `alpha` x a fraction of each joint's maximum torque (now about 25 Nm on joints 1-4 and 10 Nm on the wrist), forces `alpha` x 40 N, above the software stops of 30 N / 15 Nm. Change `alpha` only (smaller stops on lighter contact); `run_box.py` prints the resulting values before the confirmation. The tracking-torque limits (`execution.robot.track_tau_max`) are estimates: check them after the air run.
- Force sign: gripper empty and the arm at rest, `O_F_ext_hat_K` should read below 1 N; with the box hanging in the closed gripper about (0, 0, -m g). The wrench stop removes the box weight assuming this sign.
- Unlock the brakes, close the gripper, and run `.venv/bin/python experiments/calibrate_box.py 172.16.0.2` (read only). Guide the arm by hand and touch three table points (the table height should be within about 2 mm of 0; otherwise fix `robot.base`), then three top corners of the box: first the top corner of the vertical edge that carries the mark, then the other end of the longest top edge, then the corner next to it, and answer whether the mark is at the top. It prints the box pose as `--start x,y,yaw,face` (yaw over 360 deg).
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

## 6. The experiments of the paper (protocol)
`experiments/protocol.py` plans and simulates every entry of `experiments/configs/protocol.yaml` (Sec. 7.2 primitives, 7.3 object gaiting, 7.4 sequencing, 7.5 free rotation against the rigid grasp), prints GO / NO-GO and the `run_box.py` command of every run, with the `calibrate_box.py --measure` call that follows it. Run it again after any change of the config or the code.
- **The mark.** Put a sticker on one corner of the box. That corner is the (+x, +y, +z) corner of the box frame; place the box so that the mark is where the red sphere of the replay is. `calibrate_box.py` asks you to touch first the top corner of the vertical edge that carries the mark and whether the mark is at the top.
- **Order.** All runs with the bearing fingertips first (prim, gait, seq, fr free), then swap to the original fingers for the `fr ... rigid` runs. The fingers have the same geometry, so nothing else changes.
- **Expected stops.** A rigid run whose simulation stops by an infeasible QP carries `--expect-stop`; the pre-flight says at which angle of the pivot it stops. The robot then stops there and holds the tilted box: support the box, open the gripper from Desk, put the box back on its marks. `compare.json` must say `infeasible QP`; any other cause is a real failure.
- **Lag compensation.** `protocol.yaml` has `lead: false`. After the air run has shown that the measured lag matches the model (section 3), set it to `true` and run `protocol.py` again: the gait needs it (without it the hand leaves the arcs by about 4 mm at speed 0.3 in the simulation with the lag).
- **After each run.** `calibrate_box.py <host> --measure experiments/runs/box/<name>` (final pose against the goal, `measured.json`), then `analyze_run.py`. At the end `hardware_tables.py` writes the rows of the Sec. 7 tables (`experiments/results/hardware_tables.tex`; a dagger marks an error from forward kinematics, i.e. not measured) and `paper_figure.py <run> --out <Figures>/exp_pivot` makes panels (b) and (c) of the pivot figure.

## 7. Keep
The whole run folder `experiments/runs/box/<name>/` (`plan.json`, `preflight.json`, `expected.npz`, `real.npz`, `compare.json`, `analysis.pdf`, `summary.txt`), the measured final pose, a photo or video, and the config used (`git rev-parse HEAD` and `box_real.yaml`). `experiments/runs/` is ignored by git: copy it somewhere safe.
