# Hardware day: from the dry run to the robot

Everything below runs from the repo root after `unset PYTHONPATH`, with `.venv/bin/python`. Only `--execute` moves the arm, and only after you type the confirmation phrase. Keep a hand on the user stop.

## 1. Machine (once)
- Real-time priority for your user: add `vahid - rtprio 99` and `vahid - memlock unlimited` to `/etc/security/limits.d/99-realtime.conf`, log out and in, check `ulimit -r` prints 99.
- CPU governor: `sudo cpupower frequency-set -g performance` (resets at reboot).
- `.venv/bin/python experiments/check_connection.py 172.16.0.2 --base-z 0.013`: read only. It must report realtime available, no errors, joints inside the limits, a plausible TCP pose and the gripper; it locks the brakes at the end.

## 2. Measure before the first run
- Weigh the box and write `box.mass` into `experiments/configs/box_real.yaml` (the 1 kg there is a placeholder; the Panda carries 3 kg with the hand). Check `box.size`.
- Unlock the brakes, close the gripper, and run `.venv/bin/python experiments/calibrate_box.py 172.16.0.2` (read only). Guide the arm by hand and touch three table points (the table height should be within about 2 mm of 0; otherwise fix `robot.base`), then three top corners of the box. It prints the box pose as `--start x,y,yaw,face`.
- Mark the box outline on the table with tape so that a run can be repeated from the same place. The plan assumes the start pose is exact; nothing measures the box during the run.
- Keep the box inside x 0.30 to 0.75 m and |y| < 0.35 m (the pre-flight refuses other poses).

## 3. The ladder
1. Dry run and replay: `experiments/run_box.py --start ... --goal ... --replay` (no robot). Read the pre-flight; it must end in GO. Watch the replay in a normal browser (`http://localhost:52000/?53000`).
2. `--execute --level air --speed 0.3`: no box on the table. The fingers close to the box width in the air. Check that the motion matches the replay.
3. `--level low --speed 0.3` with the box, a slide-only task first (for example `--start 0.50,0.00,0,lying --goal 0.60,0.10,20,lying_flip`), then one pivot, then a slide and a pivot.
4. `--level full` (speed up to 1.0) once `low` runs are clean.
After every run: `experiments/analyze_run.py experiments/runs/box/<name>` (figure and summary), and measure the final box pose against the goal (tape or ruler), since the logged object pose is forward kinematics, not a measurement.

## 4. Reading the pre-flight
| Line | Means | If it fails |
|---|---|---|
| pose inside the table area | start and goal within the region the planner was tested on | move the pose |
| plan found | a sequence exists whose simulation passes every check | other goal or face; with `--rigid` many more fail |
| simulation: ... | the simulated run of the plan: segments finish, QP feasible, speeds, goal reached, back at ready, palm and arm clear | should not fail when a plan is found; report it |
| joint-limit margin | smallest distance to a joint limit in the simulation (refused below 0.05 rad, warning below 0.15 rad) | other start or goal |
| arm and hand clearance | smallest distance of the arm and hand to the box and table | refused below 5 mm |
| box mass within the payload / weight vs force stop | the mass in the config against 3 kg and against `safety.f_max` | check the mass; raise `f_max` only with care |
| joint accelerations | warning only (the robot integrates the commanded velocities) | lower `--speed` |
| robot: ... (after connecting) | joints inside limits, no error, no load, within 1 rad of the ready pose, gripper answers | move the arm near the ready pose in guiding mode, clear errors in Desk |

## 5. When a run stops
The arm is commanded to zero velocity and the controller is stopped; the gripper is left as it is (it may still hold the box). The cause is printed and saved (`compare.json`, `real.npz`): `external wrench`, `infeasible QP`, `robot error: ...`, `timeout`, or a failed grasp. Open the gripper from Desk or with a gripper move, put the box back on its marks, run `analyze_run.py`, and lower `--speed` or change the task before trying again. Do not rerun a task that stopped on a wrench until you know what it touched.

## 6. Keep
The whole run folder `experiments/runs/box/<name>/` (`plan.json`, `preflight.json`, `expected.npz`, `real.npz`, `compare.json`, `analysis.pdf`, `summary.txt`), the measured final pose, a photo or video, and the config used (`git rev-parse HEAD` and `box_real.yaml`). `experiments/runs/` is ignored by git: copy it somewhere safe.
