# pandaSim guide: box planner, executor, hardware path

Branch `exp/planner`. Setup: `uv sync --extra sim --extra robot --extra dev`, then run everything with `.venv/bin/python` (run `unset PYTHONPATH` first if ROS is sourced). Tests: `pytest` (headless, about 1 min). For the robot itself see [HARDWARE_DAY.md](HARDWARE_DAY.md) and the safety review [HARDWARE_SAFETY.md](HARDWARE_SAFETY.md) (watchdog, tracking monitor, payload-aware wrench stop, lag model and `--lead`, swept collision check of joint moves, placement tolerance).

## 1. What was added

| Part | Files | What it does |
|---|---|---|
| Planner | `planning/se3.py`, `box_primitives.py`, `sequencer.py`, `gait.py` | Pivot and slide plans for the 0.15 x 0.06 x 0.25 m box and the search over sequences of them (section 3); object gaiting with one arm (section 3b). numpy and scipy. |
| Checks | `planning/clearance.py`, `planning/collision.py`, `planning/feasibility.py` | Palm clearance and the limits of the bearing angle phi; arm and hand against box and table (coal through RTB); `Rollout` simulates a candidate step with the executor. |
| Control | `control/contact_qp.py`, `primitive_executor.py`, `backends.py`, `viewer.py` | `ContactQP` (free rotation or rigid, `phi_lim`); `run_plan` runs the steps; `SimBackend` / `PandaBackend(air=...)`; Swift replay with fingers at the grasp width. |
| Scripts | `experiments/run_box.py`, `run_plan.py`, `ablation.py`, `ablation_figures.py`, `check_connection.py` | Start/goal to robot; five sim tasks; random-trial ablation; figure; read-only robot check. |
| Hardware-day tools | `experiments/calibrate_box.py`, `analyze_run.py`, `paper_numbers.py` | Box pose by touching corners (read only; `--measure` for the final pose of a run); plan against measurement for a robot run; every number of the paper's Sec. 7 from the code and results, checked against `main.tex`. |
| Paper experiments | `experiments/configs/protocol.yaml`, `protocol.py`, `hardware_tables.py`, `paper_figure.py` | Every robot run of Sec. 7 checked in simulation with its command; the tables (LaTeX rows) and the pivot figure from the run folders. |
| Configs | `experiments/configs/box_plan.yaml` (sim), `box_real.yaml` (robot) | Box, robot base, speeds, gripper, safety, tolerances. |
| Results | `experiments/results/ablation_*.json`, `ablation_pivots.{pdf,png}` | 300 random pivots, 100 random tasks (rigid vs free rotation). |

## 2. How to run

```bash
# plan + simulate + pre-flight (GO / NO-GO), no robot. Pose = x,y,yaw_deg,face
python experiments/run_box.py --start 0.50,0.00,0,lying --goal 0.45,-0.15,60,standing
python experiments/run_box.py ... --replay            # Swift replay (normal browser, http://localhost:52000/?53000)
python experiments/run_box.py ... --rigid             # execute the free-rotation plan with the rigid grasp (original fingers)
python experiments/run_box.py ... --rigid --expect-stop   # same, when the simulation predicts the stop (infeasible QP)
python experiments/run_box.py --start 0.50,0.10,0,lying --goal 0.55,-0.08,30,lying --gait   # object gaiting
python experiments/protocol.py                        # every run of the paper: GO / NO-GO and the commands
python experiments/run_box.py ... --execute --host 172.16.0.2 --level air --speed 0.3   # robot
python experiments/run_plan.py --task all --viewer    # the five sim tasks
python experiments/ablation.py pivots|tasks           # results in experiments/results/
python experiments/ablation_figures.py                # figure
python experiments/ablation.py report                 # summaries, success by distance from the base, seeds
python experiments/calibrate_box.py 172.16.0.2        # box pose from touched corners (read only)
python experiments/calibrate_box.py 172.16.0.2 --measure experiments/runs/box/<name>   # final pose -> measured.json
python experiments/hardware_tables.py                 # Sec. 7 tables from the run folders
python experiments/paper_figure.py experiments/runs/box/<name> --out <Figures>/exp_pivot   # panels (b), (c)
python experiments/analyze_run.py experiments/runs/box/<name>   # after a robot run
python experiments/paper_numbers.py                   # paper parameters and results, checked in main.tex
python experiments/check_connection.py 172.16.0.2 --base-z 0.013   # read-only
```

Faces: `lying` (on the 0.06 x 0.25 face), `lying_flip`, `standing`, `standing_flip`. The largest face on the table cannot be grasped. The real box carries a mark on one corner: that corner is the (+x, +y, +z) corner of the box frame (x along 0.15 m, y along 0.06 m, z along 0.25 m), so `yaw` runs over 360 deg and a pose turned by 180 deg is a different pose (`planner.symmetric: false` in `box_real.yaml`). The replay draws the mark as a red sphere on the box and on the goal.
Output of a run: `experiments/runs/box/<name>/` with `plan.json`, `expected.npz`, `preflight.json`; after a robot run also `real.npz`, `compare.json`.

**Robot ladder (you present, hand on the stop button).** 1) `check_connection.py`. 2) `--level air` (no box, fingers close in the air). 3) `--level low` (box, slow). 4) `--level full`. Start with a slide-only task. `--execute` shows the checks and the plan and moves only after you type `RUN <level> <name>`; there is no bypass. A robot error, wrench above the limit, infeasible QP or failed grasp stops the arm (gripper stays as it is). Before the first run edit `box_real.yaml`: weigh the box (the placeholder is 1 kg; the Panda carries 3 kg), check the size and the base height.

## 3. The sequencer

Input: box (size, mass, friction), start pose, goal pose. Output: a list of steps, each a **slide** or a **pivot**, with the grasp pose, the screw motion and the grasp width, plus the cost.

- **State.** A resting pose is labelled by the box axis that points down. A pivot needs a thin box axis (< 8 cm, the gripper stroke) horizontal: it is the edge about which the box tips. Pivots only change the box between lying on the 0.06 x 0.25 face and standing; the largest face is neither grasped nor reached.
- **Template.** `[start slide] -> pivots A (90 deg each) -> [slide] -> pivots B -> goal`. A slide moves the box on the table (translation and yaw, at most 100 deg of yaw per grasp, otherwise two grasps). A plain box has four equivalent goal poses by symmetry (`box_plan.yaml`); the marked box on the robot has one.
- **Cheap tests first.** Keep-out around the base (0.30 m), tipping margin of a slide, reach of every grasp point (0.30 to 0.78 m from the base). Then candidate plans are ranked by cost.
- **Cost.** Quasi-static work: slide = mu m g (path length + effective radius x yaw); pivot = m g x rise of the box centre; 0.3 per regrasp.
- **Simulation check.** The cheapest candidates (at most 12, at most 3 pivots) are simulated with `Rollout`, the same executor code that runs the robot (1 kHz QP, joint and speed limits, palm and arm clearance). The first candidate that passes is returned, so a plan is accepted only if the executor can run it.
- **Free rotation vs rigid.** With free rotation the hand may turn about the finger axis during a pivot (bearing angle phi), within limits at which the palm still clears the box. This is what lets the Panda keep its posture; the rigid model fixes the hand to the box. `plan_free_rotation` (planner) and `free_rotation` (execution) switch it independently.
- **Execution of a step.** `pre` (0.10 m above the grasp) -> `approach` -> grasp (20 N) -> `pivot` or `slide` -> release -> `retreat` -> `ready` (joint move). Every step starts and ends at `q_ready`. Pivots are feed-forward along the planned screw; slides use pose feedback. The box pose is tracked from the arm's kinematics, not measured.

## 3b. Object gaiting (`planning/gait.py`, `run_box.py --gait`)

The box stays on the same face and turns about its vertices without sliding (manuscript Sec. 5, Case 1, with one arm). One step = one grasp across the 6 cm at the end away from the vertex, then five screw motions whose axes all pass through the vertex v_j: tilt gamma (15 deg) about the short edge that contains v_j, tilt beta (30 deg) about the box edge through v_j (the box now stands on v_j alone), turn alpha_j about the vertical through v_j, back by beta and gamma; release; regrasp at the other end. Net: a rotation by alpha_j about the vertical through v_j. Consecutive vertices are at opposite ends of the support face (bigger steps). The planner solves for the vertices and alpha (|alpha_j| <= 45 deg, exact goal, fewest steps, then the smallest |alpha|, eq:gait_a-d with eq:gait_b_single) and rolls each step out with the executor like the sequencer does. Typical goals (10-15 cm, up to 90 deg) take 3-4 steps, about 4-5 min at speed 0.3. With free rotation the bearing angle is fixed only if the vertex does not slip (the box weight presses it onto the table). The 25 screw motions accumulate the controller's lag: run gaiting with `--lead` once the air run has confirmed the lag model.

## 4. Where to look when something fails

- `no feasible plan`: start or goal outside the reach (`REACH`), largest face down, or a pivot the QP cannot do (try the free-rotation model).
- Pre-flight NO-GO: the line marked `FAIL` says which check (joint margin, clearance, payload, area, wrench threshold).
- `infeasible QP` in a segment: joint velocity or position limit of the arm; lower `--speed` first.
- Real run stops: `compare.json` and `real.npz` hold the segment, cause, joint error against the simulation and the forces.

## 5. Known limits

The hand model for the limits of phi is set by `planner.hand_model` (`box`: bounding box of the hand, cautious, used on the robot; `mesh`: collision mesh with 5 mm, used in the simulation study). 
Kinematic simulation (no slip or contact forces); only collisions of the arm, hand and box with the box and table are checked (no other objects), fingers are modelled closed; the box pose on the table is not measured (place it with marks or a jig); `planning/screw_motion_planner.generate_twist_trajectory_dq` (old code) is broken and unused.
