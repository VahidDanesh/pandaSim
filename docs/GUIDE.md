# pandaSim guide: box planner, executor, hardware path

Branch `exp/planner`. Setup: `uv sync --extra sim --extra robot --extra dev`, then run everything with `.venv/bin/python` (run `unset PYTHONPATH` first if ROS is sourced). Tests: `pytest` (87, headless, about 45 s).

## 1. What was added

| Part | Files | What it does |
|---|---|---|
| Planner | `planning/se3.py`, `box_primitives.py`, `sequencer.py` | Pivot and slide plans for the 0.15 x 0.06 x 0.25 m box and the search over sequences of them (section 3). numpy only. |
| Checks | `planning/clearance.py`, `planning/collision.py`, `planning/feasibility.py` | Palm clearance and the limits of the bearing angle phi; arm and hand against box and table (coal through RTB); `Rollout` simulates a candidate step with the executor. |
| Control | `control/contact_qp.py`, `primitive_executor.py`, `backends.py`, `viewer.py` | `ContactQP` (free rotation or rigid, `phi_lim`); `run_plan` runs the steps; `SimBackend` / `PandaBackend(air=...)`; Swift replay with fingers at the grasp width. |
| Scripts | `experiments/run_box.py`, `run_plan.py`, `ablation.py`, `ablation_figures.py`, `check_connection.py` | Start/goal to robot; five sim tasks; random-trial ablation; figure; read-only robot check. |
| Configs | `experiments/configs/box_plan.yaml` (sim), `box_real.yaml` (robot) | Box, robot base, speeds, gripper, safety, tolerances. |
| Results | `experiments/results/ablation_*.json`, `ablation_pivots.{pdf,png}` | 300 random pivots, 100 random tasks (rigid vs free rotation). |

## 2. How to run

```bash
# plan + simulate + pre-flight (GO / NO-GO), no robot. Pose = x,y,yaw_deg,face
python experiments/run_box.py --start 0.50,0.00,0,lying --goal 0.45,-0.15,60,standing
python experiments/run_box.py ... --replay            # Swift replay (normal browser, http://localhost:52000/?53000)
python experiments/run_box.py ... --rigid             # rigid grasp instead of free rotation (ablation)
python experiments/run_box.py ... --execute --host 172.16.0.2 --level air --speed 0.3   # robot
python experiments/run_plan.py --task all --viewer    # the five sim tasks
python experiments/ablation.py pivots|tasks           # results in experiments/results/
python experiments/ablation_figures.py                # figure
python experiments/check_connection.py 172.16.0.2 --base-z 0.013   # read-only
```

Faces: `lying` (on the 0.06 x 0.25 face), `lying_flip`, `standing`, `standing_flip`. The largest face on the table cannot be grasped.
Output of a run: `experiments/runs/box/<name>/` with `plan.json`, `expected.npz`, `preflight.json`; after a robot run also `real.npz`, `compare.json`.

**Robot ladder (you present, hand on the stop button).** 1) `check_connection.py`. 2) `--level air` (no box, fingers close in the air). 3) `--level low` (box, slow). 4) `--level full`. Start with a slide-only task. `--execute` shows the checks and the plan and moves only after you type `RUN <level> <name>`; there is no bypass. A robot error, wrench above the limit, infeasible QP or failed grasp stops the arm (gripper stays as it is). Before the first run edit `box_real.yaml`: weigh the box (the placeholder is 1 kg; the Panda carries 3 kg), check the size and the base height.

## 3. The sequencer

Input: box (size, mass, friction), start pose, goal pose. Output: a list of steps, each a **slide** or a **pivot**, with the grasp pose, the screw motion and the grasp width, plus the cost.

- **State.** A resting pose is labelled by the box axis that points down. A pivot needs a thin box axis (< 8 cm, the gripper stroke) horizontal: it is the edge about which the box tips. Pivots only change the box between lying on the 0.06 x 0.25 face and standing; the largest face is neither grasped nor reached.
- **Template.** `[start slide] -> pivots A (90 deg each) -> [slide] -> pivots B -> goal`. A slide moves the box on the table (translation and yaw, at most 100 deg of yaw per grasp, otherwise two grasps). There are four equivalent goal poses by box symmetry.
- **Cheap tests first.** Keep-out around the base (0.30 m), tipping margin of a slide, reach of every grasp point (0.30 to 0.78 m from the base). Then candidate plans are ranked by cost.
- **Cost.** Quasi-static work: slide = mu m g (path length + effective radius x yaw); pivot = m g x rise of the box centre; 0.3 per regrasp.
- **Simulation check.** The cheapest candidates (at most 12, at most 3 pivots) are simulated with `Rollout`, the same executor code that runs the robot (1 kHz QP, joint and speed limits, palm and arm clearance). The first candidate that passes is returned, so a plan is accepted only if the executor can run it.
- **Free rotation vs rigid.** With free rotation the hand may turn about the finger axis during a pivot (bearing angle phi), within limits at which the palm still clears the box. This is what lets the Panda keep its posture; the rigid model fixes the hand to the box. `plan_free_rotation` (planner) and `free_rotation` (execution) switch it independently.
- **Execution of a step.** `pre` (0.10 m above the grasp) -> `approach` -> grasp (20 N) -> `pivot` or `slide` -> release -> `retreat` -> `ready` (joint move). Every step starts and ends at `q_ready`. Pivots are feed-forward along the planned screw; slides use pose feedback. The box pose is tracked from the arm's kinematics, not measured.

## 4. Where to look when something fails

- `no feasible plan`: start or goal outside the reach (`REACH`), largest face down, or a pivot the QP cannot do (try the free-rotation model).
- Pre-flight NO-GO: the line marked `FAIL` says which check (joint margin, clearance, payload, area, wrench threshold).
- `infeasible QP` in a segment: joint velocity or position limit of the arm; lower `--speed` first.
- Real run stops: `compare.json` and `real.npz` hold the segment, cause, joint error against the simulation and the forces.

## 5. Known limits

Kinematic simulation (no slip or contact forces); only collisions of the arm, hand and box with the box and table are checked (no other objects), fingers are modelled closed; the box pose on the table is not measured (place it with marks or a jig); `planning/screw_motion_planner.generate_twist_trajectory_dq` (old code) is broken and unused.
