# The planner: algorithm, decisions, and how it works

This document explains how pandaSim turns "the box is here, put it there" into a sequence of robot motions, why it is
built this way, and where each part lives in the code. It describes the code on `exp/planner` (2026-11-25).

Contents
1. [The problem in one paragraph](#1-the-problem-in-one-paragraph)
2. [The one idea: every motion is a constant screw motion](#2-the-one-idea-every-motion-is-a-constant-screw-motion)
3. [The building blocks: pivot, slide, gait step, roll](#3-the-building-blocks)
4. [The sequencer: how a plan is found](#4-the-sequencer-how-a-plan-is-found)
5. [The cost](#5-the-cost)
6. [Feasibility: cheap tests, then a rollout of the real executor](#6-feasibility)
7. [The other two planners: gait and roll](#7-the-other-two-planners)
8. [What happens to the plan afterwards](#8-what-happens-to-the-plan-afterwards)
9. [The decisions and why](#9-the-decisions-and-why)
10. [Limits: what the planner does not do](#10-limits)
11. [Code map and how to look inside](#11-code-map-and-how-to-look-inside)

---

## 1. The problem in one paragraph

A box (0.15 x 0.06 x 0.25 m, one corner marked) lies on a table in front of a Franka Panda. The gripper opens at
most 8 cm, so it can only hold the box across its 0.06 m side. The fingertips are bearings: the box can turn freely
about the finger axis **n**, the line between the fingertips. The box is too heavy, or the task does not allow it, to be
lifted, so it is moved by **non-prehensile primitives** that keep it in contact with the table: it is **slid** on a face,
**pivoted** about a bottom edge onto a neighbouring face, or **gaited** (walked about its vertices). Given a start
pose and a goal pose, both resting on a face, the planner returns a list of steps. Each step is one grasp: approach,
close, one motion (one or a few screw motions), open, retreat. The plan is guaranteed to run in the executor.

The box has four faces it can rest on and be grasped in, because the 0.06 m axis must stay horizontal between the
fingers:

```
            pivot 90 deg about an edge parallel to n (the finger axis)
   lying  ───────────►  standing  ───────────►  lying_flip  ───────────►  standing_flip  ───► (back to lying)
 (0.06x0.25 face)      (0.15x0.06 face)
```

That chain of faces is the **face graph**. Here it is a cycle of four. The largest face (0.15 x 0.25) would put the
0.06 m axis vertical, so it cannot be grasped and is outside the domain.

---

## 2. The one idea: every motion is a constant screw motion

Every primitive moves the box by a **constant screw motion**: a rotation about a fixed line in space, combined with a
translation along it. In the world frame this is

```
T_obj(s) = exp([S] theta s) T_obj(0),    s in [0, 1]
```

where S = (omega, v) is the unit screw axis (Lynch & Park ordering) and theta the angle (or the distance, for a pure
translation). Anything rigidly attached to the box moves the same way, including the grasp frame {c} at the fingertips:
`T_wc(s) = exp([S] theta s) T_grasp`.

The screw axis follows from the contact between the box and the table:

| primitive | contact that stays fixed | screw |
|---|---|---|
| pivot | a bottom edge (line contact) | pure rotation about that edge: S = (e, -e x q), theta = +-90 deg |
| slide | the bottom face stays on the table | a planar displacement: a rotation about a vertical axis, or a translation |
| gait step | one vertex (point contact) | five pure rotations, each about an axis through that vertex |

So the planner never solves inverse kinematics or samples paths for the box. A primitive is *defined* by its contact,
and its path is the screw. All the search is about **which** primitives to chain and **where** to slide.

The SE(3) arithmetic (`planning/se3.py`, 93 lines: exp, log, adjoint) uses 4x4 matrices. The manuscript writes the same
motions with unit dual quaternions and ScLERP. The two give the same poses to 1.1e-11 (a unit test checks this), and the
matrix version is not slower. Section 9, decision D5, explains why the code was not rewritten.

---

## 3. The building blocks

`planning/box_primitives.py` turns a box pose and a primitive choice into a plan object: the screws, the grasp frame
{c}, the end pose, the box width between the fingers, and the limits of the bearing angle phi.

### Pivot (`plan_box_pivot`)

- **Input:** the box pose and a horizontal object axis d, the side the box tips toward.
- **Edge:** the bottom edge on side d. Its direction is e = up x d, and it is **parallel to the finger axis n**. That
  is the point of the bearing: the box turns about e, the bearing lets it turn relative to the hand about n, and the
  hand only has to follow the arc of the grasp point.
- **Screw:** S = (e, -e x q), with q a point on the edge and theta = pi/2. One pivot always moves the box onto the
  neighbouring face of the face graph.
- **Grasp:** near the top, on the side away from the edge. The grasp point is GRASP_INSET = 20 mm from the top face and
  from the far side face, and the hand points down. This gives the longest lever about the edge, so the hand pushes with
  the least force, and the largest free arc of phi before the palm meets the box. The review measured 123 deg at the
  corner and 0 deg in the middle of the face.
- **phi limits:** `clearance.phi_limits` sweeps the hand about n and finds the interval in which the palm stays 5 mm
  clear of the box. The QP keeps phi inside it during the motion.

### Slide (`plan_box_slide`)

- **Grasp:** top-down at the centre of the top face, 20 mm below its top, fingers across the thinnest horizontal axis.
- **Direct screw:** any planar displacement from start to goal is one screw, `planar_screw(T_from, T_to)`.
- **Tipping check (`tip_margin`):** table friction acts at the bottom of the box, h_g below the grasp. Dragging the box
  is quasi-statically stable only if mu h_g is smaller than the distance from the grasp to the leading edge of the
  support. Along the finger axis, the friction couple of the fingertip pads also helps. A slide needs a margin of more
  than 3 cm (SLIDE_MARGIN_MIN).
- **Tip-free chain:** if the direct screw tips the box, the slide becomes a chain of screws in the same grasp. The box
  rotates about the vertical through the grasp point, which does not move that point and so cannot tip the box. It is
  then translated along the finger axis (or, as a second try, across it), and rotated to the goal yaw. The first chain
  that passes the tipping check is used.
- **Size limits:** a slide rotates the box by at most 100 deg in one grasp; beyond that the hand swings into the box. A
  larger rotation becomes two slides with a regrasp halfway.

### Gait step (`planning/gait.py`) and roll by pi (`planning/chain.py`)

Both are several screws in one grasp. They are described with their planners in Section 7.

---

## 4. The sequencer: how a plan is found

`planning/sequencer.py`, class `Sequencer`. A plan always has this shape (the **template**):

```
[start slide]  ->  pivots A  ->  [middle slide]  ->  pivots B  ->  goal
```

- **Brackets:** the parts in brackets may be empty.
- **Pivot groups:** A and B together have at most max_pivots = 3 pivots.
- **What a slide is:** each slide is one grasp, or two if it rotates the box too far for one.
- **Why a middle slide:** each pivot moves the box 0.2 m along the table. The slide between the two pivot groups puts
  the box back where the remaining pivots can reach, and where they end exactly at the goal.

### The key trick: solve the last pivots backwards from the goal

A pivot depends only on which face the box rests on and which side it tips to, not on where the box is on the table.
In the box's own frame it is always the same displacement D, so `T_after = T_before D`. A chain of pivots B is therefore
also a fixed body-frame displacement D_B, and the pose X the box must be slid to before pivots B is

```
X = T_goal D_B^-1
```

The planner never searches for where to put the middle slide: for every choice of pivots B it computes X exactly. The
continuous part of the search (the middle slide) is solved in closed form, and only discrete choices are left.

### The search, step by step

```
plan(box, T0, Tg):
  T0, Tg  <- snap to exact resting poses (down axis vertical, height from the size)
  goals   <- [Tg]                              marked box (box_real.yaml: symmetric: false)
             or Tg and its three 180 deg turns  plain box: they look the same

  starts  <- stay at T0
             + rotate in place by 90/180/270 deg
             + slide to each of 36 staging poses (x in {0.45,0.55,0.65}, y in {-0.15,0,0.15}, yaw in 4 x 90 deg)
             each start slide must pass the slide tests (Sec. 6); cost = work + kappa per grasp

  for each start (pre, Ts):
    for each feasible pivot sequence A from Ts (BFS, at most 3 pivots; at most 2 after a start slide), ending at Tm:
      for each pivot sequence B from the face of Tm to the goal face, with displacement D_B   (cached per face)
          (|A| + |B| <= 3):
        X <- Tg D_B^-1                         where pivots B must start
        reject if X is out of reach, or a pivot of B fails keep-out / reach at X
        middle slide Tm -> X if Tm != X
        bound <- cost(start) + cost(A) + cost(B) + work(middle slide) + kappa   (middle slide counted as one grasp)
        candidate (bound, finish)              finish() plans the middle slide; it may need two grasps, or fail

  sort candidates by bound
  take them in that order; finish() each; stop when the next bound exceeds the 12th-best cost found
      -> the 12 cheapest complete candidates (exact top 12: the bound never exceeds the true cost)
  if a goal pose has no candidate: fallback = reach the goal position with another yaw (+-45, +-90, +-135 deg),
      then rotate in place to the goal yaw

  rollout check (Sec. 6), one cost group at a time, cheapest group first:
      within a group of equal cost, the plan with the larger smallest distance to a joint limit goes first
  return the first candidate that passes, with slides that do not move the box removed
```

Every complete candidate is built before any rollout, so the costly part (rollouts) only ever touches the 12 cheapest
candidates, and usually far fewer, because the search stops at the first cost group that has a feasible plan.

### Two real runs (box_real.yaml, marked box, 1 kg placeholder mass)

| task | candidates (time) | what happened | plan | rollouts | planning time |
|---|---|---|---|---|---|
| turn: `0.50,0.00,0,lying -> 0.55,0.10,30,standing` | 12 (0.11 s) | the cheapest passed | pivot, slide; cost 2.39 (pivot about 1.0 = 0.70 J of lift + kappa; slide about 1.4) | 1 candidate | 1.3 s |
| flip: `0.50,0.00,0,lying -> 0.50,0.00,0,lying_flip` | 12 (0.38 s) | the cheapest (5.19: six grasps) failed its rollout; the next group of four at 5.30 were all checked, and the one with the largest joint margin (0.545 rad) won | slide, pivot, pivot, slide | 5 candidates | 11 s |

Generating candidates takes a fraction of a second. **Almost all planning time is rollouts**, and a step that appears in
several candidates is rolled out once (cache).

### Is the template complete?

For this box, yes, within its domain. The face graph is a cycle of four, and pivots A and B enumerate every pivot
sequence of up to three pivots between any two faces. The middle slide is solved exactly. The continuous choices left
are approximated:

- the start slide uses a fixed staging grid of 36 poses plus 3 in-place turns;
- only the 12 cheapest candidates are checked.

The result is therefore "the cheapest feasible plan among the enumerated candidates", not a global optimum (review
thread D1). Of the tasks in the earlier random benchmark that had no plan, 7 of 8 and 3 of 3 had the start or goal inside
the robot's own keep-out zone. On well-posed tasks the planner found plans in 92 of 93 and 97 of 97 cases.

---

## 5. The cost

The cost (eq:seq_a of the Case 3 draft) is the quasi-static work, plus a constant kappa per grasp:

```
slide:  mu m g ( |displacement| + r_eff |yaw| ),    r_eff = half the diagonal of the support face
pivot:  m g (rise of the centre of mass)            (lifting over the edge)
grasp:  kappa = 0.3 J-equivalent, charged once per step (a slide split into two grasps pays it twice)
```

- **Why work:** it is the physical effort of each primitive under the same assumptions as the method (quasi-static, no
  slip), and it puts slides and pivots on one scale.
- **Why kappa:** time on the robot follows the number of grasps more than the work. Every grasp costs an approach, a
  close, an open, a retreat and a return to ready; at speed 0.3 a one-slide run takes 41 s in the protocol. Without
  kappa, a plan with many small slides would win.
- **Value of kappa:** 0.3 is a choice, yours, not derived. It is charged once per grasp since review D2.

The cost **ranks candidates during planning and nothing else**. The cost plays no part once the robot moves. Then the
contact QP runs every segment (Section 8).

---

## 6. Feasibility

### Cheap tests (inside the search, no simulation)

| test | where | what |
|---|---|---|
| gripper fit | `box_grasp`, `_box_pivots` | the axis between the fingers is under 8 cm and horizontal |
| reach | `Sequencer._reach_ok` | grasp points at 0.30-0.78 m (horizontal) from the base axis, at the start and end of each screw |
| base keep-out | `keepout_ok` | box corners stay 0.30 m from the base axis along the whole screw (12 samples) |
| tipping | `tip_margin` | slides: margin > 3 cm |
| slide rotation | `_slide_ok` | <= 100 deg of box rotation per grasp |

### The rollout: the plan runs in the executor before it is accepted (`planning/feasibility.py`, `Rollout`)

A candidate's steps are run one by one through **the same executor that later drives the robot**
(`primitive_executor.run_plan` on `SimBackend`, at a coarse dt = 0.01 s). A step is feasible if:

- every segment (pre pose, approach, close, motion, open, retreat, back to ready) finishes;
- the contact QP stays feasible all along. Joint position limits are enforced through the damper, joint speed limits
  are bounds, and the contact twist is a hard equality;
- at every sample, no arm link, the hand or the palm comes within 2 mm of the box or the table (coal, collision meshes).

Every step starts and ends at the ready configuration, so steps are independent. A step's result is cached by its
grasp and screws, and reused by every candidate that contains it. The result also gives the smallest distance to a
joint limit, which breaks ties between candidates of equal cost.

The model the rollout uses is set by `plan_free_rotation` (normally the bearing, free rotation). For the free-vs-rigid
experiments the plan is made with free rotation, and the rigid grasp then executes the same plan. That is how the
experiments show where the rigid grasp stops.

### Grasp observability (`feasibility.grasp_observability`)

phi is neither driven nor measured: the box contact decides it. That is only well defined if the box cannot also turn
about n while it keeps its contact. Two consequences:

- **pivot:** the grasp point must be at least 20 mm from the pivot edge;
- **slide:** the finger axis must not be vertical.

The pre-flight checks this.

---

## 7. The other two planners

They have the same interface (`plan(box, T0, Tg) -> Plan`) and the same rollout check. `run_box.py --gait` and
`--chain` select them.

### Object gaiting (`planning/gait.py`, `GaitPlanner`)

**Task:** turn and shift the box on the **same face** without sliding it, so there is no friction work and no tipping.

Each gait step is five screws in one grasp, all through one vertex v of the support face:

1. tilt by gamma = 15 deg about the short edge that contains v;
2. tilt by beta = 30 deg about the box edge through v, so the box stands on v alone;
3. turn by alpha about the vertical through v;
4. and 5. undo beta and gamma.

The net effect is a planar rotation by alpha about v. After each step the box is regrasped at the other end, and the
next step uses a vertex there.

**Search:** for k = 1..5 steps and every sequence of vertices, the angles alpha_1..k come from a small optimisation
(SLSQP, analytic Jacobian):

```
minimize  sum alpha_j^2
subject to  final centre = goal centre,  sum alpha_j = goal yaw change (+-2pi),  |alpha_j| <= 45 deg
```

Candidates are ordered by fewest steps, then smallest |alpha|. The first one whose steps all pass the rollouts wins.

**Departure from the manuscript:** the manuscript's Case 1 alternates the two vertices of one edge. With a 6 cm wide box
those steps are tiny. Consecutive vertices at **opposite ends** give larger steps; this was your decision of 2026-11-17
(flag F3 in MANUSCRIPT_FLAGS.md).

**Lag:** a gait is 15-25 screw motions long, so the robot's tracking lag adds up. The protocol runs the gait with
`--lead`. Without it the lag simulation ends 6-13 mm off the goal, which the pre-flight refuses.

### Two pivots in one grasp: the roll by pi (`planning/chain.py`, `ChainPlanner`)

**Task:** move the box onto the **opposite** face (lying to lying_flip) in one grasp, instead of two grasps with two
pivots.

**Motion:** two same-sense pivots by 90 deg about edges parallel to n. The box rolls 0.40 m (W + H). The bearing takes
the part of the 180 deg that the palm clearance allows (at most about 108 deg). The hand itself turns by the rest.

**Grasp:** the study (`experiments/double_pivot_study.py`, p2_study.md) found one region with margins.

- **Where:** the back-bottom corner band, CHAIN_INSET = 10 mm from the back face and 27.5 mm above the table.
- **Hand orientation:** tilted 68 deg from pointing down, so it starts low behind the box and ends above it.
- **Lift:** 0.25 m for the pre pose and the retreat, since the hand starts beside the box.

The sequencer's usual top-corner pivot grasp has no end pose clear of the table, which is why a separate grasp is needed.

**Search:** for each horizontal non-finger axis, both roll directions, and both finger signs, the planner keeps the
roll that ends at the goal. It then runs keep-out and reach, then the rollout. The first that passes wins.

**Result:** in experiment `chain`, free rotation completes both pivots. The rigid grasp stops in the second pivot,
because its QP becomes infeasible there.

---

## 8. What happens to the plan afterwards

`primitive_executor.run_plan` runs each step as segments:

```
ready -> pre (above the grasp, lift) -> approach -> close -> motion -> open -> retreat (straight up) -> ready
```

Every Cartesian segment (pre, approach, motion, retreat) is the **contact QP** (`control/contact_qp.py`, eq:qp). At
1 kHz it computes joint velocities that make the grasp frame {c} move with the reference twist V_c:

```
min  1/2 ||theta_dot - K (theta_bar - theta)||^2 - lambda J_m^T theta_dot + 1/2 nu phi_dot^2
s.t. J_c^aug(theta, phi) [theta_dot; phi_dot] = V_c        hard equality (the box is hinged on the table)
     damper rows near joint limits, and on phi near phi_lim (palm clearance)
     |theta_dot| <= 0.8 x the libfranka speed limit
```

- **Virtual joint:** the bearing appears as an eighth, virtual joint phi about n (J_c^aug). The QP decides how much of
  the box's rotation the bearing takes and how much the hand takes.
- **Pivots:** feed-forward, V_c = s_dot(t) theta Ad(T_c0^-1) S, with a quintic time scaling. The edge fixes where a pivot
  ends.
- **Slides and the approach:** the same screw plus a small pose feedback toward the planned path, K_e log(T^-1 T_plan).
  Slides use the rigid model (no phi): the hand stays as planned.
- **Rigid grasp:** the same QP without phi. Where the arm cannot follow the screw, the equality has no solution and the
  run stops. That is the expected rigid stop of the experiments.

So, to your earlier question: **the QP is used in every segment, before, during and after the grasp. The cost (kappa
and the work) is used only to rank plans.** The robot's collision reflex is a third, separate thing: a force threshold
in libfranka, set once per run.

---

## 9. The decisions and why

Each row gives the decision, the reason, and when it would be wrong. The code review of 2026-11-24/25 had a critic
attack each one and a defender answer (DEBATE.md). The outcomes are in the last column.

| # | Decision | Why | When it would be wrong | Review |
|---|---|---|---|---|
| 1 | **Primitives are constant screw motions defined by the box-table contact** | the contact fixes the screw axis, so there is no path search for the box. The same equations serve planner, executor and paper, and the end pose of a pivot is exact | contacts that change during a motion (rolling a curved object); motions that are not one screw (handled as chains of screws in one grasp: tip-free slides, gait steps, the roll) | kept |
| 2 | **4x4 matrices instead of unit dual quaternions** | the same screw motions (1.1e-11), and not slower. numpy and the robotics toolbox speak matrices, and a rewrite would only convert at every interface | if the code needed interpolation of arbitrary poses (ScLERP between any two) or compact pose storage; it does not | D5: kept, executor moved onto `se3.py` |
| 3 | **A fixed template (slide, pivots A, slide, pivots B) instead of a general graph search** | for this box the face graph is a cycle of four, so the template enumerates every pivot sequence of up to 3 between any two faces. The middle slide is solved exactly (X = Tg D_B^-1). It finds plans in 0.1-0.4 s before rollouts and is easy to explain in the paper | objects with more grasped face pairs, or more primitive types in one plan (gaits mixed with pivots): then a graph search over (face, primitive) pays off | D1: kept; paper wording "cheapest feasible among the enumerated candidates" |
| 4 | **Pivots only about edges parallel to the finger axis** | the bearing turns about n, so it can only absorb the box's rotation if n is parallel to the pivot axis. With the 8 cm gripper and this box, the 0.06 m axis is the only one that fits anyway | a wider gripper, or pivots about other edges (the bearing would not help; the rigid grasp is the only model there) | kept; domain stated in the paper |
| 5 | **Cost = quasi-static work + kappa per grasp** | physical, on one scale for slides and pivots, and kappa stands for the time a grasp costs on the robot | when time, not effort, is what matters (then count grasps first); kappa = 0.3 is a choice | D2: kappa per grasp (was per step type) |
| 6 | **A plan is accepted only if the real executor completes it in simulation** | IK alone does not predict what the QP does with the redundancy and the bearing, the damper, or the speed bounds. Running the same code removes the gap between check and execution. The rigid stops seen in simulation are those the robot will see | when the error is a force (compliance under 5 kg, friction, slip), which the kinematic rollout cannot see (P3 adds a load model); when the plan space grows and rollouts become the time limit | D4 made it hold exactly: the QP rates now scale with the speed factor, so rollouts (speed 1) and robot runs (0.15-0.3) follow the same joint path (difference 0.21 rad down to 1.2e-4 rad) |
| 7 | **Every step starts and ends at the ready configuration** | steps are independent, so they are checked alone and cached, and each step can be replayed and judged on its own. It also matches how the experiments are run | it costs time on the robot (the return to ready in every step). A planner that chains steps arm-state to arm-state would be faster but would lose the cache and the simple checks | kept |
| 8 | **Grasps from fixed rules (top corner for pivots, centre for slides, bottom-back band for the roll)** | the rules put the grasp where the lever and the free arc of phi are largest (D11: free arcs exist only within about 30 mm of an edge) | for other objects the rules would need the same study again; a grasp search would replace them | D11: rules kept, roll grasp from the P2 study |
| 9 | **Hard-equality QP with the bearing as a virtual joint** | the box is hinged on the table: the hand must stay on the arc. A slack variable was tried and drifted 351 mm off the plan | needs a defined recovery when the expected rigid stop happens (you: hold the box by hand) | D6: kept |
| 10 | **Feed-forward pivots, object pose from kinematics** | simple, the same in simulation and on the robot, and the edge fixes the end pose | under load (5-18 mm of hand deflection at 5 kg), and for long chains, where the gait needs `--lead` | kept; pivot feedback (P1) is for the next paper (your decision) |

---

## 10. Limits

- **Domain:** faces whose 0.06 m axis is horizontal. The largest face cannot be grasped, so a task that starts or ends on
  it has no plan.
- **Optimality:** the cheapest feasible plan among the enumerated candidates (36 staging poses plus 3 in-place turns for
  the start slide; the 12 cheapest candidates checked), not a global optimum.
- **Forces:** feasibility is kinematic: joint limits and speeds, collisions, the QP. Forces enter only through the
  slide tipping test and the cost. At 5 kg the load model of `exp/heavy-box` (P3, not merged) adds force, torque,
  friction and deflection checks; 6 of the 17 protocol entries pass at 5 kg until mu, mu_pad and the deflection are
  measured.
- **Mass:** `box_real.yaml` still uses mass 1.0 kg as a placeholder, so plans and costs are for 1 kg.
- **Gait angles:** solved by SLSQP, 4.7 s per call. A closed form exists (0.7 ms; review D10) and is left for after
  the paper.
- **Planning time:** about 1-70 s per task, almost all of it rollouts. The review measured about 9 s of 68 s for one
  task to be parallelisable and did not parallelise it.

---

## 11. Code map and how to look inside

| file | lines | role |
|---|---|---|
| `src/pandaSim/planning/se3.py` | 93 | exp, log, adjoint of SE(3) |
| `src/pandaSim/planning/box_primitives.py` | 231 | pivot and slide plans, grasp rules, tipping margin, keep-out |
| `src/pandaSim/planning/sequencer.py` | 399 | the template search, cost, ranking |
| `src/pandaSim/planning/feasibility.py` | 113 | `Rollout` (the executor as feasibility check), grasp observability |
| `src/pandaSim/planning/clearance.py` | 150 | phi limits from the hand and box geometry |
| `src/pandaSim/planning/collision.py` | 101 | arm and hand against box and table (coal) |
| `src/pandaSim/planning/gait.py` | 223 | object gaiting |
| `src/pandaSim/planning/chain.py` | 100 | the roll by pi in one grasp |
| `src/pandaSim/control/contact_qp.py` | 119 | the QP of eq:qp |
| `src/pandaSim/control/primitive_executor.py` | 419 | `run_plan`: segments, references, safety stops |
| `experiments/run_plan.py` | | `make_planner`: which planner, with which rollout model |
| `experiments/run_box.py` | | plan + simulate + pre-flight for one task (`--gait`, `--chain`) |

Ways to look inside:

```bash
unset PYTHONPATH
cd ~/pandaSim/pandaSim
# plan, simulate and pre-flight one task (no robot)
.venv/bin/python experiments/run_box.py --start 0.50,0.00,0,lying --goal 0.50,0.00,0,lying_flip
```

In Python, the candidates before any rollout:

```python
import sys; sys.path.insert(0, "experiments")
import run_box, run_plan
cfg = run_plan.load_config("experiments/configs/box_real.yaml")
robot, box = run_plan.make_robot(cfg), run_plan.make_box(cfg)
planner = run_plan.make_planner(cfg, robot)
T0 = run_plan.box_pose(box, run_box.parse_pose("0.50,0.00,0,lying"))
Tg = run_plan.box_pose(box, run_box.parse_pose("0.50,0.00,0,lying_flip"))
for cost, steps in planner.candidates(box, T0, Tg, 12):
    print(round(cost, 2), [s.kind for s in steps])
plan = planner.plan(box, T0, Tg)          # with rollouts
print([s.kind for s in plan.steps], plan.stats)
```

The notebook `experiments/plan_box.ipynb` shows all of this in order: Section 2 the primitives, Section 3 the QP,
Section 4 the sequencer with its candidates, Sections 5-6b the gait, the free-vs-rigid comparison and the roll, all
with Swift replays.
