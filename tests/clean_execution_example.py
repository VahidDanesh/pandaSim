"""
Example of clean robot execution using the new RobotExecutor.

This shows how to replace the old control loop with the new clean API.
"""
import numpy as np
import os
import getpass
import logging

# pandaSim imports
from pandaSim.geometry.rtb_adapter import RoboticsToolboxAdapter
from pandaSim.geometry.utils import create_virtual_panda
from pandaSim.planning.screw_motion_planner import ScrewMotionPlanner
from pandaSim.control import QPController, RobotExecutor, ExecutionConfig

# Optional: real robot imports
try:
    import panda_py
    from panda_py import controllers
    REAL_ROBOT_AVAILABLE = True
except ImportError:
    REAL_ROBOT_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO)

def setup_simulation():
    """Setup simulation environment."""
    adapter = RoboticsToolboxAdapter({
        "realtime": True,
        "rate": 100,
        "browser": "notebook"
    })
    
    urdf_path = "/home/vahid/repos/pandaSim/model/franka_description/robots/frankaEmikaPandaVirtual.urdf"
    panda_sim = create_virtual_panda(urdf_path=urdf_path)
    adapter.env.add(panda_sim)
    
    if hasattr(panda_sim, "grippers") and panda_sim.grippers:
        panda_sim.grippers[0].q = [0.035, 0.035]
    
    return adapter, panda_sim

def setup_real_robot():
    """Setup real robot connection."""
    if not REAL_ROBOT_AVAILABLE:
        return None
        
    hostname = os.getenv("PANDA_HOST") or input("Panda hostname/IP: ").strip()
    username = os.getenv("PANDA_USER") or input("Desk Username: ").strip() 
    password = os.getenv("PANDA_PASS") or getpass.getpass("Desk Password: ")
    
    desk = panda_py.Desk(hostname, username, password, platform='panda')
    desk.lock()
    desk.unlock()
    
    panda_real = panda_py.Panda(hostname)
    panda_real.move_to_start()
    
    return panda_real

def main():
    """Clean execution example."""
    # Setup
    adapter, panda_sim = setup_simulation()
    
    # Create controller
    controller = QPController(
        adapter=adapter,
        robot=panda_sim,
        end_effector_link="panda_finger_virtual",
        gains_translation=1.0,
        gains_rotation=0.5,
        threshold=0.001,
        lambda_q=0.05,
        lambda_m=1.0,
        lambda_j=0.1,
        ps=0.5,
        pi=0.3,
        eta=1.0,
        solver="quadprog",
        T=2.0,
    )
    
    # Execution configuration (only controller-related params)
    config = ExecutionConfig(
        frequency=100.0,
        max_runtime=15.0,
        convergence_threshold=0.001,
        safety_timeout=30.0
    )
    
    # Create executor
    executor = RobotExecutor(controller, adapter, config)
    
    # Compute target pose (example)
    planner = ScrewMotionPlanner(adapter=adapter)
    # Add a simple box for demo
    import spatialmath as sm
    import spatialgeometry as sg
    box = sg.Box(scale=[0.2, 0.07, 0.1], color='blue', pose=sm.SE3(0.7, 0.0, 0.05))
    adapter.env.add(box)
    
    grasp_pose, qs, s_axes = planner.compute_grasp(
        obj=box,
        adapter=adapter,
        grasp_height="top",
        prefer_closer_grasp=True,
        gripper_offset=sm.SE3.Rx(np.pi/2).A,
        output_type="t"
    )
    
    print("=== CASE 1: Target Approach (Simulation Only) ===")
    # Case 1: Move to grasp pose (simulation only)
    result_sim = executor.approach_target(
        target_pose=grasp_pose,
        sim_robot=panda_sim,
        optimization_type="j",
        use_real_robot=False
    )
    
    print(f"Simulation result: converged={result_sim.converged}, steps={result_sim.n_steps}")
    print(f"Execution time: {result_sim.execution_time:.2f}s")
    
    if REAL_ROBOT_AVAILABLE:
        print("\n=== CASE 1: Target Approach (Real Robot) ===")
        # Setup real robot
        panda_real = setup_real_robot()
        
        if panda_real:
            # Case 1: Move to grasp pose (real robot)
            result_real = executor.approach_target(
                target_pose=grasp_pose,
                sim_robot=panda_sim,
                real_robot=panda_real,
                optimization_type="j",
                use_real_robot=True
            )
            
            print(f"Real robot result: converged={result_real.converged}, steps={result_real.n_steps}")
            print(f"Execution time: {result_real.execution_time:.2f}s")
    
    print("\n=== CASE 2: Velocity Following ===")
    # Case 2: Follow velocity commands (e.g., for manipulation)
    # Generate some twist trajectory
    twist_trajectory = planner.generate_twist_trajectory(
        body_pose=grasp_pose,
        q=qs,
        s_axis=s_axes,
        theta=np.pi/4,
        h=0.0,
        theta_dot=0.1,
        Tf=5.0,
        time_scaling='quintic',
        body_coordinate=True
    )
    
    # Convert twists to joint velocities
    velocity_commands = []
    for twist in twist_trajectory:
        qd, _ = controller.compute_joint_velocities(twist=twist, optimization_type="j")
        velocity_commands.append(qd[:panda_sim.n])
    
    # Execute velocity following (simulation)
    result_vel = executor.follow_velocities(
        velocity_commands=velocity_commands,
        sim_robot=panda_sim,
        use_real_robot=False
    )
    
    print(f"Velocity following result: steps={result_vel.n_steps}")
    print(f"Execution time: {result_vel.execution_time:.2f}s")
    
    # Save results
    result_sim.save("approach_simulation.npz")
    result_vel.save("velocity_following.npz")
    print("\nResults saved to .npz files")

if __name__ == "__main__":
    main()
