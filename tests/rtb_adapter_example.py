"""
Example demonstrating the Robotics Toolbox adapter usage.

This example shows how to use the RTB adapter to:
1. Initialize the environment
2. Load/create robots and objects
3. Perform basic operations like forward kinematics, pose manipulation, etc.
"""

import numpy as np
import spatialmath as sm
import spatialgeometry as sg
from pandaSim.geometry.rtb_adapter import RoboticsToolboxAdapter, create_virtual_panda


def main():
    """Main example function demonstrating RTB adapter usage."""
    
    # Initialize the adapter
    print("Initializing RTB adapter...")
    adapter = RoboticsToolboxAdapter({
        "realtime": True,
        "rate": 100
    })
    
    # Create a virtual Panda robot (similar to the provided sample)
    print("Creating virtual Panda robot...")
    panda_virtual = create_virtual_panda()
    
    # Add robot to the environment through adapter
    robot_dict = {
        "id": "panda",
        "entity": panda_virtual
    }
    adapter.entities["panda"] = panda_virtual
    adapter.env.add(panda_virtual)
    
    # Open the gripper fingers (from sample code)
    if hasattr(panda_virtual, 'grippers') and len(panda_virtual.grippers) > 0:
        panda_virtual.grippers[0].q = [0.035, 0.035]
    
    # Create a box object to grasp (from sample code)
    print("Creating box object...")
    box = adapter.add_primitive(
        "box",
        scale=[0.1, 0.07, 0.03],
        color='blue',
        T=sm.SE3(0.7, 0, 0.015)
    )
    
    # Set box transparency
    box_entity = box["entity"]
    if hasattr(box_entity, 'set_alpha'):
        box_entity.set_alpha(0.5)
    
    # Demonstrate some basic operations
    print("\nDemonstrating adapter operations:")
    
    # 1. Get joint positions
    print("Current joint positions:", adapter.get_joint_positions(robot_dict))
    
    # 2. Get robot pose
    robot_pose = adapter.get_pose(robot_dict)
    print("Robot base pose (pq format):", robot_pose)
    
    # 3. Get box pose
    box_pose = adapter.get_pose(box)
    print("Box pose (pq format):", box_pose)
    
    # 4. Get box size
    box_size = adapter.get_size(box)
    print("Box size:", box_size)
    
    # 5. Get bounding box
    bbox = adapter.get_bbox(box)
    print("Box AABB min bounds:", bbox['min_bounds'])
    print("Box AABB max bounds:", bbox['max_bounds'])
    
    # 6. Forward kinematics
    try:
        ee_pose = adapter.forward_kinematics(robot_dict, output_type='pq')
        print("End-effector pose (pq format):", ee_pose)
    except Exception as e:
        print("Forward kinematics error:", e)
    
    # 7. Joint limits
    try:
        lower, upper = adapter.get_joint_limits(robot_dict)
        print("Joint limits - Lower:", lower)
        print("Joint limits - Upper:", upper)
    except Exception as e:
        print("Joint limits error:", e)
    
    # 8. Transform the box
    print("\nTransforming box...")
    transform = sm.SE3.Trans(0.1, 0.1, 0.05)  # Move box by [0.1, 0.1, 0.05]
    new_pose = adapter.transform(box, transform, apply=True, output_type='pq')
    print("New box pose after transformation:", new_pose)
    
    # 9. Create additional objects for demonstration
    print("\nCreating additional objects...")
    
    # Create a sphere
    sphere = adapter.add_primitive(
        "sphere",
        radius=0.05,
        color='red',
        pos=[0.5, 0.5, 0.5]
    )
    
    # Create a cylinder
    cylinder = adapter.add_primitive(
        "cylinder",
        radius=0.03,
        length=0.2,
        color='green',
        pos=[0.3, 0.3, 0.1]
    )
    
    print(f"Created {len(adapter.entities)} entities total")
    
    # Step the simulation to update visualizations
    print("\nStepping simulation...")
    for i in range(10):
        adapter.step_simulation(0.1)
    
    print("\nExample completed! Check the Swift visualizer window.")
    
    # Keep the environment running
    try:
        input("Press Enter to close the environment...")
    except KeyboardInterrupt:
        pass
    
    print("Closing environment...")


if __name__ == "__main__":
    main() 