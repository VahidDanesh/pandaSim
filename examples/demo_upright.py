"""
Demo for the upright task.

This script demonstrates the use of the pandaSim package for
accomplishing an upright task with a 3D object.
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pandaSim import UprightTask, SimulationConfig
from src.pandaSim.core.config import GeometryBackend, BoundingBoxType, PlannerType


def visualize_trajectory(obj, trajectory, adapter):
    """
    Visualize the object and its trajectory.
    
    Args:
        obj: Object representation
        trajectory: List of trajectory waypoints
        adapter: Geometry adapter
    """
    fig = plt.figure(figsize=(12, 6))
    
    # Get vertices for visualization
    vertices = adapter.get_vertices(obj)
    
    # Create subplots for each waypoint
    n_waypoints = min(5, len(trajectory))  # Limit to 5 for clarity
    
    for i in range(n_waypoints):
        # Get transformation matrix for this waypoint
        T = trajectory[i]
        
        # Transform vertices
        transformed_obj = adapter.transform(obj, T)
        transformed_vertices = adapter.get_vertices(transformed_obj)
        
        # Create subplot
        ax = fig.add_subplot(1, n_waypoints, i + 1, projection='3d')
        
        # Plot vertices
        ax.scatter(
            transformed_vertices[:, 0],
            transformed_vertices[:, 1],
            transformed_vertices[:, 2],
            c='b', marker='o', alpha=0.2
        )
        
        # Add coordinate axes
        origin = np.mean(transformed_vertices, axis=0)
        axes_length = np.max(np.ptp(transformed_vertices, axis=0)) / 2
        
        # X-axis (red)
        ax.quiver(
            origin[0], origin[1], origin[2],
            axes_length, 0, 0,
            color='r', arrow_length_ratio=0.1
        )
        
        # Y-axis (green)
        ax.quiver(
            origin[0], origin[1], origin[2],
            0, axes_length, 0,
            color='g', arrow_length_ratio=0.1
        )
        
        # Z-axis (blue)
        ax.quiver(
            origin[0], origin[1], origin[2],
            0, 0, axes_length,
            color='b', arrow_length_ratio=0.1
        )
        
        # Set equal aspect ratio
        ax.set_box_aspect([1, 1, 1])
        
        # Set labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Waypoint {i+1}')
    
    plt.tight_layout()
    plt.show()


def main():
    """Main function to demonstrate the upright task."""
    # Create configuration
    config = SimulationConfig(
        geometry_backend=GeometryBackend.GENESIS_AI,
        bbox_type=BoundingBoxType.PCA_OBB,
        planner_type=PlannerType.PCA,
        planner_params={"num_waypoints": 5}
    )
    
    # Create upright task
    task = UprightTask(config)
    
    # Path to the 3D object (placeholder)
    # In a real scenario, you would provide a path to an actual 3D model file
    object_path = "examples/data/object.obj"
    
    # Check if file exists, if not use a synthetic example
    if not os.path.exists(object_path):
        print(f"Object file {object_path} not found, using synthetic example.")
        
        # Get the adapter from the task
        adapter = task.geometry_adapter
        
        # Create a synthetic object (a simple cube in this case)
        # This is a placeholder for actual object loading
        obj = {"type": "synthetic"}
        
        # Execute the task with our synthetic object
        trajectory = task.plan_upright_orientation(obj, None, [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0])
        ])
    else:
        # Execute the task with the actual object file
        trajectory = task.execute(object_path)
    
    # Get the first (and probably only) trajectory
    waypoints = trajectory[0].waypoints
    
    # Visualize the trajectory
    visualize_trajectory(obj, waypoints, adapter)
    
    print("Trajectory planning completed successfully!")
    print(f"Generated {len(waypoints)} waypoints.")


if __name__ == "__main__":
    main()