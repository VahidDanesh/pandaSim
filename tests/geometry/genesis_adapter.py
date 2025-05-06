import numpy as np
import sys
import os
import genesis as gs
from pathlib import Path

from pandaSim.geometry.genesis_adapter import GenesisAdapter
from pandaSim.core.config import SimulationConfig, GeometryBackend

def test_genesis_adapter():
    """Test the Genesis AI adapter functionality."""
    
    # Create adapter with custom configuration
    config = {
        "show_viewer": True,
        "viewer_options": {
            "width": 1280,
            "height": 720,
            "camera_pos": (4.0, 0.0, 2.0),
            "camera_lookat": (0.0, 0.0, 0.5),
            "fov": 45
        }
    }
    gs.init(backend=gs.gpu, seed=42)
    adapter = GenesisAdapter(config)
    
    # 1. Test loading a robot model from MJCF XML
    # Adjust the path to a valid MJCF file on your system
    robot_path = 'xml/franka_emika_panda/panda.xml'
    try:
        robot = adapter.load(robot_path)
        print(f"Successfully loaded robot: {robot['id']}")
    except FileNotFoundError:
        print(f"Robot file not found: {robot_path} - skipping this test")
    
    # 2. Test adding primitive shapes
    # Add a ground plane
    ground = adapter.add_primitive("plane", size=(20.0, 20.0))
    print(f"Added ground plane: {ground['id']}")
    
    # Add a box
    box = adapter.add_primitive("box", size=(0.2, 0.2, 0.2), pos=(1.0, 0.0, 0.1))
    print(f"Added box: {box['id']}")
    
    # Add a cylinder
    cylinder = adapter.add_primitive("cylinder", radius=0.1, height=0.3, pos=(0.0, 1.0, 0.15))
    print(f"Added cylinder: {cylinder['id']}")
    
    # Add a sphere
    sphere = adapter.add_primitive("sphere", radius=0.15, pos=(-1.0, 0.0, 0.15))
    print(f"Added sphere: {sphere['id']}")
    
    # # 3. Test getting vertices and faces
    # try:
    #     vertices = adapter.get_vertices(box)
    #     print(f"Box vertices shape: {vertices.shape}")
        
    #     faces = adapter.get_faces(box)
    #     if faces is not None:
    #         print(f"Box faces shape: {faces.shape}")
    #     else:
    #         print("Box faces not available")
    # except Exception as e:
    #     print(f"Error getting geometry data: {e}")
    
    # # 4. Test transformation
    # # Create a translation matrix (move the box up by 0.5)
    # translation = np.eye(4)
    # translation[2, 3] = 0.5  # Move up in z direction
    
    # try:
    #     transformed_box = adapter.transform(box, translation)
    #     print("Applied transformation to box")
    # except Exception as e:
    #     print(f"Error applying transformation: {e}")
    
    # # 5. Test computing center of mass and volume
    # try:
    #     com = adapter.compute_center_of_mass(sphere)
    #     print(f"Sphere center of mass: {com}")
        
    #     volume = adapter.compute_volume(sphere)
    #     print(f"Sphere volume: {volume}")
    # except Exception as e:
    #     print(f"Error computing properties: {e}")
    
    # # Wait for user input to keep the window open (if viewer is shown)
    # if config.get("show_viewer"):
    #     input("Press Enter to exit...")

if __name__ == "__main__":
    test_genesis_adapter()