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
    
    # =============================================
    # Approach 1: Using the adapter wrapper methods
    # =============================================
    print("\n=== Using adapter methods ===")
    
    # 1. Test loading a robot model from MJCF XML
    robot_path = 'xml/franka_emika_panda/panda.xml'
    try:
        robot = adapter.load(robot_path)
        print(f"Successfully loaded robot: {robot['id']}")
    except FileNotFoundError:
        print(f"Robot file not found: {robot_path} - skipping this test")
    
    # Add primitives with colors using the adapter
    ground = adapter.add_primitive("plane", surface=gs.surfaces.Default(color=(0.3, 0.3, 0.3)))
    print(f"Added ground plane: {ground['id']}")
    
    box = adapter.add_primitive(
        "box", 
        size=(0.2, 0.2, 0.2), 
        pos=(1.0, 0.0, 0.1),
        surface=gs.surfaces.Default(color=(0.8, 0.1, 0.1))
    )
    print(f"Added red box: {box['id']}")
    
    
    # =============================================
    # Approach 2: Using direct scene access
    # =============================================
    print("\n=== Using direct scene access ===")
    
    # Get direct scene access
    scene = adapter.get_scene
    
    # Add objects directly with full Genesis API visibility
    cube = scene.add_entity(
        gs.morphs.Box(
            size=(0.1, 0.07, 0.03),
            pos=(0.5, 0.5, 0.05),
        ),
        surface=gs.surfaces.Default(
            color=(0.1, 0.8, 0.1),  # Green
        ),
    )
    print(f"Added green cube directly")
    
    # Add a sphere with metallic surface
    sphere = scene.add_entity(
        gs.morphs.Sphere(
            radius=0.05,
            pos=(-0.5, 0.5, 0.05),
        ),
        surface=gs.surfaces.Default(
            color=(0.1, 0.1, 0.8),  # Blue
            metallic=0.8,
            roughness=0.2,
        ),
    )
    print(f"Added metallic blue sphere directly")


    # =============================================
    # Build scene and get bounding box
    # =============================================

    adapter.scene.build()


    bbox = adapter.get_bbox(box)
    print(f"Box bounding box min: {bbox['min_bounds']}, max: {bbox['max_bounds']}")
    print(f"Box edges: {bbox['edges']}")

    
    # You can still use adapter methods with direct entities
    sphere_bbox = adapter.get_bbox(sphere)
    print(f"Sphere bounding box min: {sphere_bbox['min_bounds']}, max: {sphere_bbox['max_bounds']}")
    

    # Keep open until key is pressed
    input("Press Enter to exit...")

if __name__ == "__main__":
    test_genesis_adapter()