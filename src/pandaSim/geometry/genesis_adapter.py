"""
Genesis AI adapter implementation.

This module provides an adapter for the Genesis AI geometry backend.
"""
from typing import Any, Optional, Dict, Tuple, Union
import numpy as np
import genesis as gs
from pathlib import Path

class GenesisAdapter:
    """
    Adapter for Genesis AI geometry backend.
    
    Implements the GeometryAdapter protocol for Genesis AI.
    """
    
    def __init__(self, config: Optional[dict] = None):
        """
        Initialize Genesis AI adapter.
        
        Args:
            config: Optional configuration parameters
        """
        self.config = config or {}
        
        # Configure viewer options
        viewer_options = self.config.get("viewer_options", {})
        self.viewer_options = gs.options.ViewerOptions(
            res=(viewer_options.get("width", 1280), viewer_options.get("height", 960)),
            camera_pos=viewer_options.get("camera_pos", (3.5, 0.0, 2.5)),
            camera_lookat=viewer_options.get("camera_lookat", (0.0, 0.0, 0.5)),
            camera_fov=viewer_options.get("fov", 40),
            max_FPS=viewer_options.get("max_fps", 60),
        )
        
        # Configure simulation options
        sim_options = self.config.get("sim_options", {})
        self.sim_options = gs.options.SimOptions(
            dt=sim_options.get("dt", 0.01),
        )
        
        # Configure visualization options
        vis_options = self.config.get("vis_options", {})
        self.vis_options = gs.options.VisOptions(
            show_world_frame=vis_options.get("show_world_frame", True),
            world_frame_size=vis_options.get("world_frame_size", 1.0),
            show_link_frame=vis_options.get("show_link_frame", False),
            show_cameras=vis_options.get("show_cameras", False),
            plane_reflection=vis_options.get("plane_reflection", True),
            ambient_light=vis_options.get("ambient_light", (0.1, 0.1, 0.1)),
        )
        
        # Create scene
        self.scene = gs.Scene(
            show_viewer=self.config.get("show_viewer", True),
            viewer_options=self.viewer_options,
            sim_options=self.sim_options,
            vis_options=self.vis_options,
            renderer=gs.renderers.Rasterizer(),
        )
        
        # Store loaded entities
        self.entities = {}
    
    def load(self, file: str) -> Any:
        """
        Load data from file using Genesis AI.
        
        Args:
            file_path: Path to the geometry file, robot model or object file
            
        Returns:
            Genesis AI representation of the geometry
        """

        
        # Determine the type of file and load accordingly
        if file.suffix.lower() == '.xml':
            # Load MJCF file
            entity = self.scene.add_entity(
                gs.morphs.MJCF(file=str(file))
            )
        elif file.suffix.lower() == '.obj':
            # Load OBJ mesh file
            entity = self.scene.add_entity(
                gs.morphs.Mesh(file=str(file))
            )
        elif file.suffix.lower() == '.stl':
            # Load STL mesh file
            entity = self.scene.add_entity(
                gs.morphs.Mesh(file=str(file))
            )
        else:
            raise ValueError(f"Unsupported file format: {file.suffix}")
            
        # Store entity with an ID
        entity_id = str(len(self.entities))
        self.entities[entity_id] = entity
        
        return {
            "id": entity_id,
            "path": str(file),
            "entity": entity
        }
    
    def add_primitive(self, primitive_type: str, **params) -> Dict:
        """
        Add a primitive shape to the scene.
        
        Args:
            primitive_type: Type of primitive shape ("box", "cylinder", "sphere", "plane")
            **params: Shape-specific parameters (will use Genesis defaults if not provided)
        
        Returns:
            Dictionary containing the entity information
        """
        primitive_type = primitive_type.lower()
        
        if primitive_type == "box":
            entity = self.scene.add_entity(
                gs.morphs.Box(**params)
            )
            
        elif primitive_type == "cylinder":
            entity = self.scene.add_entity(
                gs.morphs.Cylinder(**params)
            )
            
        elif primitive_type == "sphere":
            entity = self.scene.add_entity(
                gs.morphs.Sphere(**params)
            )
            
        elif primitive_type == "plane":
            entity = self.scene.add_entity(
                gs.morphs.Plane(**params)
            )
            
        else:
            raise ValueError(f"Unsupported primitive shape type: {primitive_type}")
        
        # Store entity with an ID
        entity_id = str(len(self.entities))
        self.entities[entity_id] = entity
        
        return {
            "id": entity_id,
            "primitive_type": primitive_type,
            "params": params,
            "entity": entity
        }

    def get_vertices(self, obj: Dict) -> np.ndarray:
        """
        Extract vertices from Genesis AI geometry.
        
        Args:
            obj: Genesis AI geometry representation
            
        Returns:
            Numpy array of vertices (Nx3)
        """
        entity = obj["entity"]
        # Get mesh data from entity
        # Note: Exact API method may need adjustment based on Genesis documentation
        mesh_data = entity.get_mesh_data()
        
        # Convert to numpy array if needed
        if not isinstance(mesh_data.vertices, np.ndarray):
            return np.array(mesh_data.vertices)
        return mesh_data.vertices
    
    def get_faces(self, obj: Dict) -> Optional[np.ndarray]:
        """
        Extract faces from Genesis AI geometry.
        
        Args:
            obj: Genesis AI geometry representation
            
        Returns:
            Numpy array of face indices
        """
        entity = obj["entity"]
        # Get mesh data from entity
        mesh_data = entity.get_mesh_data()
        
        if hasattr(mesh_data, 'faces') and mesh_data.faces:
            # Convert to numpy array if needed
            if not isinstance(mesh_data.faces, np.ndarray):
                return np.array(mesh_data.faces)
            return mesh_data.faces
        return None
    
    def transform(self, obj: Dict, transformation: np.ndarray) -> Dict:
        """
        Apply transformation to Genesis AI geometry.
        
        Args:
            obj: Genesis AI geometry representation
            transformation: 4x4 transformation matrix
            
        Returns:
            Transformed Genesis AI geometry
        """
        entity = obj["entity"]
        
        # Extract rotation and translation from transformation matrix
        rotation = transformation[:3, :3]
        translation = transformation[:3, 3]
        
        # Apply transformation to entity
        entity.set_pose(
            pos=translation,
            rot=rotation
        )
        
        return obj
    
    def compute_center_of_mass(self, obj: Dict) -> np.ndarray:
        """
        Compute center of mass with Genesis AI.
        
        Args:
            obj: Genesis AI geometry representation
            
        Returns:
            3D coordinates of the center of mass
        """
        entity = obj["entity"]
        # Get center of mass from entity
        com = entity.get_center_of_mass()
        
        return np.array(com)
    
    def compute_volume(self, obj: Dict) -> float:
        """
        Compute volume with Genesis AI.
        
        Args:
            obj: Genesis AI geometry representation
            
        Returns:
            Volume of the geometry
        """
        entity = obj["entity"]
        # Get volume from entity
        volume = entity.get_volume()
        
        return float(volume)