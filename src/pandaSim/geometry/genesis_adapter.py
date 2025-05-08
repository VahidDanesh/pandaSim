"""
Genesis AI adapter implementation.

This module provides an adapter for the Genesis AI geometry backend.
"""
from typing import Any, Optional, Dict, Tuple, Union
import numpy as np
import genesis as gs
from pathlib import Path
from pytransform3d import (
    transformations as pt,
    rotations as pr,
    batch_rotations as pb,
    trajectories as ptr,
    plot_utils as ppu
)

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
    
    @property
    def get_scene(self):
        """Get the underlying Genesis scene for direct access"""
        return self.scene
    
    def load(self, file: str) -> Any:
        """
        Load data from file using Genesis AI.
        
        Args:
            file_path: Path to the geometry file, robot model or object file
            
        Returns:
            Genesis AI representation of the geometry
        """

        
        # Determine the type of file and load accordingly
        file_path = Path(file)

        if file_path.suffix.lower() == '.xml':
            # Load MJCF file
            entity = self.scene.add_entity(
                gs.morphs.MJCF(file=str(file_path))
            )
        elif file_path.suffix.lower() == '.obj':
            # Load OBJ mesh file
            entity = self.scene.add_entity(
                gs.morphs.Mesh(file=str(file_path))
            )
        elif file_path.suffix.lower() == '.stl':
            # Load STL mesh file
            entity = self.scene.add_entity(
                gs.morphs.Mesh(file=str(file_path))
            )
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
            
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
            **params: Shape-specific parameters including surface properties
                - pos, quat, euler: For shape position and orientation
                - size, lower, upper: For box shape
                - radius, height: For cylinder shape
                - radius: For sphere shape
                - surface: For material/color properties (e.g., gs.surfaces.Default(color=(r,g,b)))
        
        Returns:
            Dictionary containing the entity information
        """
        primitive_type = primitive_type.lower()
        
        # Extract surface parameter separately, don't pass it to the morph constructor
        surface = params.pop('surface', None)
        
        if primitive_type == "box":
            entity = self.scene.add_entity(
                gs.morphs.Box(**params),
                surface=surface
            )
            
        elif primitive_type == "cylinder":
            entity = self.scene.add_entity(
                gs.morphs.Cylinder(**params),
                surface=surface
            )
            
        elif primitive_type == "sphere":
            entity = self.scene.add_entity(
                gs.morphs.Sphere(**params),
                surface=surface
            )
            
        elif primitive_type == "plane":
            entity = self.scene.add_entity(
                gs.morphs.Plane(**params),
                surface=surface
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


    
    def get_bbox(self, obj: Union[Dict, Any]) -> Dict:
        """
        Calculate axis-aligned bounding box for an object.
        
        Args:
            obj: Object representation or direct entity
        
        Returns:
            Dictionary containing bounding box information:
            - vertices: 8 corner points of the bounding box
            - edges: 12 edges connecting the vertices
            - min_bounds: Minimum coordinate values (x,y,z)
            - max_bounds: Maximum coordinate values (x,y,z)
        """
        # Handle both dict wrapper and direct entity
        entity = obj["entity"] if isinstance(obj, dict) else obj
        
        # Get mesh data from entity
        vertices = entity.get_verts().cpu().numpy()

        
        # Calculate min and max bounds
        min_bounds = np.min(vertices, axis=0)
        max_bounds = np.max(vertices, axis=0)
        
        # Generate the 8 corners of the bounding box
        corners = []
        for x in [min_bounds[0], max_bounds[0]]:
            for y in [min_bounds[1], max_bounds[1]]:
                for z in [min_bounds[2], max_bounds[2]]:
                    corners.append([x, y, z])
        
        corners = np.array(corners)
        # Define the 12 edges as pairs of vertex indices
        edge_indices = [
            (0, 2), (4, 6), (0, 4), (2, 6),  # Bottom face
            (1, 3), (5, 7), (3, 7), (1, 5),  # Top face
            (0, 1), (2, 3), (4, 5), (6, 7)   # Connecting edges
        ]
        edges = [corners[j] - corners[i] for i, j in edge_indices]
        edges = [edge / np.linalg.norm(edge) for edge in edges]
        edges = np.array(edges)
        

        return {
            "vertices": corners,
            "edges": edges,
            "min_bounds": min_bounds,
            "max_bounds": max_bounds
        }
    
    def transform(self, obj: Union[Dict, Any], transformation: np.ndarray) -> Dict:
        """
        Apply transformation to Genesis AI geometry.
        
        Args:
            obj: Genesis AI geometry representation
            transformation: 4x4 transformation matrix
            
        Returns:
            Transformed Genesis AI geometry
        """
        entity = obj["entity"] if isinstance(obj, dict) else obj
        
        
        pq = pt.pq_from_transform(transformation)
        # Apply transformation to entity
        entity.set_qpos(
            pq
        )

