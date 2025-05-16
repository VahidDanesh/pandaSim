"""
Geometry utility functions for conversions between different representations.
"""

import numpy as np
import torch
from pytransform3d import (
    transformations as pt,
    rotations as pr,
    batch_rotations as pb,
    trajectories as ptr,
    plot_utils as ppu
)

def convert_pose(transformation: any, output_type: str = 'pq') -> np.ndarray:
    """
    Convert input to the given output_type.
    
    Args:
        transformation: Can be one of the following:
            - tuple of (position(s), quaternion(s))
            - pq(s) format (x, y, z, qw, qx, qy, qz)
            - dual quaternion(s) (qw, qx, qy, qz, tx, ty, tz)
            - transformation matrix(ices) 
        output_type: Desired output format ('pq', 'transform', 'dual_quaternion', etc.)
        
    Returns:
        np.ndarray: Converted representation in the requested format
    """

    # Convert from tuple (pos, quat)
    if isinstance(transformation, tuple):
        pos, quat = transformation
        if isinstance(pos, torch.Tensor):
            pos = pos.cpu().numpy()
        if isinstance(quat, torch.Tensor):
            quat = quat.cpu().numpy()

        pq = np.hstack([pos, quat])
    # Convert from tensor/array formats
    elif isinstance(transformation, (torch.Tensor, np.ndarray)):
        if isinstance(transformation, torch.Tensor):
            input_np = transformation.cpu().numpy()
        else:
            input_np = transformation
            
        # PQ format: (x, y, z, qw, qx, qy, qz)
        if input_np.shape[-1] == 7:
            pq = input_np
            
        # Transformation matrix format
        elif input_np.shape[-2:] == (4, 4):
            pq = ptr.pqs_from_transforms(input_np)
                
        # Dual quaternion format
        elif input_np.shape[-1] == 8:
            pq = ptr.pqs_from_dual_quaternions(input_np)
        else:
            raise ValueError(f"Unsupported input shape: {input_np.shape}")
    else:
        raise TypeError(f"Unsupported input type: {type(transformation)}")
        
    # Convert to requested output format
    if output_type.lower().startswith('t'):
        return ptr.transforms_from_pqs(pq)
    elif output_type.lower().startswith('d'):
        return ptr.dual_quaternions_from_pqs(pq)
    elif output_type.lower().startswith('p'):
        return pq
    else:
        raise ValueError(f"Unsupported output type: {output_type}") 