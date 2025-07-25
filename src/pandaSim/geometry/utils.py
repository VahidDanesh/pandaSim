"""
Geometry utility functions for conversions between different representations.
"""
from typing import Any, Optional, Dict, Tuple, Union
import numpy as np
import torch
from pytransform3d import (
    transformations as pt,
    rotations as pr,
    batch_rotations as pb,
    trajectories as ptr,
    plot_utils as ppu
)
import roboticstoolbox as rtb
import os
from pathlib import Path
import spatialmath as sm


PANDA_VIRTUAL = '../../../model/franka_description/robots/frankaEmikaPandaVirtual.urdf'

def convert_pose(transformation: tuple | torch.Tensor | np.ndarray | sm.SE3, output_type: str = 'pq') -> np.ndarray:
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

    if isinstance(transformation, sm.SE3):
        transformation = transformation.A

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
    


def create_virtual_panda(urdf_path: Union[str, None] = None) -> rtb.models.Panda:
    """
    Attaches a virtual finger link to a Panda robot model and returns the modified Panda.

    Args:
        urdf_path (str): The path to the URDF file containing the virtual finger.

    Returns:
        rtb.models.Panda: The modified Panda robot model.
    """
    if urdf_path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        urdf_path = os.path.abspath(os.path.join(current_dir, PANDA_VIRTUAL))
    else:
        urdf_path = os.path.abspath(urdf_path)

    panda = rtb.models.Panda()
    panda_virtual = rtb.Robot.URDF(file_path=urdf_path, gripper='panda_hand')

    panda_virtual.addconfiguration('qr', np.append(panda.qr, 0))
    panda_virtual.addconfiguration('qz', np.append(panda.qz, 0))
    panda_virtual.qr = np.append(panda.qr, 0)
    panda_virtual.qz = np.append(panda.qz, 0)

    panda_virtual.q = panda_virtual.qr
    # Make a variable for the upper and lower limits of the robot
    qd_lb = -np.pi * np.array([5/6]*4 + [1]*(panda_virtual.n-4))
    qd_ub = np.pi * np.array([5/6]*4 + [1]*(panda_virtual.n-4))
    panda_virtual.qdlim = np.vstack((qd_lb, qd_ub))

    #rebuild the ETS.
    panda_virtual.ets()

    return panda_virtual
