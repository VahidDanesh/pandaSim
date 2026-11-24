"""The Panda model with the virtual bearing joint, and the manipulability Jacobian."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Union

import numpy as np
import roboticstoolbox as rtb
from roboticstoolbox.models.URDF.URDFRobot import URDF_file
from xacrodoc import packages

PANDA_VIRTUAL = '../../../model/franka_description/robots/frankaEmikaPandaVirtual.urdf'


def manipulability_jacobian(J: np.ndarray, H: np.ndarray) -> np.ndarray:
    """
    Gradient of the Yoshikawa manipulability w = sqrt(det(J J^T)) w.r.t. the joints.

    Replaces rtb ``Robot.jacobm(J=, H=)``, whose shape check rejects the Hessian
    returned by ``hessiane``/``hessian0``.

    Args:
        J: Manipulator Jacobian, shape (6, n).
        H: Manipulator Hessian in the same frame as J, shape (n, 6, n).

    Returns:
        Manipulability Jacobian, shape (n,).
    """
    JJt = J @ J.T
    w = np.sqrt(max(np.linalg.det(JJt), 0.0))
    return w * np.einsum('jk,ilk,jl->i', J, H, np.linalg.inv(JJt))


def _find_panda_virtual() -> str:
    """The Panda URDF with the virtual finger: $PANDASIM_MODEL_DIR/franka_description/..., else the repository's
    `model/` folder (a source checkout or an editable install)."""
    rel = Path(PANDA_VIRTUAL).relative_to('../../../model')
    roots = ([Path(os.environ['PANDASIM_MODEL_DIR'])] if os.environ.get('PANDASIM_MODEL_DIR') else []) \
        + [Path(__file__).resolve().parents[3] / 'model']
    for root in roots:
        if (root / rel).exists():
            return str((root / rel).resolve())
    raise FileNotFoundError(f"{rel} not found in {[str(r) for r in roots]}; set PANDASIM_MODEL_DIR to the "
                            "repository's model/ folder or install the repository in editable mode")


def create_virtual_panda(urdf_path: Union[str, None] = None) -> rtb.models.Panda:
    """
    Attaches a virtual finger link to a Panda robot model and returns the modified Panda.

    Args:
        urdf_path (str): The path to the URDF file containing the virtual finger.

    Returns:
        rtb.models.Panda: The modified Panda robot model.
    """
    if urdf_path is None:
        urdf_path = _find_panda_virtual()
    else:
        urdf_path = os.path.abspath(urdf_path)

    # rtb resolves package:// mesh paths through xacrodoc, and the model folder
    # has no package.xml to be discovered by
    packages.update_package_cache({'franka_description': str(Path(urdf_path).parents[1])})
    elinks, name, _ = URDF_file(urdf_path)
    hand = next(link for link in elinks if link.name == 'panda_hand')

    panda = rtb.models.Panda()
    panda_virtual = rtb.Robot(elinks, name=name, gripper_links=hand)

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
