"""Import hygiene and basic model loading, runnable with only the `sim` dependencies."""
import subprocess
import sys

import numpy as np
import pytest

SIM_MODULES = [
    "pandaSim",
    "pandaSim.control",
    "pandaSim.geometry",
    "pandaSim.geometry.utils",
    "pandaSim.control.contact_qp",
    "pandaSim.control.backends",
    "pandaSim.control.primitive_executor",
    "pandaSim.control.viewer",
    "pandaSim.planning.primitives",
    "pandaSim.planning.se3",
    "pandaSim.planning.box_primitives",
    "pandaSim.planning.sequencer",
    "pandaSim.planning.gait",
    "pandaSim.planning.feasibility",
    "pandaSim.planning.clearance",
    "pandaSim.planning.collision",
]
HEAVY = ["torch", "genesis", "swift"]

EE = "panda_finger_virtual"


def test_sim_modules_do_not_import_heavy_backends():
    code = (
        "import sys, importlib\n"
        f"[importlib.import_module(m) for m in {SIM_MODULES!r}]\n"
        f"print([m for m in {HEAVY!r} if m in sys.modules])"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().splitlines()[-1] == "[]"


@pytest.fixture(scope="module")
def panda():
    from pandaSim.geometry.utils import create_virtual_panda

    return create_virtual_panda()


def test_virtual_panda_model(panda):
    assert panda.n == 8
    assert [link.name for link in panda.links][-1] == EE
    assert panda.qlim.shape == (2, 8)
    assert np.allclose(panda.q, panda.qr)


def test_manipulability_jacobian_matches_finite_difference(panda):
    from pandaSim.geometry.utils import manipulability_jacobian

    def manip(q):
        J = panda.jacobe(q, end=EE)
        return np.sqrt(np.linalg.det(J @ J.T))

    q = panda.qr + 0.1 * np.arange(panda.n)
    J = panda.jacobe(q, end=EE)
    Jm = manipulability_jacobian(J, panda.hessiane(Je=J, end=EE))

    eps = 1e-6
    fd = np.array([(manip(q + eps * e) - manip(q - eps * e)) / (2 * eps) for e in np.eye(panda.n)])
    assert Jm.shape == (panda.n,)
    assert np.allclose(Jm, fd, atol=1e-6)


def test_model_folder_is_found_and_missing_folder_is_reported(monkeypatch):
    from pandaSim.geometry import utils

    assert utils._find_panda_virtual().endswith("frankaEmikaPandaVirtual.urdf")
    monkeypatch.setattr(utils, "__file__", "/nonexistent/a/b/c/utils.py")       # not a source checkout
    monkeypatch.delenv("PANDASIM_MODEL_DIR", raising=False)
    with pytest.raises(FileNotFoundError, match="PANDASIM_MODEL_DIR"):
        utils._find_panda_virtual()
    monkeypatch.setenv("PANDASIM_MODEL_DIR", str(__import__("pathlib").Path(__file__).resolve().parents[2] / "model"))
    assert utils._find_panda_virtual().endswith("frankaEmikaPandaVirtual.urdf")
