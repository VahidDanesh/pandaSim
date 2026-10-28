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
    "pandaSim.geometry.rtb_adapter",
    "pandaSim.planning.screw_motion_planner",
    "pandaSim.control.qp",
    "pandaSim.control.contact_qp",
    "pandaSim.planning.primitives",
    "pandaSim.control.robot_executor",
    "pandaSim.control.data_collector",
    "pandaSim.control.config",
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


def test_lazy_names_resolve():
    import pandaSim
    from pandaSim.control import QPController, RobotExecutor

    assert pandaSim.QPController is QPController
    assert RobotExecutor.__name__ == "RobotExecutor"
    with pytest.raises(AttributeError):
        pandaSim.does_not_exist


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


def test_headless_adapter_integrates_joint_velocities(panda):
    from pandaSim.geometry.rtb_adapter import RoboticsToolboxAdapter

    adapter = RoboticsToolboxAdapter({"viewer": None, "rate": 200})
    assert adapter.dt == pytest.approx(1 / 200)
    adapter.env.add(panda)

    q0 = panda.q.copy()
    qd = np.linspace(0.1, 0.8, panda.n)
    adapter.control_joint_velocities(panda, qd)
    for _ in range(10):
        adapter.step_simulation(adapter.dt)
    assert np.allclose(panda.q, q0 + qd * 10 * adapter.dt)
    panda.q, panda.qd = q0, np.zeros(panda.n)
