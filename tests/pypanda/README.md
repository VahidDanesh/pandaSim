# panda-py from source, patched, on a fresh machine

Builds panda-py from current `main` against libfranka 0.8.0, with a fix for a
GIL bug that makes `move_to_start()` abort the interpreter, and sets up the
realtime scheduling limit the 1 kHz control loop needs.

Written for Ubuntu 22.04, Python 3.10, FER (Panda) on robot system >= 4.0.0.
Adjust the libfranka version if your robot differs — see the table below.

Files in this directory:

- `README.md` — this file
- `gil_fix.patch` — the C++ fix, applied in step 4

---

## Why build from source at all

panda-py 1.0.0 ships a bug: `Panda.move_to_start()`, `move_to_joint_position()`
and `move_to_pose()` kill the process with

```
Fatal Python error: PyEval_SaveThread: the function must be called with the GIL held,
but the GIL is released (the current Python thread state is NULL)
```

The `moveTo*` bindings are declared with `py::call_guard<py::gil_scoped_release>()`,
so the GIL is already released on entry. The trajectory generator constructors in
`src/motion/generators.cpp` then declare their own `py::gil_scoped_release`, whose
constructor calls `PyEval_SaveThread()` — legal only while the GIL is held. It
aborts every time, so `move_to_*` is unusable on the released 1.0.0 wheels.

0.8.1 was fine because both guards lived in the constructor's function scope, so
the `gil_scoped_acquire` was still holding the GIL when the release was
constructed. The 1.0.0 refactor put the acquire in its own block, and it now dies
before the release runs. `gil_fix.patch` hoists it back out.

Check whether this is still needed before you build — see step 4.

---

## Pick your libfranka version

Determined by the robot, not by preference. Read the system version in Desk
(Settings, or the footer of the web UI).

| Robot system version | Robot | Server version | libfranka |
|---|---|---|---|
| >= 4.2.1 | FER (Panda) | 5 | 0.9.2 |
| >= 4.0.0 | FER (Panda) | 4 | **0.8.0** |
| >= 3.0.0 | FER (Panda) | 3 | 0.7.1 |

FR3 needs a much newer libfranka (0.13+), which pulls in Pinocchio and is a
different, longer build — this guide does not cover it.

panda-py also prints the server version on connect (`Using FER joint limits
(robot server version 4)`), which is the definitive answer once you're running.

Set it once and reuse it:

```bash
export LIBFRANKA_VER=0.8.0
```

---

## 1. System packages

```bash
sudo apt update
sudo apt install build-essential cmake git libpoco-dev libeigen3-dev
```

`build-essential` is not optional and not just for convenience. Without it CMake
finds no `c++` driver, falls through to `clang++`, and fails with
`cannot find -lstdc++` — a confusing error that looks like a Clang problem but
means g++ is missing. Verify:

```bash
which gcc g++ cc c++     # all four must resolve
```

Poco and Eigen are all libfranka < 0.14 needs.

---

## 2. Python environment

```bash
python3 -m venv ~/envs/panda        # or: uv venv ~/envs/panda
source ~/envs/panda/bin/activate
```

Python 3.9 or newer. panda-py requires `numpy>=2.0`.

**If this machine has ROS installed, do not source it in this shell.** ROS puts
its own `site-packages` on `PYTHONPATH`, which takes priority over the venv, and
ROS Humble ships numpy 1.x. That shadows the venv's numpy 2.x and breaks the
compiled extension in confusing ways. Confirm you're clean:

```bash
echo "$PYTHONPATH"                                      # ideally empty
python -c "import numpy; print(numpy.__version__, numpy.__file__)"
```

The numpy path must be inside your venv, and the version 2.x.

---

## 3. Build libfranka

Use **`JeanElsner/libfranka`**, not `frankaemika/libfranka`. Upstream 0.7.1 and
0.8.0 date from 2021 and do not compile on a current GCC; the fork carries the
fixes. This is the same source panda-py's own CI uses for these versions
(`bin/before_install_centos.sh`).

```bash
mkdir -p ~/src && cd ~/src
git clone --recursive https://github.com/JeanElsner/libfranka.git
cd libfranka
git checkout "$LIBFRANKA_VER"
git submodule update --recursive

cmake -B build -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTS=OFF -DBUILD_EXAMPLES=OFF \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5
cmake --build build -j"$(nproc)"
sudo cmake --install build --strip
sudo ldconfig
```

`CMAKE_POLICY_VERSION_MINIMUM` is only needed on CMake 4, whose policy floor
rejects these older CMake files. Harmless on CMake 3.

Verify before continuing:

```bash
ls -l /usr/local/lib/libfranka.so*
grep -i version /usr/local/lib/cmake/Franka/FrankaConfigVersion.cmake | head -3
```

You want `libfranka.so.0.8.0` and a matching version string.

---

## 4. Clone panda-py and apply the patch

```bash
cd ~/src
git clone https://github.com/JeanElsner/panda-py.git
cd panda-py
git log -1 --oneline
```

**Check whether the bug is still there first.** If upstream has fixed it, skip
the patch:

```bash
git apply --check ~/path/to/gil_fix.patch && echo "STILL BROKEN - apply the patch"
git apply --reverse --check ~/path/to/gil_fix.patch && echo "ALREADY FIXED - skip"
```

If neither prints, the surrounding code changed; open
`src/motion/generators.cpp` and check by eye that in both the `JointTrajectory`
and `CartesianTrajectory` constructors the `py::gil_scoped_acquire` outlives the
`py::gil_scoped_release` below it.

To apply:

```bash
git apply ~/path/to/gil_fix.patch
git diff --stat        # expect: src/motion/generators.cpp | 8 ++++++--
```

Building `main` rather than the `v1.0.0` tag also gets you Desk brake-endpoint
auto-detection (the `platform` argument became optional) and the realtime warning
used in step 6.

---

## 5. Build and install

```bash
pip install .          # or: uv pip install .
```

Build isolation fetches `scikit-build-core` and `pybind11`. CMake derives the
libfranka version guards from `find_package(Franka)`, so no environment variable
is needed.

If CMake still picks the wrong compiler:

```bash
pip install . \
  -C cmake.define.CMAKE_C_COMPILER=gcc \
  -C cmake.define.CMAKE_CXX_COMPILER=g++
```

If `find_package(Franka REQUIRED)` fails (CMakeLists.txt line 13), point it at
the install prefix:

```bash
  -C cmake.define.CMAKE_PREFIX_PATH=/usr/local
```

### Verify the install

```bash
python -c "
import panda_py, panda_py._core as c
print(panda_py.__version__)
print(c.__file__)
print('built from main:', hasattr(panda_py, 'realtime_priority_available'))
"
```

`__version__` reads `1.0.0` even on `main` — it's hardcoded in
`src/panda_py/__init__.py` and hasn't been bumped, so it proves nothing. The
`realtime_priority_available` attribute is the real signal: it only exists in
commits after the 1.0.0 release.

Confirm it linked against the libfranka you built:

```bash
ldd $(python -c "import panda_py._core as c; print(c.__file__)") | grep -i franka
```

Expect `libfranka.so.0.8.0 => /usr/local/lib/libfranka.so.0.8.0`.

### Run the test suite (no robot needed)

```bash
pip install pytest
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests
```

The plugin-autoload flag avoids pytest importing unrelated `pytest11` plugins
from a ROS install and dying on a missing dependency such as `lark`.

Note this suite does **not** verify the GIL fix — its trajectory tests call the
constructors from Python, where the GIL is held, which was never the broken path.
It only tells you the build is sane. The real check is step 7.

---

## 6. Realtime scheduling

libfranka puts its control thread on `SCHED_FIFO`. That needs a non-zero
`RLIMIT_RTPRIO`, which is 0 by default, so without this the 1 kHz loop runs at
normal priority and misses deadlines when the machine is busy. The robot reports
that as `communication_constraints_violation` or a reflex abort — which looks
like a network fault and sends people off debugging ethernet.

panda-py runs libfranka with `RealtimeConfig::kIgnore`, so this fails silently
except for the warning added in `main`.

```bash
printf '%s - rtprio 99\n%s - memlock 102400\n' "$USER" "$USER" \
  | sudo tee /etc/security/limits.d/99-realtime.conf
```

**Log out and back in.** PAM applies limits at login, so a new terminal in the
same session keeps the old ones. Over SSH or VS Code Remote, drop the whole
connection and reconnect. Then:

```bash
ulimit -r        # want 99, not 0
python -c "import panda_py; print(panda_py.realtime_priority_available())"
```

### The kernel half

Priority granted is not the same as a preemptible kernel. If you still see a
warning about the kernel after the above, that's the second branch of the check:

```bash
uname -v | grep -o PREEMPT_RT || echo "stock kernel"
```

A stock kernel plus the rtprio limit is what most bench setups run and is fine
for moderate motions. Move to a `PREEMPT_RT` kernel (Franka documents this for
FCI) if you do sustained high-rate control and hit unexplained reflex aborts.

---

## 7. First robot run

The FCI must be enabled and the robot in FCI mode. Note that `Desk` refuses to
take the control token if someone else holds it — that leaves `unlock()` a no-op
with the brakes still engaged, so take control explicitly:

```python
import panda_py

hostname = "172.16.0.2"

desk = panda_py.Desk(hostname, username, password)
desk.take_control(force=True)   # press the circle button on the robot base
desk.unlock()

panda = panda_py.Panda(hostname)
panda.move_to_start()

desk.lock()
```

`force=True` requires physically pressing the button on the base within the
timeout — it cannot be scripted headlessly.

Success looks like:

```
INFO:panda:Connected to robot (172.16.0.2).
INFO:panda:Using FER joint limits (robot server version 4).
INFO:panda:Initializing motion generation (moveToJointPosition).
INFO:motion:Computed joint trajectory: 1 waypoint, duration 1.16 seconds.
INFO:panda:Starting new controller (JointTrajectory).
```

The second and third lines are the proof the patch worked: the unpatched build
aborts between them.

---

## Troubleshooting

| Symptom | Cause |
|---|---|
| `PyEval_SaveThread ... GIL is released` | The patch is not applied, or a stale wheel is installed. Check `hasattr(panda_py, 'realtime_priority_available')`. |
| `clang++ ... cannot find -lstdc++` | `build-essential` missing; CMake fell through to Clang. |
| `find_package(Franka)` fails | libfranka not installed, or installed outside `/usr/local`. |
| `ModuleNotFoundError: No module named 'lark'` from pytest | A ROS pytest plugin on `PYTHONPATH`. Use `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`. |
| numpy 1.x imported from `/opt/ros/...` | ROS sourced in this shell, shadowing the venv. |
| `Cannot take control. User X is in control.` | Someone holds the Desk token. `take_control(force=True)`, or have them release it. |
| `communication_constraints_violation` | Usually the realtime limit from step 6, not the network. |
| Realtime warning persists after step 6 | You didn't log out and back in, or it's the kernel branch of the check. |

---

## Rechecking upstream

This whole setup exists to work around one bug. Periodically:

```bash
cd ~/src/panda-py && git fetch && git log --oneline HEAD..origin/main
```

Once a release contains the fix, drop all of this and go back to
`pip install panda-python==<version>` with the matching libfranka wheel from the
release page. The `git apply --reverse --check` in step 4 tells you when that
has happened.
