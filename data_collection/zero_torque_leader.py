"""
主臂零力矩重力补偿: 后台循环发 MIT 帧 (kp=0, 小 kd, torque=重力)。
夹爪 torque=0, 可手掰。
"""

import ctypes
import os
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np

from data_collection.arm_control import ELA3Arm, JOINT_NAMES

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_BUNDLED_SDK_ROOT = _PROJECT_ROOT / "el_a3_sdk"
_FALLBACK_SDK_ROOT = Path("/home/ubuntu/AI/EDULITE_A3/el_a3_sdk")


def _resolve_sdk_root() -> Path:
    if (_BUNDLED_SDK_ROOT / "el_a3_sdk" / "kinematics.py").is_file():
        return _BUNDLED_SDK_ROOT
    if (_FALLBACK_SDK_ROOT / "el_a3_sdk" / "kinematics.py").is_file():
        return _FALLBACK_SDK_ROOT
    return _BUNDLED_SDK_ROOT


_EL_A3_SDK_ROOT = str(_resolve_sdk_root())
_DEFAULT_URDF = str(_resolve_sdk_root() / "resources" / "urdf" / "el_a3_legacy.urdf")
_DEFAULT_INERTIA = str(_resolve_sdk_root() / "resources" / "config" / "inertia_params.yaml")


def _cmeel_lib_dir() -> Optional[Path]:
    py = f"python{sys.version_info.major}.{sys.version_info.minor}"
    candidates = [Path(sys.prefix) / "lib" / py / "site-packages" / "cmeel.prefix" / "lib"]
    for p in sys.path:
        path = Path(p)
        candidates.append(path / "cmeel.prefix" / "lib")
        if path.name == "lib":
            candidates.append(path)
    seen = set()
    for cand in candidates:
        resolved = cand.resolve() if cand.exists() else cand
        if resolved in seen:
            continue
        seen.add(resolved)
        if (cand / "libeigenpy.so").is_file():
            return cand
    return None


def _preload_pinocchio_libs():
    cmeel_lib = _cmeel_lib_dir()
    if cmeel_lib is None:
        return
    for name in (
        "libeigenpy.so",
        "libcoal.so",
        "libpinocchio_default.so",
        "libpinocchio_parsers.so",
        "libpinocchio_collision.so",
    ):
        lib = cmeel_lib / name
        if lib.is_file():
            ctypes.CDLL(str(lib), mode=ctypes.RTLD_GLOBAL)


def _prefer_venv_site_packages():
    venv_site = Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
    if venv_site.is_dir() and str(venv_site) in sys.path:
        sys.path.remove(str(venv_site))
        sys.path.insert(0, str(venv_site))


def _ld_dir_has_foreign_pinocchio(path: str, cmeel_lib: Optional[Path]) -> bool:
    d = Path(path)
    if not d.is_dir():
        return False
    if cmeel_lib is not None and d.resolve() == cmeel_lib.resolve():
        return False
    if (d / "libeigenpy.so").exists():
        return True
    return any(d.glob("libpinocchio*.so*"))


def ensure_venv_native_env():
    if os.environ.get("VLA_VENV_LIBS") == "1":
        return
    cmeel_lib = _cmeel_lib_dir()
    ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    if not any(_ld_dir_has_foreign_pinocchio(p, cmeel_lib) for p in ld_path.split(os.pathsep) if p):
        return

    env = os.environ.copy()
    env["VLA_VENV_LIBS"] = "1"
    env["PYTHONPATH"] = str(_PROJECT_ROOT)
    ld_keep = []
    if cmeel_lib is not None:
        ld_keep.append(str(cmeel_lib))
    for p in ld_path.split(os.pathsep):
        if p and not _ld_dir_has_foreign_pinocchio(p, cmeel_lib):
            ld_keep.append(p)
    env["LD_LIBRARY_PATH"] = os.pathsep.join(ld_keep)

    argv0 = Path(sys.argv[0]).resolve()
    if argv0.name == "teleop_record.py":
        argv = [sys.executable, "-m", "data_collection.teleop_record", *sys.argv[1:]]
    else:
        argv = [sys.executable, *sys.argv]
    os.execvpe(sys.executable, argv, env)


def ensure_pip_pinocchio():
    ensure_venv_native_env()
    _prefer_venv_site_packages()
    _preload_pinocchio_libs()
    import pinocchio as pin
    return pin


def _ensure_el_a3_sdk_on_path():
    sdk_root = str(_resolve_sdk_root())
    if sdk_root in sys.path:
        sys.path.remove(sdk_root)
    sys.path.insert(0, sdk_root)


class ZeroTorqueLeader:
    def __init__(
        self,
        arm: ELA3Arm,
        urdf_path: str = _DEFAULT_URDF,
        inertia_config_path: str = _DEFAULT_INERTIA,
        update_rate_hz: float = 200.0,
        kd_min: Optional[dict] = None,
        kd_max: Optional[dict] = None,
        gravity_scale: Optional[dict] = None,
        kd_velocity_ref: float = 1.0,
        kd_smoothing_alpha: float = 0.25,
        gravity_smooth_alpha: float = 0.1,
        gripper_kd: float = 0.02,
        verbose: bool = False,
    ):
        self.arm = arm
        self.update_rate_hz = update_rate_hz
        self.dt = 1.0 / update_rate_hz
        self.kd_min = kd_min or {1: 0.001, 2: 0.005, 3: 0.005, 4: 0.005, 5: 0.005, 6: 0.005}
        self.kd_max = kd_max or {1: 0.15,  2: 0.25,  3: 0.25,  4: 0.10,  5: 0.05,  6: 0.05}
        self.gravity_scale = gravity_scale or {4: 2.0}
        self.kd_velocity_ref = kd_velocity_ref
        self.kd_smoothing_alpha = kd_smoothing_alpha
        self.gravity_smooth_alpha = gravity_smooth_alpha
        self.gripper_kd = gripper_kd
        self.verbose = verbose
        self._verbose_last = 0.0
        self._kin = self._load_kinematics(urdf_path, inertia_config_path)
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._running = False
        self._adaptive_kd = [self.kd_max.get(i + 1, 0.05) for i in range(6)]
        self._smoothed_q = np.zeros(6, dtype=np.float64)
        self._cache_lock = threading.Lock()
        self._cache_positions = np.zeros(6, dtype=np.float64)
        self._cache_velocities = np.zeros(6, dtype=np.float64)
        self._cache_gripper = 0.0
        self._cache_ts = 0.0
        self._loop_count = 0
        self._last_log_time = 0.0

    @staticmethod
    def _load_kinematics(urdf_path: str, inertia_config_path: str):
        ensure_pip_pinocchio()
        _ensure_el_a3_sdk_on_path()
        try:
            from el_a3_sdk.kinematics import ELA3Kinematics
        except ImportError as e:
            raise RuntimeError(
                f"无法导入 el_a3_sdk 运动学: {e}\n"
                f"路径: {_EL_A3_SDK_ROOT}, 需要 pip install pin"
            ) from e

        if not Path(urdf_path).exists():
            raise FileNotFoundError(f"URDF 文件不存在: {urdf_path}")
        if not Path(inertia_config_path).exists():
            print(f"WARNING: 惯量参数文件不存在: {inertia_config_path}, 用 URDF 内置参数")
            inertia_config_path = None

        return ELA3Kinematics(
            urdf_path=urdf_path,
            inertia_config_path=inertia_config_path,
        )

    def _compute_adaptive_kd(self, joint_idx: int, velocity: float, prev_kd: float) -> float:
        mid = joint_idx + 1
        kd_min = self.kd_min.get(mid, 0.001)
        kd_max = self.kd_max.get(mid, 0.05)
        ratio = abs(velocity) / max(self.kd_velocity_ref, 1e-6)
        kd_raw = kd_min + (kd_max - kd_min) / (1.0 + ratio * ratio)
        kd = self.kd_smoothing_alpha * kd_raw + (1.0 - self.kd_smoothing_alpha) * prev_kd
        return float(np.clip(kd, 0.0, 5.0))

    def _loop(self):
        try:
            q0 = self.arm.read_joint_positions_mit()
            g0 = self.arm.read_gripper_position_mit()
            self._smoothed_q = q0.copy()
            with self._cache_lock:
                self._cache_positions = q0.copy()
                self._cache_velocities = np.zeros(6, dtype=np.float64)
                self._cache_gripper = g0
                self._cache_ts = time.time()
        except Exception:
            pass

        last_q = self._smoothed_q.copy()
        last_v = np.zeros(6, dtype=np.float64)
        last_gripper = float(self._cache_gripper)

        while not self._stop_event.is_set():
            t_start = time.perf_counter()
            try:
                self._smoothed_q = (
                    self.gravity_smooth_alpha * last_q
                    + (1.0 - self.gravity_smooth_alpha) * self._smoothed_q
                )
                grav = self._kin.compute_gravity(self._smoothed_q.tolist())
                grav_arm = list(grav)[:6] if len(grav) >= 6 else (list(grav) + [0.0] * (6 - len(grav)))

                new_q = np.zeros(6, dtype=np.float64)
                new_v = np.zeros(6, dtype=np.float64)
                for i, joint in enumerate(JOINT_NAMES):
                    mid = i + 1
                    self._adaptive_kd[i] = self._compute_adaptive_kd(
                        i, last_v[i], self._adaptive_kd[i]
                    )
                    self.arm.bus.write_operation_frame(
                        joint,
                        position=last_q[i],
                        velocity=0.0,
                        kp=0.0,
                        kd=self._adaptive_kd[i],
                        torque=grav_arm[i] * self.gravity_scale.get(mid, 1.0),
                    )
                    pos, vel, _torque, _temp = self.arm.bus.read_operation_frame(joint)
                    new_q[i] = pos
                    new_v[i] = vel

                self.arm.bus.write_operation_frame(
                    "gripper",
                    position=0.0, velocity=0.0,
                    kp=0.0, kd=self.gripper_kd, torque=0.0,
                )
                try:
                    gpos, _gvel, _gt, _gtemp = self.arm.bus.read_operation_frame("gripper")
                except Exception:
                    gpos = last_gripper

                with self._cache_lock:
                    self._cache_positions = new_q
                    self._cache_velocities = new_v
                    self._cache_gripper = gpos
                    self._cache_ts = time.time()

                last_q = new_q
                last_v = new_v
                last_gripper = float(gpos)
                self._loop_count += 1

                if self.verbose:
                    now = time.time()
                    if now - self._verbose_last > 1.0:
                        self._verbose_last = now
                        scaled = [grav_arm[i] * self.gravity_scale.get(i + 1, 1.0) for i in range(6)]
                        print(
                            f"[zt]"
                            f" q=[" + ", ".join(f"{q:+.2f}" for q in new_q) + "]"
                            f" grav=[" + ", ".join(f"{g:+.2f}" for g in grav_arm) + "]"
                            f" sent=[" + ", ".join(f"{t:+.2f}" for t in scaled) + "]"
                            f" kd=[" + ", ".join(f"{k:.3f}" for k in self._adaptive_kd) + "]"
                        )
            except Exception as e:
                now = time.time()
                if now - self._last_log_time > 2.0:
                    print(f"  [zero_torque_leader] WARN: {e}")
                    self._last_log_time = now

            sleep_time = self.dt - (time.perf_counter() - t_start)
            if sleep_time > 0:
                time.sleep(sleep_time)

    def start(self):
        if self._running:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="zero_torque_leader")
        self._thread.start()
        self._running = True
        print(f"[zero_torque_leader] 后台循环已启动 ({self.update_rate_hz:.0f} Hz)")

    def stop(self):
        if not self._running:
            return
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._running = False
        print("[zero_torque_leader] 后台循环已停止")

    def read_joint_positions(self) -> np.ndarray:
        with self._cache_lock:
            return self._cache_positions.copy()

    def read_joint_velocities(self) -> np.ndarray:
        with self._cache_lock:
            return self._cache_velocities.copy()

    def read_gripper_position(self) -> float:
        with self._cache_lock:
            return float(self._cache_gripper)

    def read_full_state(self) -> tuple:
        with self._cache_lock:
            return self._cache_positions.copy(), float(self._cache_gripper), self._cache_ts

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def loop_count(self) -> int:
        return self._loop_count
