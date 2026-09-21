"""
arm_control.py

EL-A3 机械臂控制接口, 封装 robstride_dynamics SDK。
支持关节位置读取、位置控制、夹爪控制。
用于 leader-follower 遥操作数据采集。
"""

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

# robstride_dynamics SDK 位于 Python_Sample/ 目录下，添加到 import 路径
_SDK_DIR = str(Path(__file__).resolve().parent.parent / "Python_Sample")
if _SDK_DIR not in sys.path:
    sys.path.insert(0, _SDK_DIR)

try:
    from robstride_dynamics import RobstrideBus, Motor, ParameterType  # type: ignore
except ImportError as e:
    raise ImportError(
        f"无法导入 robstride_dynamics: {e}\n"
        "请确保已安装依赖: pip install python-can numpy tqdm"
    ) from e


# ── EL-A3 默认配置 ──────────────────────────────────────────
# 根据你的实际接线和电机 ID 修改下面的配置

@dataclass
class ArmConfig:
    """单条机械臂的电机配置"""
    channel: str = "can0"
    motors: dict = field(default_factory=lambda: {
        "J1": Motor(id=1, model="rs-00"),
        "J2": Motor(id=2, model="rs-00"),
        "J3": Motor(id=3, model="rs-00"),
        "J4": Motor(id=4, model="rs-00"),
        "J5": Motor(id=5, model="rs-00"),
        "J6": Motor(id=6, model="rs-00"),
        "gripper": Motor(id=7, model="rs-00"),
    })
    # 每个电机的校准参数: direction(+1/-1) 和 homing_offset(rad)
    # 根据你的装配实际情况修改
    calibration: dict = field(default_factory=lambda: {
        "J1": {"direction": -1, "homing_offset": 0.0},
        "J2": {"direction": 1, "homing_offset": 0.0},
        "J3": {"direction": -1, "homing_offset": 0.0},
        "J4": {"direction": 1, "homing_offset": 0.0},
        "J5": {"direction": -1, "homing_offset": 0.0},
        "J6": {"direction": 1, "homing_offset": 0.0},
        "gripper": {"direction": 1, "homing_offset": 0.0},
    })


JOINT_NAMES = ["J1", "J2", "J3", "J4", "J5", "J6"]

# URDF 关节限位 (rad)
JOINT_LIMITS = {
    "J1": (-2.79253, 2.79253),
    "J2": (0.0, 3.66519),
    "J3": (-4.01426, 0.0),
    "J4": (-1.5708, 1.5708),
    "J5": (-1.5708, 1.5708),
    "J6": (-1.5708, 1.5708),
}


class ELA3Arm:
    """EL-A3 6-DOF 机械臂 + 夹爪控制器"""

    # 单个控制周期内, 目标位置相对当前实际位置允许的最大偏差 (rad)。
    # 该值直接限制了 MIT 环的最大位置误差 (最大力矩 ≈ kp * 该值),
    # 上游读数错乱/跳变时从臂最多缓慢移动而不会猛甩。设为 None 可关闭。
    MAX_TARGET_JUMP_RAD = 0.25

    def __init__(self, config: ArmConfig, name: str = "arm"):
        self.name = name
        self.config = config
        self.bus: Optional[RobstrideBus] = None
        self._connected = False
        self._last_feedback: dict = {}   # joint -> 最近一次实测位置 (URDF 约定)
        self._jump_warn_ts = 0.0
        self._mode_warn_ts: dict = {}    # motor -> 上次"未使能"告警时间

    def connect(self):
        self.bus = RobstrideBus(
            channel=self.config.channel,
            motors=self.config.motors,
            calibration=self.config.calibration,
            bitrate=1000000,
        )
        self.bus.connect()
        self._connected = True
        print(f"[{self.name}] 已连接 (channel={self.config.channel})")

    def disconnect(self):
        if self._connected and self.bus is not None:
            self.bus.disconnect(disable_torque=True)
            self._connected = False
            print(f"[{self.name}] 已断开")

    def enable_all(self, max_retries: int = 3):
        """
        使能所有电机。单个电机偶发无响应时自动重试 (刷新接收缓冲后再试),
        重试仍失败则抛错 — 电机没使能会导致后续 MIT 帧有应答但不出力,
        表现为"该关节/夹爪完全不动"，必须在启动阶段就暴露。
        """
        for motor in JOINT_NAMES + ["gripper"]:
            last_err = None
            for attempt in range(1, max_retries + 1):
                try:
                    self.bus.enable(motor)
                    last_err = None
                    break
                except RuntimeError as e:
                    last_err = e
                    print(f"WARNING: [{self.name}] 使能 {motor} 失败 "
                          f"(第 {attempt}/{max_retries} 次): {e}")
                    self.bus._flush_rx_buffer()
                    time.sleep(0.1)
            if last_err is not None:
                raise RuntimeError(
                    f"[{self.name}] 电机 {motor} 使能失败 (已重试 {max_retries} 次)。"
                    f"请检查 CAN 接线、终端电阻、供电和电机 ID。"
                ) from last_err
        print(f"[{self.name}] 所有电机已使能")

    def disable_all(self):
        for motor in JOINT_NAMES + ["gripper"]:
            try:
                self.bus.disable(motor)
            except Exception as e:
                print(f"WARNING: 失能 {motor} 失败: {e}")
        print(f"[{self.name}] 所有电机已失能")

    # ── 安全限位 ─────────────────────────────────────────

    def set_torque_limit(self, motor: str, max_torque_nm: float):
        """
        设置某个电机的固件级扭矩上限 (TORQUE_LIMIT 参数, 0x700B)。
        即使 MIT 控制环算出更大的力矩, 电机内部也会先把它截断到这个值,
        从而避免硬卡时持续过流烧电机。

        Args:
            motor: 电机名 (如 "gripper" / "J1"...)
            max_torque_nm: 扭矩上限, 单位 Nm
        """
        self.bus.write(motor, ParameterType.TORQUE_LIMIT, float(max_torque_nm))
        print(f"[{self.name}] {motor} TORQUE_LIMIT = {max_torque_nm:.2f} Nm")

    def set_current_limit(self, motor: str, max_current_a: float):
        """
        设置某个电机的固件级电流上限 (CURRENT_LIMIT 参数, 0x7018)。
        和 TORQUE_LIMIT 类似, 用电流维度保护。

        Args:
            motor: 电机名
            max_current_a: 电流上限, 单位 A
        """
        self.bus.write(motor, ParameterType.CURRENT_LIMIT, float(max_current_a))
        print(f"[{self.name}] {motor} CURRENT_LIMIT = {max_current_a:.2f} A")

    # ── 校准工具 ───────────────────────────────────────────

    def _apply_calibration(self, joint: str, raw_pos: float) -> float:
        """电机原始角度 → URDF 约定角度：raw * direction + homing_offset"""
        cal = self.config.calibration.get(joint, {})
        direction = cal.get("direction", 1)
        offset = cal.get("homing_offset", 0.0)
        return raw_pos * direction + offset

    def _inverse_calibration(self, joint: str, urdf_pos: float) -> float:
        """URDF 约定角度 → 电机原始角度：(urdf - homing_offset) / direction"""
        cal = self.config.calibration.get(joint, {})
        direction = cal.get("direction", 1)
        offset = cal.get("homing_offset", 0.0)
        return (urdf_pos - offset) / direction

    # ── 读取 ─────────────────────────────────────────────

    def read_joint_positions(self) -> np.ndarray:
        """
        读取 6 个关节的当前角度 (rad)，已应用 calibration 修正。

        Returns:
            np.ndarray: shape (6,), [q1, q2, q3, q4, q5, q6]
        """
        positions = np.zeros(6, dtype=np.float64)
        for i, joint in enumerate(JOINT_NAMES):
            raw = float(self.bus.read(joint, ParameterType.MEASURED_POSITION))
            positions[i] = self._apply_calibration(joint, raw)
        return positions

    def read_joint_positions_mit(self) -> np.ndarray:
        """
        通过发送零力矩的 MIT 帧来同时读取关节位置。
        比逐个 read 更快，适合高频循环。
        bus 层的 read_operation_frame 已处理 calibration，此处不再重复。

        Returns:
            np.ndarray: shape (6,), [q1, q2, q3, q4, q5, q6]
        """
        positions = np.zeros(6, dtype=np.float64)
        for i, joint in enumerate(JOINT_NAMES):
            self.bus.write_operation_frame(joint, position=0, kp=0, kd=0, velocity=0, torque=0)
            pos, vel, torque, temp = self.bus.read_operation_frame(joint)
            positions[i] = pos
            self._last_feedback[joint] = pos
        return positions

    def read_gripper_position(self) -> float:
        """读取夹爪位置 (rad)，通过参数读取，已应用 calibration 修正。"""
        raw = float(self.bus.read("gripper", ParameterType.MEASURED_POSITION))
        return self._apply_calibration("gripper", raw)

    def read_gripper_position_mit(self) -> float:
        """
        通过 MIT 帧读取夹爪位置，同时保持夹爪在 MIT 模式下活跃。
        bus 层的 read_operation_frame 已处理 calibration，此处不再重复。
        """
        self.bus.write_operation_frame("gripper", position=0, kp=0, kd=0, velocity=0, torque=0)
        pos, vel, torque, temp = self.bus.read_operation_frame("gripper")
        self._last_feedback["gripper"] = pos
        return pos

    # ── 控制 ─────────────────────────────────────────────

    def _check_run_mode(self, motor: str):
        """
        检查最近一帧状态中电机是否处于运行模式 (mode=2)。
        电机锁存故障后会自动退出运行模式: 仍有应答但无力矩, 位置控制完全失效。
        """
        mode = self.bus.last_status_mode.get(motor)
        if mode is not None and mode != 2:
            now = time.time()
            if now - self._mode_warn_ts.get(motor, 0.0) > 1.0:
                print(f"  [SAFETY] [{self.name}] {motor} 不在运行模式 (mode={mode}), "
                      f"无力矩输出! 请重新使能或检查故障")
                self._mode_warn_ts[motor] = now

    def _clamp_target_jump(self, joint: str, target: float) -> float:
        """
        把目标位置限制在"当前实测位置 ± MAX_TARGET_JUMP_RAD"内。
        防止上游读数错乱 (CAN 串号) 或 leader 角度跳变时从臂猛甩。
        """
        if self.MAX_TARGET_JUMP_RAD is None:
            return target
        last = self._last_feedback.get(joint)
        if last is None:
            return target
        clamped = float(np.clip(target, last - self.MAX_TARGET_JUMP_RAD,
                                last + self.MAX_TARGET_JUMP_RAD))
        if clamped != target:
            now = time.time()
            if now - self._jump_warn_ts > 1.0:
                print(f"  [SAFETY] [{self.name}] {joint} 目标突变被限幅: "
                      f"目标 {target:+.3f} rad, 当前 {last:+.3f} rad "
                      f"(限幅 ±{self.MAX_TARGET_JUMP_RAD} rad)")
                self._jump_warn_ts = now
        return clamped

    def set_joint_positions(self, positions: np.ndarray, kp: float = 30.0, kd: float = 1.0):
        """
        MIT 模式位置控制，发送 6 个关节目标位置。
        输入为 URDF 约定角度，bus 层的 write_operation_frame 负责转换为电机原始角度。

        Args:
            positions: shape (6,), 目标关节角度 (rad), URDF 约定
            kp: 位置增益
            kd: 阻尼增益
        """
        for i, joint in enumerate(JOINT_NAMES):
            lo, hi = JOINT_LIMITS[joint]
            target = float(np.clip(positions[i], lo, hi))
            target = self._clamp_target_jump(joint, target)
            self.bus.write_operation_frame(joint, position=target, kp=kp, kd=kd)
            pos, _vel, _torque, _temp = self.bus.read_operation_frame(joint)
            self._last_feedback[joint] = pos
            self._check_run_mode(joint)

    def set_gripper(self, position: float, kp: float = 20.0, kd: float = 0.5):
        """
        控制夹爪位置。
        输入为 URDF 约定角度，bus 层的 write_operation_frame 负责转换为电机原始角度。

        Args:
            position: 夹爪目标位置 (rad), 0=关闭, 正值=打开
            kp: 位置增益
            kd: 阻尼增益
        """
        position = self._clamp_target_jump("gripper", position)
        self.bus.write_operation_frame("gripper", position=position, kp=kp, kd=kd)
        pos, _vel, _torque, _temp = self.bus.read_operation_frame("gripper")
        self._last_feedback["gripper"] = pos
        self._check_run_mode("gripper")

    # ── Leader 模式 (低刚度，可手动拖拽) ────────────────

    def set_leader_mode(self, kp: float = 0.0, kd: float = 0.3):
        """
        将机械臂设为 leader 模式：极低刚度 + 轻微阻尼。
        你可以用手自由拖拽机械臂，同时读取关节角度。
        """
        for joint in JOINT_NAMES:
            self.bus.write_operation_frame(joint, position=0, kp=kp, kd=kd)
            self.bus.read_operation_frame(joint)
        self.bus.write_operation_frame("gripper", position=0, kp=kp, kd=kd)
        self.bus.read_operation_frame("gripper")
        print(f"[{self.name}] Leader 模式 (kp={kp}, kd={kd})")

    # ── Follower 模式 (高刚度，跟随目标) ────────────────

    def follow(self, joint_positions: np.ndarray, gripper_pos: float,
               kp: float = 30.0, kd: float = 1.0,
               gripper_kp: float = 20.0, gripper_kd: float = 0.5):
        """
        Follower 模式：跟随给定的关节位置和夹爪位置。
        """
        self.set_joint_positions(joint_positions, kp=kp, kd=kd)
        self.set_gripper(gripper_pos, kp=gripper_kp, kd=gripper_kd)

    def get_cached_full_state(self) -> tuple[np.ndarray, float]:
        """
        返回最近一次 MIT 应答里的实测位置 (6 关节 + 夹爪)。

        follow()/set_joint_positions() 已经在控制帧里读过反馈, 录制时应复用这份缓存,
        不要再调 read_joint_positions_mit(): 那个函数会发 kp=0 的 MIT 帧,
        从臂会立刻失去力矩, 表现为"一按 Enter 电机失能"。
        """
        missing = [j for j in JOINT_NAMES + ["gripper"] if j not in self._last_feedback]
        if missing:
            raise RuntimeError(f"[{self.name}] 尚无电机反馈缓存: {missing}")
        joints = np.array([self._last_feedback[j] for j in JOINT_NAMES], dtype=np.float64)
        return joints, float(self._last_feedback["gripper"])

    # ── 工具方法 ─────────────────────────────────────────

    def get_full_state(self) -> dict:
        """
        读取完整状态: 6 关节角度 + 夹爪位置。

        Returns:
            dict: {"joint_positions": np.ndarray(6,), "gripper_position": float}
        """
        return {
            "joint_positions": self.read_joint_positions_mit(),
            "gripper_position": self.read_gripper_position(),
        }

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *args):
        self.disconnect()
