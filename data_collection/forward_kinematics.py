"""
forward_kinematics.py

基于 EL-A3 URDF 的正运动学计算。
从 6 个关节角度 → 末端执行器位姿 [x, y, z, roll, pitch, yaw]。
"""

import numpy as np
from typing import Tuple


def _rotation_x(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([
        [1, 0,  0, 0],
        [0, c, -s, 0],
        [0, s,  c, 0],
        [0, 0,  0, 1],
    ])


def _rotation_y(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([
        [ c, 0, s, 0],
        [ 0, 1, 0, 0],
        [-s, 0, c, 0],
        [ 0, 0, 0, 1],
    ])


def _rotation_z(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([
        [c, -s, 0, 0],
        [s,  c, 0, 0],
        [0,  0, 1, 0],
        [0,  0, 0, 1],
    ])


def _translation(x: float, y: float, z: float) -> np.ndarray:
    T = np.eye(4)
    T[0, 3] = x
    T[1, 3] = y
    T[2, 3] = z
    return T


def _rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """URDF 使用 rpy = (roll_x, pitch_y, yaw_z), 乘法顺序 Rz * Ry * Rx"""
    return _rotation_z(yaw) @ _rotation_y(pitch) @ _rotation_x(roll)


def _transform_from_origin(xyz: Tuple, rpy: Tuple) -> np.ndarray:
    """从 URDF joint origin 的 xyz + rpy 构造 4x4 齐次变换矩阵"""
    T = np.eye(4)
    T[:3, :3] = _rpy_to_matrix(rpy[0], rpy[1], rpy[2])[:3, :3]
    T[0, 3] = xyz[0]
    T[1, 3] = xyz[1]
    T[2, 3] = xyz[2]
    return T


def rotation_matrix_to_rpy(R: np.ndarray) -> np.ndarray:
    """
    从 3x3 旋转矩阵提取 roll, pitch, yaw (ZYX 欧拉角)。

    Returns:
        np.ndarray: [roll, pitch, yaw] in radians
    """
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0.0

    return np.array([roll, pitch, yaw])


# ── EL-A3 关节变换参数 (从 URDF 提取) ─────────────────────
#
# 运动链: base → L1 → L2 → L3 → L4 → L5 → L6 → end_effector
#
# 每个关节: 先施加 origin transform (固定偏移+旋转), 再绕 Z 轴旋转 q_i
#
# URDF 中所有 joint axis 都是 z = (0,0,1)

# (xyz, rpy) — 直接从 URDF joint origin 提取
JOINT_ORIGINS = [
    # L1_joint: base → link1
    ((0.0, 0.0, 0.0054),      (0.0, 0.0, 0.0)),
    # L2_joint: link1 → link2_3 (注意 rpy 含 π/2 绕 X 轴旋转)
    ((-0.0176846, 0.0, 0.0606), (np.pi / 2, 0.0, 0.0)),
    # L3_joint: link2_3 → link3
    ((0.19, 0.0, 0.0),         (0.0, 0.0, 0.0)),
    # L4_joint: link3 → link4_5
    ((-0.15, 0.06, 0.0),       (0.0, 0.0, 0.0)),
    # L5_joint: link4_5 → part_9
    ((-0.0492, 0.038, 0.0),    (np.pi / 2, 0.0, 0.0)),
    # L6_joint: part_9 → end_effector
    ((0.00805, 0.0, 0.038),    (np.pi / 2, 0.0, -np.pi / 2)),
]

# 末端执行器相对于 L6 link 的额外偏移 (到夹爪中心)
# 根据 URDF 中 l5_l6_urdf_asm 的 mesh 估算，夹爪中心在 Z 方向约 0.0755m 处
EEF_OFFSET = _translation(0.0, 0.0, 0.0755)


def forward_kinematics(joint_angles: np.ndarray) -> np.ndarray:
    """
    EL-A3 正运动学：从 6 个关节角度计算末端执行器位姿。

    Args:
        joint_angles: np.ndarray, shape (6,), 关节角度 [q1..q6] (rad)

    Returns:
        np.ndarray: shape (6,), [x, y, z, roll, pitch, yaw]
                    位置单位: 米 (m), 角度单位: 弧度 (rad)
    """
    assert len(joint_angles) == 6, f"需要 6 个关节角度, 收到 {len(joint_angles)}"

    T = np.eye(4)  # 从基座开始累积

    for i in range(6):
        xyz, rpy = JOINT_ORIGINS[i]
        T_origin = _transform_from_origin(xyz, rpy)
        T_joint = _rotation_z(joint_angles[i])
        T = T @ T_origin @ T_joint

    # 加上末端偏移
    T = T @ EEF_OFFSET

    # 提取位置和姿态
    position = T[:3, 3]
    rpy = rotation_matrix_to_rpy(T[:3, :3])

    return np.concatenate([position, rpy])


def forward_kinematics_matrix(joint_angles: np.ndarray) -> np.ndarray:
    """
    返回完整 4x4 齐次变换矩阵 (用于更精确的计算)。
    """
    T = np.eye(4)
    for i in range(6):
        xyz, rpy = JOINT_ORIGINS[i]
        T_origin = _transform_from_origin(xyz, rpy)
        T_joint = _rotation_z(joint_angles[i])
        T = T @ T_origin @ T_joint
    T = T @ EEF_OFFSET
    return T


# ── 测试 ───────────────────────────────────────────────────

if __name__ == "__main__":
    print("EL-A3 正运动学测试")
    print("=" * 50)

    # 零位
    q_zero = np.zeros(6)
    pose = forward_kinematics(q_zero)
    print(f"\n零位 q={q_zero}")
    print(f"  EEF 位置: x={pose[0]:.4f}, y={pose[1]:.4f}, z={pose[2]:.4f} (m)")
    print(f"  EEF 姿态: roll={np.degrees(pose[3]):.1f}, pitch={np.degrees(pose[4]):.1f}, yaw={np.degrees(pose[5]):.1f} (deg)")

    # 仅 J1 转 45°
    q_test = np.array([np.pi / 4, 0, 0, 0, 0, 0])
    pose = forward_kinematics(q_test)
    print(f"\nJ1=45° q={np.degrees(q_test)}")
    print(f"  EEF 位置: x={pose[0]:.4f}, y={pose[1]:.4f}, z={pose[2]:.4f} (m)")
    print(f"  EEF 姿态: roll={np.degrees(pose[3]):.1f}, pitch={np.degrees(pose[4]):.1f}, yaw={np.degrees(pose[5]):.1f} (deg)")

    # 仅 J2 转 45°
    q_test = np.array([0, np.pi / 4, 0, 0, 0, 0])
    pose = forward_kinematics(q_test)
    print(f"\nJ2=45° q={np.degrees(q_test)}")
    print(f"  EEF 位置: x={pose[0]:.4f}, y={pose[1]:.4f}, z={pose[2]:.4f} (m)")
    print(f"  EEF 姿态: roll={np.degrees(pose[3]):.1f}, pitch={np.degrees(pose[4]):.1f}, yaw={np.degrees(pose[5]):.1f} (deg)")
