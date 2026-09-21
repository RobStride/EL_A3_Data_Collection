"""
test_realsense_usb3.py

USB 3.0 版本：彩色 + 左红外 + 深度, 640x480@30fps。
采集若干帧保存为 PNG, 无 GUI 依赖。

运行前请确认:
  1. 使用 USB 3.0 数据线（非充电线）
  2. 插在 USB 3.0 口（蓝色/橙红色）
"""

import os
import time

import numpy as np
import pyrealsense2 as rs
from PIL import Image

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "realsense_output_usb3")
WIDTH, HEIGHT, FPS = 640, 480, 30
NUM_FRAMES = 5


def check_usb_speed(device):
    """检查 USB 连接速率，返回版本字符串。"""
    try:
        usb_type = device.get_info(rs.camera_info.usb_type_descriptor)
        return usb_type
    except Exception:
        return "未知"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.rgb8, FPS)
    config.enable_stream(rs.stream.infrared, 1, WIDTH, HEIGHT, rs.format.y8, FPS)
    config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)

    print(f"[RealSense] 启动中 ({WIDTH}x{HEIGHT}@{FPS}fps, 彩色+红外+深度) ...")

    try:
        profile = pipeline.start(config)
    except RuntimeError as e:
        print(f"[错误] 无法启动: {e}")
        print("[提示] 这个配置需要 USB 3.0，请检查线缆和接口")
        return

    device = profile.get_device()
    usb_ver = check_usb_speed(device)
    name = device.get_info(rs.camera_info.name)
    sn = device.get_info(rs.camera_info.serial_number)

    print(f"[RealSense] 设备: {name}")
    print(f"[RealSense] 序列号: {sn}")
    print(f"[RealSense] USB 类型: {usb_ver}")

    if usb_ver.startswith("2"):
        print("[警告] 当前为 USB 2.x 连接，此配置可能不稳定！")
    else:
        print(f"[OK] USB {usb_ver} 连接，带宽充足")

    # 暖机
    print("[RealSense] 暖机中 ...")
    for _ in range(30):
        try:
            pipeline.wait_for_frames(timeout_ms=3000)
        except RuntimeError:
            pass
    print("[RealSense] 暖机完成，开始采集")

    saved = 0
    for attempt in range(NUM_FRAMES + 10):
        if saved >= NUM_FRAMES:
            break
        try:
            frames = pipeline.wait_for_frames(timeout_ms=5000)
        except RuntimeError:
            print(f"[警告] 第 {attempt} 次取帧超时")
            continue

        color_frame = frames.get_color_frame()
        ir_frame = frames.get_infrared_frame(1)
        depth_frame = frames.get_depth_frame()
        if not color_frame or not ir_frame or not depth_frame:
            continue

        color_img = np.asanyarray(color_frame.get_data())
        ir_img = np.asanyarray(ir_frame.get_data())
        depth_img = np.asanyarray(depth_frame.get_data())

        # 深度图归一化到 8bit 方便查看
        depth_vis = (depth_img / depth_img.max() * 255).astype(np.uint8) if depth_img.max() > 0 else depth_img.astype(np.uint8)

        color_path = os.path.join(OUTPUT_DIR, f"color_{saved:04d}.png")
        ir_path = os.path.join(OUTPUT_DIR, f"ir_left_{saved:04d}.png")
        depth_path = os.path.join(OUTPUT_DIR, f"depth_{saved:04d}.png")

        Image.fromarray(color_img).save(color_path)
        Image.fromarray(ir_img, mode="L").save(ir_path)
        Image.fromarray(depth_vis, mode="L").save(depth_path)

        print(f"[帧 {saved}] color {color_img.shape} | ir {ir_img.shape} | depth {depth_img.shape} (max={depth_img.max()}mm)")
        saved += 1
        time.sleep(0.05)

    pipeline.stop()
    print(f"[RealSense] 完成，共保存 {saved} 帧到 {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
