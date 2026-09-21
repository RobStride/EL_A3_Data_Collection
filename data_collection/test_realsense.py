"""
test_realsense.py

打开 RealSense D435, 采集彩色图 + 左红外图，保存为 PNG。
无 GUI 依赖，适用于远程/无显示器环境。
"""

import os

import numpy as np
import pyrealsense2 as rs
from PIL import Image

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "realsense_output")
WIDTH, HEIGHT, FPS = 640, 480, 6
NUM_FRAMES = 3


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.rgb8, FPS)
    config.enable_stream(rs.stream.infrared, 1, WIDTH, HEIGHT, rs.format.y8, FPS)

    print(f"[RealSense] 启动中 ({WIDTH}x{HEIGHT}@{FPS}fps, 彩色+红外) ...")
    profile = pipeline.start(config)
    device = profile.get_device()
    print(f"[RealSense] 设备: {device.get_info(rs.camera_info.name)}")
    print(f"[RealSense] 序列号: {device.get_info(rs.camera_info.serial_number)}")

    saved = 0
    for attempt in range(NUM_FRAMES + 10):
        if saved >= NUM_FRAMES:
            break
        try:
            frames = pipeline.wait_for_frames(timeout_ms=10000)
        except RuntimeError:
            print(f"[警告] 第 {attempt} 次取帧超时")
            continue

        color_frame = frames.get_color_frame()
        ir_frame = frames.get_infrared_frame(1)
        if not color_frame or not ir_frame:
            continue

        color_img = np.asanyarray(color_frame.get_data())
        ir_img = np.asanyarray(ir_frame.get_data())

        color_path = os.path.join(OUTPUT_DIR, f"color_{saved:04d}.png")
        ir_path = os.path.join(OUTPUT_DIR, f"ir_left_{saved:04d}.png")

        Image.fromarray(color_img).save(color_path)
        Image.fromarray(ir_img, mode="L").save(ir_path)
        print(f"[帧 {saved}] color {color_img.shape} -> {color_path}")
        print(f"         ir    {ir_img.shape} -> {ir_path}")
        saved += 1

    pipeline.stop()
    print(f"[RealSense] 完成，共保存 {saved} 帧到 {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
