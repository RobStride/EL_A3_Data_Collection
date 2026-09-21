"""
test_realsense_show.py

USB 3.0 版本: 彩色 + 左红外 + 深度, 640x480@30fps. 
实时 imshow 显示, 按 q 退出, 按 s 保存当前帧. 
"""

import os

import cv2
import numpy as np
import pyrealsense2 as rs

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "realsense_output_show")
WIDTH, HEIGHT, FPS = 640, 480, 30


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
    config.enable_stream(rs.stream.infrared, 1, WIDTH, HEIGHT, rs.format.y8, FPS)
    config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)

    print(f"[RealSense] 启动中 ({WIDTH}x{HEIGHT}@{FPS}fps, 彩色+红外+深度) ...")

    try:
        profile = pipeline.start(config)
    except RuntimeError as e:
        print(f"[错误] 无法启动: {e}")
        print("[提示] 需要 USB 3.0 线缆和接口")
        return

    device = profile.get_device()
    try:
        usb_ver = device.get_info(rs.camera_info.usb_type_descriptor)
    except Exception:
        usb_ver = "未知"

    print(f"[RealSense] 设备: {device.get_info(rs.camera_info.name)}")
    print(f"[RealSense] USB 类型: {usb_ver}")
    print("[RealSense] 按 q 退出, 按 s 保存当前帧")

    for _ in range(30):
        try:
            pipeline.wait_for_frames(timeout_ms=3000)
        except RuntimeError:
            pass

    frame_count = 0
    try:
        while True:
            try:
                frames = pipeline.wait_for_frames(timeout_ms=5000)
            except RuntimeError:
                continue

            color_frame = frames.get_color_frame()
            ir_frame = frames.get_infrared_frame(1)
            depth_frame = frames.get_depth_frame()
            if not color_frame or not ir_frame or not depth_frame:
                continue

            color_img = np.asanyarray(color_frame.get_data())
            ir_img = np.asanyarray(ir_frame.get_data())
            depth_img = np.asanyarray(depth_frame.get_data())

            ir_bgr = cv2.cvtColor(ir_img, cv2.COLOR_GRAY2BGR)
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_img, alpha=0.03),
                cv2.COLORMAP_JET,
            )

            cv2.putText(color_img, "Color", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(ir_bgr, "IR Left", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(depth_colormap, "Depth", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            combined = np.hstack([color_img, ir_bgr, depth_colormap])
            cv2.imshow("RealSense - Color | IR Left | Depth", combined)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                cv2.imwrite(os.path.join(OUTPUT_DIR, f"color_{frame_count:04d}.png"), color_img)
                cv2.imwrite(os.path.join(OUTPUT_DIR, f"ir_left_{frame_count:04d}.png"), ir_img)
                cv2.imwrite(os.path.join(OUTPUT_DIR, f"depth_{frame_count:04d}.png"), depth_colormap)
                print(f"[保存] 帧 {frame_count}")

            frame_count += 1

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print(f"[RealSense] 已关闭，共显示 {frame_count} 帧")


if __name__ == "__main__":
    main()
   