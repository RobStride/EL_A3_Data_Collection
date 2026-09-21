"""Leader-Follower 遥操作采集, 写出 LeRobotDataset v3.0。"""

import os
import re
import signal
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

os.environ.setdefault("QT_LOGGING_RULES", "*.warning=false")

from data_collection.arm_control import ELA3Arm, ArmConfig, JOINT_NAMES
from data_collection.lerobot_dataset import LeRobotDatasetWriter
from data_collection.realsense_camera import RealSenseCamera, DummyCamera
from data_collection.usb_camera import USBCamera

try:
    from data_collection.zero_torque_leader import ZeroTorqueLeader
    HAS_ZERO_TORQUE = True
except ImportError as _e:
    ZeroTorqueLeader = None
    HAS_ZERO_TORQUE = False
    print(f"WARNING: zero_torque_leader 不可用: {_e}")


@dataclass
class RecordConfig:
    hz: int = 5
    dataset_root: str = "lerobot_dataset"
    robot_type: str = "ela3"
    use_dummy_camera: bool = False

    camera_width: int = 640
    camera_height: int = 480
    camera_fps: int = 30
    camera_serial: str = ""
    realsense_image_key: str = "observation.images.front"

    use_usb_camera: bool = True
    usb_camera_device: int = 0
    usb_camera_width: int = 640
    usb_camera_height: int = 480
    usb_camera_fps: int = 30
    usb_camera_fourcc: str = "MJPG"
    usb_image_key: str = "observation.images.wrist"

    show_preview: bool = True
    preview_fps: float = 30.0
    preview_scale: float = 1.0

    leader_channel: str = "can0"
    leader_motor_ids: list = field(default_factory=lambda: [1, 2, 3, 4, 5, 6, 7])
    leader_motor_model: str = "rs-00"

    follower_channel: str = "can1"
    follower_motor_ids: list = field(default_factory=lambda: [1, 2, 3, 4, 5, 6, 7])
    follower_motor_model: str = "rs-00"

    leader_kp: float = 0.0
    leader_kd: float = 0.3

    follower_kp: float = 100.0
    follower_kd: float = 5.0
    follower_gripper_kp: float = 40.0
    follower_gripper_kd: float = 0.5

    gripper_torque_limit_nm: float = 1.2
    follower_gripper_torque_limit_nm: float = 1.2

    use_zero_torque_gravity: bool = True
    zero_torque_update_rate: float = 200.0
    zero_torque_kd_min: dict = field(default_factory=lambda: {
        1: 0.001, 2: 0.001, 3: 0.001, 4: 0.005, 5: 0.005, 6: 0.005,
    })
    zero_torque_kd_max: dict = field(default_factory=lambda: {
        1: 0.15,  2: 0.25,  3: 0.35,  4: 0.10,  5: 0.05,  6: 0.05,
    })
    zero_torque_gravity_scale: dict = field(default_factory=lambda: {2: 1.3, 3: 1.3, 4: 2.5})
    zero_torque_gripper_kd: float = 0.02
    zero_torque_kd_scale: float = 1.0
    zero_torque_verbose: bool = False


class CameraPreview:
    WINDOW_NAME = "Teleop Cameras"

    def __init__(self, cameras: dict, fps: float = 30.0, scale: float = 1.0):
        self.cameras = {k: v for k, v in cameras.items() if v is not None}
        self.interval = 1.0 / max(fps, 1.0)
        self.scale = max(scale, 0.1)
        self._status = ""
        self._status_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def set_status(self, text: str) -> None:
        with self._status_lock:
            self._status = text

    def start(self) -> None:
        if not self.cameras:
            return
        try:
            import cv2  # noqa: F401
        except ImportError:
            print("WARNING: opencv 未安装, 无法显示相机预览")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="camera_preview")
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join(timeout=2.0)
        self._thread = None

    def _compose(self, cv2) -> Optional[np.ndarray]:
        tiles = []
        target_h = None
        for label, cam in self.cameras.items():
            rgb = cam.peek_rgb()
            if rgb is None:
                continue
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            if target_h is None:
                target_h = bgr.shape[0]
            elif bgr.shape[0] != target_h:
                new_w = int(bgr.shape[1] * target_h / bgr.shape[0])
                bgr = cv2.resize(bgr, (new_w, target_h))
            cv2.putText(bgr, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            tiles.append(bgr)
        if not tiles:
            return None
        canvas = np.hstack(tiles) if len(tiles) > 1 else tiles[0]

        with self._status_lock:
            status = self._status
        if status:
            color = (0, 0, 255) if status.startswith("REC") else (0, 255, 255)
            cv2.putText(canvas, status, (10, canvas.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        return canvas

    def _loop(self) -> None:
        import cv2
        window_sized = False
        try:
            cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            while not self._stop_event.is_set():
                t0 = time.monotonic()
                canvas = self._compose(cv2)
                if canvas is not None:
                    if not window_sized:
                        cv2.resizeWindow(
                            self.WINDOW_NAME,
                            int(canvas.shape[1] * self.scale),
                            int(canvas.shape[0] * self.scale),
                        )
                        window_sized = True
                    cv2.imshow(self.WINDOW_NAME, canvas)
                cv2.waitKey(1)
                remain = self.interval - (time.monotonic() - t0)
                if remain > 0:
                    time.sleep(remain)
        except Exception as e:
            print(f"WARNING: 相机预览窗口异常, 已关闭预览 (录制不受影响): {e}")
        finally:
            try:
                cv2.destroyWindow(self.WINDOW_NAME)
                cv2.waitKey(1)
            except Exception:
                pass


def _make_arm_config(channel: str, motor_ids: list, model: str) -> ArmConfig:
    from data_collection.arm_control import Motor
    motors = {}
    joint_names = ["J1", "J2", "J3", "J4", "J5", "J6", "gripper"]
    for name, mid in zip(joint_names, motor_ids):
        motors[name] = Motor(id=mid, model=model)
    return ArmConfig(channel=channel, motors=motors)


class TeleopRecorder:
    MOTOR_NAMES = ["J1.pos", "J2.pos", "J3.pos", "J4.pos", "J5.pos", "J6.pos", "gripper.pos"]

    def __init__(self, config: RecordConfig):
        self.config = config
        self.leader: Optional[ELA3Arm] = None
        self.zt_leader: Optional["ZeroTorqueLeader"] = None
        self.follower: Optional[ELA3Arm] = None
        self.camera = None
        self.usb_camera = None
        self.preview: Optional[CameraPreview] = None
        self.writer: Optional[LeRobotDatasetWriter] = None
        self._use_zero_torque = False
        self._idle_follow_warn_ts = 0.0
        self._teardown_done = False

    def setup(self):
        use_zero_torque = self.config.use_zero_torque_gravity and HAS_ZERO_TORQUE
        if self.config.use_zero_torque_gravity and not HAS_ZERO_TORQUE:
            print("WARNING: use_zero_torque_gravity=True 但 zero_torque_leader 不可用，回退到低刚度模式")

        if use_zero_torque:
            try:
                from data_collection.zero_torque_leader import ensure_pip_pinocchio
                ensure_pip_pinocchio()
            except Exception as e:
                print(f"WARNING: Pinocchio 不可用 ({e})，回退到低刚度拖拽模式")
                use_zero_torque = False

        leader_cfg = _make_arm_config(
            self.config.leader_channel,
            self.config.leader_motor_ids,
            self.config.leader_motor_model,
        )
        self.leader = ELA3Arm(leader_cfg, name="leader")
        self.leader.connect()
        self.leader.enable_all()

        follower_cfg = _make_arm_config(
            self.config.follower_channel,
            self.config.follower_motor_ids,
            self.config.follower_motor_model,
        )
        self.follower = ELA3Arm(follower_cfg, name="follower")
        self.follower.connect()
        self.follower.enable_all()

        if self.config.use_dummy_camera:
            self.camera = DummyCamera(self.config.camera_width, self.config.camera_height)
        else:
            self.camera = RealSenseCamera(
                width=self.config.camera_width,
                height=self.config.camera_height,
                fps=self.config.camera_fps,
                serial=self.config.camera_serial,
            )
        self.camera.connect()

        if self.config.use_usb_camera:
            if self.config.use_dummy_camera:
                self.usb_camera = DummyCamera(
                    self.config.usb_camera_width,
                    self.config.usb_camera_height,
                )
            else:
                self.usb_camera = USBCamera(
                    device=self.config.usb_camera_device,
                    width=self.config.usb_camera_width,
                    height=self.config.usb_camera_height,
                    fps=self.config.usb_camera_fps,
                    fourcc=self.config.usb_camera_fourcc,
                )
            self.usb_camera.connect()

        if self.config.show_preview:
            preview_cams = {"front (RealSense)": self.camera}
            if self.usb_camera is not None:
                preview_cams["wrist (USB)"] = self.usb_camera
            self.preview = CameraPreview(
                preview_cams,
                fps=self.config.preview_fps,
                scale=self.config.preview_scale,
            )
            self.preview.start()
            self._set_preview_status("IDLE")

        cameras = {
            self.config.realsense_image_key: (self.config.camera_height, self.config.camera_width),
        }
        if self.config.use_usb_camera:
            cameras[self.config.usb_image_key] = (
                self.config.usb_camera_height, self.config.usb_camera_width,
            )
        self.writer = LeRobotDatasetWriter(
            root=self.config.dataset_root,
            fps=int(self.config.hz),
            state_names=self.MOTOR_NAMES,
            cameras=cameras,
            robot_type=self.config.robot_type,
        )

        try:
            self.leader.set_torque_limit("gripper", self.config.gripper_torque_limit_nm)
        except Exception as e:
            print(f"WARNING: leader 写入夹爪 TORQUE_LIMIT 失败: {e}")
        try:
            self.follower.set_torque_limit(
                "gripper", self.config.follower_gripper_torque_limit_nm
            )
        except Exception as e:
            print(f"WARNING: follower 写入夹爪 TORQUE_LIMIT 失败: {e}")

        if use_zero_torque:
            try:
                kd_scale = max(self.config.zero_torque_kd_scale, 0.0)
                scaled_kd_max = {mid: v * kd_scale for mid, v in self.config.zero_torque_kd_max.items()}

                self.zt_leader = ZeroTorqueLeader(
                    arm=self.leader,
                    update_rate_hz=self.config.zero_torque_update_rate,
                    kd_min=self.config.zero_torque_kd_min,
                    kd_max=scaled_kd_max,
                    gravity_scale=self.config.zero_torque_gravity_scale,
                    gripper_kd=self.config.zero_torque_gripper_kd,
                    verbose=self.config.zero_torque_verbose,
                )
                self.zt_leader.start()
                self._use_zero_torque = True
                time.sleep(0.2)
                print(f"[leader] 零力矩重力补偿模式已启用 "
                      f"(update_rate={self.config.zero_torque_update_rate}Hz, "
                      f"kd_scale={kd_scale:.2f})")
            except Exception as e:
                print(f"WARNING: ZeroTorqueLeader 初始化失败 ({e})，回退到低刚度拖拽模式")
                self.zt_leader = None
                self.leader.set_leader_mode(kp=self.config.leader_kp, kd=self.config.leader_kd)
                self._use_zero_torque = False
        else:
            self.leader.set_leader_mode(kp=self.config.leader_kp, kd=self.config.leader_kd)
            self._use_zero_torque = False

        print("\n硬件初始化完成!")
        print(f"  Leader:    {self.config.leader_channel} ({'零力矩重力补偿+夹爪可拖' if self._use_zero_torque else '低刚度拖拽'})")
        print(f"  Follower:  {self.config.follower_channel}")
        print(f"  RealSense: {'Dummy' if self.config.use_dummy_camera else 'Real'}")
        if self.usb_camera is not None:
            print(
                f"  USB cam:   {'Dummy' if self.config.use_dummy_camera else f'device={self.config.usb_camera_device}'}"
            )
        else:
            print(f"  USB cam:   (已禁用)")
        preview_desc = f'开启 (窗口 "{CameraPreview.WINDOW_NAME}")' if self.preview is not None else "关闭"
        print(f"  预览窗口:  {preview_desc}")
        print(f"  频率:      {self.config.hz} Hz")
        print(f"  数据来源:  从臂实测 (state=当前帧, action=下一帧; 主臂不写入数据集)")
        print(
            f"  夹爪跟随:  follower_kp={self.config.follower_gripper_kp}, "
            f"follower_torque_limit={self.config.follower_gripper_torque_limit_nm} Nm"
        )
        try:
            if self._use_zero_torque and self.zt_leader is not None:
                _, leader_gripper, _ = self.zt_leader.read_full_state()
            else:
                leader_gripper = self.leader.read_gripper_position_mit()
            follower_gripper = self.follower.read_gripper_position_mit()
            print(
                f"  夹爪初值:  leader={leader_gripper:+.3f} rad, "
                f"follower={follower_gripper:+.3f} rad"
            )
        except Exception as e:
            print(f"  夹爪初值读取失败: {e}")

    def _set_preview_status(self, text: str) -> None:
        if self.preview is not None:
            self.preview.set_status(text)

    @staticmethod
    def _recover_arm_fault(arm: ELA3Arm, error: Exception) -> bool:
        # 只发 ENABLE 清 fault, 不要 disable+enable (会掉臂)
        arm.bus._flush_rx_buffer()
        match = re.search(r"Motor (\S+) has critical fault", str(error))
        if not match:
            return False
        motor_name = match.group(1)
        try:
            arm.bus.enable(motor_name)
            print(f"  [RECOVER] {arm.name}/{motor_name} 已通过 ENABLE 复位 (扭矩保持)")
            return True
        except RuntimeError as recover_err:
            print(
                f"  [SAFETY] {arm.name}/{motor_name} ENABLE 复位失败: {recover_err}\n"
                f"  [SAFETY] 不会执行 disable+enable (会释放扭矩导致机械臂下坠)\n"
                f"  [SAFETY] 请手动托住机械臂, 排查过载/装配/供电后再重启录制"
            )
            return False

    def _follow_once(self) -> bool:
        if self.leader is None or self.follower is None:
            return False

        try:
            if self._use_zero_torque:
                leader_joints, leader_gripper, _ts = self.zt_leader.read_full_state()
            else:
                leader_joints = self.leader.read_joint_positions_mit()
                leader_gripper = self.leader.read_gripper_position_mit()

            self.follower.follow(
                leader_joints,
                leader_gripper,
                kp=self.config.follower_kp,
                kd=self.config.follower_kd,
                gripper_kp=self.config.follower_gripper_kp,
                gripper_kd=self.config.follower_gripper_kd,
            )
            return True
        except RuntimeError as e:
            now = time.time()
            if now - self._idle_follow_warn_ts > 2.0:
                print(f"  [WARN] 空闲跟随异常: {e}")
                self._idle_follow_warn_ts = now
            if "follower" in str(e).lower() and self.follower is not None:
                self._recover_arm_fault(self.follower, e)
            return False

    def _start_idle_follow(self, stop_event: threading.Event) -> threading.Thread:
        def _loop():
            while not stop_event.is_set():
                self._follow_once()
                time.sleep(0.001)

        thread = threading.Thread(target=_loop, daemon=True, name="idle_follower_follow")
        thread.start()
        return thread

    def _wait_with_idle_follow(self, prompt: str) -> None:
        stop_event = threading.Event()
        thread = self._start_idle_follow(stop_event)
        try:
            input(prompt)
        finally:
            stop_event.set()
            thread.join(timeout=1.0)

    def teardown(self):
        if getattr(self, "_teardown_done", False):
            return
        self._teardown_done = True

        old_sigint = None
        try:
            old_sigint = signal.signal(signal.SIGINT, signal.SIG_IGN)
        except ValueError:
            pass

        try:
            if self.leader is not None or self.follower is not None:
                try:
                    input(
                        "\n[SAFETY] 即将释放电机扭矩 (disable_all)，机械臂会失去支撑。\n"
                        "[SAFETY] 请先用手托住 leader 和 follower 机械臂，确认后按 Enter 继续...\n"
                    )
                except (EOFError, KeyboardInterrupt):
                    print("[SAFETY] 输入中断/不可用，直接执行 disable。请注意托住机械臂!")

            if self.zt_leader is not None:
                try:
                    self.zt_leader.stop()
                except Exception as e:
                    print(f"WARNING: zt_leader 停止异常: {e}")
                self.zt_leader = None

            for arm in (self.leader, self.follower):
                if arm is None:
                    continue
                try:
                    arm.disable_all()
                except Exception as e:
                    print(f"WARNING: [{arm.name}] disable_all 异常: {e}")
                try:
                    arm.disconnect()
                except Exception as e:
                    print(f"WARNING: [{arm.name}] disconnect 异常: {e}")
            self.leader = None
            self.follower = None

            if self.preview is not None:
                try:
                    self.preview.stop()
                except Exception as e:
                    print(f"WARNING: 预览窗口关闭异常: {e}")
                self.preview = None

            for cam in (self.camera, self.usb_camera):
                if cam is None:
                    continue
                try:
                    cam.disconnect()
                except Exception as e:
                    print(f"WARNING: 相机断开异常: {e}")
            self.camera = None
            self.usb_camera = None

            if self.writer is not None:
                try:
                    self.writer.finalize()
                except Exception as e:
                    print(f"WARNING: 数据集 finalize 异常: {e}")
                self.writer = None
        finally:
            if old_sigint is not None:
                signal.signal(signal.SIGINT, old_sigint)

    def record_one_trajectory(self, instruction: str, trajectory_idx: int) -> Optional[int]:
        dt = 1.0 / self.config.hz
        images, states = [], []
        usb_images = [] if self.usb_camera is not None else None

        print(f"\n{'='*50}")
        print(f"轨迹 #{trajectory_idx:04d}")
        print(f"指令: \"{instruction}\"")
        print(f"从臂已开始跟随主臂，请调整到起始位置")
        print(f"按 Enter 开始录制...")
        self._set_preview_status(f"IDLE  #{trajectory_idx:04d}  (Enter = start)")

        start_recording = threading.Event()
        stop_flag = threading.Event()

        def _wait_for_enter():
            input()
            start_recording.set()
            input()
            stop_flag.set()

        t = threading.Thread(target=_wait_for_enter, daemon=True)
        t.start()

        step = 0
        control_cycles = 0
        camera_errors = 0
        motor_errors = 0
        max_camera_errors = 10
        max_motor_errors = 10
        record_interval = dt
        last_record_time = 0.0
        recording_started = False

        try:
            while not stop_flag.is_set():
                try:
                    if self._use_zero_torque:
                        leader_joints, leader_gripper, _ts = self.zt_leader.read_full_state()
                    else:
                        leader_joints = self.leader.read_joint_positions_mit()
                        leader_gripper = self.leader.read_gripper_position_mit()
                except RuntimeError as e:
                    motor_errors += 1
                    print(f"  [WARN] leader 异常 ({motor_errors}/{max_motor_errors}): {e}")
                    if not self._use_zero_torque:
                        recovered = self._recover_arm_fault(self.leader, e)
                        if not recovered:
                            print("  [ERROR] leader 故障无法自动复位，停止 (扭矩保持)")
                            break
                    if motor_errors >= max_motor_errors:
                        print("  [ERROR] 电机连续失败次数过多，停止")
                        break
                    continue

                try:
                    self.follower.follow(
                        leader_joints,
                        leader_gripper,
                        kp=self.config.follower_kp,
                        kd=self.config.follower_kd,
                        gripper_kp=self.config.follower_gripper_kp,
                        gripper_kd=self.config.follower_gripper_kd,
                    )
                except RuntimeError as e:
                    motor_errors += 1
                    print(f"  [WARN] follower 异常 ({motor_errors}/{max_motor_errors}): {e}")
                    recovered = self._recover_arm_fault(self.follower, e)
                    if not recovered:
                        print("  [ERROR] follower 故障无法自动复位，停止 (扭矩保持)")
                        break
                    if motor_errors >= max_motor_errors:
                        print("  [ERROR] 电机连续失败次数过多，停止")
                        break
                    continue

                motor_errors = 0
                control_cycles += 1

                if not recording_started:
                    if start_recording.is_set():
                        recording_started = True
                        print("录制中... (按 Enter 停止)")
                        self._set_preview_status(f"REC  #{trajectory_idx:04d}  step 0")
                        last_record_time = time.time()
                    continue

                now = time.time()
                if now - last_record_time < record_interval:
                    continue
                last_record_time = now

                try:
                    image = self.camera.capture_rgb()
                    usb_image = self.usb_camera.capture_rgb() if self.usb_camera is not None else None
                except RuntimeError as e:
                    camera_errors += 1
                    print(f"  [WARN] 相机异常 ({camera_errors}/{max_camera_errors}): {e}")
                    if camera_errors >= max_camera_errors:
                        print("  [ERROR] 相机连续失败次数过多，停止录制")
                        break
                    continue

                camera_errors = 0

                # 复用 follow() 反馈, 不要再 read_*_mit() (会给从臂发 kp=0)
                try:
                    follower_joints, follower_gripper = self.follower.get_cached_full_state()
                except RuntimeError as e:
                    motor_errors += 1
                    print(f"  [WARN] follower 状态读取异常 ({motor_errors}/{max_motor_errors}): {e}")
                    if motor_errors >= max_motor_errors:
                        print("  [ERROR] 电机连续失败次数过多，停止")
                        break
                    continue

                follower_q = np.append(follower_joints, follower_gripper).astype(np.float32)

                images.append(image)
                if usb_images is not None:
                    usb_images.append(usb_image)
                states.append(follower_q)

                step += 1
                self._set_preview_status(
                    f"REC  #{trajectory_idx:04d}  step {step}  ({step / self.config.hz:.1f}s)"
                )
                if step % 10 == 0:
                    joints_str = ", ".join(f"{q:+.2f}" for q in follower_q[:6])
                    print(f"  step {step:4d} | joints: [{joints_str}]"
                          f" | gripper: {follower_q[6]:+.3f}"
                          f" | ctrl: {control_cycles} cycles")

        except KeyboardInterrupt:
            print("\n录制被中断")

        print(f"录制完成: {step} 步 ({step / self.config.hz:.1f} 秒)")

        if step < 2:
            print("WARNING: 本条轨迹有效帧不足 (需要至少 2 帧才能构成从臂 state→下一帧 action)，跳过保存")
            self._set_preview_status("IDLE  (skipped, too few frames)")
            return None

        self._set_preview_status(f"SAVING  #{trajectory_idx:04d}  ({step} frames)")

        state_arr = np.stack(states)
        action_arr = state_arr[1:]
        state_arr = state_arr[:-1]
        images = images[:-1]
        if usb_images is not None:
            usb_images = usb_images[:-1]

        episode_images = {self.config.realsense_image_key: images}
        if usb_images is not None:
            episode_images[self.config.usb_image_key] = usb_images

        idle_stop = threading.Event()
        idle_thread = self._start_idle_follow(idle_stop)
        try:
            episode_index = self.writer.save_episode(
                task=instruction,
                images=episode_images,
                state=state_arr,
                action=action_arr,
            )
        finally:
            idle_stop.set()
            idle_thread.join(timeout=1.0)

        print(f"已保存 LeRobot episode #{episode_index} ({len(state_arr)} 帧, 从臂 state/action) -> {self.config.dataset_root}/")
        self._set_preview_status(f"IDLE  (saved episode #{episode_index})")
        return episode_index

    def run(self, num_trajectories: int = 10, default_instruction: str = ""):
        failed = False
        try:
            self.setup()

            for i in range(num_trajectories):
                print(f"\n{'='*60}")
                print(f"准备录制第 {i+1}/{num_trajectories} 条轨迹")
                print(f"{'='*60}")

                if default_instruction:
                    instruction = default_instruction
                    print(f"使用默认指令: \"{instruction}\"")
                else:
                    instruction = input("请输入任务指令 (英文): ").strip()
                    if not instruction:
                        instruction = "do the task"

                self.record_one_trajectory(instruction, trajectory_idx=i)

                if i < num_trajectories - 1:
                    print("\n请将物体复位, 准备下一条轨迹...")
                    self._wait_with_idle_follow("按 Enter 继续...（等待期间 follower 持续跟随）")

        except KeyboardInterrupt:
            print("\n\n采集被用户中断")
        except Exception:
            import traceback
            print("\n" + "=" * 60)
            print("[ERROR] 程序异常, 即将安全退出 (先失能电机):")
            traceback.print_exc()
            print("=" * 60)
            failed = True
        finally:
            self.teardown()

        if failed:
            raise SystemExit(1)

        print(f"\n采集完成! LeRobotDataset (v3.0) 保存在: {self.config.dataset_root}/")
        print("可直接拷贝到训练服务器, 用 lerobot 加载 (repo_id 任意, root 指向该目录) 训练 SmolVLA。")


def _parse_joint_map(text: str) -> dict:
    result = {}
    for pair in text.split(","):
        pair = pair.strip()
        if not pair:
            continue
        key, value = pair.split("=")
        result[int(key)] = float(value)
    return result


if __name__ == "__main__":
    from data_collection.zero_torque_leader import ensure_venv_native_env
    ensure_venv_native_env()

    import argparse

    parser = argparse.ArgumentParser(description="EL-A3 Leader-Follower 遥操作数据采集 (LeRobot 格式)")
    parser.add_argument("--num", type=int, default=10, help="采集轨迹数量")
    parser.add_argument("--hz", type=int, default=5, help="采集频率 (lerobot fps, 整数)")
    parser.add_argument("--instruction", type=str, default="", help="默认指令 (为空则每次手动输入)")
    parser.add_argument("--output_dir", type=str, default="lerobot_dataset",
                        help="LeRobot 数据集输出目录 (已存在则续录)")
    parser.add_argument("--leader_channel", type=str, default="can0", help="Leader CAN 通道")
    parser.add_argument("--follower_channel", type=str, default="can1", help="Follower CAN 通道")
    parser.add_argument("--dummy_camera", action="store_true", help="使用假相机 (调试)")
    parser.add_argument("--no_usb_camera", action="store_true", help="禁用 USB 副相机")
    parser.add_argument("--usb_device", type=int, default=10, help="USB 相机 cv2 索引 (0/2/4)")
    parser.add_argument("--usb_width", type=int, default=640, help="USB 相机宽度")
    parser.add_argument("--usb_height", type=int, default=480, help="USB 相机高度")
    parser.add_argument("--usb_fps", type=int, default=30, help="USB 相机帧率")
    parser.add_argument("--no_preview", action="store_true", help="不显示相机实时预览窗口 (无图形界面/SSH 时用)")
    parser.add_argument("--preview_scale", type=float, default=1.0,
                        help="预览窗口缩放比例 (屏幕小可设 0.5; 不影响录制数据)")
    parser.add_argument("--no_zero_torque", action="store_true", help="禁用零力矩重力补偿模式，改用低刚度拖拽")
    parser.add_argument("--gravity_scale", type=str, default="",
                        help='重力补偿放大系数, 如 "2=1.3,3=1.3" (在默认值基础上覆盖对应关节; 下坠就调大)')
    parser.add_argument("--kd_scale", type=float, default=1.0,
                        help="零力矩模式 kd_max 全局缩放 (拖不动调小, 抖/坠调大)")
    parser.add_argument("--gripper_kp", type=float, default=40.0,
                        help="follower 夹爪跟随 kp (夹爪跟不上调大)")
    parser.add_argument("--gripper_torque_limit", type=float, default=1.2,
                        help="follower 夹爪最大夹持力矩 Nm (固件级硬上限; 夹不稳调大, 怕夹坏调小)")
    parser.add_argument("--zt_verbose", action="store_true",
                        help="每秒打印零力矩重力补偿调试信息 (调参用)")
    args = parser.parse_args()

    config = RecordConfig(
        hz=args.hz,
        dataset_root=args.output_dir,
        use_dummy_camera=args.dummy_camera,
        leader_channel=args.leader_channel,
        follower_channel=args.follower_channel,
        use_usb_camera=not args.no_usb_camera,
        usb_camera_device=args.usb_device,
        usb_camera_width=args.usb_width,
        usb_camera_height=args.usb_height,
        usb_camera_fps=args.usb_fps,
        show_preview=not args.no_preview,
        preview_scale=args.preview_scale,
        use_zero_torque_gravity=not args.no_zero_torque,
        zero_torque_kd_scale=args.kd_scale,
        follower_gripper_kp=args.gripper_kp,
        follower_gripper_torque_limit_nm=args.gripper_torque_limit,
        zero_torque_verbose=args.zt_verbose,
    )
    if args.gravity_scale:
        config.zero_torque_gravity_scale.update(_parse_joint_map(args.gravity_scale))

    recorder = TeleopRecorder(config)
    recorder.run(num_trajectories=args.num, default_instruction=args.instruction)
