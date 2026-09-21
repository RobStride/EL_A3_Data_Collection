# EL-A3 LeRobot 数据采集

Leader-Follower 遥操作采集，直接写出 LeRobotDataset v3.0，可拿去训练 SmolVLA 等。

## 操作系统

- **Linux（Ubuntu 22.04 / 24.04）**
- 需要 SocketCAN（主臂 `can0`、从臂 `can1`）
- 需要图形界面才能开相机预览；SSH / 无桌面时加 `--no_preview`

## Python 环境

- **Python 3.10+**（本机原先用 3.11）
- 建议单独建虚拟环境：

```bash
cd /home/ubuntu/vla_data_collection
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

主要依赖：`numpy`、`pandas`、`pyarrow`、`av`、`opencv-python`、`pyrealsense2`、`python-can`、`pin`（Pinocchio）。

零力矩重力补偿用本仓库的 `el_a3_sdk/` 和 `pip` 装的 `pin`（Pinocchio）。没有 Pinocchio 就加 `--no_zero_torque`，改用低刚度拖拽。

## 采集前

1. 接好主臂 / 从臂 CAN、RealSense、USB 腕部相机。
2. 拉起 CAN（比特率按实际电机，常见 1Mbps）：

```bash
sudo ip link set can0 type can bitrate 1000000
sudo ip link set can0 up
sudo ip link set can1 type can bitrate 1000000
sudo ip link set can1 up
```

3. USB 相机默认索引是 `--usb_device 10`，对不上就改。

## 采集流程

在项目根目录运行：

```bash
source .venv/bin/activate
python -m data_collection.teleop_record --num 10 --hz 5 --output_dir lerobot_dataset
```

每条轨迹：

1. 输入英文任务指令（或用 `--instruction "..."` 固定一条）。
2. 拖主臂，从臂会跟着；摆好起始姿态后按 **Enter** 开始录。
3. 做完动作再按 **Enter** 停止，自动写成 LeRobot episode。
4. 把场景复位，按 **Enter** 录下一条。
5. `Ctrl+C` 可提前结束；退出前会先让你托住机械臂，再失能电机。

数据写在 `lerobot_dataset/`（目录已存在则续录）。拷到训练机后，用 lerobot 把 `root` 指到这个目录即可。微调 SmolVLA 的命令和 TensorBoard 包装见 [training/](training/)。

常用参数：

| 参数 | 说明 |
| --- | --- |
| `--num` | 本轮要录几条 |
| `--hz` | 采样频率（整数 fps） |
| `--instruction` | 固定任务文本，不再每次手输 |
| `--no_usb_camera` | 只录 RealSense |
| `--no_preview` | 关掉预览窗口 |
| `--no_zero_torque` | 不用重力补偿 |
| `--dummy_camera` | 假相机，只测机械臂 |

