# SmolVLA 训练

采集脚本只负责写出 [LeRobot dataset v3.0](https://github.com/huggingface/lerobot)。微调本身完全走 HuggingFace **LeRobot**，本目录只提供 TensorBoard 日志包装和一条可直接跑的命令。

不改 lerobot 源码。采集环境（本仓库 `.venv`）和训练环境是两套，训练机单独装 lerobot。

## 环境

需要 GPU 机器。建议用独立虚拟环境：

```bash
git clone https://github.com/huggingface/lerobot.git
cd lerobot
python -m venv .venv && source .smolvla_venv/bin/activate
pip install -e ".[smolvla]"
pip install tensorboard
```

本仓库实测过 **LeRobot 0.5.2**（`lerobot/smolvla_base`）。数据集已是 v3.0，采集机上不用再做格式转换。

把本仓库采好的 `lerobot_dataset/` 拷到训练机，`--dataset.root` 指过去即可。

## 数据集和 SmolVLA 相机键

本仓库默认写出：

- `observation.images.front`（RealSense）
- `observation.images.wrist`（USB 腕部相机）
- `action` / `observation.state`：7 维（J1–J6 + gripper）

SmolVLA 内部相机名固定是 `camera1/camera2/camera3`，所以命令里用 `--rename_map` 做映射，并用 `--policy.empty_cameras=1` 补第三个空位。`--dataset.repo_id` 只是本地标识，不会去 Hub 下载。

`--wandb.enable=true` 只用来打开 lerobot 的日志分支；本包装器把数据写到本地 TensorBoard，不会连 wandb。

## 训练

在能 `import lerobot` 的环境里执行（把 `--dataset.root` 换成数据集目录）：

```bash
unset ALL_PROXY all_proxy          # httpx 不支持 socks:// 代理

python training/lerobot_train_tb.py \
  --wandb.enable=true \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=local/ela3_dataset \
  --dataset.root=lerobot_dataset \
  --rename_map='{"observation.images.front":"observation.images.camera1","observation.images.wrist":"observation.images.camera2"}' \
  --policy.empty_cameras=1 \
  --policy.freeze_vision_encoder=false \
  --policy.train_expert_only=false \
  --policy.n_action_steps=1 \
  --policy.push_to_hub=false \
  --batch_size=32 \
  --steps=15000 \
  --save_freq=5000 \
  --log_freq=10 \
  --output_dir=outputs/train/ela3_smolvla \
  --job_name=ela3_smolvla
```

| 参数 | 说明 |
| --- | --- |
| `--policy.path` | 微调起点，官方 `lerobot/smolvla_base` |
| `--dataset.root` | 采集得到的数据集目录 |
| `--log_freq` | 每隔多少步写一次 TensorBoard，默认 200 |
| `--batch_size` / `--steps` | 显存不够就把 batch 降到 12–16 |
| `--save_freq` | checkpoint 间隔 |

`--output_dir` 必须是新目录，已存在且不 `--resume` 时 lerobot 会直接报错退出。

## 看曲线

```bash
tensorboard --logdir outputs/train/ela3_smolvla/tensorboard --port 6006 --bind_all
```

浏览器打开 `http://localhost:6006`，Scalars 里选 `train/loss`。

checkpoint 在 `outputs/train/<job_name>/checkpoints/<step>/pretrained_model/`，部署时把这个目录当作 `--policy.path`。
