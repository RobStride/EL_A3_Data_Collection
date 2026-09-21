#!/usr/bin/env python
"""带 TensorBoard 日志的 lerobot-train 包装入口（不修改 lerobot 源码）。

lerobot 训练循环里所有指标都走 WandBLogger。这里在启动前把它替换成
TensorBoard 实现，指标写入 <output_dir>/tensorboard，不会连接 wandb。

用法与 lerobot-train 完全一致，但必须加 --wandb.enable=true 以激活日志分支。
"""

import atexit
import logging
from pathlib import Path


class TensorBoardLogger:
    """与 lerobot WandBLogger 同接口的 TensorBoard 实现。"""

    def __init__(self, cfg):
        from torch.utils.tensorboard import SummaryWriter

        log_dir = Path(cfg.output_dir) / "tensorboard"
        self.writer = SummaryWriter(log_dir=str(log_dir))
        atexit.register(self.writer.close)
        logging.info(f"TensorBoard 日志目录: {log_dir}")

    def log_dict(self, d: dict, step=None, mode="train", custom_step_key=None):
        if step is None and custom_step_key is not None:
            step = d.get(custom_step_key)
        for k, v in d.items():
            if isinstance(v, (int, float)):
                self.writer.add_scalar(f"{mode}/{k}", v, step)
        self.writer.flush()

    def log_policy(self, checkpoint_dir):
        """checkpoint 已由训练循环存到磁盘，这里无需上传 artifact。"""

    def log_video(self, video_path, step, mode="train"):
        """评估视频已保存在 output_dir/eval 下，TensorBoard 不重复记录。"""


def main():
    import lerobot.scripts.lerobot_train as lerobot_train

    lerobot_train.WandBLogger = TensorBoardLogger
    lerobot_train.main()


if __name__ == "__main__":
    main()
