"""
lerobot_dataset.py

不依赖 lerobot 框架的 LeRobotDataset v3.0 写入器。

在采集机上直接生成 lerobot (>=0.4, 数据集格式 codebase_version "v3.0") 可以直接
加载/训练 (如 SmolVLA) 的数据集目录，训练服务器上无需任何格式转换:

    <root>/
    ├── meta/
    │   ├── info.json                             # schema + 计数 + 路径模板
    │   ├── stats.json                            # 全局归一化统计
    │   ├── tasks.parquet                         # 任务文本 -> task_index
    │   └── episodes/chunk-000/file-000.parquet   # 每条 episode 的元数据 + 统计
    ├── data/chunk-000/file-000.parquet           # 逐帧表格数据 (每 episode 一个文件)
    └── videos/<camera_key>/chunk-000/file-000.mp4

格式细节 (列名 / stats 形状 / 路径模板) 与 lerobot v0.6.0 源码逐一对齐:
- data parquet 列: action, observation.state, timestamp, frame_index,
  episode_index, index, task_index
- episodes parquet 列: episode_index, tasks, length, data/*, dataset_from_index,
  dataset_to_index, meta/episodes/*, videos/<key>/*, stats/<feature>/<stat>
- 图像 stats 形状 (3,1,1) 且归一化到 [0,1]; 向量 stats 形状 (D,); 标量 (1,)

仅依赖: numpy, pyarrow, pandas, av (PyAV, 用 libx264 编码 mp4)。
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

CODEBASE_VERSION = "v3.0"
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_DATA_FILE_SIZE_IN_MB = 100
DEFAULT_VIDEO_FILE_SIZE_IN_MB = 200

DATA_PATH = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
VIDEO_PATH = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
EPISODES_PATH = "meta/episodes/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
INFO_PATH = "meta/info.json"
STATS_PATH = "meta/stats.json"
TASKS_PATH = "meta/tasks.parquet"

# lerobot DEFAULT_FEATURES: 由写入器自动填充，调用者不需要提供
DEFAULT_FEATURES = {
    "timestamp": {"dtype": "float32", "shape": [1], "names": None},
    "frame_index": {"dtype": "int64", "shape": [1], "names": None},
    "episode_index": {"dtype": "int64", "shape": [1], "names": None},
    "index": {"dtype": "int64", "shape": [1], "names": None},
    "task_index": {"dtype": "int64", "shape": [1], "names": None},
}

QUANTILES = {"q01": 0.01, "q10": 0.10, "q50": 0.50, "q90": 0.90, "q99": 0.99}
STAT_KEYS = ["min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99"]

# 每条 episode 图像统计最多采样的帧数 (与 lerobot estimate_num_samples 量级一致)
MAX_IMAGE_STAT_SAMPLES = 100


# ── 视频编码 ─────────────────────────────────────────────────


def encode_video_h264(frames: Sequence[np.ndarray], path: Path, fps: int,
                      crf: int = 23, gop: int = 2, preset: str = "fast") -> float:
    """
    将 RGB uint8 帧序列编码为 mp4 (libx264, yuv420p)。

    gop=2 与 lerobot 默认一致: 关键帧间隔小, 训练时随机 seek 解码快。

    Returns:
        视频时长 (秒)。
    """
    from fractions import Fraction

    import av

    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]

    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=int(fps))
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        stream.codec_context.time_base = Fraction(1, int(fps))
        stream.options = {"crf": str(crf), "g": str(gop), "preset": preset}

        for i, frame_np in enumerate(frames):
            frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(frame_np), format="rgb24")
            frame.pts = i
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)

    return len(frames) / float(fps)


# ── 统计 (与 lerobot compute_stats 形状对齐) ────────────────


def _vector_stats(data: np.ndarray) -> Dict[str, np.ndarray]:
    """(T, D) -> 每维统计, 形状 (D,); (T,) -> 形状 (1,)。"""
    arr = np.asarray(data, dtype=np.float64)
    scalar = arr.ndim == 1
    if scalar:
        arr = arr[:, None]
    stats = {
        "min": arr.min(axis=0),
        "max": arr.max(axis=0),
        "mean": arr.mean(axis=0),
        "std": arr.std(axis=0),
        "count": np.array([len(arr)]),
    }
    for name, q in QUANTILES.items():
        stats[name] = np.quantile(arr, q, axis=0)
    return stats


def _image_stats(frames: Sequence[np.ndarray]) -> Dict[str, np.ndarray]:
    """采样帧计算每通道统计, 归一化到 [0,1], 形状 (3,1,1)。"""
    num = len(frames)
    indices = np.unique(np.linspace(0, num - 1, min(num, MAX_IMAGE_STAT_SAMPLES)).astype(int))
    # (N, H, W, 3) -> (N*H*W, 3), float64 避免溢出
    pixels = np.stack([frames[i] for i in indices]).reshape(-1, 3).astype(np.float64) / 255.0
    stats = {
        "min": pixels.min(axis=0).reshape(3, 1, 1),
        "max": pixels.max(axis=0).reshape(3, 1, 1),
        "mean": pixels.mean(axis=0).reshape(3, 1, 1),
        "std": pixels.std(axis=0).reshape(3, 1, 1),
        "count": np.array([len(indices)]),
    }
    for name, q in QUANTILES.items():
        stats[name] = np.quantile(pixels, q, axis=0).reshape(3, 1, 1)
    return stats


def _aggregate_feature_stats(stats_list: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    """多条 episode 同一特征的统计聚合 (与 lerobot aggregate_feature_stats 相同的数学)。"""
    means = np.stack([s["mean"] for s in stats_list])
    variances = np.stack([np.asarray(s["std"]) ** 2 for s in stats_list])
    counts = np.stack([s["count"] for s in stats_list])
    total_count = counts.sum(axis=0)

    while counts.ndim < means.ndim:
        counts = np.expand_dims(counts, axis=-1)

    total_mean = (means * counts).sum(axis=0) / total_count
    delta_means = means - total_mean
    total_variance = ((variances + delta_means ** 2) * counts).sum(axis=0) / total_count

    aggregated = {
        "min": np.min(np.stack([s["min"] for s in stats_list]), axis=0),
        "max": np.max(np.stack([s["max"] for s in stats_list]), axis=0),
        "mean": total_mean,
        "std": np.sqrt(np.maximum(total_variance, 0.0)),
        "count": total_count,
    }
    for q_key in QUANTILES:
        if all(q_key in s for s in stats_list):
            q_values = np.stack([s[q_key] for s in stats_list])
            aggregated[q_key] = (q_values * counts).sum(axis=0) / total_count
    return aggregated


def aggregate_stats(stats_list: List[Dict[str, Dict[str, np.ndarray]]]) -> Dict[str, Dict[str, np.ndarray]]:
    keys = {k for stats in stats_list for k in stats}
    return {
        k: _aggregate_feature_stats([s[k] for s in stats_list if k in s])
        for k in keys
    }


# ── 写入器 ───────────────────────────────────────────────────


class LeRobotDatasetWriter:
    """
    直接生成 LeRobotDataset v3.0 目录结构的最小写入器。

    - 每条 episode 一个 data parquet 文件 + 每相机一个 mp4 (合法的 v3.0 布局,
      metadata 中的 chunk/file 索引逐条指向对应文件)
    - 支持 resume: root 已存在时继续追加 episode
    - finalize() 关闭 episodes 元数据的 parquet writer, 录制结束必须调用
    """

    def __init__(
        self,
        root: str,
        fps: int,
        state_names: List[str],
        cameras: Dict[str, tuple],   # {camera_key: (height, width)}
        robot_type: str = "ela3",
        video_crf: int = 23,
    ):
        self.root = Path(root)
        self.fps = int(fps)
        self.state_names = list(state_names)
        self.cameras = dict(cameras)
        self.robot_type = robot_type
        self.video_crf = video_crf

        dim = len(self.state_names)
        self.features: Dict[str, dict] = {
            "action": {"dtype": "float32", "shape": [dim], "names": self.state_names},
            "observation.state": {"dtype": "float32", "shape": [dim], "names": self.state_names},
        }
        for cam_key, (h, w) in self.cameras.items():
            self.features[cam_key] = {
                "dtype": "video",
                "shape": [h, w, 3],
                "names": ["height", "width", "channels"],
                "info": {
                    "video.height": h,
                    "video.width": w,
                    "video.codec": "h264",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "video.fps": self.fps,
                    "video.channels": 3,
                    "has_audio": False,
                    "is_depth_map": False,
                },
            }
        self.features.update(DEFAULT_FEATURES)

        self._data_schema = pa.schema(
            [("action", pa.list_(pa.float32(), dim)),
             ("observation.state", pa.list_(pa.float32(), dim)),
             ("timestamp", pa.float32()),
             ("frame_index", pa.int64()),
             ("episode_index", pa.int64()),
             ("index", pa.int64()),
             ("task_index", pa.int64())]
        )

        # 当前 episodes 元数据文件 (meta/episodes/chunk-XXX/file-YYY.parquet) 的全部行。
        # 每保存一条 episode 就把整表原子重写一次 (写 .tmp 再 os.replace),
        # 任何时刻磁盘上的 parquet 都是完整可读的, 程序被强杀也不会损坏元数据。
        # (旧实现用常驻 ParquetWriter, 只有 finalize() 时才写 footer, 一旦被 kill 整个文件报废)
        self._episode_rows: List[dict] = []
        self._episode_stats: List[Dict[str, Dict[str, np.ndarray]]] = []

        if (self.root / INFO_PATH).exists():
            self._resume()
        else:
            self._create()

    # ── 初始化 / resume ──────────────────────────────────

    def _create(self):
        (self.root / "meta").mkdir(parents=True, exist_ok=True)
        self.total_episodes = 0
        self.total_frames = 0
        self.tasks: Dict[str, int] = {}
        self._meta_chunk_idx, self._meta_file_idx = 0, 0
        self._write_info()

    def _resume(self):
        info = json.loads((self.root / INFO_PATH).read_text(encoding="utf-8"))
        if info["codebase_version"] != CODEBASE_VERSION:
            raise ValueError(f"已有数据集版本 {info['codebase_version']} != {CODEBASE_VERSION}")
        if info["fps"] != self.fps:
            raise ValueError(f"已有数据集 fps={info['fps']} 与当前 {self.fps} 不一致")
        existing_keys = set(info["features"].keys())
        if existing_keys != set(self.features.keys()):
            raise ValueError(
                f"已有数据集特征与当前配置不一致:\n  已有: {sorted(existing_keys)}\n  当前: {sorted(self.features)}"
            )

        self.total_episodes = info["total_episodes"]
        self.total_frames = info["total_frames"]

        # tasks.parquet 在保存第一条 episode 时才创建;
        # 上次运行如果在录制前就中断, 数据集目录里只有 info.json
        tasks_path = self.root / TASKS_PATH
        if tasks_path.exists():
            tasks_df = pd.read_parquet(tasks_path)
            self.tasks = {task: int(row["task_index"]) for task, row in tasks_df.iterrows()}
        else:
            self.tasks = {}

        # 重建每条 episode 的统计 (聚合 stats.json 用), 并把最后一个元数据文件的行
        # 载入内存: 之后新 episode 继续追加到该文件 (整表原子重写)。
        episodes_dir = self.root / "meta" / "episodes"
        ep_files = sorted(episodes_dir.glob("*/*.parquet"))
        last_chunk, last_file = 0, 0
        all_rows: List[dict] = []
        for f in ep_files:
            try:
                rows = pq.read_table(f).to_pylist()
            except (pa.ArrowInvalid, pa.ArrowIOError, OSError) as e:
                raise RuntimeError(
                    f"episodes 元数据文件损坏, 无法续录: {f}\n"
                    f"  原因: {e}\n"
                    f"  这通常是上次录制被强杀 (kill -9 / SIGQUIT / 崩溃) 导致。\n"
                    f"  修复: python data_collection/rebuild_lerobot_dataset.py "
                    f"--src {self.root} --dst {self.root}_rebuilt\n"
                    f"        然后把 {self.root} 改名备份, 再把 {self.root}_rebuilt 改名为 {self.root}"
                ) from e
            all_rows.extend(rows)
            if rows:
                last_chunk = int(rows[-1]["meta/episodes/chunk_index"])
                last_file = int(rows[-1]["meta/episodes/file_index"])

        # 只保留 info.json 认可的 episode (防御: 上次在写完元数据行、还没更新 info.json 时被杀)
        valid_rows = [r for r in all_rows if int(r["episode_index"]) < self.total_episodes]
        if len(valid_rows) != len(all_rows):
            print(f"[LeRobotDatasetWriter] WARN: 丢弃 {len(all_rows) - len(valid_rows)} 条 "
                  f"info.json 未记录的 episode 元数据行")
        if len(valid_rows) != self.total_episodes:
            raise RuntimeError(
                f"episodes 元数据行数 ({len(valid_rows)}) 与 info.json total_episodes "
                f"({self.total_episodes}) 不一致, 数据集不完整。\n"
                f"  修复: python data_collection/rebuild_lerobot_dataset.py "
                f"--src {self.root} --dst {self.root}_rebuilt"
            )

        for row in valid_rows:
            ep_stats: Dict[str, Dict[str, np.ndarray]] = {}
            for col, value in row.items():
                if not col.startswith("stats/"):
                    continue
                _, feat, stat = col.split("/", 2)
                ep_stats.setdefault(feat, {})[stat] = np.asarray(value)
            self._episode_stats.append(ep_stats)

        # 续写最后一个元数据文件 (内存里持有它的全部行)
        self._meta_chunk_idx, self._meta_file_idx = last_chunk, last_file
        self._episode_rows = [
            r for r in valid_rows
            if int(r["meta/episodes/chunk_index"]) == last_chunk
            and int(r["meta/episodes/file_index"]) == last_file
        ]
        if len(valid_rows) != len(all_rows):
            self._write_episodes_meta()   # 立刻把丢弃的孤儿行从磁盘清掉
        print(f"[LeRobotDatasetWriter] resume: 已有 {self.total_episodes} 条 episode, "
              f"{self.total_frames} 帧, {len(self.tasks)} 个任务")

    # ── 元数据文件 (全部原子写入: 先写 .tmp 再 os.replace, 强杀也不会留下半个文件) ──

    @staticmethod
    def _atomic_write_text(path: Path, text: str):
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)

    @staticmethod
    def _atomic_write_table(table: pa.Table, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        pq.write_table(table, tmp, compression="snappy", use_dictionary=True)
        os.replace(tmp, path)

    def _write_episodes_meta(self):
        """把当前元数据文件的全部行整表重写到磁盘。"""
        path = self.root / EPISODES_PATH.format(
            chunk_index=self._meta_chunk_idx, file_index=self._meta_file_idx
        )
        if not self._episode_rows:
            if path.exists():
                path.unlink()
            return
        self._atomic_write_table(pa.Table.from_pylist(self._episode_rows), path)

    def _write_info(self):
        info = {
            "codebase_version": CODEBASE_VERSION,
            "fps": self.fps,
            "features": self.features,
            "total_episodes": self.total_episodes,
            "total_frames": self.total_frames,
            "total_tasks": len(getattr(self, "tasks", {})),
            "chunks_size": DEFAULT_CHUNK_SIZE,
            "data_files_size_in_mb": DEFAULT_DATA_FILE_SIZE_IN_MB,
            "video_files_size_in_mb": DEFAULT_VIDEO_FILE_SIZE_IN_MB,
            "data_path": DATA_PATH,
            "video_path": VIDEO_PATH,
            "robot_type": self.robot_type,
            "splits": {"train": f"0:{self.total_episodes}"},
        }
        path = self.root / INFO_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        self._atomic_write_text(path, json.dumps(info, indent=4, ensure_ascii=False))

    def _write_tasks(self):
        df = pd.DataFrame(
            {"task_index": list(self.tasks.values())},
            index=pd.Index(list(self.tasks.keys()), name="task"),
        )
        path = self.root / TASKS_PATH
        tmp = path.with_name(path.name + ".tmp")
        df.to_parquet(tmp)
        os.replace(tmp, path)

    def _write_stats(self):
        aggregated = aggregate_stats(self._episode_stats)
        serializable = {
            feat: {k: np.asarray(v).tolist() for k, v in stats.items()}
            for feat, stats in aggregated.items()
        }
        self._atomic_write_text(self.root / STATS_PATH, json.dumps(serializable, indent=4))

    def _get_task_index(self, task: str) -> int:
        if task not in self.tasks:
            self.tasks[task] = len(self.tasks)
            self._write_tasks()
        return self.tasks[task]

    # ── 保存一条 episode ─────────────────────────────────

    def save_episode(
        self,
        task: str,
        images: Dict[str, Sequence[np.ndarray]],  # {camera_key: T 帧 RGB uint8 (H,W,3)}
        state: np.ndarray,                        # (T, D) float
        action: np.ndarray,                       # (T, D) float
    ) -> int:
        """
        保存一条完整 episode。返回 episode_index。
        """
        state = np.asarray(state, dtype=np.float32)
        action = np.asarray(action, dtype=np.float32)
        num_frames = len(state)
        if num_frames == 0:
            raise ValueError("episode 为空")
        if len(action) != num_frames:
            raise ValueError(f"state({num_frames}) 与 action({len(action)}) 帧数不一致")
        if set(images.keys()) != set(self.cameras.keys()):
            raise ValueError(f"相机键不匹配: {sorted(images)} != {sorted(self.cameras)}")
        for cam_key, frames in images.items():
            if len(frames) != num_frames:
                raise ValueError(f"{cam_key} 帧数 {len(frames)} != {num_frames}")

        episode_index = self.total_episodes
        chunk_idx, file_idx = divmod(episode_index, DEFAULT_CHUNK_SIZE)
        task_index = self._get_task_index(task)

        timestamps = (np.arange(num_frames) / self.fps).astype(np.float32)
        frame_indices = np.arange(num_frames, dtype=np.int64)
        global_indices = np.arange(self.total_frames, self.total_frames + num_frames, dtype=np.int64)

        # 1. data parquet (每 episode 一个文件)
        data_path = self.root / DATA_PATH.format(chunk_index=chunk_idx, file_index=file_idx)
        data_path.parent.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pydict(
            {
                "action": action.tolist(),
                "observation.state": state.tolist(),
                "timestamp": timestamps,
                "frame_index": frame_indices,
                "episode_index": np.full(num_frames, episode_index, dtype=np.int64),
                "index": global_indices,
                "task_index": np.full(num_frames, task_index, dtype=np.int64),
            },
            schema=self._data_schema,
        )
        pq.write_table(table, data_path, compression="snappy", use_dictionary=True)

        # 2. 视频 (每 episode 每相机一个文件, from_timestamp 从 0 开始)
        video_metadata = {}
        for cam_key, frames in images.items():
            video_path = self.root / VIDEO_PATH.format(
                video_key=cam_key, chunk_index=chunk_idx, file_index=file_idx
            )
            duration = encode_video_h264(frames, video_path, self.fps, crf=self.video_crf)
            video_metadata[f"videos/{cam_key}/chunk_index"] = chunk_idx
            video_metadata[f"videos/{cam_key}/file_index"] = file_idx
            video_metadata[f"videos/{cam_key}/from_timestamp"] = 0.0
            video_metadata[f"videos/{cam_key}/to_timestamp"] = duration

        # 3. 本条 episode 的统计
        ep_stats: Dict[str, Dict[str, np.ndarray]] = {
            "action": _vector_stats(action),
            "observation.state": _vector_stats(state),
            "timestamp": _vector_stats(timestamps),
            "frame_index": _vector_stats(frame_indices),
            "episode_index": _vector_stats(np.full(num_frames, episode_index)),
            "index": _vector_stats(global_indices),
            "task_index": _vector_stats(np.full(num_frames, task_index)),
        }
        for cam_key, frames in images.items():
            ep_stats[cam_key] = _image_stats(frames)
        self._episode_stats.append(ep_stats)

        # 4. episodes 元数据 (本次录制 session 追加到同一个 parquet)
        episode_row = {
            "episode_index": episode_index,
            "tasks": [task],
            "length": num_frames,
            "data/chunk_index": chunk_idx,
            "data/file_index": file_idx,
            "dataset_from_index": self.total_frames,
            "dataset_to_index": self.total_frames + num_frames,
            "meta/episodes/chunk_index": self._meta_chunk_idx,
            "meta/episodes/file_index": self._meta_file_idx,
        }
        episode_row.update(video_metadata)
        for feat, stats in ep_stats.items():
            for stat_name in STAT_KEYS:
                episode_row[f"stats/{feat}/{stat_name}"] = np.asarray(stats[stat_name]).tolist()

        # 整表原子重写 (行数 = 本文件内 episode 数, 很小, 开销可忽略)
        self._episode_rows.append(episode_row)
        self._write_episodes_meta()

        # 5. 更新全局元数据 (最后写 info.json: 它是"这条 episode 已完整落盘"的提交标记,
        #    resume 时以它为准, 之前步骤写了一半被杀的文件会被当作孤儿忽略/清理)
        self.total_episodes += 1
        self.total_frames += num_frames
        self._write_stats()
        self._write_info()

        return episode_index

    def discard_episode_files(self, episode_index: int):
        """删除某条 episode 已写入的 data/video 文件 (仅用于清理失败的写入)。"""
        chunk_idx, file_idx = divmod(episode_index, DEFAULT_CHUNK_SIZE)
        data_path = self.root / DATA_PATH.format(chunk_index=chunk_idx, file_index=file_idx)
        if data_path.exists():
            data_path.unlink()
        for cam_key in self.cameras:
            video_path = self.root / VIDEO_PATH.format(
                video_key=cam_key, chunk_index=chunk_idx, file_index=file_idx
            )
            if video_path.exists():
                video_path.unlink()

    def finalize(self):
        """
        结束写入。每条 episode 保存时元数据都已完整落盘, 这里不再有必须执行的动作,
        保留此方法只为兼容调用方 (teardown / with 语句)。
        """
        return None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.finalize()
