"""
rebuild_lerobot_dataset.py

从每条 episode 的 data parquet + mp4 重建 LeRobotDataset v3.0 的全部元数据
(meta/episodes、stats.json、info.json)，并可选剔除步数过短的 episode。

适用场景:
- 录制程序被强杀导致 episodes 元数据 parquet 缺 footer 而损坏
- 需要删掉某些 episode 后重排索引

用法:
    python data_collection/rebuild_lerobot_dataset.py \
        --src lerobot_dataset --dst lerobot_dataset_rebuilt --min_steps 10

数据本体 (data parquet / mp4) 不重编码, 只重写索引列并复制视频文件。
图像统计从 mp4 解码帧重新计算 (与原始帧有轻微压缩差异, 对归一化无影响)。
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_collection.lerobot_dataset import (
    CODEBASE_VERSION,
    DATA_PATH,
    DEFAULT_CHUNK_SIZE,
    EPISODES_PATH,
    INFO_PATH,
    STAT_KEYS,
    STATS_PATH,
    TASKS_PATH,
    VIDEO_PATH,
    _image_stats,
    _vector_stats,
    aggregate_stats,
)


def decode_video_frames(path: Path) -> list:
    """解码 mp4 全部帧为 RGB uint8 数组列表。"""
    import av

    frames = []
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            frames.append(frame.to_ndarray(format="rgb24"))
    return frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=str, required=True, help="源数据集目录")
    parser.add_argument("--dst", type=str, required=True, help="输出目录 (不能已存在)")
    parser.add_argument("--min_steps", type=int, default=1,
                        help="步数小于该值的 episode 会被剔除")
    args = parser.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    if dst.exists():
        raise SystemExit(f"输出目录已存在: {dst}")

    info = json.loads((src / INFO_PATH).read_text(encoding="utf-8"))
    assert info["codebase_version"] == CODEBASE_VERSION
    fps = info["fps"]
    camera_keys = [k for k, v in info["features"].items() if v["dtype"] == "video"]

    tasks_df = pd.read_parquet(src / TASKS_PATH)
    index_to_task = {int(row["task_index"]): task for task, row in tasks_df.iterrows()}

    src_data_files = sorted((src / "data").glob("chunk-*/file-*.parquet"))
    print(f"源数据集: {len(src_data_files)} 个 episode 数据文件, fps={fps}, 相机={camera_keys}")

    (dst / "meta").mkdir(parents=True)

    episodes_writer = None
    episode_stats_list = []
    new_ep_idx = 0
    global_frame_idx = 0
    dropped = []

    for src_file in src_data_files:
        df = pd.read_parquet(src_file)
        old_ep_idx = int(df["episode_index"].iloc[0])
        num_frames = len(df)
        if num_frames < args.min_steps:
            dropped.append((old_ep_idx, num_frames))
            continue

        old_chunk, old_file = divmod(old_ep_idx, DEFAULT_CHUNK_SIZE)
        new_chunk, new_file = divmod(new_ep_idx, DEFAULT_CHUNK_SIZE)
        task_index = int(df["task_index"].iloc[0])
        task = index_to_task[task_index]

        # 1. 重写 data parquet (只更新索引列)
        df["episode_index"] = np.int64(new_ep_idx)
        df["index"] = np.arange(global_frame_idx, global_frame_idx + num_frames, dtype=np.int64)
        dst_data = dst / DATA_PATH.format(chunk_index=new_chunk, file_index=new_file)
        dst_data.parent.mkdir(parents=True, exist_ok=True)
        dim = len(df["action"].iloc[0])
        schema = pa.schema(
            [("action", pa.list_(pa.float32(), dim)),
             ("observation.state", pa.list_(pa.float32(), dim)),
             ("timestamp", pa.float32()),
             ("frame_index", pa.int64()),
             ("episode_index", pa.int64()),
             ("index", pa.int64()),
             ("task_index", pa.int64())]
        )
        table = pa.Table.from_pydict(
            {col: df[col].tolist() for col in
             ["action", "observation.state", "timestamp", "frame_index",
              "episode_index", "index", "task_index"]},
            schema=schema,
        )
        pq.write_table(table, dst_data, compression="snappy", use_dictionary=True)

        # 2. 复制视频 + 从解码帧计算图像统计
        video_metadata = {}
        ep_stats = {
            "action": _vector_stats(np.stack(df["action"].to_numpy())),
            "observation.state": _vector_stats(np.stack(df["observation.state"].to_numpy())),
            "timestamp": _vector_stats(df["timestamp"].to_numpy()),
            "frame_index": _vector_stats(df["frame_index"].to_numpy()),
            "episode_index": _vector_stats(np.full(num_frames, new_ep_idx)),
            "index": _vector_stats(df["index"].to_numpy()),
            "task_index": _vector_stats(df["task_index"].to_numpy()),
        }
        for cam_key in camera_keys:
            src_video = src / VIDEO_PATH.format(
                video_key=cam_key, chunk_index=old_chunk, file_index=old_file)
            dst_video = dst / VIDEO_PATH.format(
                video_key=cam_key, chunk_index=new_chunk, file_index=new_file)
            dst_video.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_video, dst_video)
            frames = decode_video_frames(src_video)
            ep_stats[cam_key] = _image_stats(frames)
            video_metadata[f"videos/{cam_key}/chunk_index"] = new_chunk
            video_metadata[f"videos/{cam_key}/file_index"] = new_file
            video_metadata[f"videos/{cam_key}/from_timestamp"] = 0.0
            video_metadata[f"videos/{cam_key}/to_timestamp"] = len(frames) / float(fps)
        episode_stats_list.append(ep_stats)

        # 3. episodes 元数据行
        row = {
            "episode_index": new_ep_idx,
            "tasks": [task],
            "length": num_frames,
            "data/chunk_index": new_chunk,
            "data/file_index": new_file,
            "dataset_from_index": global_frame_idx,
            "dataset_to_index": global_frame_idx + num_frames,
            "meta/episodes/chunk_index": 0,
            "meta/episodes/file_index": 0,
        }
        row.update(video_metadata)
        for feat, stats in ep_stats.items():
            for stat_name in STAT_KEYS:
                row[f"stats/{feat}/{stat_name}"] = np.asarray(stats[stat_name]).tolist()

        ep_table = pa.Table.from_pydict({k: [v] for k, v in row.items()})
        if episodes_writer is None:
            ep_meta_path = dst / EPISODES_PATH.format(chunk_index=0, file_index=0)
            ep_meta_path.parent.mkdir(parents=True, exist_ok=True)
            episodes_writer = pq.ParquetWriter(
                ep_meta_path, schema=ep_table.schema,
                compression="snappy", use_dictionary=True)
        episodes_writer.write_table(ep_table)

        print(f"  episode {old_ep_idx:3d} -> {new_ep_idx:3d}  ({num_frames} 步)")
        new_ep_idx += 1
        global_frame_idx += num_frames

    if episodes_writer is not None:
        episodes_writer.close()

    # 4. tasks / stats / info
    shutil.copy2(src / TASKS_PATH, dst / TASKS_PATH)

    aggregated = aggregate_stats(episode_stats_list)
    serializable = {
        feat: {k: np.asarray(v).tolist() for k, v in stats.items()}
        for feat, stats in aggregated.items()
    }
    (dst / STATS_PATH).write_text(json.dumps(serializable, indent=4), encoding="utf-8")

    info["total_episodes"] = new_ep_idx
    info["total_frames"] = global_frame_idx
    info["total_tasks"] = len(index_to_task)
    info["splits"] = {"train": f"0:{new_ep_idx}"}
    (dst / INFO_PATH).write_text(
        json.dumps(info, indent=4, ensure_ascii=False), encoding="utf-8")

    print(f"\n完成: {new_ep_idx} 条 episode, {global_frame_idx} 帧 -> {dst}")
    if dropped:
        print(f"剔除 {len(dropped)} 条过短 episode: " +
              ", ".join(f"episode {e} ({n} 步)" for e, n in dropped))


if __name__ == "__main__":
    main()
