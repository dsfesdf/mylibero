"""把 LIBERO HDF5 里 demo 的相机图像序列导出成 mp4，方便直接用播放器观看。

两种定位方式（二选一）：

1) 直接给 HDF5 文件路径：
    python scripts/export_demo_video.py \
        --hdf5 /home/zx/LIBERO/libero/datasets/libero_spatial/<task>_demo.hdf5 \
        --demo-id all --output libero/datasets/video

2) 给 benchmark + task-id（脚本自动找对应 HDF5，等价于 rollout 的 --task-id）：
    python scripts/export_demo_video.py \
        --benchmark LIBERO_SPATIAL --task-id 8 \
        --demo-id all --output libero/datasets/video

--demo-id 可以是单个整数（只导那一条），也可以是 "all"（导出该文件全部 demo）。

输出目录下按 demo 分子目录，每个 demo 内按相机命名，例如：

    <output>/demo_000/agentview.mp4
    <output>/demo_000/eye_in_hand.mp4
    <output>/demo_001/agentview.mp4
    ...

注意：图像在 HDF5 里是上下翻转存储的（LIBERO 的约定），
导出时默认做 [::-1] 翻转，和 rollout 脚本里的做法保持一致。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import imageio.v2 as imageio
import numpy as np

# HDF5 obs 组里的相机键 -> 输出文件名。缺哪个就跳过哪个。
CAMERA_KEYS = {
    "agentview_rgb": "agentview.mp4",
    "eye_in_hand_rgb": "eye_in_hand.mp4",
}


def parse_args():
    p = argparse.ArgumentParser(description="Export LIBERO demo camera frames to mp4")
    # 定位 HDF5 的两种方式
    p.add_argument("--hdf5", default=None, help="LIBERO demo 的 HDF5 文件路径")
    p.add_argument("--benchmark", default=None, help="benchmark 名，如 LIBERO_SPATIAL（配合 --task-id）")
    p.add_argument("--task-id", type=int, default=None, help="benchmark 里的任务序号（配合 --benchmark）")
    # 选哪些 demo
    p.add_argument("--demo-id", default="all",
                   help="导出哪条 demo：单个整数，或 'all' 导出全部（默认 all）")
    p.add_argument("--output", required=True, help="输出目录")
    p.add_argument("--fps", type=int, default=20, help="导出视频帧率（LIBERO 录制约 20fps）")
    p.add_argument("--flip", dest="flip", action="store_true", default=True,
                   help="上下翻转图像（默认开启，符合 LIBERO 约定）")
    p.add_argument("--no-flip", dest="flip", action="store_false")
    return p.parse_args()


def resolve_hdf5(args) -> Path:
    """根据 --hdf5 或 --benchmark/--task-id 得到 HDF5 路径。"""
    if args.hdf5:
        path = Path(args.hdf5).expanduser().resolve()
    elif args.benchmark is not None and args.task_id is not None:
        # 复用 LIBERO 自己的 benchmark 接口来定位演示文件。
        from libero.libero import get_libero_path
        from libero.libero.benchmark import get_benchmark
        benchmark = get_benchmark(args.benchmark)()
        rel = benchmark.get_task_demonstration(args.task_id)  # 如 libero_spatial/xxx_demo.hdf5
        path = Path(get_libero_path("datasets")) / rel
    else:
        raise SystemExit("请用 --hdf5，或同时提供 --benchmark 与 --task-id")
    if not path.is_file():
        raise FileNotFoundError(f"找不到 HDF5 文件: {path}")
    return path


def export_one(demo, demo_tag: str, output_dir: Path, fps: int, flip: bool):
    obs = demo["obs"]
    num_frames = demo["actions"].shape[0]
    demo_dir = output_dir / demo_tag
    demo_dir.mkdir(parents=True, exist_ok=True)
    exported = []
    for key, filename in CAMERA_KEYS.items():
        if key not in obs:
            continue
        frames = np.asarray(obs[key], dtype=np.uint8)
        if flip:
            frames = frames[:, ::-1]  # 上下翻转，保持和 rollout 一致的朝向
        out_path = demo_dir / filename
        imageio.mimsave(out_path, list(frames), fps=fps)
        exported.append((key, out_path))
    print(f"{demo_tag}: 帧数={num_frames}  导出 {len(exported)} 个视频 -> {demo_dir}", flush=True)


def main():
    args = parse_args()
    hdf5_path = resolve_hdf5(args)
    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"HDF5: {hdf5_path}\n输出: {output_dir}", flush=True)

    # demo-id 解析
    want_all = str(args.demo_id).lower() == "all"
    single_id = None if want_all else int(args.demo_id)

    with h5py.File(hdf5_path, "r") as handle:
        demo_names = sorted(handle["data"].keys(), key=lambda name: int(name.split("_")[-1]))
        if not want_all and not (0 <= single_id < len(demo_names)):
            raise IndexError(f"demo-id={single_id} 超出范围，共 {len(demo_names)} 条 demo")
        targets = range(len(demo_names)) if want_all else [single_id]
        print(f"共 {len(demo_names)} 条 demo，即将导出 {len(targets)} 条", flush=True)
        for demo_id in targets:
            demo = handle["data"][demo_names[demo_id]]
            # demo_0 -> demo_000，保证排序整齐
            demo_tag = f"demo_{demo_id:03d}"
            export_one(demo, demo_tag, output_dir, args.fps, args.flip)


if __name__ == "__main__":
    main()
