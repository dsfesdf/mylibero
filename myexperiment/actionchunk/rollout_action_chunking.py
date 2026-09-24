"""在 LIBERO 中闭环执行 action-chunking BC。

默认每次只执行最新 chunk 的第一个动作；加上
``--temporal-aggregation`` 后，会融合历史 chunk 对当前时刻的重叠预测。
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch

# 允许从 myexperiment/actionchunk 直接执行，也允许作为包导入。
import sys
HERE = Path(__file__).resolve().parent
BC_DIR = HERE.parent / "bc"
for path in (HERE, BC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from action_chunking import SmallChunkBC
from bc import NormalizationStats
from libero.libero.benchmark import get_benchmark
from libero.libero.envs import OffScreenRenderEnv


def parse_args():
    p = argparse.ArgumentParser(description="Roll out an action-chunking BC policy in LIBERO")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--benchmark", default="LIBERO_SPATIAL")
    p.add_argument("--task-id", type=int, required=True)
    p.add_argument("--num-rollouts", type=int, default=5)
    p.add_argument("--max-steps", type=int, default=600)
    p.add_argument("--output", default="myexperiment/actionchunk/outputs/rollout_k10")
    p.add_argument("--temporal-aggregation", action="store_true", help="融合重叠 chunk 预测")
    p.add_argument("--aggregation-decay", type=float, default=0.2, help="指数衰减系数，越大越信任新预测")
    p.add_argument("--history-size", type=int, default=None, help="最多保留多少个历史 chunk，默认等于 chunk_size")
    return p.parse_args()


def preprocess_obs(obs, stats, image_size, device):
    image = torch.from_numpy(obs["agentview_image"]).permute(2, 0, 1).float() / 255.0
    image = torch.nn.functional.interpolate(
        image[None], (image_size, image_size), mode="bilinear", align_corners=False
    ).to(device)
    state = np.concatenate([obs["robot0_joint_pos"], obs["robot0_gripper_qpos"]]).astype(np.float32)
    state = (state - stats.state_mean) / stats.state_std
    return image, torch.from_numpy(state).float()[None].to(device)


def choose_action(predictions, current_step, chunk_size, action_mean, action_std, aggregate, decay):
    """从历史 normalized chunks 中取当前动作，并反归一化到 LIBERO action 空间。"""
    candidates = []
    for start_step, normalized_chunk in predictions:
        relative_index = current_step - start_step
        if 0 <= relative_index < chunk_size:
            age = relative_index
            weight = np.exp(-decay * age) if aggregate else (1.0 if age == 0 else 0.0)
            if weight > 0:
                candidates.append((weight, normalized_chunk[relative_index]))
    if not candidates:
        raise RuntimeError("没有找到当前时间步对应的 action chunk")
    total_weight = sum(weight for weight, _ in candidates)
    normalized_action = sum(weight * action for weight, action in candidates) / total_weight
    return normalized_action * action_std + action_mean, len(candidates)


def main():
    args = parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")
    output_dir = Path(args.output).expanduser().resolve(); output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(Path(args.checkpoint).expanduser(), map_location="cpu")
    stats = NormalizationStats(**checkpoint["stats"])
    cfg = checkpoint["config"]
    chunk_size = int(cfg["chunk_size"])
    history_size = args.history_size or chunk_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmallChunkBC(len(stats.state_mean), len(stats.action_mean), chunk_size, cfg.get("pool_size", 2)).to(device)
    model.load_state_dict(checkpoint["model"]); model.eval()
    action_mean = stats.action_mean.astype(np.float32)
    action_std = stats.action_std.astype(np.float32)

    benchmark = get_benchmark(args.benchmark)()
    task = benchmark.get_task(args.task_id)
    init_states = benchmark.get_task_init_states(args.task_id)
    env = OffScreenRenderEnv(bddl_file_name=benchmark.get_task_bddl_file_path(args.task_id), camera_heights=128, camera_widths=128)
    successes, lengths, candidate_counts = [], [], []
    try:
        for rollout_id in range(args.num_rollouts):
            env.reset(); obs = env.set_init_state(init_states[rollout_id % len(init_states)])
            for _ in range(5): obs, _, _, _ = env.step(np.zeros(7, dtype=np.float32))
            frames = [obs["agentview_image"][::-1]]
            predictions = []
            success = False
            for step in range(args.max_steps):
                image, state = preprocess_obs(obs, stats, cfg.get("image_size", 84), device)
                with torch.no_grad():
                    normalized_chunk = model(image, state)[0].cpu().numpy()
                predictions.append((step, normalized_chunk))
                predictions = predictions[-history_size:]
                action, count = choose_action(predictions, step, chunk_size, action_mean, action_std, args.temporal_aggregation, args.aggregation_decay)
                candidate_counts.append(count)
                obs, _, done, _ = env.step(np.clip(action, -1.0, 1.0))
                frames.append(obs["agentview_image"][::-1])
                if done:
                    success = True; break
            video_path = output_dir / f"rollout_{rollout_id:03d}_{'success' if success else 'fail'}.mp4"
            imageio.mimsave(video_path, frames, fps=20)
            successes.append(success); lengths.append(step + 1)
            print(f"rollout {rollout_id + 1}/{args.num_rollouts}: success={success} steps={step + 1}", flush=True)
    finally:
        env.close()

    metrics = {
        "benchmark": args.benchmark, "task_id": args.task_id, "task": task.name,
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "chunk_size": chunk_size, "temporal_aggregation": args.temporal_aggregation,
        "aggregation_decay": args.aggregation_decay, "history_size": history_size,
        "num_rollouts": args.num_rollouts, "successes": [bool(x) for x in successes],
        "episode_lengths": lengths, "success_rate": float(np.mean(successes)),
        "mean_candidate_count": float(np.mean(candidate_counts)) if candidate_counts else 0.0,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__": main()
