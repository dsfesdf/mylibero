"""ACT 在 LIBERO 中的闭环 rollout。"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import deque
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
BC_DIR = HERE.parent / "bc"
for path in (HERE, BC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from act import ACT
from bc import NormalizationStats
from libero.libero.benchmark import get_benchmark
from libero.libero.envs import OffScreenRenderEnv
from rollout_action_chunking import choose_action


def parse_args():
    parser = argparse.ArgumentParser(description="Roll out ACT in LIBERO")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--benchmark", default="LIBERO_SPATIAL")
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--num-rollouts", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--output", default="myexperiment/actionchunk/outputs/act_rollout")
    parser.add_argument("--temporal-aggregation", action="store_true")
    parser.add_argument("--aggregation-decay", type=float, default=0.2)
    parser.add_argument("--history-size", type=int, default=None)
    return parser.parse_args()


def observation_to_tensors(obs, stats, image_size):
    image = torch.from_numpy(obs["agentview_image"]).permute(2, 0, 1).float() / 255.0
    image = torch.nn.functional.interpolate(image[None], (image_size, image_size), mode="bilinear", align_corners=False)[0]
    state = np.concatenate([obs["robot0_joint_pos"], obs["robot0_gripper_qpos"]]).astype(np.float32)
    state = (state - stats.state_mean) / stats.state_std
    return image, torch.from_numpy(state).float()


def main():
    args = parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint["config"]
    stats = NormalizationStats(**checkpoint["stats"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ACT(
        state_dim=len(stats.state_mean), action_dim=len(stats.action_mean),
        history_length=config["history_length"], chunk_size=config["chunk_size"],
        latent_dim=config["latent_dim"], d_model=config["d_model"],
        num_heads=config["num_heads"], num_layers=config["num_layers"], dropout=config["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    action_mean = stats.action_mean.astype(np.float32)
    action_std = stats.action_std.astype(np.float32)
    chunk_size = int(config["chunk_size"])
    history_length = int(config["history_length"])
    history_size = args.history_size or chunk_size

    benchmark = get_benchmark(args.benchmark)()
    task = benchmark.get_task(args.task_id)
    init_states = benchmark.get_task_init_states(args.task_id)
    env = OffScreenRenderEnv(
        bddl_file_name=benchmark.get_task_bddl_file_path(args.task_id),
        camera_heights=128, camera_widths=128,
    )
    successes, lengths, candidate_counts = [], [], []
    try:
        for rollout_id in range(args.num_rollouts):
            env.reset()
            obs = env.set_init_state(init_states[rollout_id % len(init_states)])
            for _ in range(5):
                obs, _, _, _ = env.step(np.zeros(7, dtype=np.float32))
            first = observation_to_tensors(obs, stats, config.get("image_size", 84))
            observations = deque([first] * history_length, maxlen=history_length)
            predictions, frames = [], [obs["agentview_image"][::-1]]
            success = False
            for step in range(args.max_steps):
                images = torch.stack([item[0] for item in observations])[None].to(device)
                states = torch.stack([item[1] for item in observations])[None].to(device)
                with torch.no_grad():
                    normalized_chunk = model.predict(images, states)[0].cpu().numpy()
                predictions.append((step, normalized_chunk))
                predictions = predictions[-history_size:]
                action, count = choose_action(
                    predictions, step, chunk_size, action_mean, action_std,
                    args.temporal_aggregation, args.aggregation_decay,
                )
                candidate_counts.append(count)
                obs, _, done, _ = env.step(np.clip(action, -1.0, 1.0))
                observations.append(observation_to_tensors(obs, stats, config.get("image_size", 84)))
                frames.append(obs["agentview_image"][::-1])
                if done:
                    success = True
                    break
            imageio.mimsave(output_dir / f"rollout_{rollout_id:03d}_{'success' if success else 'fail'}.mp4", frames, fps=20)
            successes.append(success)
            lengths.append(step + 1)
            print(f"rollout {rollout_id + 1}/{args.num_rollouts}: success={success} steps={step + 1}", flush=True)
    finally:
        env.close()
    metrics = {
        "benchmark": args.benchmark, "task_id": args.task_id, "task": task.name,
        "checkpoint": str(checkpoint_path), "history_length": history_length,
        "chunk_size": chunk_size, "latent_dim": config["latent_dim"],
        "temporal_aggregation": args.temporal_aggregation,
        "aggregation_decay": args.aggregation_decay, "history_size": history_size,
        "num_rollouts": args.num_rollouts, "successes": [bool(x) for x in successes],
        "episode_lengths": lengths, "success_rate": float(np.mean(successes)),
        "mean_candidate_count": float(np.mean(candidate_counts)) if candidate_counts else 0.0,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
