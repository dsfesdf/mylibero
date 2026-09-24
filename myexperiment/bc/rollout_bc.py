"""在 LIBERO 仿真环境中闭环评测最小 BC，并保存 rollout 视频。"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch

from bc import NormalizationStats, SmallVisualBC
from libero.libero.benchmark import get_benchmark
from libero.libero.envs import OffScreenRenderEnv


def parse_args():
    parser = argparse.ArgumentParser(description="Roll out a minimal BC policy in LIBERO")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--benchmark", default="LIBERO_SPATIAL")
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--num-rollouts", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--output", default="myexperiment/bc/outputs/rollout_bc")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def make_model_input(obs, stats, image_size, device):
    """将 LIBERO 原始 observation 处理成训练时相同的模型输入。"""
    image = torch.from_numpy(obs["agentview_image"]).permute(2, 0, 1).float() / 255.0
    image = torch.nn.functional.interpolate(
        image[None], (image_size, image_size), mode="bilinear", align_corners=False
    ).to(device)
    state = np.concatenate([obs["robot0_joint_pos"], obs["robot0_gripper_qpos"]]).astype(np.float32)
    state = (state - stats.state_mean) / stats.state_std
    return image, torch.from_numpy(state).float()[None].to(device)


def main():
    args = parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")
    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(Path(args.checkpoint).expanduser(), map_location="cpu")
    stats = NormalizationStats(**checkpoint["stats"])
    image_size = checkpoint["config"].get("image_size", 84)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmallVisualBC(len(stats.state_mean), len(stats.action_mean)).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    action_mean = torch.from_numpy(stats.action_mean).to(device)
    action_std = torch.from_numpy(stats.action_std).to(device)

    benchmark = get_benchmark(args.benchmark)()
    task = benchmark.get_task(args.task_id)
    init_states = benchmark.get_task_init_states(args.task_id)
    env = OffScreenRenderEnv(
        bddl_file_name=benchmark.get_task_bddl_file_path(args.task_id),
        camera_heights=128,
        camera_widths=128,
    )
    successes, lengths = [], []
    try:
        for rollout_id in range(args.num_rollouts):
            env.reset()
            obs = env.set_init_state(init_states[rollout_id % len(init_states)])
            # 与官方评测一致，先执行 5 个零动作稳定物理状态。
            for _ in range(5):
                obs, _, _, _ = env.step(np.zeros(7, dtype=np.float32))

            frames = [obs["agentview_image"][::-1]]  # 仅视频显示时上下翻转。
            success = False
            for step in range(1, args.max_steps + 1):
                image, state = make_model_input(obs, stats, image_size, device)
                with torch.no_grad():
                    normalized_action = model(image, state)[0]
                    action = (normalized_action * action_std + action_mean).cpu().numpy()
                # LIBERO OSC controller 的动作范围为 [-1, 1]。
                obs, _, done, _ = env.step(np.clip(action, -1.0, 1.0))
                frames.append(obs["agentview_image"][::-1])
                if done:
                    success = True
                    break

            video_path = output_dir / f"rollout_{rollout_id:03d}_{'success' if success else 'fail'}.mp4"
            imageio.mimsave(video_path, frames, fps=20)
            successes.append(success)
            lengths.append(step)
            print(f"rollout {rollout_id + 1}/{args.num_rollouts}: success={success} steps={step}", flush=True)
    finally:
        env.close()

    metrics = {
        "benchmark": args.benchmark,
        "task_id": args.task_id,
        "task": task.name,
        "num_rollouts": args.num_rollouts,
        "successes": [bool(value) for value in successes],
        "episode_lengths": lengths,
        "success_rate": float(np.mean(successes)),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
