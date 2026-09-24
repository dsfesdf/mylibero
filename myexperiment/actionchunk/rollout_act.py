"""ACT 的闭环评测、距离分桶和输入消融。"""
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


ABLATIONS = (
    "normal",
    "zero_image",
    "zero_state",
    "current_image",
    "current_state",
    "shuffle_image",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Roll out ACT in LIBERO")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--benchmark", default="LIBERO_SPATIAL")
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--num-rollouts", type=int, default=5)
    parser.add_argument("--all-init-states", action="store_true")
    parser.add_argument("--init-state-offset", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--output", default="myexperiment/actionchunk/outputs/act_rollout")
    parser.add_argument("--temporal-aggregation", action="store_true")
    parser.add_argument("--aggregation-decay", type=float, default=0.2)
    parser.add_argument("--history-size", type=int, default=None)
    parser.add_argument("--input-ablation", choices=ABLATIONS, default="normal")
    parser.add_argument(
        "--distance-objects",
        nargs=2,
        default=("akita_black_bowl_1", "plate_1"),
        metavar=("OBJECT_A", "OBJECT_B"),
    )
    parser.add_argument(
        "--distance-thresholds",
        default=None,
        help="两个米制阈值，例如 0.12,0.24；默认使用所有 init states 的三分位数",
    )
    parser.add_argument("--settle-steps", type=int, default=5)
    return parser.parse_args()


def observation_to_tensors(obs, stats, image_size):
    image = torch.from_numpy(obs["agentview_image"]).permute(2, 0, 1).float() / 255.0
    image = torch.nn.functional.interpolate(
        image[None], (image_size, image_size), mode="bilinear", align_corners=False
    )[0]
    state = np.concatenate([obs["robot0_joint_pos"], obs["robot0_gripper_qpos"]]).astype(np.float32)
    state = (state - stats.state_mean) / stats.state_std
    return image, torch.from_numpy(state).float()


def apply_input_ablation(images, states, mode):
    images = images.clone()
    states = states.clone()
    if mode == "zero_image":
        images.zero_()
    elif mode == "current_image":
        images[:, :-1].zero_()
    elif mode == "zero_state":
        states.zero_()
    elif mode == "current_state":
        states[:, :-1].zero_()
    return images, states


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint["config"]
    stats = NormalizationStats(**checkpoint["stats"])
    model = ACT(
        state_dim=len(stats.state_mean),
        action_dim=len(stats.action_mean),
        history_length=config["history_length"],
        chunk_size=config["chunk_size"],
        latent_dim=config["latent_dim"],
        d_model=config["d_model"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
        visual_representation=config.get("visual_representation", "global_pool"),
        spatial_grid=config.get("spatial_grid", 6),
        observation_mode=config.get("observation_mode", "both"),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint, stats


def get_object_distance(env, object_names):
    task_env = env.env
    body_ids = task_env.obj_body_id
    missing = [name for name in object_names if name not in body_ids]
    if missing:
        raise KeyError(
            f"找不到距离测量对象 {missing}；当前对象包括: {sorted(body_ids.keys())}"
        )
    first = env.sim.data.body_xpos[body_ids[object_names[0]]]
    second = env.sim.data.body_xpos[body_ids[object_names[1]]]
    return float(np.linalg.norm(first - second))


def measure_init_state_distances(env, init_states, object_names, settle_steps):
    distances = []
    zero_action = np.zeros(7, dtype=np.float32)
    for init_state in init_states:
        env.reset()
        env.set_init_state(init_state)
        for _ in range(settle_steps):
            env.step(zero_action)
        distances.append(get_object_distance(env, object_names))
    return distances


def parse_thresholds(raw, distances):
    if raw is None:
        thresholds = np.quantile(np.asarray(distances, dtype=np.float64), [1 / 3, 2 / 3])
    else:
        try:
            thresholds = np.asarray([float(value) for value in raw.split(",")], dtype=np.float64)
        except ValueError as exc:
            raise ValueError("--distance-thresholds 必须是两个逗号分隔的数字") from exc
        if thresholds.shape != (2,) or thresholds[0] >= thresholds[1]:
            raise ValueError("--distance-thresholds 必须满足 near_threshold < far_threshold")
    return thresholds.tolist()


def distance_bucket(distance, thresholds):
    if distance <= thresholds[0]:
        return "near"
    if distance <= thresholds[1]:
        return "medium"
    return "far"


def _settle(env, init_state, settle_steps):
    env.reset()
    obs = env.set_init_state(init_state)
    zero_action = np.zeros(7, dtype=np.float32)
    for _ in range(settle_steps):
        obs, _, _, _ = env.step(zero_action)
    return obs


def run_rollouts(args, model=None, checkpoint=None, stats=None, device=None):
    os.environ.setdefault("MUJOCO_GL", "egl")
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model is None or checkpoint is None or stats is None:
        model, checkpoint, stats = load_model(checkpoint_path, device)
    config = checkpoint["config"]

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
    init_count = len(init_states)
    num_rollouts = init_count if args.all_init_states else args.num_rollouts
    if num_rollouts < 1:
        raise ValueError("num-rollouts 必须 >= 1")

    distance_env = OffScreenRenderEnv(
        bddl_file_name=benchmark.get_task_bddl_file_path(args.task_id),
        camera_heights=128,
        camera_widths=128,
    )
    try:
        distances = measure_init_state_distances(
            distance_env, init_states, args.distance_objects, args.settle_steps
        )
    finally:
        distance_env.close()

    env = OffScreenRenderEnv(
        bddl_file_name=benchmark.get_task_bddl_file_path(args.task_id),
        camera_heights=128,
        camera_widths=128,
    )
    donor_env = None
    try:
        thresholds = parse_thresholds(args.distance_thresholds, distances)
        buckets = [distance_bucket(distance, thresholds) for distance in distances]
        if args.input_ablation == "shuffle_image":
            donor_env = OffScreenRenderEnv(
                bddl_file_name=benchmark.get_task_bddl_file_path(args.task_id),
                camera_heights=128,
                camera_widths=128,
            )

        successes = []
        lengths = []
        candidate_counts = []
        rollout_records = []
        image_size = config.get("image_size", 84)

        for rollout_id in range(num_rollouts):
            init_index = (args.init_state_offset + rollout_id) % init_count
            target_obs = _settle(env, init_states[init_index], args.settle_steps)
            donor_index = (init_index + 1) % init_count
            donor_obs = None
            if donor_env is not None:
                donor_obs = _settle(donor_env, init_states[donor_index], args.settle_steps)

            target_image, target_state = observation_to_tensors(target_obs, stats, image_size)
            if donor_obs is None:
                input_image = target_image
            else:
                input_image, _ = observation_to_tensors(donor_obs, stats, image_size)
            observations = deque(
                [(input_image, target_state)] * history_length,
                maxlen=history_length,
            )
            predictions = []
            frames = [target_obs["agentview_image"][::-1]]
            success = False
            episode_candidate_counts = []

            for step in range(args.max_steps):
                images = torch.stack([item[0] for item in observations])[None]
                states = torch.stack([item[1] for item in observations])[None]
                images, states = apply_input_ablation(images, states, args.input_ablation)
                images, states = images.to(device), states.to(device)
                with torch.no_grad():
                    normalized_chunk = model.predict(images, states)[0].cpu().numpy()
                predictions.append((step, normalized_chunk))
                predictions = predictions[-history_size:]
                action, count = choose_action(
                    predictions,
                    step,
                    chunk_size,
                    action_mean,
                    action_std,
                    args.temporal_aggregation,
                    args.aggregation_decay,
                )
                episode_candidate_counts.append(count)

                target_obs, _, done, _ = env.step(np.clip(action, -1.0, 1.0))
                if donor_env is not None:
                    donor_obs, _, _, _ = donor_env.step(np.clip(action, -1.0, 1.0))
                target_image, target_state = observation_to_tensors(target_obs, stats, image_size)
                if donor_obs is None:
                    input_image = target_image
                else:
                    input_image, _ = observation_to_tensors(donor_obs, stats, image_size)
                observations.append((input_image, target_state))
                frames.append(target_obs["agentview_image"][::-1])
                if done:
                    success = True
                    break

            video_path = output_dir / (
                f"rollout_{rollout_id:03d}_init{init_index:03d}_"
                f"{'success' if success else 'fail'}.mp4"
            )
            imageio.mimsave(video_path, frames, fps=20)
            bucket = buckets[init_index]
            successes.append(success)
            lengths.append(step + 1)
            candidate_counts.extend(episode_candidate_counts)
            rollout_records.append(
                {
                    "rollout_id": rollout_id,
                    "init_state_index": init_index,
                    "image_source_init_state_index": donor_index if donor_env is not None else init_index,
                    "distance": distances[init_index],
                    "distance_bucket": bucket,
                    "success": bool(success),
                    "steps": step + 1,
                    "mean_candidate_count": float(np.mean(episode_candidate_counts)),
                }
            )
            print(
                f"rollout {rollout_id + 1}/{num_rollouts}: "
                f"init={init_index} distance={distances[init_index]:.4f} "
                f"bucket={bucket} success={success} steps={step + 1}",
                flush=True,
            )
    finally:
        if donor_env is not None:
            donor_env.close()
        env.close()

    bucket_metrics = {}
    for bucket in ("near", "medium", "far"):
        records = [record for record in rollout_records if record["distance_bucket"] == bucket]
        bucket_metrics[bucket] = {
            "count": len(records),
            "successes": sum(record["success"] for record in records),
            "success_rate": (
                float(np.mean([record["success"] for record in records])) if records else None
            ),
        }
    metrics = {
        "benchmark": args.benchmark,
        "task_id": args.task_id,
        "task": task.name,
        "checkpoint": str(checkpoint_path),
        "history_length": history_length,
        "chunk_size": chunk_size,
        "latent_dim": config["latent_dim"],
        "visual_representation": config.get("visual_representation", "global_pool"),
        "spatial_grid": config.get("spatial_grid", 1),
        "observation_mode": config.get("observation_mode", "both"),
        "input_ablation": args.input_ablation,
        "temporal_aggregation": args.temporal_aggregation,
        "aggregation_decay": args.aggregation_decay,
        "history_size": history_size,
        "num_rollouts": num_rollouts,
        "init_state_offset": args.init_state_offset,
        "distance_objects": list(args.distance_objects),
        "distance_thresholds": thresholds,
        "all_init_state_distances": distances,
        "all_init_state_buckets": buckets,
        "successes": [bool(x) for x in successes],
        "episode_lengths": lengths,
        "success_rate": float(np.mean(successes)),
        "mean_candidate_count": float(np.mean(candidate_counts)) if candidate_counts else 0.0,
        "by_distance_bucket": bucket_metrics,
        "rollouts": rollout_records,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return metrics


def main():
    args = parse_args()
    run_rollouts(args)


if __name__ == "__main__":
    main()
