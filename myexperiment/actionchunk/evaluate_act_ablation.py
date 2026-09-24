"""批量运行 ACT 距离分桶和输入消融评测。"""
from __future__ import annotations

import argparse
import json
import os
import sys
from argparse import Namespace
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from rollout_act import ABLATIONS, load_model, run_rollouts


def parse_args():
    parser = argparse.ArgumentParser(description="Run ACT ablation rollouts with shared settings")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--benchmark", default="LIBERO_SPATIAL")
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--num-rollouts", type=int, default=40)
    parser.add_argument("--all-init-states", action="store_true")
    parser.add_argument("--init-state-offset", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--output", default="myexperiment/actionchunk/outputs/act_ablation")
    parser.add_argument("--temporal-aggregation", action="store_true")
    parser.add_argument("--aggregation-decay", type=float, default=0.2)
    parser.add_argument("--history-size", type=int, default=None)
    parser.add_argument(
        "--ablations",
        nargs="+",
        choices=ABLATIONS,
        default=["normal", "shuffle_image", "zero_image", "zero_state", "current_image", "current_state"],
    )
    parser.add_argument(
        "--distance-objects",
        nargs=2,
        default=("akita_black_bowl_1", "plate_1"),
        metavar=("OBJECT_A", "OBJECT_B"),
    )
    parser.add_argument("--distance-thresholds", default=None)
    parser.add_argument("--settle-steps", type=int, default=5)
    return parser.parse_args()


def clone_args(args, ablation, output):
    return Namespace(
        checkpoint=args.checkpoint,
        benchmark=args.benchmark,
        task_id=args.task_id,
        num_rollouts=args.num_rollouts,
        all_init_states=args.all_init_states,
        init_state_offset=args.init_state_offset,
        max_steps=args.max_steps,
        output=str(output),
        temporal_aggregation=args.temporal_aggregation,
        aggregation_decay=args.aggregation_decay,
        history_size=args.history_size,
        input_ablation=ablation,
        distance_objects=args.distance_objects,
        distance_thresholds=args.distance_thresholds,
        settle_steps=args.settle_steps,
    )


def compact(metrics):
    row = {
        "input_ablation": metrics["input_ablation"],
        "success_rate": metrics["success_rate"],
        "num_rollouts": metrics["num_rollouts"],
    }
    for bucket, values in metrics["by_distance_bucket"].items():
        row[f"{bucket}_count"] = values["count"]
        row[f"{bucket}_success_rate"] = values["success_rate"]
    return row


def main():
    args = parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")
    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint, stats = load_model(checkpoint_path, device)

    all_metrics = []
    for ablation in args.ablations:
        print(f"\n=== input_ablation={ablation} ===", flush=True)
        metrics = run_rollouts(
            clone_args(args, ablation, output_dir / ablation),
            model=model,
            checkpoint=checkpoint,
            stats=stats,
            device=device,
        )
        all_metrics.append(metrics)

    summary = {
        "checkpoint": str(checkpoint_path),
        "benchmark": args.benchmark,
        "task_id": args.task_id,
        "ablations": [compact(metrics) for metrics in all_metrics],
    }
    (output_dir / "ablation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
