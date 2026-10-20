"""离线评测 ACT checkpoint 的 reconstruction MSE、KL 和总 loss。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BC_DIR = HERE.parent / "bc"
for path in (HERE, BC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import torch
from torch.utils.data import DataLoader

from act import ACT, LIBEROACTDataset, act_loss
from action_chunking import masked_chunk_mse
from bc import NormalizationStats, list_demos
from train_act import run_epoch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--flow-steps", type=int, default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--split", choices=["train", "val", "all"], default="val")
    parser.add_argument("--batch-size", type=int, default=None, help="默认复用训练 batch size，保证 Flow 验证噪声分组一致")
    parser.add_argument("--val-seed", type=int, default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint["config"]
    dataset_path = Path(args.dataset or config["dataset"]).expanduser()
    if not dataset_path.is_absolute():
        dataset_path = Path.cwd() / dataset_path
    dataset_path = dataset_path.resolve()
    stats = NormalizationStats(**checkpoint["stats"])
    if args.split == "train":
        demos = checkpoint["train_demos"]
    elif args.split == "val":
        demos = checkpoint["val_demos"]
    else:
        demos = list_demos(dataset_path)
    dataset = LIBEROACTDataset(dataset_path, demos, stats, config["history_length"], config["chunk_size"], config.get("image_size", 84), cameras=config.get("cameras", "agentview"))
    batch_size = args.batch_size or config.get("batch_size", 64)
    loader = DataLoader(dataset, batch_size, shuffle=False, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ACT(
        state_dim=len(stats.state_mean), action_dim=len(stats.action_mean),
        history_length=config["history_length"], chunk_size=config["chunk_size"],
        latent_dim=config["latent_dim"], d_model=config["d_model"],
        num_heads=config["num_heads"], num_layers=config["num_layers"],
        dropout=config["dropout"],
        visual_representation=config.get("visual_representation", "global_pool"),
        spatial_grid=config.get("spatial_grid", 6),
        observation_mode=config.get("observation_mode", "both"),
        cameras=config.get("cameras", "agentview"),
        action_head_type=config.get("action_head", "transformer"), flow_steps=config.get("flow_steps", 10),
    ).to(device)
    if args.flow_steps is not None:
        if args.flow_steps < 1:
            raise ValueError("flow-steps 必须 >= 1")
        model.flow_steps = args.flow_steps
    model.load_state_dict(checkpoint["model"])
    model.eval()
    val_seed = config.get("val_seed", 12345) if args.val_seed is None else args.val_seed
    torch.manual_seed(val_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(val_seed)
    with torch.no_grad():
        metrics = run_epoch(model, loader, None, device, config["kl_weight"])
    result = {
        "checkpoint": str(checkpoint_path), "dataset": str(dataset_path), "split": args.split,
        "history_length": config["history_length"], "chunk_size": config["chunk_size"],
        "latent_dim": config["latent_dim"], "num_frames": len(dataset),
        "cameras": config.get("cameras", "agentview"),
        "action_head": config.get("action_head", "transformer"),
        "flow_steps": model.flow_steps,
        "val_seed": val_seed, "batch_size": batch_size,
        **metrics,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.output:
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
