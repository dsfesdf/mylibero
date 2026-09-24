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
from bc import NormalizationStats, list_demos


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--split", choices=["train", "val", "all"], default="val")
    parser.add_argument("--batch-size", type=int, default=128)
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
    dataset = LIBEROACTDataset(dataset_path, demos, stats, config["history_length"], config["chunk_size"], config.get("image_size", 84))
    loader = DataLoader(dataset, args.batch_size, shuffle=False, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ACT(
        state_dim=len(stats.state_mean), action_dim=len(stats.action_mean),
        history_length=config["history_length"], chunk_size=config["chunk_size"],
        latent_dim=config["latent_dim"], d_model=config["d_model"],
        num_heads=config["num_heads"], num_layers=config["num_layers"],
        dropout=config["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    totals = {"loss": 0.0, "reconstruction": 0.0, "kl": 0.0, "count": 0}
    with torch.no_grad():
        for images, states, target, mask in loader:
            images, states = images.to(device), states.to(device)
            target, mask = target.to(device), mask.to(device)
            prediction, mean, logvar = model(images, states, target, mask)
            loss, reconstruction, kl = act_loss(prediction, target, mask, mean, logvar, config["kl_weight"])
            count = images.size(0)
            totals["loss"] += loss.item() * count
            totals["reconstruction"] += reconstruction.item() * count
            totals["kl"] += kl.item() * count
            totals["count"] += count
    count = max(totals.pop("count"), 1)
    result = {
        "checkpoint": str(checkpoint_path), "dataset": str(dataset_path), "split": args.split,
        "history_length": config["history_length"], "chunk_size": config["chunk_size"],
        "latent_dim": config["latent_dim"], "num_frames": count,
        **{key: value / count for key, value in totals.items()},
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.output:
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
