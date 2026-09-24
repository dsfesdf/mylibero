"""离线评测小型时序 Transformer checkpoint。"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BC_DIR = HERE.parent / "bc"
for path in (HERE, BC_DIR):
    if str(path) not in sys.path: sys.path.insert(0, str(path))

import torch
from torch.utils.data import DataLoader

from bc import NormalizationStats, list_demos
from sequence_transformer import LIBEROSequenceChunkDataset, SmallSequenceTransformer, masked_chunk_mse


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset", default=None)
    p.add_argument("--split", choices=["train", "val", "all"], default="val")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--output", default=None)
    args = p.parse_args()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    dataset_path = Path(args.dataset or checkpoint["config"]["dataset"]).expanduser()
    if not dataset_path.is_absolute(): dataset_path = Path.cwd() / dataset_path
    dataset_path = dataset_path.resolve()
    cfg = checkpoint["config"]
    stats = NormalizationStats(**checkpoint["stats"])
    if args.split == "train": demos = checkpoint["train_demos"]
    elif args.split == "val": demos = checkpoint["val_demos"]
    else: demos = list_demos(dataset_path)
    dataset = LIBEROSequenceChunkDataset(dataset_path, demos, stats, cfg["history_length"], cfg["chunk_size"], cfg.get("image_size", 84))
    loader = DataLoader(dataset, args.batch_size, shuffle=False, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmallSequenceTransformer(
        9, 7, cfg["history_length"], cfg["chunk_size"], cfg["d_model"], cfg["num_heads"], cfg["num_layers"], cfg["dropout"]
    ).to(device)
    model.load_state_dict(checkpoint["model"]); model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for images, states, target, mask in loader:
            prediction = model(images.to(device), states.to(device))
            loss = masked_chunk_mse(prediction, target.to(device), mask.to(device))
            total += loss.item() * images.size(0); count += images.size(0)
    result = {"checkpoint": str(checkpoint_path), "dataset": str(dataset_path), "split": args.split,
              "history_length": cfg["history_length"], "chunk_size": cfg["chunk_size"],
              "num_frames": count, "masked_normalized_mse": total / max(count, 1)}
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.output:
        output = Path(args.output).expanduser().resolve(); output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__": main()
