"""离线评测 action-chunking checkpoint 的 masked MSE。"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# 让本文件直接运行时能找到同目录的 action_chunking.py 和 ../bc/bc.py。
_HERE = os.path.dirname(os.path.abspath(__file__))
_BC_DIR = os.path.join(os.path.dirname(_HERE), "bc")
for _path in (_HERE, _BC_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import torch
from torch.utils.data import DataLoader

from action_chunking import LIBEROChunkDataset, SmallChunkBC, masked_chunk_mse
from bc import NormalizationStats, list_demos


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
    stats = NormalizationStats(**checkpoint["stats"])
    if args.split == "train": demos = checkpoint["train_demos"]
    elif args.split == "val": demos = checkpoint["val_demos"]
    else: demos = list_demos(dataset_path)
    cfg = checkpoint["config"]
    dataset = LIBEROChunkDataset(dataset_path, demos, stats, cfg["chunk_size"], cfg.get("image_size", 84))
    loader = DataLoader(dataset, args.batch_size, shuffle=False, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmallChunkBC(9, 7, cfg["chunk_size"], cfg.get("pool_size", 2)).to(device)
    model.load_state_dict(checkpoint["model"]); model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for image, state, target, mask in loader:
            loss = masked_chunk_mse(model(image.to(device), state.to(device)), target.to(device), mask.to(device))
            total += loss.item() * image.size(0); count += image.size(0)
    result = {"checkpoint": str(checkpoint_path), "dataset": str(dataset_path), "split": args.split,
              "chunk_size": cfg["chunk_size"], "num_frames": count, "masked_normalized_mse": total / max(count, 1)}
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.output:
        output = Path(args.output).expanduser().resolve(); output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__": main()
