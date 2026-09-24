"""训练最小 action chunking BC；不会修改原有 train_bc.py。"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

# 让本文件直接运行时能找到同目录的 action_chunking.py 和 ../bc/bc.py。
_HERE = os.path.dirname(os.path.abspath(__file__))
_BC_DIR = os.path.join(os.path.dirname(_HERE), "bc")
for _path in (_HERE, _BC_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import numpy as np
import torch
from torch.utils.data import DataLoader

from action_chunking import LIBEROChunkDataset, SmallChunkBC, masked_chunk_mse
from bc import compute_stats, list_demos


def parse_args():
    p = argparse.ArgumentParser(description="Train minimal action-chunking BC on LIBERO")
    p.add_argument("--dataset", required=True)
    p.add_argument("--output", default="myexperiment/actionchunk/outputs/chunk_bc")
    p.add_argument("--chunk-size", type=int, default=10, help="每次预测未来 K 个动作")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0, help="只控制模型初始化、shuffle 和训练随机性")
    p.add_argument("--split-seed", type=int, default=0, help="只控制 train/val demo 划分")
    p.add_argument("--image-size", type=int, default=84)
    p.add_argument("--pool-size", type=int, default=2, choices=[1, 2, 4])
    p.add_argument("--num-workers", type=int, default=0, help="dataset 已缓存到内存；多 worker 会复制缓存")
    return p.parse_args()


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    # 同一个 seed 重跑时尽量复现；不同 seed 仍会改变初始化和训练随机性。
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_epoch(model, loader, optimizer, device):
    training = optimizer is not None
    model.train(training)
    total, count = 0.0, 0
    for image, state, target, mask in loader:
        image, state = image.to(device), state.to(device)
        target, mask = target.to(device), mask.to(device)
        with torch.set_grad_enabled(training):
            loss = masked_chunk_mse(model(image, state), target, mask)
            if training:
                optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        total += loss.item() * image.size(0); count += image.size(0)
    return total / max(count, 1)


def main():
    args = parse_args(); set_seed(args.seed)
    dataset_path = Path(args.dataset).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve(); output_dir.mkdir(parents=True, exist_ok=True)
    if not dataset_path.is_file(): raise FileNotFoundError(f"找不到数据集: {dataset_path}")

    demos = list_demos(dataset_path)
    if len(demos) < 2: raise ValueError("至少需要 2 个 demo")
    # split_seed 固定后，多个 train seed 会使用完全相同的 train/val demos。
    rng = random.Random(args.split_seed); rng.shuffle(demos)
    val_count = max(1, int(round(len(demos) * args.val_ratio)))
    val_demos, train_demos = demos[:val_count], demos[val_count:]
    stats = compute_stats(dataset_path, train_demos)
    train_set = LIBEROChunkDataset(dataset_path, train_demos, stats, args.chunk_size, args.image_size)
    val_set = LIBEROChunkDataset(dataset_path, val_demos, stats, args.chunk_size, args.image_size)
    loader_generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_set, args.batch_size, shuffle=True, num_workers=args.num_workers,
        generator=loader_generator,
    )
    val_loader = DataLoader(val_set, args.batch_size, shuffle=False, num_workers=args.num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmallChunkBC(9, 7, args.chunk_size, args.pool_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_val, history = float("inf"), []
    best_checkpoint_saved = False
    print(f"device={device} chunk_size={args.chunk_size} train_frames={len(train_set)} val_frames={len(val_set)}", flush=True)
    interrupted = False
    try:
        for epoch in range(1, args.epochs + 1):
            train_loss = run_epoch(model, train_loader, optimizer, device)
            val_loss = run_epoch(model, val_loader, None, device)
            history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
            if val_loss < best_val:
                best_val = val_loss
                torch.save({
                    "model": model.state_dict(), "stats": stats.to_dict(), "config": vars(args),
                    "train_demos": train_demos, "val_demos": val_demos,
                }, output_dir / "best.pt")
                best_checkpoint_saved = True
            (output_dir / "metrics.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
            print(f"epoch {epoch:03d}/{args.epochs:03d} train_loss={train_loss:.6f} val_loss={val_loss:.6f}", flush=True)
    except KeyboardInterrupt:
        interrupted = True
        print("\n收到 Ctrl+C：停止训练，正在保存已完成 epoch 的指标。", flush=True)
    finally:
        (output_dir / "metrics.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(f"saved metrics: {output_dir / 'metrics.json'}", flush=True)
    if interrupted:
        checkpoint_path = output_dir / "best.pt"
        if best_checkpoint_saved:
            print(f"保留当前最佳 checkpoint: {checkpoint_path}", flush=True)
        else:
            print("本次运行尚未完成任何 epoch，因此没有生成新的 best.pt。", flush=True)
    else:
        print(f"saved checkpoint: {output_dir / 'best.pt'}", flush=True)


if __name__ == "__main__": main()
