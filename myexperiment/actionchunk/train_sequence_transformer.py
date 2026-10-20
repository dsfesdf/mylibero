"""训练小型时序 Transformer；原有单步/Chunk MLP 脚本保持不变。"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BC_DIR = HERE.parent / "bc"
for path in (HERE, BC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import torch
from torch.utils.data import DataLoader

from bc import compute_stats, list_demos
from experiment_protocol import split_demos, set_seed
from sequence_transformer import LIBEROSequenceChunkDataset, SmallSequenceTransformer, masked_chunk_mse


def parse_args():
    p = argparse.ArgumentParser(description="Train a small history Transformer on LIBERO")
    p.add_argument("--dataset", required=True)
    p.add_argument("--output", default="myexperiment/actionchunk/outputs/transformer_h5_k10")
    p.add_argument("--history-length", type=int, default=5)
    p.add_argument("--chunk-size", type=int, default=10)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0, help="只控制模型初始化、shuffle 和训练随机性")
    p.add_argument("--split-seed", type=int, default=0, help="只控制 train/val demo 划分")
    p.add_argument("--split-file", default=None)
    p.add_argument("--image-size", type=int, default=84)
    p.add_argument("--cameras", choices=["none", "agentview", "wrist", "both"], default="agentview")
    p.add_argument("--observation-mode", choices=["both", "image", "state"], default="both")
    p.add_argument("--num-workers", type=int, default=0, help="dataset 已缓存到内存；多 worker 会复制缓存")
    return p.parse_args()


def run_epoch(model, loader, optimizer, device):
    training = optimizer is not None
    model.train(training)
    total, count = 0.0, 0
    for images, states, target, mask in loader:
        images, states = images.to(device), states.to(device)
        target, mask = target.to(device), mask.to(device)
        with torch.set_grad_enabled(training):
            loss = masked_chunk_mse(model(images, states), target, mask)
            if training:
                optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        valid = mask.sum().item()
        total += loss.item() * valid; count += valid
    return total / max(count, 1)


def main():
    args = parse_args(); set_seed(args.seed)
    dataset_path = Path(args.dataset).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve(); output_dir.mkdir(parents=True, exist_ok=True)
    if not dataset_path.is_file(): raise FileNotFoundError(f"找不到数据集: {dataset_path}")
    demos = list_demos(dataset_path)
    train_demos, val_demos = split_demos(demos, args.val_ratio, args.split_seed, args.split_file, output_dir)
    stats = compute_stats(dataset_path, train_demos)
    train_set = LIBEROSequenceChunkDataset(dataset_path, train_demos, stats, args.history_length, args.chunk_size, args.image_size, cameras=args.cameras)
    val_set = LIBEROSequenceChunkDataset(dataset_path, val_demos, stats, args.history_length, args.chunk_size, args.image_size, cameras=args.cameras)
    loader_generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_set, args.batch_size, shuffle=True, num_workers=args.num_workers,
        generator=loader_generator,
    )
    val_loader = DataLoader(val_set, args.batch_size, shuffle=False, num_workers=args.num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmallSequenceTransformer(
        state_dim=9, action_dim=7, history_length=args.history_length, chunk_size=args.chunk_size,
        d_model=args.d_model, num_heads=args.num_heads, num_layers=args.num_layers, dropout=args.dropout,
        cameras=args.cameras, observation_mode=args.observation_mode,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_val, history = float("inf"), []
    best_checkpoint_saved = False
    print(f"device={device} history={args.history_length} chunk={args.chunk_size} train_frames={len(train_set)} val_frames={len(val_set)}", flush=True)
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
                    "selection_metric": "val_mse", "best_epoch": epoch, "best_val_mse": best_val,
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
