"""训练 CVAE + Transformer decoder 版 ACT。"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BC_DIR = HERE.parent / "bc"
for path in (HERE, BC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np
import torch
from torch.utils.data import DataLoader

from act import ACT, LIBEROACTDataset, act_loss
from bc import compute_stats, list_demos


def parse_args():
    p = argparse.ArgumentParser(description="Train a compact ACT policy on LIBERO")
    p.add_argument("--dataset", required=True)
    p.add_argument("--output", default="myexperiment/actionchunk/outputs/act_h1_k10")
    p.add_argument("--history-length", type=int, default=1)
    p.add_argument("--chunk-size", type=int, default=10)
    p.add_argument("--latent-dim", type=int, default=32)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--kl-weight", type=float, default=10.0)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0, help="模型、shuffle 和训练随机性")
    p.add_argument("--split-seed", type=int, default=0, help="只控制 episode 划分")
    p.add_argument("--image-size", type=int, default=84)
    p.add_argument("--num-workers", type=int, default=0)
    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_epoch(model, loader, optimizer, device, kl_weight):
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "reconstruction": 0.0, "kl": 0.0, "count": 0}
    for images, states, target, mask in loader:
        images, states = images.to(device), states.to(device)
        target, mask = target.to(device), mask.to(device)
        with torch.set_grad_enabled(training):
            prediction, mean, logvar = model(images, states, target, mask)
            loss, reconstruction, kl = act_loss(prediction, target, mask, mean, logvar, kl_weight)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        n = images.size(0)
        totals["loss"] += loss.item() * n
        totals["reconstruction"] += reconstruction.item() * n
        totals["kl"] += kl.item() * n
        totals["count"] += n
    count = max(totals.pop("count"), 1)
    return {key: value / count for key, value in totals.items()}


def save_metrics(path: Path, history):
    # 直接写临时文件再替换，避免 Ctrl+C 恰好发生在写文件时留下半个 JSON。
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(history, indent=2), encoding="utf-8")
    temporary.replace(path)


def main():
    args = parse_args()
    set_seed(args.seed)
    dataset_path = Path(args.dataset).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"找不到数据集: {dataset_path}")

    demos = list_demos(dataset_path)
    if len(demos) < 2:
        raise ValueError("至少需要 2 个 demo")
    split_rng = random.Random(args.split_seed)
    split_rng.shuffle(demos)
    val_count = max(1, int(round(len(demos) * args.val_ratio)))
    val_demos, train_demos = demos[:val_count], demos[val_count:]
    stats = compute_stats(dataset_path, train_demos)
    train_set = LIBEROACTDataset(dataset_path, train_demos, stats, args.history_length, args.chunk_size, args.image_size)
    val_set = LIBEROACTDataset(dataset_path, val_demos, stats, args.history_length, args.chunk_size, args.image_size)
    loader_generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_set, args.batch_size, shuffle=True, num_workers=args.num_workers, generator=loader_generator)
    val_loader = DataLoader(val_set, args.batch_size, shuffle=False, num_workers=args.num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ACT(
        state_dim=len(stats.state_mean), action_dim=len(stats.action_mean),
        history_length=args.history_length, chunk_size=args.chunk_size,
        latent_dim=args.latent_dim, d_model=args.d_model,
        num_heads=args.num_heads, num_layers=args.num_layers, dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_val, history, interrupted = float("inf"), [], False
    best_checkpoint_saved = False
    metrics_path = output_dir / "metrics.json"
    print(f"device={device} history={args.history_length} chunk={args.chunk_size} latent={args.latent_dim} train_frames={len(train_set)} val_frames={len(val_set)}", flush=True)
    try:
        for epoch in range(1, args.epochs + 1):
            train_metrics = run_epoch(model, train_loader, optimizer, device, args.kl_weight)
            with torch.no_grad():
                val_metrics = run_epoch(model, val_loader, None, device, args.kl_weight)
            record = {"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"val_{k}": v for k, v in val_metrics.items()}}
            history.append(record)
            if val_metrics["reconstruction"] < best_val:
                best_val = val_metrics["reconstruction"]
                torch.save({
                    "model": model.state_dict(), "stats": stats.to_dict(), "config": vars(args),
                    "train_demos": train_demos, "val_demos": val_demos,
                }, output_dir / "best.pt")
                best_checkpoint_saved = True
            save_metrics(metrics_path, history)
            print(
                f"epoch {epoch:03d}/{args.epochs:03d} "
                f"train_loss={train_metrics['loss']:.6f} train_mse={train_metrics['reconstruction']:.6f} "
                f"val_loss={val_metrics['loss']:.6f} val_mse={val_metrics['reconstruction']:.6f} "
                f"val_kl={val_metrics['kl']:.6f}", flush=True,
            )
    except KeyboardInterrupt:
        interrupted = True
        print("\n收到 Ctrl+C：停止 ACT 训练，正在保存已完成 epoch 的指标。", flush=True)
    finally:
        save_metrics(metrics_path, history)
        print(f"saved metrics: {metrics_path}", flush=True)
    if interrupted:
        if best_checkpoint_saved:
            print(f"保留当前最佳 checkpoint: {output_dir / 'best.pt'}", flush=True)
        else:
            print("尚未完成任何 epoch，因此没有新的 best.pt。", flush=True)
    else:
        print(f"saved checkpoint: {output_dir / 'best.pt'}", flush=True)


if __name__ == "__main__":
    main()
