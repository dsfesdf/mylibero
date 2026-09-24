"""训练最小 LIBERO 视觉 BC baseline。"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from bc import LIBEROFrameDataset, SmallVisualBC, compute_stats, list_demos


def parse_args():
    p = argparse.ArgumentParser(description="Train image+state BC on one LIBERO HDF5 task")
    p.add_argument("--dataset", required=True, help="单个 demonstrations .hdf5 文件")
    p.add_argument("--output", default="myexperiment/bc/outputs/bc_baseline")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--image-size", type=int, default=84)
    p.add_argument("--num-workers", type=int, default=0, help="dataset 已缓存到内存；多 worker 会复制缓存")
    return p.parse_args()


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, optimizer, device):
    training = optimizer is not None
    model.train(training)
    criterion = torch.nn.MSELoss()
    total, count = 0.0, 0
    for image, state, action in loader:
        image, state, action = image.to(device), state.to(device), action.to(device)
        with torch.set_grad_enabled(training):
            loss = criterion(model(image, state), action)
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
    if len(demos) < 2: raise ValueError("至少需要 2 个 demo 才能划分 train/validation")
    rng = random.Random(args.seed); rng.shuffle(demos)
    val_count = max(1, int(round(len(demos) * args.val_ratio)))
    val_demos, train_demos = demos[:val_count], demos[val_count:]
    stats = compute_stats(dataset_path, train_demos)
    train_set = LIBEROFrameDataset(dataset_path, train_demos, stats, args.image_size)
    val_set = LIBEROFrameDataset(dataset_path, val_demos, stats, args.image_size)
    train_loader = DataLoader(train_set, args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_set, args.batch_size, shuffle=False, num_workers=args.num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmallVisualBC(state_dim=9, action_dim=7).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_val, history = float("inf"), []
    best_checkpoint_saved = False
    print(f"device={device} train_frames={len(train_set)} val_frames={len(val_set)}", flush=True)
    interrupted = False
    try:
        for epoch in range(1, args.epochs + 1):
            train_loss = run_epoch(model, train_loader, optimizer, device)
            val_loss = run_epoch(model, val_loader, None, device)
            history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
            if val_loss < best_val:
                best_val = val_loss
                torch.save({"model": model.state_dict(), "stats": stats.to_dict(), "config": vars(args), "train_demos": train_demos, "val_demos": val_demos}, output_dir / "best.pt")
                best_checkpoint_saved = True
            # 每个完整 epoch 后立即落盘；看到下面的 loss 输出时，文件已经写好。
            (output_dir / "metrics.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
            print(f"epoch {epoch:03d}/{args.epochs:03d} train_loss={train_loss:.6f} val_loss={val_loss:.6f}", flush=True)
    except KeyboardInterrupt:
        interrupted = True
        print("\n收到 Ctrl+C：停止训练，正在保存已完成 epoch 的指标。", flush=True)
    finally:
        # 即使在 epoch 之间按 Ctrl+C，也把已经完成的 epoch 写入磁盘。
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
