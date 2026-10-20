"""训练 ACT 的 Transformer/CVAE 或 Flow Matching 分支。"""
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

from act import ACT, LIBEROACTDataset, act_loss
from bc import compute_stats, list_demos
from experiment_protocol import split_demos, set_seed


def parse_args():
    p = argparse.ArgumentParser(description="Train a compact ACT policy on LIBERO")
    p.add_argument("--dataset", required=True)
    p.add_argument("--output", default="myexperiment/actionchunk/outputs/act_h1_k10")
    p.add_argument("--history-length", type=int, default=1)
    p.add_argument("--chunk-size", type=int, default=10)
    p.add_argument("--action-head", choices=["transformer", "deterministic", "flow"], default="transformer")
    p.add_argument("--flow-steps", type=int, default=10)
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
    p.add_argument("--split-file", default=None, help="共用 demo 划分 JSON；不存在时创建")
    p.add_argument("--val-seed", type=int, default=12345, help="固定验证生成噪声，与训练 RNG 独立")
    p.add_argument("--image-size", type=int, default=84)
    p.add_argument("--cameras", choices=["none", "agentview", "wrist", "both"], default="agentview")
    p.add_argument("--visual-representation", choices=["global_pool", "spatial_tokens"], default="global_pool")
    p.add_argument("--spatial-grid", type=int, default=6)
    p.add_argument("--observation-mode", choices=["both", "image", "state"], default="both")
    p.add_argument("--num-workers", type=int, default=0, help="CPU 张量缓存，默认单进程")
    amp_group = p.add_mutually_exclusive_group()
    amp_group.add_argument("--amp", dest="amp", action="store_true", help="在 CUDA 上使用 mixed precision")
    amp_group.add_argument("--no-amp", dest="amp", action="store_false", help="关闭 CUDA mixed precision")
    p.set_defaults(amp=True)
    return p.parse_args()


def run_epoch(model, loader, optimizer, device, kl_weight, use_amp=False, scaler=None):
    training = optimizer is not None
    model.train(training)
    totals = {"reconstruction": 0.0, "kl": 0.0, "flow": 0.0, "prior_mse": 0.0}
    frame_count, action_count = 0, 0.0
    for images, states, target, mask in loader:
        images = images.to(device, non_blocking=True)
        states = states.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        with torch.set_grad_enabled(training), torch.cuda.amp.autocast(enabled=use_amp):
            if model.action_head_type == "flow":
                loss = model.flow_loss(images, states, target, mask)
                flow = loss
                reconstruction = loss.detach().new_zeros(())
                kl = reconstruction
            else:
                prediction, mean, logvar = model(images, states, target, mask)
                loss, reconstruction, kl = act_loss(prediction, target, mask, mean, logvar, kl_weight)
                flow = loss.detach().new_zeros(())
        prior_mse = loss.detach().new_zeros(())
        if not training:
            # 用与 rollout 一致的纯观测推理选择 checkpoint，避免后验标签辅助的偏差。
            with torch.no_grad():
                prediction = model.predict(images, states)
                error = (prediction.float() - target.float()).square().mean(-1)
                prior_mse = (error * mask).sum() / mask.sum().clamp_min(1)
                if model.action_head_type == "flow":
                    reconstruction = prior_mse
        if training:
            optimizer.zero_grad(set_to_none=True)
            if scaler is None:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            else:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
        n = images.size(0)
        valid = mask.sum().item()
        totals["reconstruction"] += reconstruction.item() * valid
        totals["kl"] += kl.item() * n
        totals["flow"] += flow.item() * valid
        totals["prior_mse"] += prior_mse.item() * valid
        frame_count += n
        action_count += valid
    metrics = {key: value / max(frame_count if key == "kl" else action_count, 1)
               for key, value in totals.items()}
    metrics["loss"] = (metrics["flow"] if model.action_head_type == "flow" else
                       metrics["reconstruction"] + kl_weight * metrics["kl"] / max(model.latent_dim, 1))
    return metrics


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
    train_demos, val_demos = split_demos(demos, args.val_ratio, args.split_seed, args.split_file, output_dir)
    stats = compute_stats(dataset_path, train_demos)
    train_set = LIBEROACTDataset(dataset_path, train_demos, stats, args.history_length, args.chunk_size, args.image_size, cameras=args.cameras)
    val_set = LIBEROACTDataset(dataset_path, val_demos, stats, args.history_length, args.chunk_size, args.image_size, cameras=args.cameras)
    loader_generator = torch.Generator().manual_seed(args.seed)
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if args.num_workers > 0:
        loader_kwargs.update({"persistent_workers": True, "prefetch_factor": 2})
    train_loader = DataLoader(
        train_set, shuffle=True, generator=loader_generator, **loader_kwargs
    )
    val_loader = DataLoader(val_set, shuffle=False, **loader_kwargs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ACT(
        state_dim=len(stats.state_mean), action_dim=len(stats.action_mean),
        history_length=args.history_length, chunk_size=args.chunk_size,
        latent_dim=args.latent_dim, d_model=args.d_model,
        num_heads=args.num_heads, num_layers=args.num_layers, dropout=args.dropout,
        visual_representation=args.visual_representation,
        spatial_grid=args.spatial_grid,
        observation_mode=args.observation_mode,
        cameras=args.cameras, action_head_type=args.action_head, flow_steps=args.flow_steps,
    ).to(device)
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_val, history, interrupted = float("inf"), [], False
    best_checkpoint_saved = False
    metrics_path = output_dir / "metrics.json"
    print(
        f"device={device} amp={use_amp} workers={args.num_workers} "
        f"history={args.history_length} chunk={args.chunk_size} latent={args.latent_dim} "
        f"train_frames={len(train_set)} val_frames={len(val_set)}",
        flush=True,
    )
    try:
        for epoch in range(1, args.epochs + 1):
            train_metrics = run_epoch(
                model, train_loader, optimizer, device, args.kl_weight, use_amp, scaler
            )
            # 固定验证噪声，且不改变训练 RNG 状态；各 epoch 的 flow MSE 可比较。
            devices = [torch.cuda.current_device()] if device.type == "cuda" else []
            with torch.random.fork_rng(devices=devices), torch.no_grad():
                torch.manual_seed(args.val_seed)
                if device.type == "cuda":
                    torch.cuda.manual_seed(args.val_seed)
                val_metrics = run_epoch(model, val_loader, None, device, args.kl_weight)
            record = {"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"val_{k}": v for k, v in val_metrics.items()}}
            # 不计算的指标写 null，避免把占位 0 误读成完美拟合。
            record["train_prior_mse"] = None
            if model.action_head_type == "flow":
                record["train_reconstruction"] = None
            history.append(record)
            if val_metrics["prior_mse"] < best_val:
                best_val = val_metrics["prior_mse"]
                torch.save({
                    "model": model.state_dict(), "stats": stats.to_dict(), "config": vars(args),
                    "selection_metric": "val_prior_mse",
                    "best_epoch": epoch, "best_val_prior_mse": best_val,
                    "train_demos": train_demos, "val_demos": val_demos,
                }, output_dir / "best.pt")
                best_checkpoint_saved = True
            save_metrics(metrics_path, history)
            train_mse_label = f"{train_metrics['reconstruction']:.6f}" if model.action_head_type != "flow" else "n/a"
            print(
                f"epoch {epoch:03d}/{args.epochs:03d} "
                f"train_loss={train_metrics['loss']:.6f} train_mse={train_mse_label} "
                f"train_flow={train_metrics['flow']:.6f} "
                f"val_loss={val_metrics['loss']:.6f} val_mse={val_metrics['reconstruction']:.6f} "
                f"val_kl={val_metrics['kl']:.6f} val_flow={val_metrics['flow']:.6f} "
                f"val_prior_mse={val_metrics['prior_mse']:.6f}", flush=True,
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
