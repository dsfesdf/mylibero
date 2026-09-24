"""在 checkpoint 保存的 held-out demos 上离线评测最小 BC。

离线 MSE 用于确认模型和 checkpoint 是否正常，但它不能替代 LIBERO 环境中的
闭环 rollout 成功率。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from bc import LIBEROFrameDataset, NormalizationStats, SmallVisualBC, list_demos


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a minimal BC checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", default=None, help="默认使用 checkpoint config 中的数据路径")
    parser.add_argument("--split", choices=["val", "train", "all"], default="val")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--output", default=None, help="可选的 JSON 指标输出路径")
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    dataset_value = args.dataset or checkpoint["config"]["dataset"]
    dataset_path = Path(dataset_value).expanduser()
    if not dataset_path.is_absolute():
        # checkpoint 中保存的是从 LIBERO 根目录运行时使用的相对路径。
        dataset_path = Path.cwd() / dataset_path
    dataset_path = dataset_path.resolve()
    if not dataset_path.is_file():
        raise FileNotFoundError(f"找不到数据集: {dataset_path}，请通过 --dataset 指定服务器上的实际路径")

    stats = NormalizationStats(**checkpoint["stats"])
    if args.split == "val":
        demo_names = checkpoint["val_demos"]
    elif args.split == "train":
        demo_names = checkpoint["train_demos"]
    else:
        demo_names = list_demos(dataset_path)

    image_size = checkpoint["config"].get("image_size", 84)
    dataset = LIBEROFrameDataset(dataset_path, demo_names, stats, image_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmallVisualBC(state_dim=len(stats.state_mean), action_dim=len(stats.action_mean)).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    normalized_squared_error = np.zeros(len(stats.action_mean), dtype=np.float64)
    raw_squared_error = np.zeros(len(stats.action_mean), dtype=np.float64)
    count = 0
    action_mean = torch.from_numpy(stats.action_mean).to(device)
    action_std = torch.from_numpy(stats.action_std).to(device)

    with torch.no_grad():
        for image, state, normalized_target in loader:
            image, state = image.to(device), state.to(device)
            normalized_target = normalized_target.to(device)
            normalized_prediction = model(image, state)
            normalized_error = normalized_prediction - normalized_target

            # 将预测与标签反归一化，得到 LIBERO 原始动作空间中的误差。
            raw_prediction = normalized_prediction * action_std + action_mean
            raw_target = normalized_target * action_std + action_mean
            raw_error = raw_prediction - raw_target
            normalized_squared_error += normalized_error.square().sum(0).cpu().numpy()
            raw_squared_error += raw_error.square().sum(0).cpu().numpy()
            count += image.size(0)

    normalized_mse_by_dim = normalized_squared_error / count
    raw_mse_by_dim = raw_squared_error / count
    metrics = {
        "checkpoint": str(checkpoint_path),
        "dataset": str(dataset_path),
        "split": args.split,
        "num_demos": len(demo_names),
        "num_frames": count,
        "normalized_mse": float(normalized_mse_by_dim.mean()),
        "raw_mse": float(raw_mse_by_dim.mean()),
        "raw_rmse": float(np.sqrt(raw_mse_by_dim.mean())),
        "normalized_mse_by_action_dim": normalized_mse_by_dim.tolist(),
        "raw_rmse_by_action_dim": np.sqrt(raw_mse_by_dim).tolist(),
    }
    print(json.dumps(metrics, indent=2, ensure_ascii=False))

    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"saved metrics: {output_path}")


if __name__ == "__main__":
    main()
