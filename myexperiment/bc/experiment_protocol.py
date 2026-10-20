"""各模型共用的数据划分、随机种子与闭环评测协议。"""
import json
import random
from pathlib import Path

import numpy as np
import torch

PROTOCOL_VERSION = 2


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def split_demos(demos, val_ratio, split_seed, split_file=None, output_dir=None):
    """按完整 episode 划分；可复用显式名单，绝不读取训练 RNG。

    split 文件不存在时创建；存在时校验，避免模型悄悄使用不同划分。
    名单检查不能替代数据内容哈希，跨机器仍需保证 HDF5 内容一致。
    """
    demos = list(demos)
    if len(demos) < 2 or not 0 < val_ratio < 1:
        raise ValueError("至少需要两个 demo，且 0 < val_ratio < 1")
    count = max(1, int(round(len(demos) * val_ratio)))
    if count >= len(demos):
        raise ValueError("val_ratio 导致训练集为空")
    shuffled = demos.copy()
    random.Random(split_seed).shuffle(shuffled)
    record = {"protocol_version": PROTOCOL_VERSION, "split_seed": split_seed,
              "val_ratio": val_ratio, "train_demos": shuffled[count:],
              "val_demos": shuffled[:count]}
    path = Path(split_file).expanduser().resolve() if split_file else None
    if path and path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        train, val = saved["train_demos"], saved["val_demos"]
        if (not train or not val or len(set(train + val)) != len(train + val)
                or set(train + val) != set(demos)):
            raise ValueError("split 文件存在重复、遗漏或不属于当前数据集的 demo")
        if saved.get("split_seed") != split_seed or saved.get("val_ratio") != val_ratio:
            raise ValueError("split 文件的 split_seed/val_ratio 与命令行不一致")
        record = saved
    elif path:
        path.parent.mkdir(parents=True, exist_ok=True)
        # 排他创建：同时启动多组训练时，不覆盖另一进程创建的名单。
        try:
            with path.open("x", encoding="utf-8") as handle:
                json.dump(record, handle, indent=2)
        except FileExistsError:
            raise RuntimeError("split 文件正在被另一训练创建，请完成首次创建后再并行运行")
    if output_dir:
        (Path(output_dir) / "split.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record["train_demos"], record["val_demos"]


def add_rollout_args(parser, include_seed=True):
    if include_seed:
        parser.add_argument("--seed", type=int, default=0, help="环境 seed，与训练 seed 独立")
    parser.add_argument("--sampling-seed", type=int, default=None, help="动作采样 seed；默认等于环境 seed")
    parser.add_argument("--all-init-states", action="store_true")
    parser.add_argument("--init-state-offset", type=int, default=0)
    parser.add_argument("--init-state-indices", type=int, nargs="+", default=None,
                        help="显式初始状态索引，所有模型使用同一名单")
    parser.add_argument("--settle-steps", type=int, default=5)


def rollout_indices(args, count):
    if count < 1 or args.max_steps < 1 or args.settle_steps < 0:
        raise ValueError("初始状态数/max_steps 必须为正，settle_steps 不能为负")
    explicit = getattr(args, "init_state_indices", None)
    if explicit is not None:
        if args.all_init_states or args.init_state_offset:
            raise ValueError("显式 indices 不能与 all-init-states/offset 混用")
        indices = list(explicit)
    else:
        n = count if args.all_init_states else args.num_rollouts
        indices = list(range(args.init_state_offset, args.init_state_offset + n))
    if not indices or len(set(indices)) != len(indices) or any(i < 0 or i >= count for i in indices):
        raise ValueError("初始状态索引必须非空、唯一且不越界；不再通过取模重复测试")
    return indices


def seed_episode(env, args, init_index):
    """逐场景重置随机性，结果不依赖之前跑了多少个 episode。"""
    environment_seed = args.seed + init_index
    sampling_seed = getattr(args, "sampling_seed", None)
    sampling_seed = args.seed if sampling_seed is None else sampling_seed
    set_seed(sampling_seed + init_index)
    env.seed(environment_seed)


def rollout_metadata(args, indices):
    sampling_seed = getattr(args, "sampling_seed", None)
    return {"protocol_version": PROTOCOL_VERSION, "seed": args.seed,
            "sampling_seed": args.seed if sampling_seed is None else sampling_seed,
            "init_state_indices": indices, "num_rollouts": len(indices),
            "settle_steps": args.settle_steps, "max_steps": args.max_steps}
