"""顺序训练与闭环评测；默认只预览，--execute 才运行，不实现 epoch 恢复。"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shlex
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "bc": ("bc", "bc", 1, 1, None),
    "chunk1": ("actionchunk", "action_chunking", 1, 1, None),
    "chunk10": ("actionchunk", "action_chunking", 1, 10, None),
    "sequence1": ("actionchunk", "sequence_transformer", 1, 10, None),
    "sequence5": ("actionchunk", "sequence_transformer", 5, 10, None),
    "act": ("actionchunk", "act", 5, 10, "transformer"),
    "deterministic": ("actionchunk", "act", 5, 10, "deterministic"),
    "flow": ("actionchunk", "act", 5, 10, "flow"),
}
DEFAULT_MODELS = [name for name in MODELS if name != "bc"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default=None, help="训练必填；rollout 阶段从 suite.json 恢复配置")
    p.add_argument("--output-root", default="myexperiment/results_v2/vision_suite")
    p.add_argument("--models", nargs="+", choices=list(MODELS), default=DEFAULT_MODELS)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--cameras", nargs="+", choices=["none", "agentview", "wrist", "both"], default=["agentview"])
    p.add_argument("--observation-mode", choices=["image", "both", "state"], default="image")
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--val-ratio", type=float, default=.2)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=2, help="训练 DataLoader worker 数，可设为0")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--image-size", type=int, default=84)
    p.add_argument("--flow-steps", type=int, default=10)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--benchmark", default="LIBERO_SPATIAL")
    p.add_argument("--task-id", type=int, default=8)
    p.add_argument("--max-steps", type=int, default=270)
    p.add_argument("--settle-steps", type=int, default=5)
    p.add_argument("--init-state-indices", nargs="+", type=int, default=None,
                   help="默认全部官方场景；可指定同一组调试或最终场景")
    p.add_argument("--rollout-seed", type=int, default=0)
    p.add_argument("--sampling-seed", type=int, default=123)
    p.add_argument("--aggregation-decay", type=float, default=.2)
    p.add_argument("--policies", nargs="+", choices=["first", "ta"], default=["first", "ta"])
    p.add_argument("--stage", choices=["all", "train", "rollout"], default="all")
    p.add_argument("--gpu", default=None, help="CUDA_VISIBLE_DEVICES，例：2；不指定则继承当前环境")
    p.add_argument("--execute", action="store_true")
    return p.parse_args()


def build_jobs(args):
    """纯函数式生成命令；顺序执行，同一权重的 first/TA 不重复训练。"""
    jobs = []
    base = [sys.executable]
    for camera in args.cameras:
        if args.observation_mode == "image" and camera == "none":
            raise ValueError("image 模式不能选择 cameras=none")
        for name in args.models:
            folder, script, history, chunk, head = MODELS[name]
            for seed in args.seeds:
                run = args.output_root / (name + "_" + camera + "_" + args.observation_mode) / ("split%d_seed%d" % (args.split_seed, seed))
                train = base + [str(ROOT / "myexperiment" / folder / ("train_" + script + ".py")),
                    "--dataset", str(args.dataset), "--output", str(run),
                    "--split-file", str(args.output_root / "split.json"),
                    "--split-seed", str(args.split_seed), "--seed", str(seed),
                    "--val-ratio", str(args.val_ratio), "--epochs", str(args.epochs),
                    "--batch-size", str(args.batch_size), "--lr", str(args.lr),
                    "--image-size", str(args.image_size), "--num-workers", str(args.num_workers),
                    "--cameras", camera, "--observation-mode", args.observation_mode]
                if name != "bc":
                    train += ["--chunk-size", str(chunk), "--weight-decay", str(args.weight_decay)]
                if script in ("sequence_transformer", "act"):
                    train += ["--history-length", str(history), "--d-model", "128",
                              "--num-heads", "4", "--num-layers", "2", "--dropout", "0.1"]
                if head:
                    train += ["--action-head", head, "--flow-steps", str(args.flow_steps),
                              "--visual-representation", "global_pool", "--latent-dim", "32",
                              "--kl-weight", "10", "--val-seed", "12345",
                              "--no-amp" if args.no_amp else "--amp"]
                jobs.append(dict(kind="train", model=name, camera=camera, seed=seed,
                                 policy="", run=run, output=run, command=train))
                for policy in args.policies:
                    if chunk == 1 and policy == "ta":
                        continue
                    output = run / ("eval_" + policy)
                    cmd = base + [str(ROOT / "myexperiment" / folder / ("rollout_" + script + ".py")),
                        "--checkpoint", str(run / "best.pt"), "--output", str(output),
                        "--benchmark", args.benchmark, "--task-id", str(args.task_id),
                        "--max-steps", str(args.max_steps), "--settle-steps", str(args.settle_steps),
                        "--seed", str(args.rollout_seed), "--sampling-seed", str(args.sampling_seed)]
                    if args.init_state_indices is None:
                        cmd += ["--all-init-states"]
                    else:
                        cmd += ["--init-state-indices"] + list(map(str, args.init_state_indices))
                    if policy == "ta":
                        cmd += ["--temporal-aggregation", "--aggregation-decay", str(args.aggregation_decay),
                                "--history-size", str(chunk)]
                    if head == "flow":
                        cmd += ["--flow-steps", str(args.flow_steps)]
                    jobs.append(dict(kind="rollout", model=name, camera=camera, seed=seed,
                                     policy=policy, run=run, output=output, command=cmd))
    # 稳定分组：所有模型/seed 训练结束后才开始评测。
    return [job for job in jobs if job["kind"] == "train"] + [job for job in jobs if job["kind"] == "rollout"]


def restore_rollout_config(args):
    """跨机器评测直接复用训练协议，不要求复制 HDF5 或保留服务器绝对路径。

    rollout 除 output-root/gpu/execute 外的参数由 manifest 决定，避免无意使用默认值。
    """
    manifest = args.output_root / "suite.json"
    if not manifest.is_file():
        raise FileNotFoundError("rollout 阶段需要复制完整套件目录（含 suite.json）：%s" % manifest)
    saved = json.loads(manifest.read_text())
    for key, value in saved.items():
        if key in vars(args) and key not in ("output_root", "stage", "gpu", "execute"):
            setattr(args, key, value)
    # 旧套件硬编码 workers0；保持原协议，不按新默认值误认成 workers2。
    args.num_workers = saved.get("num_workers", 0)
    args.dataset = Path(saved["dataset"])
    print("Rollout settings restored from:", manifest, flush=True)


def run_job(job, args, env):
    output = job["output"]
    marker = output / ("train.complete.json" if job["kind"] == "train" else "rollout.complete.json")
    if marker.exists():
        saved_command = json.loads(marker.read_text())["command"]
        current_command = job["command"]
        if job["kind"] == "rollout":
            # 数据目录移动后解释器、脚本、权重和输出的绝对路径会变化；
            # 模型/seed 由所在目录确定，仍严格比较所有评测参数。
            def portable(command):
                result = [Path(command[1]).name] + command[2:]
                for option in ("--checkpoint", "--output"):
                    index = result.index(option)
                    result[index + 1] = "<relocated>"
                return result
            saved_command, current_command = portable(saved_command), portable(current_command)
        if saved_command != current_command:
            raise RuntimeError("完成标记与当前命令不一致：%s" % output)
        if not (output / "metrics.json").exists() or (job["kind"] == "train" and not (output / "best.pt").exists()):
            raise RuntimeError("完成标记存在但产物缺失：%s" % output)
        print("SKIP completed:", output, flush=True)
        return
    # 不覆盖中断的训练或未经此脚本确认的结果。
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("目录非空但没有完成标记，请用新 output-root 或自行归档该目录：%s" % output)
    if job["kind"] == "rollout" and not (job["run"] / "train.complete.json").exists():
        raise RuntimeError("缺少完整训练标记：%s" % job["run"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "command.json").write_text(json.dumps(job["command"], indent=2))
    with (output / (job["kind"] + ".log")).open("w") as log:
        # 前台同进程组，Ctrl+C 会发给训练子进程；不把中断运行标记为完成。
        result = subprocess.run(job["command"], cwd=str(ROOT), env=env, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError("子进程失败，请检查日志：%s" % output)
    metrics = json.loads((output / "metrics.json").read_text())
    if job["kind"] == "train":
        if not metrics or metrics[-1]["epoch"] != args.epochs or not (output / "best.pt").exists():
            raise RuntimeError("训练未完成规定 epoch，不进入 rollout：%s" % output)
    elif not metrics.get("successes") or len(metrics["successes"]) != metrics["num_rollouts"]:
        raise RuntimeError("rollout 结果不完整：%s" % output)
    marker.write_text(json.dumps({"command": job["command"]}, indent=2))


def summarize(jobs, root):
    rows, groups = [], {}
    for job in jobs:
        if job["kind"] != "rollout" or not (job["output"] / "rollout.complete.json").exists():
            continue
        metrics = json.loads((job["output"] / "metrics.json").read_text())
        row = {key: job[key] for key in ("model", "camera", "seed", "policy")}
        row.update(successes=sum(metrics["successes"]), episodes=metrics["num_rollouts"],
                   success_rate=metrics["success_rate"], path=str(job["output"]))
        rows.append(row)
        groups.setdefault((job["model"], job["camera"], job["policy"]), []).append(row["success_rate"])
    if rows:
        with (root / "summary.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        aggregate = [{"model": key[0], "camera": key[1], "policy": key[2], "seeds_completed": len(values),
                      "mean": statistics.mean(values), "std": statistics.stdev(values) if len(values) > 1 else None}
                     for key, values in groups.items()]
        (root / "aggregate.json").write_text(json.dumps(aggregate, indent=2))


def main():
    args = parse_args()
    args.output_root = Path(args.output_root).expanduser().resolve()
    if args.stage == "rollout":
        restore_rollout_config(args)
    else:
        if args.dataset is None:
            raise ValueError("训练阶段必须指定 --dataset")
        args.dataset = Path(args.dataset).expanduser().resolve()
        if not args.dataset.is_file():
            raise FileNotFoundError(args.dataset)
    if min(args.epochs, args.batch_size, args.max_steps, args.image_size, args.flow_steps) < 1 or not 0 < args.val_ratio < 1:
        raise ValueError("训练/评测尺寸必须为正，且 0 < val_ratio < 1")
    if args.num_workers < 0:
        raise ValueError("num-workers 不能为负")
    for values in (args.seeds, args.models, args.cameras, args.policies):
        if len(set(values)) != len(values):
            raise ValueError("列表参数不能包含重复项")
    jobs = build_jobs(args)
    selected = [job for job in jobs if args.stage == "all" or job["kind"] == args.stage]
    for job in selected:
        print(shlex.join(job["command"]), flush=True)
    if not args.execute:
        print("Preview only. Add --execute to run. No files created.")
        return
    # 固定套件配置，避免在同一结果目录混入不同预算、数据或场景。
    record = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
              if key not in ("execute", "stage", "gpu")}
    if args.stage != "rollout":
        digest = hashlib.sha256()
        with args.dataset.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
        record["dataset_sha256"] = digest.hexdigest()
        args.output_root.mkdir(parents=True, exist_ok=True)
        manifest = args.output_root / "suite.json"
        if manifest.exists():
            saved = json.loads(manifest.read_text())
            saved.setdefault("num_workers", 0)
            if saved != record:
                differences = ["%s: saved=%r, current=%r" % (key, saved.get(key), record.get(key))
                               for key in sorted(set(saved) | set(record)) if saved.get(key) != record.get(key)]
                raise RuntimeError("套件配置已冻结，请使用新的 output-root；差异：\n" + "\n".join(differences))
        manifest.write_text(json.dumps(record, indent=2))
    env = os.environ.copy()
    env.setdefault("MUJOCO_GL", "egl")
    if args.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.gpu
    try:
        for i, job in enumerate(selected):
            print("[%d/%d] %s %s seed=%s -> %s" % (i + 1, len(selected), job["kind"], job["model"], job["seed"], job["output"]), flush=True)
            run_job(job, args, env)
            summarize(jobs, args.output_root)
    except KeyboardInterrupt:
        print("Interrupted; incomplete jobs were not marked complete.", file=sys.stderr)
        raise SystemExit(130)
    print("Finished:", args.output_root, flush=True)


if __name__ == "__main__":
    main()
