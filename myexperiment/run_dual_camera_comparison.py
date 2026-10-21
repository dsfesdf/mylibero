"""四模型双相机纯视觉对比；默认只训练，--stage rollout 在笔记本评测。"""
from run_suite import main


DEFAULTS = {
    "models": ["sequence5", "act", "deterministic", "flow"],
    "seeds": [0], "split_seed": 0,
    "cameras": ["both"], "observation_mode": "image",
    "epochs": 200, "batch_size": 256, "lr": 1e-4, "num_workers": 6,
    "stage": "train", "distance_buckets": True,
    "output_root": "myexperiment/results_dual_camera/compare_seed0",
    # 主比较固定 TA；可在首次训练前设 --policies first ta 增加同权重执行消融。
    "policies": ["ta"],
    # Sequence 目前 FP32，四模型主比较统一 FP32，避免混合精度成为额外差异。
    "no_amp": True,
}


if __name__ == "__main__":
    main(DEFAULTS)
