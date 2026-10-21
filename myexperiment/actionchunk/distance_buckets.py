"""Sequence/ACT 共用的物体中心距离分桶，距离单位为米。"""
import numpy as np


def get_object_distance(env, object_names):
    body_ids = env.env.obj_body_id
    missing = [name for name in object_names if name not in body_ids]
    if missing:
        raise KeyError("距离对象不存在: %s；可用对象: %s" % (missing, sorted(body_ids)))
    first, second = [env.sim.data.body_xpos[body_ids[name]] for name in object_names]
    return float(np.linalg.norm(first - second))


def measure_init_state_distances(env, init_states, object_names, settle_steps, seed=0):
    distances = []
    for index, init_state in enumerate(init_states):
        # 不依赖模型、执行策略或测量前的全局随机状态。
        env.seed(seed + index)
        env.reset()
        env.set_init_state(init_state)
        for _ in range(settle_steps):
            env.step(np.zeros(7, dtype=np.float32))
        distances.append(get_object_distance(env, object_names))
    return distances


def parse_thresholds(raw, distances):
    if raw is None:
        thresholds = np.quantile(np.asarray(distances, dtype=np.float64), [1 / 3, 2 / 3])
    else:
        thresholds = np.asarray([float(value) for value in raw.split(",")], dtype=np.float64)
    if thresholds.shape != (2,) or not np.isfinite(thresholds).all() or thresholds[0] > thresholds[1]:
        raise ValueError("距离阈值需要两个递增的有限数值")
    if raw is not None and thresholds[0] == thresholds[1]:
        raise ValueError("显式距离阈值必须严格递增")
    return thresholds.tolist()


def distance_bucket(distance, thresholds):
    if distance <= thresholds[0]:
        return "near"
    return "medium" if distance <= thresholds[1] else "far"


def bucket_summary(records):
    result = {}
    for bucket in ("near", "medium", "far"):
        selected = [r for r in records if r["distance_bucket"] == bucket]
        successes = sum(bool(r["success"]) for r in selected)
        result[bucket] = {"count": len(selected), "successes": successes,
                          "success_rate": successes / len(selected) if selected else None}
    return result
