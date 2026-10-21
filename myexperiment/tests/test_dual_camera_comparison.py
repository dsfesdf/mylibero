"""四模型对比矩阵、距离分桶与汇总契约。"""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "myexperiment"), str(ROOT / "myexperiment/actionchunk")]
import run_suite
from run_dual_camera_comparison import DEFAULTS
from distance_buckets import measure_init_state_distances, parse_thresholds, distance_bucket, bucket_summary


class DualCameraTests(unittest.TestCase):
    def test_defaults_commands_and_stages(self):
        with patch.object(sys, "argv", ["run_dual_camera_comparison.py", "--dataset", "/tmp/data.hdf5"]):
            args = run_suite.parse_args(DEFAULTS)
        args.dataset, args.output_root = Path(args.dataset), Path("/tmp/results")
        jobs = run_suite.build_jobs(args)
        self.assertEqual(args.stage, "train")
        self.assertEqual([j["kind"] for j in jobs], ["train"] * 4 + ["rollout"] * 4)
        self.assertEqual({j["seed"] for j in jobs}, {0})
        for job in jobs[:4]:
            cmd = job["command"]
            for key, value in [("--cameras", "both"), ("--observation-mode", "image"),
                               ("--num-workers", "6"), ("--epochs", "200"),
                               ("--batch-size", "256"), ("--history-length", "5"),
                               ("--split-seed", "0"), ("--seed", "0")]:
                self.assertEqual(cmd[cmd.index(key) + 1], value)
            self.assertEqual(float(cmd[cmd.index("--lr") + 1]), 1e-4)
        for job in jobs[4:]:
            self.assertIn("--distance-objects", job["command"])
            self.assertIn("--temporal-aggregation", job["command"])
        self.assertIn("--distance-buckets", jobs[4]["command"])

    def test_measurement_seed_boundaries_and_empty_bucket(self):
        class Env:
            def __init__(self):
                from types import SimpleNamespace
                self.env = SimpleNamespace(obj_body_id={"a": 0, "b": 1})
                self.sim = SimpleNamespace(data=SimpleNamespace(body_xpos=np.zeros((2, 3))))
                self.seeds = []

            def seed(self, seed): self.seeds.append(seed)
            def reset(self): pass
            def set_init_state(self, state): self.sim.data.body_xpos[1, 0] = state
            def step(self, action): pass
        env = Env()
        distances = measure_init_state_distances(env, [.1, .2, .3], ["a", "b"], 5, seed=7)
        self.assertEqual(env.seeds, [7, 8, 9])
        self.assertEqual(distances, [.1, .2, .3])
        thresholds = parse_thresholds(None, distances)
        self.assertEqual([distance_bucket(d, thresholds) for d in distances], ["near", "medium", "far"])
        self.assertEqual(distance_bucket(.2, [.2, .3]), "near")
        self.assertEqual(distance_bucket(.3, [.2, .3]), "medium")
        with self.assertRaises(ValueError): parse_thresholds("nan,1", distances)
        result = bucket_summary([dict(distance_bucket="near", success=True), dict(distance_bucket="near", success=False)])
        self.assertEqual(result["near"]["success_rate"], .5)
        self.assertIsNone(result["far"]["success_rate"])

    def test_bucket_csv(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output / "rollout.complete.json").write_text('{}')
            (output / "metrics.json").write_text(json.dumps({
                "successes": [True], "num_rollouts": 1, "success_rate": 1.,
                "distance_thresholds": [.1, .2],
                "by_distance_bucket": bucket_summary([dict(distance_bucket="near", success=True)])}))
            run_suite.summarize([dict(kind="rollout", model="sequence5", camera="both", seed=0,
                                      policy="ta", output=output)], output)
            content = (output / "distance_summary.csv").read_text()
            self.assertIn("near,1,1,1.0", content)
            self.assertIn("thresholds_m", content)


if __name__ == "__main__":
    unittest.main()
