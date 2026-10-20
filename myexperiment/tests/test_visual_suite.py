"""纯视觉信息隔离、权重加载和自动化矩阵回归测试。"""
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[2]
for relative in ("myexperiment", "myexperiment/bc", "myexperiment/actionchunk"):
    sys.path.insert(0, str(ROOT / relative))
from bc import SmallVisualBC
from action_chunking import SmallChunkBC
from sequence_transformer import SmallSequenceTransformer
from act import ACT
from test_cameras_and_flow import make_fixture
import run_suite
import train_bc
import train_action_chunking
import train_sequence_transformer
import evaluate_bc
import evaluate_action_chunking
import evaluate_sequence_transformer


class VisualSuiteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_image_models_ignore_even_nan_state(self):
        for cameras, channels in (("agentview", 3), ("wrist", 3), ("both", 6)):
            images, states = torch.rand(2, 2, channels, 16, 16), torch.rand(2, 2, 9)
            policies = [SmallVisualBC(cameras=cameras, observation_mode="image"),
                        SmallChunkBC(cameras=cameras, observation_mode="image"),
                        SmallSequenceTransformer(history_length=2, d_model=32, num_layers=1,
                                                 cameras=cameras, observation_mode="image", dropout=0)]
            for model in policies:
                model.eval()
                image, state = (images, states) if isinstance(model, SmallSequenceTransformer) else (images[:, 0], states[:, 0])
                with torch.no_grad():
                    expected = model(image, state)
                    actual = model(image, torch.full_like(state, float("nan")))
                torch.testing.assert_close(expected, actual)
            for head in ("transformer", "deterministic", "flow"):
                model = ACT(history_length=2, d_model=32, num_layers=1, cameras=cameras,
                            observation_mode="image", action_head_type=head, flow_steps=2, dropout=0)
                model.eval()
                torch.manual_seed(4)
                expected = model.predict(images, states)
                torch.manual_seed(4)
                actual = model.predict(images, torch.full_like(states, float("nan")))
                torch.testing.assert_close(expected, actual)
        for cls in (SmallVisualBC, SmallChunkBC, SmallSequenceTransformer):
            with self.assertRaises(ValueError):
                cls(cameras="none", observation_mode="image")

    def test_cli_visual_configs_restore_in_offline_loaders(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "tiny.hdf5"
            make_fixture(dataset)
            for index, (trainer, evaluator, extra) in enumerate([
                (train_bc, evaluate_bc, []),
                (train_action_chunking, evaluate_action_chunking, ["--chunk-size", "3"]),
                (train_sequence_transformer, evaluate_sequence_transformer,
                 ["--history-length", "2", "--chunk-size", "3", "--d-model", "32", "--num-layers", "1"]),
            ]):
                output = root / str(index)
                command = [trainer.__file__, "--dataset", str(dataset), "--output", str(output),
                           "--epochs", "1", "--batch-size", "4", "--image-size", "16",
                           "--cameras", "both", "--observation-mode", "image"] + extra
                with patch.object(sys, "argv", command), patch("torch.cuda.is_available", return_value=False), contextlib.redirect_stdout(io.StringIO()):
                    trainer.main()
                with patch.object(sys, "argv", [evaluator.__file__, "--checkpoint", str(output / "best.pt")]), patch("torch.cuda.is_available", return_value=False), contextlib.redirect_stdout(io.StringIO()):
                    evaluator.main()
                checkpoint = torch.load(output / "best.pt", map_location="cpu")
                self.assertEqual(checkpoint["config"]["observation_mode"], "image")

    def test_default_matrix_and_completion_checks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "tiny.hdf5"
            make_fixture(dataset)
            with patch.object(sys, "argv", [run_suite.__file__, "--dataset", str(dataset)]):
                args = run_suite.parse_args()
            args.dataset, args.output_root = dataset, root / "suite"
            jobs = run_suite.build_jobs(args)
            self.assertEqual(sum(j["kind"] == "train" for j in jobs), 21)
            self.assertEqual(sum(j["kind"] == "rollout" for j in jobs), 39)
            for job in jobs:
                command = job["command"]
                if job["kind"] == "train":
                    self.assertEqual(command[command.index("--observation-mode") + 1], "image")
                else:
                    self.assertIn("--all-init-states", command)
            job = jobs[0]
            # 模拟用户 Ctrl+C 被训练器捕获、子进程正常退出，但没跑完预算。
            def incomplete(*unused, **kwargs):
                (job["output"] / "metrics.json").write_text('[{"epoch": 1}]')
                (job["output"] / "best.pt").touch()
                return type("Result", (), {"returncode": 0})()
            with patch("run_suite.subprocess.run", side_effect=incomplete):
                with self.assertRaisesRegex(RuntimeError, "未完成"):
                    run_suite.run_job(job, args, {})
            self.assertFalse((job["output"] / "train.complete.json").exists())
            with self.assertRaisesRegex(RuntimeError, "目录非空"):
                run_suite.run_job(job, args, {})


if __name__ == "__main__":
    unittest.main()
