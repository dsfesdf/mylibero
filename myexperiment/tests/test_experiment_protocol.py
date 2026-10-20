"""覆盖数据边界/缓存、无标签确定性路径、验证归约和各训练入口。"""
import contextlib
import io
import json
import pickle
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "myexperiment/bc"))
sys.path.insert(0, str(ROOT / "myexperiment/actionchunk"))
from bc import compute_stats, read_demo_arrays
from sequence_transformer import LIBEROSequenceChunkDataset
from act import ACT
from experiment_protocol import split_demos, rollout_indices, seed_episode
from test_cameras_and_flow import make_fixture
import train_bc
import train_action_chunking
import train_sequence_transformer
import train_act
import evaluate_act


class ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="libero-protocol-test-")
        self.root = Path(self.temp.name)
        self.path = self.root / "task.hdf5"
        make_fixture(self.path)
        self.stats = compute_stats(self.path, ["demo_0", "demo_1"])

    def tearDown(self):
        self.temp.cleanup()

    def test_split_manifest_validation_and_rng_independence(self):
        demos = ["demo_%d" % i for i in range(10)]
        manifest = self.root / "split.json"
        torch.manual_seed(0)
        first = split_demos(demos, .2, 0, manifest)
        torch.manual_seed(999)
        self.assertEqual(first, split_demos(demos, .2, 0, manifest))
        self.assertEqual((len(first[0]), len(first[1])), (8, 2))
        with self.assertRaises(ValueError):
            split_demos(demos, .2, 1, manifest)
        with self.assertRaises(ValueError):
            split_demos(demos[:-1], .2, 0, manifest)
        with self.assertRaises(ValueError):
            split_demos(demos, .99, 0)

    def test_cached_windows_match_original_data_and_are_picklable(self):
        for cameras in ("none", "agentview", "wrist", "both"):
            dataset = LIBEROSequenceChunkDataset(self.path, ["demo_0"], self.stats, 3, 4, 12, cameras)
            raw_images, states, actions = read_demo_arrays(self.path, "demo_0", cameras)
            with patch("h5py.File", side_effect=AssertionError("sample must not reopen HDF5")):
                for t in (0, 2, 4):
                    images, history, target, mask = dataset[t]
                    indices = [max(0, t - 2 + i) for i in range(3)]
                    torch.testing.assert_close(history, torch.from_numpy((states[indices] - self.stats.state_mean) / self.stats.state_std))
                    valid = min(4, 5 - t)
                    expected = (actions[t:t + valid] - self.stats.action_mean) / self.stats.action_std
                    torch.testing.assert_close(target[:valid], torch.from_numpy(expected))
                    self.assertEqual(mask.tolist(), [1.] * valid + [0.] * (4 - valid))
                    if valid < 4:
                        torch.testing.assert_close(target[valid:], target[valid - 1].expand(4 - valid, -1))
                    if cameras != "none":
                        reference = torch.from_numpy(raw_images[indices]).permute(0, 3, 1, 2).float() / 255
                        reference = torch.nn.functional.interpolate(reference, (12, 12), mode="bilinear", align_corners=False)
                        torch.testing.assert_close(images, reference)
                restored = pickle.loads(pickle.dumps(dataset))
                torch.testing.assert_close(dataset[-1][0], restored[-1][0])

    def test_deterministic_head_never_uses_labels_or_latent(self):
        for representation in ("global_pool", "spatial_tokens"):
            model = ACT(history_length=2, chunk_size=3, d_model=32, num_layers=1,
                        dropout=0, action_head_type="deterministic", visual_representation=representation,
                        spatial_grid=2, cameras="none")
            names = list(model.state_dict())
            self.assertFalse(any("posterior" in name or "latent" in name for name in names))
            images, states = torch.empty(2, 2, 0, 16, 16), torch.randn(2, 2, 9)
            target, mask = torch.randn(2, 3, 7), torch.ones(2, 3)
            pred, mean, logvar = model(images, states, target, mask)
            changed = model(images, states, target * 1000, mask * 0)[0]
            torch.testing.assert_close(pred, changed)
            torch.testing.assert_close(pred, model.predict(images, states))
            self.assertEqual(mean.abs().sum().item() + logvar.abs().sum().item(), 0)
            pred.square().mean().backward()
            self.assertIsNotNone(model.decoder.layers[0].linear1.weight.grad)

    def test_validation_mse_is_weighted_by_valid_actions(self):
        class Zero(torch.nn.Module):
            action_head_type, latent_dim = "deterministic", 2

            def forward(self, images, states, *unused):
                z = states.new_zeros(len(states), 2)
                return states.new_zeros(len(states), 3, 7), z, z

            def predict(self, images, states):
                return self(images, states)[0]

        target = torch.stack([torch.ones(3, 7), torch.ones(3, 7) * 3, torch.ones(3, 7) * 5])
        mask = torch.tensor([[1, 1, 1], [1, 0, 0], [1, 1, 0.]])
        dataset = TensorDataset(torch.empty(3, 1, 0, 8, 8), torch.zeros(3, 1, 9), target, mask)
        expected = (3 + 9 + 50) / 6
        for batch in (1, 2, 3):
            result = train_act.run_epoch(Zero(), DataLoader(dataset, batch_size=batch), None, torch.device("cpu"), 10)
            self.assertAlmostEqual(result["prior_mse"], expected, places=5)
            self.assertAlmostEqual(result["loss"], expected, places=5)

    def test_image_only_cvae_posterior_does_not_read_state(self):
        model = ACT(history_length=2, chunk_size=3, d_model=32, num_layers=1,
                    dropout=0, observation_mode="image", action_head_type="transformer")
        model.eval()
        images, states = torch.rand(2, 2, 3, 16, 16), torch.randn(2, 2, 9)
        actions, mask = torch.randn(2, 3, 7), torch.ones(2, 3)
        with torch.no_grad():
            first = model(images, states, actions, mask)
            second = model(images, states * 100 + 500, actions, mask)
        for expected, actual in zip(first, second):
            torch.testing.assert_close(expected, actual)

    def test_rollout_indices_and_order_independent_sampling(self):
        args = Namespace(seed=7, sampling_seed=42, max_steps=270, settle_steps=5,
                         all_init_states=False, init_state_offset=0, num_rollouts=2,
                         init_state_indices=[3, 1])
        self.assertEqual(rollout_indices(args, 5), [3, 1])
        class Env:
            def seed(self, seed):
                self.last_seed = seed
        env = Env()
        seed_episode(env, args, 3)
        reference = torch.randn(5)
        self.assertEqual(env.last_seed, 10)
        seed_episode(env, args, 1)
        torch.randn(500)
        seed_episode(env, args, 3)
        torch.testing.assert_close(reference, torch.randn(5))
        args.init_state_indices = [1, 1]
        with self.assertRaises(ValueError):
            rollout_indices(args, 5)
        args.init_state_indices = None
        args.num_rollouts = 6
        with self.assertRaises(ValueError):
            rollout_indices(args, 5)

    def test_all_trainers_share_split_and_act_best_matches_offline_evaluation(self):
        manifest = self.root / "common_split.json"
        reference_split = None
        cases = [(train_bc, []), (train_action_chunking, ["--chunk-size", "3"]),
                 (train_sequence_transformer, ["--history-length", "2", "--chunk-size", "3", "--d-model", "32", "--num-layers", "1"])]
        for head in ("transformer", "deterministic", "flow"):
            cases.append((train_act, ["--action-head", head, "--history-length", "2", "--chunk-size", "3",
                                      "--d-model", "32", "--num-layers", "1", "--flow-steps", "2", "--no-amp"]))
        for index, (module, extra) in enumerate(cases):
            output = self.root / ("run%d" % index)
            argv = [module.__file__, "--dataset", str(self.path), "--output", str(output),
                    "--epochs", "2", "--batch-size", "4", "--image-size", "16", "--cameras", "none",
                    "--split-seed", "0", "--seed", str(index), "--split-file", str(manifest)] + extra
            with patch.object(sys, "argv", argv), patch("torch.cuda.is_available", return_value=False), contextlib.redirect_stdout(io.StringIO()):
                module.main()
            ckpt = torch.load(output / "best.pt", map_location="cpu")
            split = (ckpt["train_demos"], ckpt["val_demos"])
            if reference_split is None:
                reference_split = split
            self.assertEqual(split, reference_split)
            self.assertTrue((output / "split.json").is_file())
            history = json.loads((output / "metrics.json").read_text())
            metric = "val_prior_mse" if module is train_act else "val_loss"
            best = min(history, key=lambda record: record[metric])
            self.assertEqual(ckpt["best_epoch"], best["epoch"])
            if module is train_act:
                self.assertEqual(ckpt["selection_metric"], "val_prior_mse")
                result_path = output / "eval.json"
                with patch.object(sys, "argv", [evaluate_act.__file__, "--checkpoint", str(output / "best.pt"), "--output", str(result_path)]), patch("torch.cuda.is_available", return_value=False), contextlib.redirect_stdout(io.StringIO()):
                    evaluate_act.main()
                result = json.loads(result_path.read_text())
                self.assertAlmostEqual(result["prior_mse"], best["val_prior_mse"], places=5)


if __name__ == "__main__":
    unittest.main()
