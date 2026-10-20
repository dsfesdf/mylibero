"""验证相机顺序、零视觉输入、末尾 padding、流匹配训练与旧权重兼容。"""
import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "myexperiment/bc"))
sys.path.insert(0, str(ROOT / "myexperiment/actionchunk"))
from bc import (LIBEROFrameDataset, SmallVisualBC, camera_channels,
                compute_stats, observation_image)
from action_chunking import LIBEROChunkDataset, SmallChunkBC
from sequence_transformer import LIBEROSequenceChunkDataset, SmallSequenceTransformer
from act import ACT, act_loss


def make_fixture(path):
    with h5py.File(path, "w") as handle:
        for index in range(3):
            demo = handle.create_group(f"data/demo_{index}")
            obs = demo.create_group("obs")
            obs["agentview_rgb"] = np.full((5, 16, 16, 3), 30, dtype=np.uint8)
            obs["eye_in_hand_rgb"] = np.full((5, 16, 16, 3), 180, dtype=np.uint8)
            obs["joint_states"] = np.arange(35, dtype=np.float32).reshape(5, 7)
            obs["gripper_states"] = np.ones((5, 2), dtype=np.float32)
            demo["actions"] = np.linspace(-1, 1, 35, dtype=np.float32).reshape(5, 7)


class CameraFlowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temp = tempfile.TemporaryDirectory(prefix="libero-camera-test-")
        cls.path = Path(cls.temp.name) / "task.hdf5"
        make_fixture(cls.path)
        cls.stats = compute_stats(cls.path, ["demo_0", "demo_1"])

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_datasets_match_rollout_and_all_policies_backward(self):
        obs = {"agentview_image": np.full((16, 16, 3), 30, dtype=np.uint8),
               "robot0_eye_in_hand_image": np.full((16, 16, 3), 180, dtype=np.uint8)}
        for cameras in ("none", "agentview", "wrist", "both"):
            with self.subTest(cameras=cameras):
                frame = LIBEROFrameDataset(self.path, ["demo_0"], self.stats, 16, cameras)
                chunk = LIBEROChunkDataset(self.path, ["demo_0"], self.stats, 3, 16, cameras)
                seq = LIBEROSequenceChunkDataset(self.path, ["demo_0"], self.stats, 2, 3, 16, cameras)
                image, state, target = frame[0]
                self.assertEqual(image.shape[0], camera_channels(cameras))
                torch.testing.assert_close(image, observation_image(obs, 16, cameras))
                model = SmallVisualBC(cameras=cameras)
                model(image[None], state[None]).square().mean().backward()
                image, state, target, mask = chunk[-1]
                self.assertEqual(mask.tolist(), [1, 0, 0])
                model = SmallChunkBC(chunk_size=3, cameras=cameras)
                model(image[None], state[None]).square().mean().backward()
                images, states, actions, mask = seq[-1]
                model = SmallSequenceTransformer(history_length=2, chunk_size=3, d_model=32, num_layers=1, cameras=cameras)
                self.assertEqual(tuple(model(images[None], states[None]).shape), (1, 3, 7))
                model(images[None], states[None]).square().mean().backward()
                for representation in ("global_pool", "spatial_tokens"):
                    for head in ("transformer", "deterministic", "flow"):
                        model = ACT(history_length=2, chunk_size=3, d_model=32, num_layers=1,
                                    latent_dim=4, cameras=cameras, visual_representation=representation,
                                    spatial_grid=2, action_head_type=head, flow_steps=2)
                        if head == "flow":
                            loss = model.flow_loss(images[None], states[None], actions[None], mask[None])
                        else:
                            pred, mean, var = model(images[None], states[None], actions[None], mask[None])
                            loss = act_loss(pred, actions[None], mask[None], mean, var)[0]
                        self.assertTrue(torch.isfinite(loss))
                        loss.backward()
                        model.eval()
                        pred = model.predict(images[None], states[None])
                        self.assertEqual(tuple(pred.shape), (1, 3, 7))
                        self.assertTrue(torch.isfinite(pred).all())

    def test_zero_cameras_need_no_image_keys(self):
        with h5py.File(self.path, "a") as handle:
            obs = handle["data/demo_2/obs"]
            del obs["agentview_rgb"]
            del obs["eye_in_hand_rgb"]
        compute_stats(self.path, ["demo_2"])
        frame = LIBEROFrameDataset(self.path, ["demo_2"], self.stats, 16, "none")
        seq = LIBEROSequenceChunkDataset(self.path, ["demo_2"], self.stats, 2, 3, 16, "none")
        self.assertEqual(frame[0][0].shape[0], 0)
        self.assertEqual(seq[0][0].shape[1], 0)
        self.assertEqual(observation_image({}, 16, "none").shape[0], 0)
        with self.assertRaises(KeyError):
            observation_image({}, 16, "wrist")

    def test_flow_ignores_padding_and_integrates_velocity(self):
        model = ACT(history_length=2, chunk_size=3, d_model=32, num_layers=1,
                    cameras="none", action_head_type="flow", dropout=0)
        images = torch.empty(2, 2, 0, 16, 16)
        states = torch.randn(2, 2, 9)
        actions = torch.randn(2, 3, 7)
        mask = torch.tensor([[1, 0, 0], [1, 1, 0.]])
        torch.manual_seed(9)
        first = model.flow_loss(images, states, actions, mask)
        changed = actions.clone()
        changed[mask == 0] = 10000
        torch.manual_seed(9)
        second = model.flow_loss(images, states, changed, mask)
        torch.testing.assert_close(first, second)
        memory = model.encode_observation(images, states)
        noise = torch.zeros(2, 3, 7)
        # 常速度 v=1 的 Euler 积分，t=0->1 后应该精确得到 1。
        from unittest.mock import patch
        with patch.object(model.flow_head, "forward", side_effect=lambda x, t, m: torch.ones_like(x)):
            torch.testing.assert_close(model.flow_head.sample(memory, 5, noise), torch.ones_like(noise))

    def test_existing_checkpoints_load_strictly(self):
        bc_path = ROOT / "myexperiment/bc/outputs/22poolBC/best.pt"
        if bc_path.exists():
            checkpoint = torch.load(bc_path, map_location="cpu")
            SmallVisualBC().load_state_dict(checkpoint["model"], strict=True)
        paths = {
            "chunk_bc_k10": lambda c: SmallChunkBC(chunk_size=c["chunk_size"], pool_size=c.get("pool_size", 2)),
            "transformer_h5_k10_split0_seed0": lambda c: SmallSequenceTransformer(
                history_length=c["history_length"], chunk_size=c["chunk_size"], d_model=c["d_model"],
                num_heads=c["num_heads"], num_layers=c["num_layers"], dropout=c["dropout"]),
            "act_h10_k10_split0_seed0": lambda c: ACT(
                history_length=c["history_length"], chunk_size=c["chunk_size"], d_model=c["d_model"],
                num_heads=c["num_heads"], num_layers=c["num_layers"], latent_dim=c["latent_dim"], dropout=c["dropout"]),
        }
        for name, factory in paths.items():
            path = ROOT / "myexperiment/actionchunk/outputs" / name / "best.pt"
            if path.exists():
                checkpoint = torch.load(path, map_location="cpu")
                factory(checkpoint["config"]).load_state_dict(checkpoint["model"], strict=True)


if __name__ == "__main__":
    unittest.main()
