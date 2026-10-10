# LIBERO BC、Action Chunking 与小型时序 Transformer

这里保留了三个逐步增强、可以独立训练和评测的策略：单步视觉 BC、
MLP action chunking，以及带历史观测和 action query 的小型时序 Transformer。
三者都不依赖 OpenVLA，便于在同一份 LIBERO Spatial demonstrations 上做受控对照。

## 模型输入和输出

```text
agentview_rgb (128, 128, 3) -> 小型 CNN -> 图像特征
joint_states (7) + gripper_states (2) -> MLP -> 状态特征
图像特征 + 状态特征 -> MLP -> 7 维 action
```

单步 BC 是最小基线。`actionchunk/` 中还提供：

```text
当前图像 + 当前状态 -> Chunk MLP -> 未来 K 个动作
最近 H 帧图像/状态 -> Transformer + K 个 action query -> 未来 K 个动作
```

两个 chunk 策略在 rollout 时都可选择 temporal aggregation。

## 文件

```text
myexperiment/
├── bc/
│   ├── bc.py
│   ├── train_bc.py
│   ├── evaluate_bc.py
│   ├── rollout_bc.py
│   └── outputs/
├── actionchunk/
│   ├── action_chunking.py
│   ├── train_action_chunking.py
│   ├── evaluate_action_chunking.py
│   ├── rollout_action_chunking.py
│   ├── sequence_transformer.py
│   ├── train_sequence_transformer.py
│   ├── evaluate_sequence_transformer.py
│   ├── rollout_sequence_transformer.py
│   ├── act.py
│   ├── train_act.py
│   ├── evaluate_act.py
│   ├── rollout_act.py
│   └── outputs/
├── README.md
└── BC总结.md
```

## 运行

```bash
cd ~/LIBERO
conda activate libero

python myexperiment/bc/train_bc.py \
  --dataset libero/datasets/libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate_demo.hdf5 \
  --output myexperiment/bc/outputs/bc_next_to_plate \
  --epochs 10 \
  --batch-size 64
```

3050 显存不足时减小 batch：

```bash
python myexperiment/bc/train_bc.py \
  --dataset libero/datasets/libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate_demo.hdf5 \
  --output myexperiment/bc/outputs/bc_debug \
  --epochs 2 \
  --batch-size 16
```

默认 `--num-workers 0` 是有意的：当前 LIBERO 的 HDF5 数据不能安全地在 `spawn` 多进程中 pickle。训练和验证按 episode 划分，而不是随机按帧划分，避免同一条轨迹同时出现在 train 和 validation。

## 输出

```text
myexperiment/bc/outputs/bc_next_to_plate/
├── best.pt       # 模型参数、归一化统计、配置和 demo 划分
└── metrics.json  # 每个 epoch 的 train/validation MSE
```

`best.pt` 保存了 state/action 的均值和标准差，之后反归一化动作时必须复用它们。

## 测试 checkpoint

### 1. 离线验证集评测

下面的命令会加载 `best.pt`，并在训练时预留的 10 条 validation demos 上重新计算误差：

```bash
cd ~/LIBERO
conda activate libero

python myexperiment/bc/evaluate_bc.py \
  --checkpoint myexperiment/bc/outputs/bc_smoke/best.pt \
  --split val \
  --output myexperiment/bc/outputs/bc_smoke/eval_val.json
```

输出中的主要指标：

- `normalized_mse`：归一化动作空间中的平均 MSE，方便与训练日志对照。
- `raw_mse`：LIBERO 原始 7 维 action 空间中的平均 MSE。
- `raw_rmse`：原始 action 空间中的整体 RMSE。
- `raw_rmse_by_action_dim`：每个 action 维度各自的 RMSE，用于发现某个方向或 gripper 特别难学。

也可以检查训练集误差：

```bash
python myexperiment/bc/evaluate_bc.py \
  --checkpoint myexperiment/bc/outputs/bc_smoke/best.pt \
  --split train
```

如果服务器上的数据路径和 checkpoint 中记录的本地路径不同，显式指定：

```bash
python myexperiment/bc/evaluate_bc.py \
  --checkpoint /path/to/best.pt \
  --dataset /share/2026zx/data/libero_spatial/task_demo.hdf5 \
  --split val
```

离线误差只能确认数据、模型和 checkpoint 能正常工作，不能说明机器人能完成任务。

### 2. LIBERO 闭环 rollout

该 checkpoint 使用 `LIBERO_SPATIAL` 的 task 8，因此运行：

```bash
cd ~/LIBERO
conda activate libero

MUJOCO_GL=egl python myexperiment/bc/rollout_bc.py \
  --checkpoint myexperiment/bc/outputs/bc_smoke/best.pt \
  --benchmark LIBERO_SPATIAL \
  --task-id 8 \
  --num-rollouts 5 \
  --max-steps 600 \
  --output myexperiment/bc/outputs/bc_smoke/rollouts
```

输出目录包含每条轨迹的视频及 `metrics.json`。`success_rate` 才是闭环任务效果。两轮 smoke 训练得到的模型很可能成功率为 0，这是合理结果：它主要用于验证代码闭环，而不是最终性能结论。

## 验收标准

先运行一个小实验：

```bash
python myexperiment/bc/train_bc.py \
  --dataset libero/datasets/libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate_demo.hdf5 \
  --output myexperiment/bc/outputs/bc_smoke \
  --epochs 2 \
  --batch-size 16
```

应看到：

```text
device=cuda train_frames=... val_frames=...
epoch 001/002 train_loss=... val_loss=...
epoch 002/002 train_loss=... val_loss=...
saved checkpoint: .../best.pt
```

重点检查：`device=cuda`、loss 不是 `nan`、`best.pt` 和 `metrics.json` 成功生成，并且 train loss 有下降趋势。

## 严格的多 seed 对照

训练脚本将两类随机性拆开：

- `--split-seed`：只决定哪些完整 demos 属于 train/validation。
- `--seed`：决定模型初始化、DataLoader shuffle、dropout 和训练随机性。

正式比较时固定 `split_seed=0`，只改变训练 seed。这样三个模型看到完全相同的
train/validation episodes，成功率差异不会混入“数据恰好分得更容易”的影响：

```bash
cd ~/LIBERO
conda activate libero

for seed in 0 1 2; do
  python myexperiment/actionchunk/train_action_chunking.py \
    --dataset libero/datasets/libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate_demo.hdf5 \
    --output "myexperiment/actionchunk/outputs/chunk_bc_k10_split0_seed${seed}" \
    --chunk-size 10 \
    --epochs 50 \
    --batch-size 64 \
    --lr 3e-4 \
    --weight-decay 1e-4 \
    --split-seed 0 \
    --seed "$seed"
done
```

每个 `best.pt` 都保存了 `seed`、`split_seed`、`train_demos` 和 `val_demos`。
正式报告应对三个 seed 的相同 rollout 协议汇报成功率的 `mean ± sample std`，
不要只对 validation loss 求平均。

### 提前停止训练

三个训练脚本都捕获 `Ctrl+C`。在终端看到 validation loss 开始上升时可以直接按：

```text
Ctrl+C
```

程序会把已经完成的 epoch 写入 `metrics.json`，并保留截至目前 validation loss 最低的
`best.pt`。正在执行但尚未完成的 epoch 不会写入，避免把半截 loss 当成完整 epoch。若在第一个
epoch 完成前就中断，则只会生成 `metrics.json`，此时还没有可用的 `best.pt`。

## 小型时序 Transformer

### 模型结构

```text
最近 H 帧图像 -> 共享 CNN ┐
                         ├-> H 个观测 token ┐
最近 H 帧状态 -> 共享 MLP ┘                  ├-> Transformer -> K 个动作
K 个可学习 action query --------------------┘                 (B, K, 7)
```

历史开头不足 `H` 帧时重复最早帧，轨迹末尾不足 `K` 个动作时用 mask 排除 padding loss。
默认模型为 `H=5, K=10, d_model=128, heads=4, layers=2`，约 58 万参数，仍属于用于验证
序列建模的小模型。它已经实现 `learnpath.md` 阶段三的核心机制，但不是完整论文 ACT：
目前没有 CVAE latent、KL loss 和独立 Transformer decoder。

### 训练

```bash
python myexperiment/actionchunk/train_sequence_transformer.py \
  --dataset libero/datasets/libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate_demo.hdf5 \
  --output myexperiment/actionchunk/outputs/transformer_h5_k10_split0_seed0 \
  --history-length 5 \
  --chunk-size 10 \
  --d-model 128 \
  --num-heads 4 \
  --num-layers 2 \
  --dropout 0.1 \
  --epochs 30 \
  --batch-size 64 \
  --lr 3e-4 \
  --weight-decay 1e-4 \
  --split-seed 0 \
  --seed 0
```

运行三个训练 seed 时，只需像上一节一样循环 `0 1 2`，并让输出目录包含 seed。
HDF5 读取默认保持 `--num-workers 0`，避免 `h5py objects cannot be pickled`。

### 离线评测

```bash
python myexperiment/actionchunk/evaluate_sequence_transformer.py \
  --checkpoint myexperiment/actionchunk/outputs/transformer_h5_k10_split0_seed0/best.pt \
  --split val \
  --output myexperiment/actionchunk/outputs/transformer_h5_k10_split0_seed0/eval_val.json
```

`masked_normalized_mse` 只计算真实的未来动作，不包含轨迹尾部 padding。它用于检查训练，
最终性能仍以闭环成功率为准。

### LIBERO rollout

```bash
MUJOCO_GL=egl python myexperiment/actionchunk/rollout_sequence_transformer.py \
  --checkpoint myexperiment/actionchunk/outputs/transformer_h5_k10_split0_seed0/best.pt \
  --benchmark LIBERO_SPATIAL \
  --task-id 8 \
  --num-rollouts 40 \
  --max-steps 270 \
  --temporal-aggregation \
  --aggregation-decay 0.2 \
  --history-size 10 \
  --output myexperiment/actionchunk/outputs/transformer_h5_k10_split0_seed0/rollouts_aggregate
```

去掉 `--temporal-aggregation` 即为每一步只执行最新 chunk 的第一个动作。公平对比
Chunk MLP 与 Transformer 时，应固定数据划分、训练 seed、`K`、rollout 初始状态、
最大步数和 aggregation 参数，只改变模型结构。

## 完整 ACT 框架（CVAE + Transformer decoder）

`actionchunk/act.py` 在已有 Chunk MLP 和小型时序 Transformer 之外，加入了 ACT 的核心
训练机制：

```text
训练：observation + demonstration future actions
      -> posterior Transformer -> q(z|observation, action)
      -> sample z
      -> Transformer decoder(action queries, observation memory, z)
      -> K 个动作
      -> reconstruction MSE + beta * KL(q(z)||N(0,I))

推理：observation -> z=0 -> Transformer decoder -> K 个动作
```

代码入口：

```text
act.py             # ACT 模型、CVAE posterior、KL/MSE loss、dataset
train_act.py      # 训练和 Ctrl+C 安全保存 metrics.json
evaluate_act.py   # train/val/all 离线评测
rollout_act.py    # LIBERO 闭环 rollout 和 temporal aggregation
```

这里的“完整”指 ACT 的核心 CVAE + action query + Transformer decoder 流程已经打通；
它是适合当前单任务实验的小型实现，不是论文官方代码的逐行复刻。默认 `H=1, K=10,
latent_dim=32`，先用当前观测验证机制，再做历史长度和 chunk size 消融。

### 训练 ACT

```bash
python myexperiment/actionchunk/train_act.py \
  --dataset libero/datasets/libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate_demo.hdf5 \
  --output myexperiment/actionchunk/outputs/act_h1_k10_split0_seed0 \
  --history-length 1 \
  --chunk-size 10 \
  --latent-dim 32 \
  --d-model 128 \
  --num-heads 4 \
  --num-layers 2 \
  --kl-weight 10 \
  --epochs 50 \
  --batch-size 64 \
  --lr 3e-4 \
  --weight-decay 1e-4 \
  --split-seed 0 \
  --seed 0
```

可以把 `--history-length` 改成 `5` 使用最近五帧。正式多 seed 实验固定
`--split-seed 0`，循环 `--seed 0 1 2`，每个 seed 使用独立输出目录。
训练日志中的：

- `train/val_mse`：动作 chunk 的 masked reconstruction MSE；
- `train/val_kl`：CVAE posterior 与标准正态的 KL；
- `train/val_loss`：`mse + kl_weight * kl / latent_dim`。

`best.pt` 默认按 validation reconstruction MSE 保存，而不是按总 loss 保存，方便和之前
的 Chunk MLP、小型 Transformer 做动作误差对比。按 `Ctrl+C` 会保存已经完成的 epoch 到
`metrics.json`，并保留当前最优 checkpoint。

### ACT 离线评测

```bash
python myexperiment/actionchunk/evaluate_act.py \
  --checkpoint myexperiment/actionchunk/outputs/act_h1_k10_split0_seed0/best.pt \
  --split val \
  --output myexperiment/actionchunk/outputs/act_h1_k10_split0_seed0/eval_val.json
```

### ACT 闭环 rollout

```bash
MUJOCO_GL=egl python myexperiment/actionchunk/rollout_act.py \
  --checkpoint myexperiment/actionchunk/outputs/act_h1_k10_split0_seed0/best.pt \
  --benchmark LIBERO_SPATIAL \
  --task-id 8 \
  --num-rollouts 40 \
  --max-steps 270 \
  --temporal-aggregation \
  --aggregation-decay 0.2 \
  --history-size 10 \
  --output myexperiment/actionchunk/outputs/act_h1_k10_split0_seed0/rollouts_aggregate
```

去掉 `--temporal-aggregation` 可以测每次只执行最新 chunk 第一个动作的结果。ACT 的
推理没有 demonstration action，因此代码会固定使用 `z=0`；不要在 rollout 中把训练
集的真实 future action 传给模型，否则会发生标签泄漏。

## 后续扩展

1. 对 Chunk MLP 与时序 Transformer 做三个训练 seed 的统一闭环评测。
2. 对 `H`、`K`、action query 和 temporal aggregation 做消融。
3. 对 ACT 的 latent_dim、KL 权重、history 和 chunk size 做消融。
4. 若 ACT 在统一 rollout 上稳定优于 BC/Chunk，再与 OpenVLA-OFT 使用同一任务和协议比较。

## 在双 A100 服务器训练

### 1. 当前 GPU 状态怎么判断

服务器包含：

```text
GPU 0: GeForce GT 1010 2GB
GPU 1: A100-SXM4-80GB，当前已有其他进程占用约 8.8GB
GPU 2: A100-SXM4-80GB，当前空闲
```

这个最小 BC 只有约几十万参数，一张 A100 已经远远足够，不需要双卡。当前应使用物理 `GPU 2`，不要占用他人正在使用的 `GPU 1`，也不要误用 `GPU 0` 的 GT 1010。

```bash
export CUDA_VISIBLE_DEVICES=2
```

设置后，PyTorch 只看得到这一张卡，并会把它重新编号成逻辑 `cuda:0`：

```bash
python - <<'PY'
import torch
print("available:", torch.cuda.is_available())
print("count:", torch.cuda.device_count())
print("device:", torch.cuda.get_device_name(0))
PY
```

输出应包含 `NVIDIA A100-SXM4-80GB`。代码中显示 `cuda:0` 是正常的，它对应物理 GPU 2。

### 2. 先确认服务器使用规则

```bash
which sbatch
which srun
which tmux
df -h /share/2026zx
du -sh /share/2026zx
nvidia-smi
```

如果存在 Slurm，先向管理员确认 GPU 分区、最长运行时间和提交方式，并优先使用 Slurm。若没有调度器，再使用 `tmux`，不要在 SSH 登录窗口中裸跑长期任务。

### 3. 获取代码

在服务器执行：

```bash
cd /share/2026zx
git clone git@github.com:dsfesdf/mylibero.git LIBERO
cd LIBERO
git remote -v
```
有 tmux，很好。训练必须用 tmux，因为：

ssh 一旦断开，直接跑的进程会被杀掉。
用 tmux 会话跑，断线也不影响。
tmux new -s train      # 新建名为 train 的会话
# 在会话里跑训练……
# 按 Ctrl+b 然后按 d 脱离（detach），训练继续跑
tmux attach -t train   # 下次回来重新接入
tmux ls                # 列出所有会话

如果已经克隆过：

```bash
cd /share/2026zx/LIBERO
git pull origin master
```

数据集和 checkpoint 被 `.gitignore` 忽略，不会随 Git 上传，必须单独同步。

### 4. 安装 Miniforge 和环境

不要修改服务器系统 Python、NVIDIA 驱动或系统 CUDA。驱动 `550.163.01` 已能识别 A100；PyTorch 使用环境内的 CUDA runtime。

```bash
cd /share/2026zx
wget -O Miniforge3.sh \
  https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3.sh -b -p /share/2026zx/miniforge3
source /share/2026zx/miniforge3/etc/profile.d/conda.sh

conda create -n libero python=3.8.13 pip -y
conda activate libero
python -m pip install --upgrade pip setuptools wheel
```

先安装与本地已验证环境一致、支持 A100 的 PyTorch CUDA 11.3 wheel。服务器的 550 驱动向后兼容该 CUDA runtime：

```bash
python -m pip install \
  torch==1.11.0+cu113 torchvision==0.12.0+cu113 \
  --extra-index-url https://download.pytorch.org/whl/cu113
```


再安装 LIBERO 和最小 BC 所需依赖：

```bash
cd /share/2026zx/LIBERO
python -m pip install -r requirements.txt
python -m pip install h5py==3.11.0 imageio==2.35.1 imageio-ffmpeg==0.5.1
python -m pip install -e .
```

不要根据 `nvidia-smi` 显示的 CUDA 12.4 随意替换项目原本的 PyTorch 版本。`nvidia-smi` 中的 12.4 表示驱动能支持的最高 CUDA 版本，不等于必须安装 CUDA 12.4 Toolkit。先按 LIBERO 当前依赖安装并验证：

```bash
python - <<'PY'
import h5py, torch
print("torch:", torch.__version__)
print("torch CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
PY
```

### 5. 同步数据和 checkpoint

以下命令在本地 3050 笔记本执行，将 `<server>` 替换成学校服务器地址或 SSH 配置中的主机名：

```bash
rsync -avhP \
  ~/LIBERO/libero/datasets/libero_spatial/ \
  2026zx@10.246.1.32:/share/2026zx/LIBERO/libero/datasets/libero_spatial/

rsync -avhP \
  ~/LIBERO/myexperiment/bc/outputs/bc_smoke/ \
  2026zx@10.246.1.32:/share/2026zx/LIBERO/myexperiment/bc/outputs/bc_smoke/
```

在服务器检查：

```bash
find /share/2026zx/LIBERO/libero/datasets/libero_spatial \
  -maxdepth 1 -name '*.hdf5' | wc -l
du -sh /share/2026zx/LIBERO/libero/datasets/libero_spatial
ls -lh /share/2026zx/LIBERO/myexperiment/bc/outputs/bc_smoke/best.pt
```

### 6. 先在服务器测试已有权重

```bash
cd /share/2026zx/LIBERO
source /share/2026zx/miniforge3/etc/profile.d/conda.sh
conda activate libero
export CUDA_VISIBLE_DEVICES=2

python myexperiment/bc/evaluate_bc.py \
  --checkpoint myexperiment/bc/outputs/bc_smoke/best.pt \
  --dataset libero/datasets/libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate_demo.hdf5 \
  --split val \
  --output myexperiment/bc/outputs/bc_smoke/eval_a100.json
```

### 7. 使用 tmux 训练

仅在服务器没有 Slurm，且管理员允许直接使用 GPU 时执行：

```bash
tmux new -s bc_train
cd /share/2026zx/LIBERO
source /share/2026zx/miniforge3/etc/profile.d/conda.sh
conda activate libero
export CUDA_VISIBLE_DEVICES=2
mkdir -p myexperiment/bc/outputs/bc_a100

python -u myexperiment/bc/train_bc.py \
  --dataset libero/datasets/libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate_demo.hdf5 \
  --output myexperiment/bc/outputs/bc_a100 \
  --epochs 50 \
  --batch-size 256 \
  2>&1 | tee myexperiment/bc/outputs/bc_a100/train.log
```

按 `Ctrl+B`，再按 `D`，可退出 tmux 而不中止训练。重新进入：

```bash
tmux attach -t bc_train
```

另开终端监控物理 GPU 2：

```bash
watch -n 2 nvidia-smi -i 2
```

如果显存不足，依次把 batch size 从 `256` 降为 `128`、`64`。本项目仍保持 `num_workers=0`，避免 `h5py objects cannot be pickled`。

### 8. Slurm 示例

只有在学校服务器启用了 Slurm 时使用，并将 `<gpu-partition>` 替换为管理员提供的分区名：

```bash
#!/bin/bash
#SBATCH --job-name=libero-bc
#SBATCH --partition=<gpu-partition>
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=myexperiment/bc/outputs/bc_a100/slurm-%j.out

source /share/2026zx/miniforge3/etc/profile.d/conda.sh
conda activate libero
cd /share/2026zx/LIBERO
mkdir -p myexperiment/bc/outputs/bc_a100

python -u myexperiment/bc/train_bc.py \
  --dataset libero/datasets/libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate_demo.hdf5 \
  --output myexperiment/bc/outputs/bc_a100 \
  --epochs 50 \
  --batch-size 256
```

作业提交和查看：

```bash
mkdir -p myexperiment/bc/outputs/bc_a100
sbatch myexperiment/scripts/train_bc.slurm
squeue -u "$USER"
```

Slurm 会负责分配 GPU，此时通常不要手动写死 `CUDA_VISIBLE_DEVICES=2`。

### 9. 回传结果

在本地笔记本执行：

```bash
rsync -avhP \
  2026zx@<server>:/share/2026zx/LIBERO/myexperiment/bc/outputs/bc_a100/ \
  ~/LIBERO/myexperiment/bc/outputs/bc_a100/
```

建议只回传 `best.pt`、`metrics.json`、评测 JSON 和日志。原始数据无需反复复制。

### 10. 性能优化

#### 10.1 优化前：每个样本都重新读取整条 demo

最初的 `__getitem__` 是这样实现的：

```python
def __getitem__(self, index):
    demo_name, timestep = self.index[index]
    images, states, actions = read_demo_arrays(self.hdf5_path, demo_name)  # 读整条 demo
    # 然后再对当前帧做 resize、归一化……
```

而 `read_demo_arrays` 每次调用都会：

```python
def read_demo_arrays(hdf5_path, demo_name):
    with h5py.File(hdf5_path, "r") as handle:          # ① 重新打开 HDF5 文件
        demo = handle["data"][demo_name]
        images = np.asarray(demo["obs"]["agentview_rgb"], dtype=np.uint8)   # ② 深拷贝整条图像
        joint_states = np.asarray(demo["obs"]["joint_states"], dtype=np.float32)
        gripper_states = np.asarray(demo["obs"]["gripper_states"], dtype=np.float32)
        actions = np.asarray(demo["actions"], dtype=np.float32)
    ...
```

问题很明确：**取一帧，却把整条 demo 重新读一遍**。如果一条 demo 有 500 帧，
要取第 3 帧，也会打开文件并把 500 帧图像全部读进内存再做深拷贝。而且 resize、
`/255` 归一化、state/action 标准化也都在 `__getitem__` 里**每帧重复做一遍**。

这条数据管线成为真正的瓶颈：A100 算力远没打满，显存占用只有百分之几，GPU 大部分时间在
“等数据”。瓶颈不在 GPU，而在 I/O 与重复预处理。

#### 10.2 优化后：一次性缓存到内存 + 预处理器化

核心思路：**在 dataset 初始化时，一次性把用到的每条 demo 读出来，并提前完成
resize 和归一化，之后 `__getitem__` 只做纯内存索引。**

新增一个预处理器：

```python
def prepare_demo_tensors(hdf5_path, demo_name, stats, image_size):
    images, states, actions = read_demo_arrays(hdf5_path, demo_name)   # 每条 demo 只读一次
    # 图像：HWC uint8 -> CHW float [0,1]，并提前 resize 到 image_size
    image_tensor = torch.from_numpy(images).permute(0, 3, 1, 2).float().div_(255.0)
    image_tensor = F.interpolate(
        image_tensor, (image_size, image_size), mode="bilinear", align_corners=False
    )
    # state/action：提前标准化，训练时直接取用
    state_tensor = torch.from_numpy(((states - stats.state_mean) / stats.state_std).astype(np.float32, copy=False))
    action_tensor = torch.from_numpy(((actions - stats.action_mean) / stats.action_std).astype(np.float32, copy=False))
    return image_tensor, state_tensor, action_tensor
```

各个 dataset 改为缓存实现，以逐帧 BC 的 `LIBEROFrameDataset` 为例：

```python
class LIBEROFrameDataset(Dataset):
    def __init__(self, hdf5_path, demo_names, stats, image_size=84):
        self.hdf5_path, self.stats, self.image_size = str(hdf5_path), stats, image_size
        self.demos = {}
        self.index = []
        for demo_name in demo_names:
            # 初始化时一次性读取 + resize + 归一化，缓存到 self.demos
            self.demos[demo_name] = prepare_demo_tensors(
                self.hdf5_path, demo_name, self.stats, self.image_size
            )
            length = self.demos[demo_name][2].shape[0]
            self.index.extend((demo_name, t) for t in range(length))

    def __getitem__(self, index):
        demo_name, timestep = self.index[index]
        images, states, actions = self.demos[demo_name]   # 纯内存索引，零 I/O
        return images[timestep], states[timestep], actions[timestep]
```

同样的缓存机制应用到了其它两个 dataset：

```text
bc/bc.py                            -> LIBEROFrameDataset           (逐帧 BC)
actionchunk/action_chunking.py      -> LIBEROChunkDataset           (Chunk MLP)
actionchunk/sequence_transformer.py -> LIBEROSequenceChunkDataset   (时序 Transformer / ACT)
```

`actionchunk/act.py` 中的 `LIBEROACTDataset` 直接继承 `LIBEROSequenceChunkDataset`，
所以时序 Transformer 和 ACT 共用同一份缓存数据集，无需重复实现。

`LIBEROSequenceChunkDataset` 的缓存版 `__getitem__`（返回历史 H 帧观测和未来 K 个动作）：

```python
def __getitem__(self, index):
    demo_name, timestep = self.index[index]
    images, states, actions = self.demos[demo_name]    # 已 resize、已归一化

    # 历史不足 H 帧时重复最早一帧，保持张量尺寸固定；只是索引，无需再处理
    history_indices = [max(0, timestep - self.history_length + 1 + i)
                       for i in range(self.history_length)]
    image_batch = images[history_indices]
    state_batch = states[history_indices]

    # 轨迹末尾不足 K 个动作时用最后动作填充，mask=0 让 loss 忽略这些位置
    end = min(timestep + self.chunk_size, len(actions))
    valid = end - timestep
    chunk = torch.empty((self.chunk_size, actions.shape[-1]), dtype=actions.dtype)
    mask = torch.zeros(self.chunk_size, dtype=torch.float32)
    chunk[:valid] = actions[timestep:end]
    chunk[valid:] = actions[end - 1]
    mask[:valid] = 1.0
    return image_batch, state_batch, chunk, mask
```

关键点：缓存版把 `/255`、`interpolate`、state/action 标准化都**提前到 `__init__`**，
`__getitem__` 里只剩索引和拼接，**数值与原实现完全等价**，训练结果不变，只是快很多。

#### 10.3 DataLoader 侧：多进程、锁页内存、预取

缓存之后，dataset 不再持有 h5py 文件句柄，因此可以安全地开启多进程 `DataLoader`，
不再有 `h5py objects cannot be pickled` 的问题。训练脚本（以 `train_act.py` 为例）配置：

```python
p.add_argument("--num-workers", type=int, default=4, help="缓存 dataset 的 DataLoader worker 数")

loader_kwargs = {
    "batch_size": args.batch_size,
    "num_workers": args.num_workers,          # 多进程并行取数据
    "pin_memory": torch.cuda.is_available(),  # 锁页内存，加速 CPU->GPU 拷贝
}
if args.num_workers > 0:
    loader_kwargs.update({
        "persistent_workers": True,           # worker 常驻，跨 epoch 复用
        "prefetch_factor": 2,                 # 每个 worker 预取更多 batch
    })
```

各项含义：

- `num_workers=4`：4 个子进程并行准备 batch，主进程不必等 I/O。
- `pin_memory=True`：数据放入锁页内存，`to(device, non_blocking=True)` 时拷贝更快。
- `persistent_workers=True`：epoch 之间不重建 worker，省去反复启动开销。
- `prefetch_factor=2`：每个 worker 提前准备更多 batch，减少 GPU 空等。

注意内存：缓存大小约为 (1 + `num_workers`) 份，worker 越多内存占用越高。先用
`free -h` 评估，内存紧张就把 `--num-workers` 降到 2 或 0。

#### 10.4 训练侧：异步搬运与混合精度

训练循环里，数据搬运改为异步，并可选地用 AMP 混合精度：

```python
images = images.to(device, non_blocking=True)   # 配合 pin_memory 异步传输
states = states.to(device, non_blocking=True)
target = target.to(device, non_blocking=True)
mask   = mask.to(device, non_blocking=True)

with torch.set_grad_enabled(training), torch.cuda.amp.autocast(enabled=use_amp):
    prediction, mean, logvar = model(images, states, target, mask)
    loss, reconstruction, kl = act_loss(prediction, target, mask, mean, logvar, kl_weight)

if training:
    optimizer.zero_grad(set_to_none=True)
    if scaler is None:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    else:
        scaler.scale(loss).backward()             # 放大 loss，防止 fp16 下溢
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
```

- `non_blocking=True`：与 `pin_memory` 配合，实现 CPU->GPU 异步拷贝。
- AMP：`autocast` 下前向用 fp16，`GradScaler` 处理梯度缩放，默认开启。
- `--amp` / `--no-amp` 控制是否启用。

注意：本项目的策略只有几十万参数，算力不是主要瓶颈，**最大的加速来自内存缓存（10.2）**；
AMP 对这类小模型收益有限，属于锦上添花。

#### 10.5 优化总结

| 优化 | 位置 | 作用 |
| --- | --- | --- |
| 一次性缓存 demo 到内存 | 三个 dataset 的 `__init__` | 消除每样本重复读整条 HDF5，收益最大 |
| 提前 resize 与归一化 | `prepare_demo_tensors` | `__getitem__` 只做索引，无重复计算 |
| 多进程 DataLoader | `--num-workers` | 并行准备数据，避免 GPU 等 I/O |
| 锁页内存 + 异步搬运 | `pin_memory` / `non_blocking` | 加速 CPU->GPU 拷贝 |
| 常驻 worker + 预取 | `persistent_workers` / `prefetch_factor` | 减少 epoch 间重建与空等 |
| 混合精度 | AMP | 小模型收益有限，可选 |

一句话：**瓶颈从“每个样本读整条 demo”变为“一次性入内存 + 多进程喂数据”**，
从而让 A100 不再被数据 I/O 拖慢。

## 常用操作：Git 与 rsync

### 1. Git：把代码改动同步到服务器

代码和文档用 Git 同步，**不要**用它传数据集和权重（那些被 `.gitignore` 忽略）。

```bash
# 本地：提交并推送
cd ~/LIBERO
git add -A
git commit -m "描述这次改动"
git push origin master

# 服务器：拉取最新代码
cd /share/2026zx/LIBERO
git pull origin master
```

查看当前状态：

```bash
git status            # 看有哪些改动
git log --oneline -5  # 看最近 5 次提交
```

提交前确认没有把大文件加进去（`*.pt`、`*.mp4`、`outputs/` 应被忽略）：

```bash
git add -A --dry-run    # 只预览，不真正添加
```

### 2. rsync：传数据集、权重和结果

数据集、checkpoint、rollout 视频走 rsync。**方向由“源 目标”决定**，`user@ip:路径`
一端是服务器。

服务器地址统一用 IP `2026zx@10.246.1.32`。

**上传（本地 -> 服务器）**，例如把数据集传到服务器：

```bash
rsync -avhP \
  ~/LIBERO/libero/datasets/libero_spatial/ \
  2026zx@10.246.1.32:/share/2026zx/LIBERO/libero/datasets/libero_spatial/
```

**下载（服务器 -> 本地）**，例如把训练结果和 rollout 拉回来：

```bash
rsync -avhP \
  2026zx@10.246.1.32:/share/2026zx/LIBERO/myexperiment/actionchunk/outputs/act_h1_k10_split0_seed0_a100/ \
  ~/LIBERO/myexperiment/actionchunk/outputs/act_h1_k10_split0_seed0_a100/
```

**只拉一个文件**（例如权重）：

```bash
rsync -avhP \
  2026zx@10.246.1.32:/share/2026zx/LIBERO/myexperiment/actionchunk/outputs/act_h1_k10_split0_seed0_a100/best.pt \
  ~/LIBERO/myexperiment/actionchunk/outputs/act_h1_k10_split0_seed0_a100/
```

参数说明：

- `-a`：归档模式，保留权限、时间、目录结构。
- `-v`：显示进度信息。
- `-h`：人类可读的大小。
- `-P`：显示进度并支持断点续传。

要点：

- **路径末尾的 `/` 很重要**：带 `/` 表示拷贝目录**内容**，不带表示连同目录本身一起拷贝。
- 用 IP，不要用主机名（避免解析失败）。
- 建议只回传 `best.pt`、`metrics.json`、评测 JSON、日志和需要的视频，原始数据不必反复复制。
- 想免密码，先运行一次 `ssh-copy-id 2026zx@10.246.1.32`。
