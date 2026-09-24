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

### 10性能提升
__getitem__ 每次重开文件

Apply
def __getitem__(self, index):
    demo_name, timestep = self.index[index]
    images, states, actions = read_demo_arrays(self.hdf5_path, demo_name)  # 读整条 demo
    ...
缺点很明确：取一帧，却把整条 demo 重新读一遍。如果一条 demo 有 500 帧，你要取第 3 帧，它把 500 帧全读了。效率很低（我上次建议的改进点就是这个）。正确做法一般是缓存到内存，或用 num_workers + 每进程保存句柄。
