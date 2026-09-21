# 最小视觉 Behavior Cloning

这是 `mylibero` 仓库中的第一个个人实验：使用一个 LIBERO Spatial demonstration 文件，训练一个不依赖 OpenVLA 的最小视觉 BC 策略。

## 模型输入和输出

```text
agentview_rgb (128, 128, 3) -> 小型 CNN -> 图像特征
joint_states (7) + gripper_states (2) -> MLP -> 状态特征
图像特征 + 状态特征 -> MLP -> 7 维 action
```

这是单步 BC：当前图像和当前状态预测当前动作。暂时没有语言、历史帧、action chunking 或闭环 rollout，目的是先验证数据读取、归一化、训练和 checkpoint。

## 文件

```text
myexperiment/
├── bc.py          # HDF5 Dataset、归一化统计和 CNN+MLP 策略
├── train_bc.py    # 训练入口
├── evaluate_bc.py # checkpoint 离线评测入口
├── rollout_bc.py  # LIBERO 闭环 rollout、成功率和视频
├── README.md
└── outputs/       # 运行后生成的 checkpoint 和指标
```

## 运行

```bash
cd ~/LIBERO
conda activate libero

python myexperiment/train_bc.py \
  --dataset libero/datasets/libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate_demo.hdf5 \
  --output myexperiment/outputs/bc_next_to_plate \
  --epochs 10 \
  --batch-size 64
```

3050 显存不足时减小 batch：

```bash
python myexperiment/train_bc.py \
  --dataset libero/datasets/libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate_demo.hdf5 \
  --output myexperiment/outputs/bc_debug \
  --epochs 2 \
  --batch-size 16
```

默认 `--num-workers 0` 是有意的：当前 LIBERO 的 HDF5 数据不能安全地在 `spawn` 多进程中 pickle。训练和验证按 episode 划分，而不是随机按帧划分，避免同一条轨迹同时出现在 train 和 validation。

## 输出

```text
myexperiment/outputs/bc_next_to_plate/
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

python myexperiment/evaluate_bc.py \
  --checkpoint myexperiment/outputs/bc_smoke/best.pt \
  --split val \
  --output myexperiment/outputs/bc_smoke/eval_val.json
```

输出中的主要指标：

- `normalized_mse`：归一化动作空间中的平均 MSE，方便与训练日志对照。
- `raw_mse`：LIBERO 原始 7 维 action 空间中的平均 MSE。
- `raw_rmse`：原始 action 空间中的整体 RMSE。
- `raw_rmse_by_action_dim`：每个 action 维度各自的 RMSE，用于发现某个方向或 gripper 特别难学。

也可以检查训练集误差：

```bash
python myexperiment/evaluate_bc.py \
  --checkpoint myexperiment/outputs/bc_smoke/best.pt \
  --split train
```

如果服务器上的数据路径和 checkpoint 中记录的本地路径不同，显式指定：

```bash
python myexperiment/evaluate_bc.py \
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

MUJOCO_GL=egl python myexperiment/rollout_bc.py \
  --checkpoint myexperiment/outputs/bc_smoke/best.pt \
  --benchmark LIBERO_SPATIAL \
  --task-id 8 \
  --num-rollouts 5 \
  --max-steps 600 \
  --output myexperiment/outputs/bc_smoke/rollouts
```

输出目录包含每条轨迹的视频及 `metrics.json`。`success_rate` 才是闭环任务效果。两轮 smoke 训练得到的模型很可能成功率为 0，这是合理结果：它主要用于验证代码闭环，而不是最终性能结论。

## 验收标准

先运行一个小实验：

```bash
python myexperiment/train_bc.py \
  --dataset libero/datasets/libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate_demo.hdf5 \
  --output myexperiment/outputs/bc_smoke \
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

## 后续扩展

1. 增加最近多帧输入。
2. 增加 action chunking。
3. 实现 ACT Transformer。
4. 与官方 lifelong BC 和 OpenVLA-OFT 做统一评测。

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
  2026zx@<server>:/share/2026zx/LIBERO/libero/datasets/libero_spatial/

rsync -avhP \
  ~/LIBERO/myexperiment/outputs/bc_smoke/ \
  2026zx@<server>:/share/2026zx/LIBERO/myexperiment/outputs/bc_smoke/
```

在服务器检查：

```bash
find /share/2026zx/LIBERO/libero/datasets/libero_spatial \
  -maxdepth 1 -name '*.hdf5' | wc -l
du -sh /share/2026zx/LIBERO/libero/datasets/libero_spatial
ls -lh /share/2026zx/LIBERO/myexperiment/outputs/bc_smoke/best.pt
```

### 6. 先在服务器测试已有权重

```bash
cd /share/2026zx/LIBERO
source /share/2026zx/miniforge3/etc/profile.d/conda.sh
conda activate libero
export CUDA_VISIBLE_DEVICES=2

python myexperiment/evaluate_bc.py \
  --checkpoint myexperiment/outputs/bc_smoke/best.pt \
  --dataset libero/datasets/libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate_demo.hdf5 \
  --split val \
  --output myexperiment/outputs/bc_smoke/eval_a100.json
```

### 7. 使用 tmux 训练

仅在服务器没有 Slurm，且管理员允许直接使用 GPU 时执行：

```bash
tmux new -s bc_train
cd /share/2026zx/LIBERO
source /share/2026zx/miniforge3/etc/profile.d/conda.sh
conda activate libero
export CUDA_VISIBLE_DEVICES=2
mkdir -p myexperiment/outputs/bc_a100

python -u myexperiment/train_bc.py \
  --dataset libero/datasets/libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate_demo.hdf5 \
  --output myexperiment/outputs/bc_a100 \
  --epochs 50 \
  --batch-size 256 \
  2>&1 | tee myexperiment/outputs/bc_a100/train.log
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
#SBATCH --output=myexperiment/outputs/bc_a100/slurm-%j.out

source /share/2026zx/miniforge3/etc/profile.d/conda.sh
conda activate libero
cd /share/2026zx/LIBERO
mkdir -p myexperiment/outputs/bc_a100

python -u myexperiment/train_bc.py \
  --dataset libero/datasets/libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate_demo.hdf5 \
  --output myexperiment/outputs/bc_a100 \
  --epochs 50 \
  --batch-size 256
```

作业提交和查看：

```bash
mkdir -p myexperiment/outputs/bc_a100
sbatch myexperiment/scripts/train_bc.slurm
squeue -u "$USER"
```

Slurm 会负责分配 GPU，此时通常不要手动写死 `CUDA_VISIBLE_DEVICES=2`。

### 9. 回传结果

在本地笔记本执行：

```bash
rsync -avhP \
  2026zx@<server>:/share/2026zx/LIBERO/myexperiment/outputs/bc_a100/ \
  ~/LIBERO/myexperiment/outputs/bc_a100/
```

建议只回传 `best.pt`、`metrics.json`、评测 JSON 和日志。原始数据无需反复复制。
