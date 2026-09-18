# LIBERO Policy Study

一个面向机器人模仿学习的实践项目：从数据审计和最小 Behavior Cloning（BC）开始，逐步实现 Action Chunking Transformer（ACT），最后在相同任务、数据和评测协议下与 OpenVLA-OFT 对比。

本项目的重点不是堆叠更大的模型，而是建立一条可复现、可解释的实验闭环：

```text
数据审计 -> Behavior Cloning -> 闭环 Rollout -> ACT -> 消融实验 -> OpenVLA-OFT 对照
```

> 当前状态：项目规划阶段。本文中的 `train_bc.py`、`train_act.py` 和统一评测脚本需要在独立学习项目中实现，不是 LIBERO 原仓库现有命令。LIBERO 原仓库负责提供任务、仿真环境、示范数据、基线策略和持续学习框架。

## 1. 项目目标

- 先理解 LIBERO 原生 HDF5 数据，再理解转换后的 RLDS 数据中 observation、state、action 和 language instruction 的组织方式。
- 独立实现一个不依赖 OpenVLA 的视觉 Behavior Cloning 策略。
- 理解 action chunking、历史观测和 temporal aggregation 的作用。
- 在统一实验条件下比较 BC、ACT 和 OpenVLA-OFT。
- 记录训练成本、推理延迟、成功率与失败类型，形成可复现实验报告。

### 1.1 项目边界

本学习项目不是重写 LIBERO。两者分工如下：

| 项目 | 负责内容 | 不负责内容 |
| --- | --- | --- |
| `LIBERO` | BDDL 任务、MuJoCo/robosuite 环境、HDF5 demonstrations、固定初始状态、官方 BC 与持续学习基线 | ACT、OpenVLA-OFT、面向本项目的统一 BC/ACT 实验接口 |
| `libero-policy-study` | 数据审计、最小 BC、ACT、统一 rollout、实验记录和模型对比 | 重新实现物理引擎、机器人控制器和全部 LIBERO 任务 |

第一轮只使用 LIBERO 原生 HDF5。等 HDF5 训练与 rollout 闭环通过后，再增加 RLDS adapter；不要让两种数据格式同时成为第一阶段的调试变量。

## 2. 硬件分工

| 设备 | 主要用途 | 不建议承担的工作 |
| --- | --- | --- |
| RTX 3050 4GB | 数据审计、可视化、小型 BC、单元测试、代码调试 | OpenVLA 全量训练、大规模多任务实验 |
| 双 A100 服务器 | ACT 训练、消融实验、批量 rollout、OpenVLA-OFT 低成本微调 | 未验证数据和评测流程前的长时间训练 |
| RTX 4090D 40GB | 空闲后用于中型训练和部署验证 | 当前项目不依赖该设备 |

推荐工作流：在 3050 上开发与检查，在 A100 上训练和批量评测，在本地整理图表与报告。

## 3. 计划中的目录结构

“在仓库旁边实现”是指保留官方仓库 `/home/zx/LIBERO`，另建同级目录 `/home/zx/libero-policy-study`。学习代码通过 `pip install -e /home/zx/LIBERO` 导入 LIBERO，不直接塞进 `libero/lifelong`。这样可以清楚区分官方代码和自己的实现，也方便更新上游、做 Git 对比和复现实验。

```text
/home/zx/
├── LIBERO/                  # 官方上游仓库，尽量少改
└── libero-policy-study/     # 自己实现和维护的实验项目
```

```text
libero-policy-study/
├── configs/                 # 数据、模型、训练和评测配置
├── data_audit/              # 数据统计、轨迹可视化与对齐检查
├── datasets/                # 数据加载与预处理代码（不提交原始数据）
├── policies/
│   ├── bc.py                # 单步 Behavior Cloning
│   ├── act.py               # Action Chunking Transformer
│   └── diffusion_policy.py  # 可选扩展
├── evaluation/              # LIBERO rollout 与指标统计
├── scripts/                 # 训练、评测和服务器作业脚本
├── tests/                   # 数据与模型接口测试
├── results/
│   ├── tables/
│   ├── plots/
│   └── videos/
├── report.md
└── README.md
```

## 4. 环境准备

### 4.1 适用范围和安装原则

以下步骤适用于 Ubuntu 22.04/24.04、x86_64、RTX 3050 本地机器。Python 和 PyTorch 放在 Conda 环境中；PyTorch 官方 wheel 自带 CUDA runtime，因此通常不需要单独安装完整 CUDA Toolkit。只有需要编译 CUDA 扩展时才安装 Toolkit。

共享 A100 服务器的 NVIDIA 驱动、系统 CUDA 和基础镜像通常由管理员维护。没有管理员许可时，只在自己的 Conda 环境中安装 Python 包，不要执行系统级驱动或 CUDA 安装。

### 4.2 系统基础工具

```bash
sudo apt update
sudo apt full-upgrade -y
sudo apt install -y git git-lfs curl wget unzip zip tmux rsync htop tree \
build-essential pkg-config cmake ninja-build \
  libgl1 libglib2.0-0 libsm6 libxext6 libxrender1

git lfs install
git --version
```

### 4.3 NVIDIA 驱动（本地机器）

先检查 GPU 和当前驱动状态：

```bash
lspci | grep -i nvidia
nvidia-smi
ubuntu-drivers devices
```

如果 `nvidia-smi` 不存在或没有识别 GPU，可以让 Ubuntu 选择推荐驱动：

```bash
sudo ubuntu-drivers autoinstall
sudo reboot
```

重启后必须确认：

```bash
nvidia-smi
```

如果这里仍然失败，先解决驱动问题，不要继续安装 PyTorch。驱动版本需要满足所选 CUDA runtime 的最低要求；不要混用多个来源的 NVIDIA 驱动。

### 4.4 CUDA Toolkit（可选）

运行 PyTorch 训练不要求 `nvcc`。先确认：

```bash
command -v nvcc || true
```

只有需要编译自定义 CUDA 算子时，才按照 NVIDIA 官方 CUDA 安装页面选择与 Ubuntu 版本匹配的 Toolkit。安装后验证：

```bash
nvcc --version
```

不要为了让 `nvcc --version` 有输出而额外安装 Toolkit；它和 PyTorch wheel 自带的 CUDA runtime 是两套不同组件。

### 4.5 安装 Miniconda

```bash
cd /tmp
wget -O miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash miniconda.sh -b -p "$HOME/miniconda3"
"$HOME/miniconda3/bin/conda" init bash
source "$HOME/.bashrc"
conda config --set channel_priority strict
conda --version
```

如果 shell 不是 bash，把 `conda init bash` 改成对应 shell，或重新打开终端。

### 4.6 使用已验证的 LIBERO 环境

```bash
conda activate libero
cd /home/zx/LIBERO
python --version
python -c "import torch, robosuite, robomimic; print(torch.__version__, torch.cuda.is_available())"
```

当前机器已验证的组合是 Python 3.8.13、PyTorch 1.11.0+cu113、robosuite 1.4.0 和 robomimic 0.2.0。这个仓库依赖较旧，不要在原环境里直接升级到最新版 Python、NumPy、PyTorch 或 Gym。

如果需要重建官方环境，优先严格按照当前仓库的 `README.md` 和 `requirements.txt` 建立 `libero` 环境。自己的学习项目可以先克隆已验证环境，避免实验依赖污染官方环境：

```bash
conda create --name libero-study --clone libero
conda activate libero-study
python -m pip install -e /home/zx/LIBERO
```

后续 OpenVLA-OFT 使用另一个独立环境，避免新旧 PyTorch、Transformers 和 robosuite 依赖冲突。

### 4.7 验证 PyTorch 与离屏仿真

不要只验证 `torch.cuda.is_available()`，还要验证 MuJoCo EGL 离屏渲染。训练依赖 PyTorch GPU，批量 rollout 同时依赖 EGL 和显卡设备号。

```bash
python - <<'PY'
import torch

print("torch:", torch.__version__)
print("cuda runtime:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    x = torch.randn(2, 3, device="cuda")
    print("tensor device:", x.device)
PY
```

`cuda available` 必须为 `True`。运行 rollout 时设置 `MUJOCO_GL=egl` 和 `MUJOCO_EGL_DEVICE_ID=<GPU_ID>`。如果失败，依次检查 `nvidia-smi`、驱动、EGL、当前 Conda 环境和 PyTorch wheel。

### 4.8 安装项目通用依赖

下面的工具依赖安装在克隆出的 `libero-study` 环境中。先运行 `python -m pip freeze > before-study.txt` 留下快照，不要修改已验证的原始 `libero` 环境。

```bash
python -m pip install \
  numpy matplotlib seaborn pandas \
  opencv-python imageio imageio-ffmpeg \
  pyyaml tqdm h5py einops scipy scikit-learn \
  jupyterlab pytest
```

确认基础 Python 包可导入：

```bash
python - <<'PY'
import cv2, h5py, imageio, matplotlib, numpy, pandas, torch, yaml
print("基础依赖导入成功")
PY
```

只有开始读取 RLDS 时才在单独环境中增加 TensorFlow 和 TensorFlow Datasets。最小 BC 和 ACT 使用原生 HDF5 不需要 TensorFlow。

### 4.9 安装 LIBERO 和 OpenVLA-OFT

当前机器已经有 `/home/zx/LIBERO`，不需要重复 clone。下面只适用于新机器。两个项目的依赖和启动接口会随上游仓库变化，应分别按对应 commit 的官方 README 安装，并记录仓库 commit：

```bash
mkdir -p "$HOME/src"
cd "$HOME/src"
git clone <LIBERO-official-repository-url> LIBERO
git clone <OpenVLA-OFT-official-repository-url> openvla-oft

cd LIBERO
git rev-parse HEAD
# 按该仓库 README 执行安装命令

cd "$HOME/src/openvla-oft"
git rev-parse HEAD
# 按该仓库 README 执行安装命令
```

安装完成后，把实际使用的仓库地址、commit、额外依赖和命令写入项目的 `environment.yml`、`requirements.txt` 或 `report.md`，保证以后可以复现。

### 4.10 保存环境快照

```bash
conda env export --from-history > environment.yml
python -m pip freeze > requirements-lock.txt
python -c "import torch; print(torch.__version__, torch.version.cuda)"
```

`environment.yml` 保存 Conda 层依赖，`requirements-lock.txt` 保存 pip 层依赖。两个文件都应提交到 Git。

### 4.11 A100 服务器检查

首次登录服务器只进行环境和资源确认：

```bash
ssh <username>@<server-address>
nvidia-smi
df -h
du -sh "$HOME"
which tmux
which sbatch
which srun
```

同时确认以下信息：

- A100 型号和单卡显存。
- 是否使用 Slurm，以及 GPU 分区名称。
- 单个作业的最长运行时间和 GPU 申请规则。
- 用户目录配额、数据盘和 checkpoint 存放路径。
- CUDA、驱动、PyTorch 和 Python 版本。
- 是否允许使用 Docker 或 Apptainer。

如果服务器使用 Slurm，训练应通过作业脚本提交，不应长期占用登录节点。

## 5. 数据准备

第一阶段只选择一个 `LIBERO-Spatial` 任务和固定数量的 demonstrations。原始数据、checkpoint 和视频不提交到 Git。

当前机器的 `libero/datasets/libero_spatial` 已包含 10 个任务。首个任务实测有 50 条轨迹、5068 个 transition；图像为双相机 `128 x 128 x 3`，action 为范围 `[-1, 1]` 的 7 维向量。

建议通过环境变量管理机器之间不同的数据路径：

```bash
export LIBERO_DATA_ROOT=/path/to/libero/dataset
export EXPERIMENT_ROOT=/path/to/experiment/output
```

数据加载完成后，必须先确认：

- episode 数量及长度分布。
- observation 字段、相机名称、图像 shape 和数据类型。
- robot state 的维度、含义和数值范围。
- action 的维度、范围、均值和标准差。
- gripper action 的编码方式。
- instruction 的种类和频率。
- 空轨迹、异常值和 train/test 重复情况。
- 图像、state 和 action 在时间上是否对齐。

## 6. 阶段一：数据审计

计划提供以下命令接口：

```bash
python -m data_audit.inspect_dataset \
  --data-root "$LIBERO_DATA_ROOT" \
  --task <task-name>

python -m data_audit.visualize_episode \
  --data-root "$LIBERO_DATA_ROOT" \
  --task <task-name> \
  --episode-id 0 \
  --output results/videos/episode_000.mp4

python -m data_audit.plot_action_distribution \
  --data-root "$LIBERO_DATA_ROOT" \
  --task <task-name> \
  --output-dir results/plots
```

本阶段验收标准：

- 能打印一条样本的全部字段和 tensor shape。
- 能生成 episode 长度及 action 分布图。
- 能播放一条随机轨迹并检查图像与动作对齐。
- 能明确说明 action normalization 和 unnormalization 的规则。
- 产出一份 `results/data_audit.md` 报告。

## 7. 阶段二：最小 Behavior Cloning

模型结构：

```text
image -> small CNN -> image feature
state -> MLP       -> state feature
concat(image feature, state feature) -> MLP -> one action
```

建议先使用单张低分辨率图像、当前 robot state 和单步 action。第一版训练目标使用 MSE，输出层使用 `tanh`，rollout 前仍执行 `clip(-1, 1)`。原始 LIBERO action 已在 `[-1, 1]`，不要未经实验就再次标准化；如果后来使用 z-score，必须把训练集统计量存入 checkpoint，并在 rollout 时严格反标准化。

计划命令：

```bash
python scripts/train_bc.py --config configs/bc_single_task.yaml

python scripts/evaluate.py \
  --config configs/bc_single_task.yaml \
  --checkpoint outputs/bc/best.pt \
  --num-rollouts 20
```

本阶段验收标准：

- 模型能够在一个很小的数据子集上过拟合。
- train/validation 划分以 episode 为单位，避免帧级数据泄漏。
- checkpoint 包含模型、优化器、配置和 normalization statistics。
- 推理输出维度和动作范围正确。
- 可以完成 LIBERO 闭环 rollout，并保存视频和失败信息。

若模型无法拟合训练集，优先检查数据错位、图像数值范围、NaN、动作定义以及归一化是否一致。

## 8. 阶段三：ACT 与消融实验

ACT 根据当前或历史观测一次预测未来一段动作：

```text
最近 N 帧观测 -> Transformer -> 未来 K 个动作
```

实现时，数据集对时间点 `t` 返回 `actions[t:t+K]`、`is_pad[K]` 和当前/历史观测。模型使用 K 个可学习 action query，输出 `(B, K, 7)`；loss 只计算非 padding 位置。推理先实现“每 K 步重新预测并顺序执行”，再实现每一步重预测和 temporal aggregation。LIBERO 自带的 `BCTransformerPolicy` 是历史观测 Transformer，但每次只输出当前动作，不是 ACT。

第一轮只改变一个变量，其余训练条件保持一致：

| 实验组 | 变量 |
| --- | --- |
| A/B/C | `chunk_size = 1 / 10 / 20` |
| D/E | 当前帧 / 最近 10 帧 |
| F/G | 仅图像 / 图像加 robot state |
| H/I | 不使用 / 使用 temporal aggregation |

计划命令：

```bash
python scripts/train_act.py --config configs/act_single_task.yaml

python scripts/evaluate.py \
  --config configs/act_single_task.yaml \
  --checkpoint outputs/act/best.pt \
  --num-rollouts 20 \
  --save-video
```

每次实验至少记录：任务、随机种子、数据量、配置、训练步数、训练时间、显存峰值、validation loss、成功率、推理延迟和失败类型。

## 9. 阶段四：OpenVLA-OFT 对照

首先追踪完整推理链路：

```text
image
-> image processor
-> vision encoder
-> multimodal connector
-> language model
-> action token / continuous action head
-> action unnormalization
-> robot control
```

对每个模块记录输入输出 shape、冻结状态、数值范围和耗时。之后再进行单任务、100 至 500 steps 的小规模 LoRA/OFT 验证，依次确认 loss、checkpoint 加载和 rollout 均正常。

建议对照项：

- 冻结 vision encoder 与否。
- 不同 LoRA rank。
- 不同 action chunk size。
- 不同 demonstrations 数量。
- 是否加入 robot state。

OpenVLA-OFT 的实际启动命令取决于所使用仓库的版本。项目落地时应在 `scripts/` 中封装并记录上游仓库 commit，避免 README 中的命令随上游接口变化而失效。

## 10. 统一评测协议

第一轮实验固定为：

- Benchmark：`LIBERO-Spatial`。
- 任务数：1 个，流程稳定后扩展到 3 个。
- 数据：所有模型使用相同 demonstrations。
- 随机种子：先使用 1 个，正式结果至少使用 3 个。
- Rollout：调试阶段每个任务 20 次，正式评测增加次数。
- 指标：成功率、平均 episode 长度、推理延迟和失败类型。

比较模型时必须同时报告输入模态、参数规模、训练数据量和计算预算。训练 loss 不能替代闭环成功率。

建议结果表：

| Model | Input | Action horizon | Demos | Success rate | Latency | GPU hours |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| BC | image + state | 1 | TBD | TBD | TBD | TBD |
| ACT | history + state | TBD | TBD | TBD | TBD | TBD |
| OpenVLA-OFT | image + language + state config | TBD | TBD | TBD | TBD | TBD |

失败类型统一标记为：感知错误、抓取失败、轨迹偏移、动作震荡、提前终止、超时或环境异常。

## 11. 六阶段里程碑

| 阶段 | 工作内容 | 交付物 |
| --- | --- | --- |
| 1 | 跑通 LIBERO 环境，阅读 BDDL、observation 和 action | 环境 smoke test 与概念笔记 |
| 2 | 审计原生 HDF5、轨迹回放和时间对齐 | `data_audit/` 与数据报告 |
| 3 | 最小 BC、按 episode 划分和小样本过拟合 | 可复现 BC checkpoint |
| 4 | 固定初始状态的闭环 rollout 和失败分类 | BC 成功率与视频 |
| 5 | 单任务 ACT、padding mask 和 chunk size 对照 | BC/ACT 对比表 |
| 6 | RLDS adapter 或 OpenVLA-OFT 二选一，再做统一评测 | 扩展实验报告 |

阶段不是强制等于一周。初学者应以验收条件推进，不以日历推进；上一个阶段没有通过时，不增加模型复杂度。

## 12. 实验记录规范

每次运行保存一份完整配置和机器信息，输出目录建议为：

```text
outputs/<model>/<task>/<date>-<run-name>/
├── config.yaml
├── environment.txt
├── train.log
├── metrics.json
├── checkpoints/
└── videos/
```

实验报告需要明确写出：

- 哪些结果使用了相同数据和评测协议。
- 哪些结果由于输入模态或计算预算不同而不能公平比较。
- 模型失败发生在哪个环节。
- 当前结论是否经过多个随机种子验证。

## 13. 推荐实施顺序

不要从 OpenVLA 微调开始。按以下顺序推进，每一步通过验收后再进入下一步：

1. 在本地 `libero` 环境创建一个 LIBERO-Spatial 环境，完成 `reset -> set_init_state -> step`。
2. 读取一个 LIBERO HDF5 episode，打印所有字段和 shape。
3. 保存一条轨迹视频，检查 observation 与 action 的时间对齐。
4. 实现 BC 并在少量样本上过拟合。
5. 接入 LIBERO 完成 20 次闭环 rollout。
6. 实现 ACT，完成 chunk size 消融。
7. 根据研究重点选择增加 RLDS adapter 或运行 OpenVLA-OFT，不要同时引入两者。
8. 使用同一评测协议运行 OpenVLA-OFT。
9. 汇总成功率、延迟、训练成本和失败案例。

项目完成的判断标准不是运行过多少模型，而是能够解释从数据进入模型到动作进入环境的每一步，并能用受控实验定位失败原因。
