# LIBERO 项目导读与学习手册

本文面向第一次接触机器人模仿学习的学习者。目标不是一次读完全部源码，而是建立一条可以运行、观察、修改和验证的主链路。

## 1. 先建立整体认识

LIBERO 是一个机器人操作任务基准，也是一个模仿学习与持续学习实验框架。它提供：

- 使用 BDDL 描述的 130 个任务。
- 基于 robosuite 和 MuJoCo 的 Panda 机械臂仿真环境。
- 人类遥操作采集的 HDF5 demonstrations。
- RNN、Transformer、ViLT 三类视觉语言 BC 策略。
- Sequential、SingleTask、Multitask、ER、EWC、AGEM、PackNet 等训练方法。
- 使用固定初始状态进行闭环 rollout 的评测代码。

它不直接提供：

- ACT（Action Chunking Transformer）实现。
- OpenVLA 或 OpenVLA-OFT 实现。
- 当前学习路线中的 `train_bc.py`、`train_act.py` 和统一评测接口。
- 完整可靠的 train/validation episode 划分流程。
- RLDS 训练管线。仓库原生格式是 HDF5/robomimic。

因此，LIBERO 最适合承担“任务、仿真、数据和评测后端”；最小 BC、ACT 和 OpenVLA 适配代码放在自己的实验项目中。

## 2. “在仓库旁边实现”是什么意思

推荐保留两个同级 Git 仓库：

```text
/home/zx/
├── LIBERO/                  # 官方仓库
└── libero-policy-study/     # 自己的学习项目
```

在学习项目所用环境中执行：

```bash
python -m pip install -e /home/zx/LIBERO
```

之后自己的代码可以直接导入：

```python
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
```

这样组织有三个好处：官方仓库容易更新；自己的实验历史清楚；BC、ACT 和 OpenVLA 可以共享同一评测器，而不必修改 LIBERO 核心代码。只有发现明确的 LIBERO bug，或者需要新增 BDDL 任务、对象、谓词时，才修改官方仓库。

## 3. 仓库架构

```text
LIBERO/
├── benchmark_scripts/       # 下载官方数据
├── scripts/                 # 数据采集、转换和检查工具
├── notebooks/               # 环境、算法和自定义对象示例
├── libero/
│   ├── configs/             # Hydra 训练配置
│   ├── datasets/            # 下载后的 HDF5 数据
│   ├── libero/              # 任务与仿真系统
│   │   ├── benchmark/       # 任务套件、顺序和任务元信息
│   │   ├── bddl_files/      # 语言、物体、初始条件、目标条件
│   │   ├── init_files/      # 评测使用的固定 MuJoCo 初始状态
│   │   ├── assets/          # 物体、场景、纹理和碰撞模型
│   │   ├── envs/            # 环境、机器人、对象、区域和谓词
│   │   └── utils/           # 数据、视频和任务生成工具
│   └── lifelong/            # 模仿学习与持续学习
│       ├── main.py          # 训练总入口
│       ├── datasets.py      # HDF5 到时间序列 Dataset
│       ├── models/          # BC 策略和编码模块
│       ├── algos/           # 训练与抗遗忘算法
│       ├── metric.py        # loss 和闭环成功率评测
│       └── evaluate.py      # 独立加载 checkpoint 评测
├── requirements.txt
└── setup.py
```

## 4. 一次完整运行发生了什么

```text
Hydra 配置
  -> Benchmark 选出任务
  -> 根据任务路径打开 HDF5
  -> robomimic 生成长度为 T 的训练样本
  -> BERT 把语言指令编码成 task embedding
  -> Policy 根据图像、robot state 和语言预测动作
  -> Algorithm 计算 loss、反向传播并保存 checkpoint
  -> Evaluator 创建 MuJoCo 环境
  -> Policy 与环境闭环交互
  -> BDDL goal predicate 决定任务是否成功
```

重要入口：

| 想理解的问题 | 文件 | 建议关注 |
| --- | --- | --- |
| 有哪些任务 | `libero/libero/benchmark/__init__.py` | `Task`、`Benchmark`、`get_task()` |
| 任务如何描述 | `libero/libero/bddl_files/` | `:language`、`:regions`、`:init`、`:goal` |
| 环境如何创建 | `libero/libero/envs/env_wrapper.py` | `ControlEnv`、`OffScreenRenderEnv` |
| 成功如何判定 | `libero/libero/envs/problems/*.py` | `_check_success()`、`_eval_predicate()` |
| 数据如何读取 | `libero/lifelong/datasets.py` | `get_dataset()`、`SequenceVLDataset` |
| 模型如何预测 | `libero/lifelong/models/bc_rnn_policy.py` | `forward()`、`get_action()`、`reset()` |
| 如何训练 | `libero/lifelong/algos/base.py` | `observe()`、`learn_one_task()` |
| 如何闭环评测 | `libero/lifelong/metric.py` | `raw_obs_to_tensor_obs()`、`evaluate_one_task_success()` |
| 如何组装一切 | `libero/lifelong/main.py` | `main()` |

## 5. BDDL、环境和成功条件

一个 BDDL 文件主要描述：

```lisp
(:language Pick up the black bowl and place it on the plate)
(:objects ...)
(:regions ...)
(:init (On bowl table_region))
(:goal (And (On bowl plate)))
```

环境解析 BDDL 后加载对象和区域。每次执行 `env.step(action)`，MuJoCo 更新物理状态，环境再用 `On`、`In`、`Open`、`Close`、`TurnOn` 等谓词检查目标。如果所有 goal 条件成立，返回 `done=True`，稀疏奖励为 `1`。

这里要区分三种 state：

- `joint_states`、`gripper_states`：适合作为策略输入的本体感知状态。
- `robot_states`：数据集保存的组合机器人状态。
- `states`：完整 MuJoCo 状态，用来恢复仿真，不应该作为普通视觉策略的输入。

## 6. Observation 和 action

当前第一个 LIBERO-Spatial 数据集实测结构如下：

| 字段 | 单个 episode 形状 | 含义 |
| --- | --- | --- |
| `agentview_rgb` | `(T, 128, 128, 3)` | 外部相机 |
| `eye_in_hand_rgb` | `(T, 128, 128, 3)` | 腕部相机 |
| `joint_states` | `(T, 7)` | Panda 七个关节位置 |
| `gripper_states` | `(T, 2)` | 夹爪关节状态 |
| `ee_states` | `(T, 6)` | 末端位置和轴角姿态 |
| `actions` | `(T, 7)` | 专家控制动作 |
| `states` | `(T, 92)` | 完整 MuJoCo 状态 |

7 维 OSC_POSE 动作可理解为：

```text
[delta_x, delta_y, delta_z, delta_rx, delta_ry, delta_rz, gripper]
```

数据中的动作范围已经是 `[-1, 1]`。控制器会把前三维缩放为末端位移，把中间三维缩放为旋转增量，最后一维控制夹爪。

训练数据经 robomimic 处理后，图像通常从 `(B,T,H,W,C)` 转为 `(B,T,C,H,W)` 并转换为浮点张量。自己写数据集时必须明确完成转置和像素缩放，不能只看 shape 数量相同就认为输入正确。

## 7. 如何运行当前仓库

### 7.1 激活已安装环境

```bash
conda activate libero
cd /home/zx/LIBERO
python --version
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

当前机器已验证：Python 3.8.13、PyTorch 1.11.0+cu113、CUDA 可用、robosuite 1.4.0、robomimic 0.2.0。

### 7.2 查看数据结构

```bash
python scripts/get_dataset_info.py \
  --dataset libero/datasets/libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo.hdf5
```

预期看到 50 条轨迹、5068 个 transition、双相机图像、7 维 action，action 范围为 `[-1,1]`。

### 7.3 环境 smoke test

保存为自己的临时学习脚本后运行以下逻辑：

```python
import os
import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

suite = benchmark.get_benchmark_dict()["libero_spatial"]()
task = suite.get_task(0)
bddl = os.path.join(
    get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
)

env = OffScreenRenderEnv(
    bddl_file_name=bddl,
    camera_heights=128,
    camera_widths=128,
)
env.seed(0)
env.reset()
obs = env.set_init_state(suite.get_task_init_states(0)[0])
obs, reward, done, info = env.step(np.zeros(7))
print(obs["agentview_image"].shape, reward, done)
env.close()
```

```bash
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 python smoke_env.py
```

本机已经验证该流程可以得到 `(128, 128, 3)` 图像并执行 7 维动作。

### 7.4 运行官方训练框架

官方训练入口会加载一个 benchmark 的全部任务，不是最小单任务教学脚本：

```bash
CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 \
python libero/lifelong/main.py \
  benchmark_name=LIBERO_SPATIAL \
  policy=bc_rnn_policy \
  lifelong=single_task \
  train.n_epochs=1 \
  eval.eval=false \
  train.num_workers=1
```

这条命令适合验证代码链路，不代表有效训练。默认 BERT task embedding 首次运行可能需要下载模型。`SingleTask` 会在每个新任务前重置策略，仍会遍历套件中的 10 个任务。初学阶段不要把它当成自己的最小 BC 实现。

## 8. 什么是最小 BC

Behavior Cloning 把专家示范当成监督学习数据：给定专家在时间 `t` 看到的观测，预测专家在同一时间执行的动作。

```text
agentview image -> small CNN -> image feature
joint + gripper -> small MLP -> state feature
image feature + state feature -> MLP -> one 7-D action
```

最小意味着第一版主动舍弃复杂度：

- 只做一个任务。
- 只用外部相机，不用腕部相机和语言。
- 只用当前帧，不使用历史。
- 使用确定性 `tanh` 动作头和 MSE。
- 不使用 GMM、LSTM、Transformer、数据增强和持续学习。

它的目的不是获得最高成功率，而是验证数据、模型、checkpoint 和 rollout 的接口全部正确。

### 8.1 最小 BC 数据集

先以 episode 为单位划分，例如 40 条训练、5 条验证、5 条测试。不能随机拆帧，否则同一轨迹的相邻画面会同时进入训练和验证，造成数据泄漏。

每个样本返回：

```python
{
    "image": float_tensor,  # (3, H, W), [0, 1]
    "state": float_tensor,  # (9,), joint 7 + gripper 2
    "action": float_tensor, # (7,), [-1, 1]
}
```

建立索引时保存 `(episode_name, timestep)`，`__getitem__` 再从 HDF5 读取对应帧。多进程 DataLoader 中不要在主进程长期共享同一个 `h5py.File` 句柄。

### 8.2 最小 BC 模型和 loss

```python
image_feature = image_encoder(image)
state_feature = state_encoder(state)
action = torch.tanh(action_head(torch.cat([image_feature, state_feature], dim=-1)))
loss = mse(action, expert_action)
```

第一项验收不是 rollout，而是让模型在 100 至 500 个样本上过拟合到很低的训练误差。做不到时优先检查 observation/action 是否错位、HWC/CHW、像素范围、动作范围和夹爪编码。

### 8.3 最小 BC rollout

```text
固定 init state
  -> 读取当前 obs
  -> 使用与训练相同的预处理
  -> model(image, state)
  -> clip 到 [-1, 1]
  -> env.step(action)
  -> 成功或达到 max_steps 后停止
```

必须在每个 episode 开始时清空策略历史；虽然最小 BC 没有历史，这个接口会在 ACT、RNN 中继续使用。训练 loss 低不代表闭环成功，因为一点预测误差会改变下一帧观测并逐步累积。

## 9. 什么是 ACT

普通 BC 每次预测一个动作：

```text
observation_t -> action_t
```

ACT 每次预测未来一段动作：

```text
observation_t -> [action_t, action_t+1, ..., action_t+K-1]
```

动作块能学习一段连贯运动，例如“靠近碗、闭合夹爪、抬起”，通常比完全独立的逐帧预测更平滑。完整 ACT 论文模型还包含用于多模态动作建模的 CVAE；学习项目第一版可以先实现不带 CVAE 的 deterministic action chunking，再增加 CVAE。

### 9.1 ACT 数据

对每个时间点 `t` 返回：

```python
{
    "image": ...,                  # 当前帧或最近 N 帧
    "state": ...,
    "action_chunk": ...,           # (K, 7)
    "is_pad": ...,                 # (K,), True 表示轨迹末尾 padding
}
```

轨迹末尾不足 K 步时补零，并用 `is_pad` 保证这些位置不参与 loss：

```python
per_step_loss = torch.abs(pred_chunk - target_chunk).mean(dim=-1)
loss = per_step_loss[~is_pad].mean()
```

### 9.2 ACT 模型

最小实现可以使用：

```text
image encoder + state encoder
              -> observation token
K learned action query tokens
              -> Transformer decoder
              -> linear layer
              -> (B, K, 7)
```

第一版推荐参数：`K=10`、`d_model=256`、4 层 Transformer、8 个 attention heads。显存不足时先降低图像分辨率、batch size 和 `d_model`，不要一开始删除 padding mask 或改变任务。

### 9.3 ACT 如何执行

按难度分三步实现：

1. `K=1`，验证 ACT 数据和模型能退化成单步 BC。
2. `K=10`，每次预测后顺序执行整个 chunk，再重新预测。
3. 每一步都重新预测 chunk，把多个历史预测对当前时刻的动作做指数加权平均，这就是 temporal aggregation。

现有 `BCTransformerPolicy` 使用最近若干观测预测当前动作。它解决的是 observation history，不包含未来 action chunk、`is_pad` 和 temporal aggregation，因此不能直接称为 ACT，但其中的图像编码和 Transformer 写法可以作为阅读参考。

## 10. 实现 BC 和 ACT 需要哪些模块

建议自己的项目最终包含：

```text
libero-policy-study/
├── configs/
│   ├── bc_single_task.yaml
│   └── act_single_task.yaml
├── data_audit/
│   ├── inspect_dataset.py
│   └── visualize_episode.py
├── datasets/
│   ├── libero_hdf5.py
│   └── splits.py
├── policies/
│   ├── bc.py
│   └── act.py
├── evaluation/
│   ├── libero_rollout.py
│   └── video.py
├── scripts/
│   ├── train_bc.py
│   ├── train_act.py
│   └── evaluate.py
└── tests/
    ├── test_dataset_alignment.py
    ├── test_policy_shapes.py
    └── test_action_range.py
```

BC 和 ACT 应共享：episode split、图像与状态预处理、checkpoint 格式、LIBERO 环境创建、固定初始状态、视频保存和成功率计算。只有模型和目标动作形状不同。共享评测器是公平比较的核心。

## 11. 推荐学习与修改顺序

### 阶段一：只观察，不训练

- 阅读一个 BDDL 文件。
- 运行数据检查脚本。
- 创建环境并打印 observation。
- 能解释 7 维 action 和三种 state。

### 阶段二：数据审计

- 统计每个 episode 的长度。
- 绘制七个动作维度的分布和时间曲线。
- 导出双相机轨迹视频。
- 检查第 `t` 帧是否对应第 `t` 个动作。

### 阶段三：最小 BC

- 先在几百帧上过拟合。
- 再使用 episode 级 train/validation split。
- 保存模型、配置、split 和预处理信息。
- 最后接闭环 rollout。

### 阶段四：ACT

- 先让 `chunk_size=1` 通过 BC 的 shape 测试。
- 再使用 `chunk_size=10` 和 padding mask。
- 最后增加 temporal aggregation。
- 每次只改变一个变量做消融。

### 阶段五：持续学习或 OpenVLA

先选择一个方向。想理解灾难性遗忘，就阅读 ER/EWC/PackNet；想研究大模型机器人策略，再接 OpenVLA-OFT。不要在 BC rollout 尚未成功时同时调试 RLDS、ACT 和 OpenVLA。

## 12. 修改官方仓库时的导航

| 修改目标 | 修改位置 |
| --- | --- |
| 新增任务 | BDDL 文件和 benchmark task map |
| 新增物体 | `assets/`、`envs/objects/` 和对象注册表 |
| 新增成功关系 | `envs/predicates/` |
| 新增场景类型 | `envs/problems/` 和 arena |
| 新增官方风格策略 | `lifelong/models/`、模型注册和 Hydra 配置 |
| 新增持续学习算法 | `lifelong/algos/` 和对应配置 |
| 修改 rollout | 优先在自己的 `evaluation/` 包装；确认通用后再改 `metric.py` |

修改时始终先写最小测试。例如新增 policy，至少验证输入 `(B,T,C,H,W)` 后输出 action shape 正确；新增任务，至少验证环境能 reset、随机动作能 step、goal predicate 能独立触发。

## 13. 常见误区

- 把低训练 loss 当成任务成功。机器人策略最终必须看闭环 rollout。
- 按帧随机划分数据。必须按 episode 划分。
- 把完整 MuJoCo `states` 当作部署时可获得的 robot state。
- 训练时用 RGB，评测时忘记通道转置、上下翻转或数值缩放。
- 对已经位于 `[-1,1]` 的 action 反复归一化，却没有保存反归一化规则。
- 把 LIBERO 的 BC Transformer 当成 ACT。
- 一开始就运行全部 10 个任务、多个种子和 20 路并行 rollout。
- 在同一环境里强行安装旧 LIBERO 和新 OpenVLA 的冲突依赖。

## 14. 第一组实践作业

完成下面四项后再开始训练：

1. 用 `get_dataset_info.py` 写下一个 episode 的所有字段和 shape。
2. 写脚本导出 `demo_0` 的 agentview 视频。
3. 画出 `demo_0` 七个 action 维度随时间变化的曲线。
4. 创建环境，使用固定初始状态执行 20 个零动作并保存首尾图像。

验收时应该能够回答：模型在时间 `t` 看到了什么、目标 action 的每一维是什么、预测动作经过哪些变换进入控制器、环境依据什么条件判断成功。
