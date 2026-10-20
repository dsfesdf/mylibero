# 纯视觉自动训练与 rollout

入口：[run_suite.py](run_suite.py)。在已配置好的 `libero` 环境中运行，不需要安装额外的调度框架。

## 1. 可以不输入 state 吗？

可以。所有模型的训练入口现在都支持：

```bash
--observation-mode image --cameras agentview
```

相机可选 `agentview`、`wrist`、`both`。纯视觉模式不能搭配 `none`。`--observation-mode both` 使用图像与本体状态；`state` 为仅状态对照。

ACT 的 CVAE、deterministic、Flow 三种头在 image 模式下都不依赖真实 state，CVAE 的 posterior 也不读取真实 state。BC/Chunk/Sequence 为兼容旧权重结构，保留原 state 分支，但在进入模型计算时将 state 换成固定零张量；它只是可学习的常量特征，不提供关节或夹爪状态信息。这与完全删除该分支在参数量上不同，应在论文式对比中说明。

数据集与 rollout 接口仍读取/传递 state，并保留归一化统计量，但 image 模式的预测不使用它。现有 HDF5 因此仍需包含 joint/gripper 字段。本次没有实现缺失 state 字段的数据格式。自动测试确认：真实 state 改成随机值或 NaN 不影响纯视觉预测。

必须按纯视觉模式重新训练；不要直接把 image+state 的旧权重切换成 image 模式当作已训练的纯视觉基线。权重会保存 observation_mode，离线评测和 rollout 自动恢复，旧权重缺少该字段时按 both 加载。

## 2. 默认实验矩阵

| 名称 / `--models` 值 | 模型 | H | K | rollout |
|---|---|---:|---:|---|
| `chunk1` | 单步 BC，用 Chunk MLP K1 实现 | 1 | 1 | first |
| `chunk10` | Chunk MLP | 1 | 10 | first、TA |
| `sequence1` | Sequence Transformer | 1 | 10 | first、TA |
| `sequence5` | Sequence Transformer | 5 | 10 | first、TA |
| `act` | ACT/CVAE | 5 | 10 | first、TA |
| `deterministic` | 无 CVAE 的确定性 ACT | 5 | 10 | first、TA |
| `flow` | 条件 Flow | 5 | 10 | first、TA |

默认三种训练 seed=0/1/2，split seed=0，外部相机、纯视觉输入，50 epochs、batch64、lr=1e-4、weight_decay=1e-4、84×84、workers2。可用 `--num-workers 0/2/4` 指定训练 worker 数。Sequence/ACT 使用 d128、4 heads、2 layers、dropout0.1；ACT KL=10、latent32，Flow Euler10步；ACT 默认 AMP，其他模型当前 FP32，可给套件加 `--no-amp` 关闭 ACT AMP。

共 **21 次训练、39 次 rollout 调用**。如果任务有 50 个官方初始状态，则是 1,950 个 episode，不包含训练。first/TA 复用同一份 best，不重复训练。first 是每步重规划只执行最新预测的第一个动作，不是开环连续执行 K 步。

还可选 `--models bc` 运行原单步 BC；它保留 Adam，不使用 weight_decay，所以默认使用 Chunk K1 做主单步基线。

统一闭环条件：环境 seed0、采样 seed123、max_steps270、稳定5步、TA decay0.2、TA history_size=K。默认全部官方初始状态；也可显式指定调试/最终名单。训练 best 按验证动作 MSE，ACT 三头按 `val_prior_mse`，不用 rollout 成功率挑权重。

## 3. 先预览，确认后执行

在服务器仓库根目录运行，激活环境并设置真实数据路径。脚本用当前 Python 解释器启动子进程，不硬编码本机 conda 路径。

```bash
conda activate libero
cd /你的服务器路径/LIBERO
export DATASET="$PWD/libero/datasets/libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate_demo.hdf5"

python myexperiment/run_suite.py \
  --dataset "$DATASET" \
  --output-root myexperiment/results_v2/vision_agentview \
  --gpu 2
```

以上仅打印完整命令，不创建结果目录或启动训练。`--gpu 2` 表示物理编号2；运行前用 nvidia-smi 确认当前分配。子进程内部的 cuda:0 就是该卡。脚本按顺序运行，不使用双卡分布式训练；不传 gpu 时继承现有 CUDA_VISIBLE_DEVICES。

正式运行：

```bash
python myexperiment/run_suite.py \
  --dataset "$DATASET" \
  --output-root myexperiment/results_v2/vision_agentview \
  --gpu 2 --execute
```

默认先依次完成全部21次训练，再开始39次 rollout，不再交替训练和评测。脚本每次只启动一个实验进程，尚未实现多个模型同时训练。终端显示任务进度，每个子进程输出写入对应 train.log 或 rollout.log。第一次执行会计算数据 SHA256，较大的 HDF5 需要等待磁盘读取完成。

需要关闭终端后继续运行时，可由 shell 后台托管：

```bash
nohup python -u myexperiment/run_suite.py \
  --dataset "$DATASET" \
  --output-root myexperiment/results_v2/vision_agentview \
  --gpu 2 --execute > suite_vision.log 2>&1 &
```

进度查看：

```bash
tail -f suite_vision.log
```

## 4. 推荐先跑最小验证

先只跑 chunk1、seed0、2 epochs、3 个初始状态，确认服务器 EGL、数据和 GPU 没问题；它不进入正式主表：

```bash
python myexperiment/run_suite.py \
  --dataset "$DATASET" \
  --output-root myexperiment/results_v2/vision_smoke \
  --models chunk1 --seeds 0 --epochs 2 \
  --init-state-indices 0 1 2 --max-steps 30 \
  --gpu 2 --execute
```

冒烟目录和正式目录分开。改预算后不复用冒烟结果。

## 5. 其他对比方案

### 只比较 ACT 的三种头

```bash
python myexperiment/run_suite.py \
  --dataset "$DATASET" \
  --output-root myexperiment/results_v2/vision_heads \
  --models act deterministic flow --seeds 0 1 2 \
  --cameras agentview --observation-mode image \
  --gpu 2 --execute
```

共9次训练，18次 rollout 调用。三者共享观测编码器设计；Flow 与回归/CVAE 仍有生成过程差异，报告成功率时也应关注计算成本。

### 对一个模型比较相机

```bash
python myexperiment/run_suite.py \
  --dataset "$DATASET" \
  --output-root myexperiment/results_v2/vision_camera_ablation \
  --models deterministic --seeds 0 1 2 \
  --cameras agentview wrist both --observation-mode image \
  --gpu 2 --execute
```

共9次训练，不加入 cameras=none。双相机为 RGB 通道拼接，并非独立相机 token attention。

### 图像+状态对照

把相同命令改为 `--observation-mode both`，使用新的 output-root；其他参数保持一致。单跑完整21组纯视觉并不能回答去掉 state 的影响，需要这一组匹配对照。

仅状态对照可用 `--observation-mode state --cameras none`。不要把默认纯视觉模式与 cameras=none 混用。

### A100 只训练，笔记本只 rollout

服务器（使用实际 dataset 和 GPU 编号）：

```bash
python myexperiment/run_suite.py \
  --dataset "$DATASET" --output-root myexperiment/results_v2/vision_agentview \
  --batch-size 256 --num-workers 2 --gpu 2 --stage train --execute
```

训练完成后把整个 `vision_agentview` 结果目录复制到笔记本的仓库中。务必保留根目录的 `suite.json`、`split.json`，以及每组的 `best.pt`、`metrics.json`、`train.complete.json`；保留完整目录最简单。训练日志可保留供核查，rollout 不需要复制 HDF5。

笔记本：

```bash
conda activate libero
cd /home/zx/LIBERO
python myexperiment/run_suite.py \
  --output-root myexperiment/results_v2/vision_agentview \
  --stage rollout --execute
```

`--stage rollout` 自动从该目录的 suite.json 恢复训练和评测配置，models、seeds、batch、场景等无需重写，命令行对这些配置的修改也不会生效；本阶段仅使用你指定的 output-root、gpu、execute。不会检查服务器的 dataset 路径，也不会改写原 suite.json。所有权重和输出路径重建为笔记本本地路径，已完成的评测在迁移后仍可跳过。

rollout 阶段要求存在本套件生成的完整训练标记，不用于导入任意旧权重；要改变评测场景或其他协议，用独立输出目录直接调用原 rollout 脚本。

### 已经启动的旧套件如何继续

旧版本把 workers 固定为0。如果保留原结果目录继续训练，需要显式 `--num-workers 0` 并保留原来的 batch/其他配置，加 `--stage train` 即可先完成剩余训练。已完整完成的训练会跳过；这次执行顺序变化不会使已有权重失效。

若想切换到默认 workers2，使用新的 output-root，避免新旧 worker 设置混入同一套件。中断训练仍不能续训；没有 complete 标记的目录需自行归档或改用新目录。

## 6. 完成检查与中断行为

- 默认不覆盖非空结果目录。完成的任务有 command 匹配的 complete 标记，重新执行相同命令会跳过；指标或 best 缺失则报错。
- 子进程失败立即停止套件，不悄悄跳过继续跑其他模型；检查日志后处理。
- 即使训练器捕获 Ctrl+C 后以退出码0结束，套件仍会检查最终 epoch 是否等于规定预算；不足则不写 train.complete.json，也不进入 rollout。
- 这里只支持**跳过已完整完成的任务**，不支持从中断 epoch 恢复。中断目录没有完成标记，重新执行会报错；先自行归档该具体目录，或使用新 output-root。脚本不会删除结果。
- suite.json 固定数据路径/哈希和实验参数，包含 num_workers。训练阶段修改配置须新建 output-root；rollout 阶段恢复原配置并允许结果目录迁移。不要同时用多个进程启动同一个 output-root。
- 代码本身不由 suite.json 冻结。正式启动前保存 Git 提交或代码快照，整批实验期间不要修改模型实现；不应在改过代码后把旧 complete 标记当作新实现的验证结果。

## 7. 输出与分析

```text
vision_agentview/
  suite.json                  # 固定套件参数与数据SHA256
  split.json                  # 所有模型共用的demo名单
  summary.csv                 # 各模型/相机/训练seed/执行策略结果
  aggregate.json              # 按模型/相机/执行策略汇总均值、样本标准差
  deterministic_agentview_image/
    split0_seed0/
      command.json
      train.log
      split.json
      metrics.json
      best.pt
      train.complete.json
      eval_first/
        command.json
        rollout.log
        metrics.json
        rollout.complete.json
        rollout_*.mp4
      eval_ta/
        ...
```

每完成一个任务更新汇总，只纳入有 rollout.complete.json 的结果。aggregate 中 seeds_completed 可用于检查是否已完成3个训练 seed；只有1个时 std 为 null，不能当成波动为0。

分析顺序：

1. chunk1 first 与 chunk10 first：预测 chunk 本身的效果。
2. chunk10 first 与 TA：同权重动作融合的效果。
3. sequence1 与 sequence5：历史图像信息是否有帮助。
4. act、deterministic、flow：固定 H5/K10 的动作生成路线比较。
5. 同模型 image 与 both：去掉本体状态的影响。

各模型骨干仍有差异，跨路线结果是系统比较；成功率差几个 episode 时，先看3个 seed 波动和逐场景失败视频。纯视觉会缺少直接的关节/夹爪状态，尤其单帧模型可能需要从图像估计运动状态，因此不保证效果超过图像+状态输入。

默认任务8必须与传入数据匹配。脚本不会从文件名自动判定 task-id；换任务要同时改 dataset、benchmark、task-id，并用新目录。最终场景不要拿来反复调参，可为预先划定的调试/最终集合分别建立套件。

## 8. 验证范围

```bash
python -m unittest discover -s myexperiment/tests -v
```

覆盖全部模型的纯视觉 state 独立性、相机合法性、训练与离线 loader 配置恢复、默认21训练/39评测矩阵，以及不完整训练不会获得完成标记。另用微型数据做自动化训练→真实仿真短 rollout→汇总→已完成任务跳过的冒烟验证；不代表正式任务成功率。未启动21组长训练。
