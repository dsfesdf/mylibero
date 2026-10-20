# 统一实验协议与确定性 ACT

本轮实现了数据划分、随机性、rollout 条件和 best 选择规则的统一，优化了 Sequence/ACT 数据读取，并增加不含 CVAE 的确定性动作头。旧结果不变，新实验建议写入 `myexperiment/results_v2/`。暂不增加训练恢复。

## 1. 数据划分与训练 seed

四个训练入口 `train_bc.py`、`train_action_chunking.py`、`train_sequence_transformer.py`、`train_act.py` 现在统一支持：

| 参数 | 用途 |
|---|---|
| `--split-seed 0` | 仅控制按完整 demo 划分 train/val，默认 0 |
| `--seed 0` | 初始化、训练随机性及 shuffle；重复实验改为 0/1/2 |
| `--split-file PATH` | 共用划分名单；不存在时创建，存在时检查并复用 |

每次训练额外保存 `output/split.json`。共用文件中的名单如果重复、遗漏或与当前数据集不匹配，或 split seed / val ratio 与命令行不一致，会直接报错，不会悄悄重新划分。第一次创建公共文件时先启动一组，文件创建完成后再并行启动其他组。

原 BC 的 `--seed` 现在不再控制数据划分：旧命令若使用非零 seed，想重现旧划分，应显式设置相同的 `--split-seed`。旧权重及其已保存的 train/val 名单不受影响。

注意 split 文件检查的是 demo 名单，不是数据内容哈希。不同任务经常都有 `demo_0` 等名字，所以公共文件必须按任务存放；跨机器仍需保证 HDF5 内容一致。

## 2. best 的统一含义

- BC、Chunk、Sequence：最小的验证动作 MSE。
- ACT/CVAE、确定性 ACT、Flow：最小的 `val_prior_mse`，全部通过 `model.predict(images, states)` 生成动作，不读取未来动作标签。
- ACT 的 posterior reconstruction、KL，以及 Flow 的 velocity MSE 保留作诊断，不用于选择 best。
- checkpoint 增加 `selection_metric`、`best_epoch` 和对应最佳分数。每个完整 epoch 后的 metrics 保存方式继续保留。

当前版本开始，Chunk/Sequence/ACT 的动作指标以**有效动作误差总和 / 有效动作数**汇总。KL 单独按样本数汇总。由此消除尾部 padding 比例、最后一个 batch 大小对统计权重的影响；旧版加权方式产生的数值不保证与新版完全相同。

ACT 验证默认使用 `--val-seed 12345`，在独立 RNG 上下文里执行，不推进训练 RNG。`evaluate_act.py` 复用训练时的验证函数，默认恢复 checkpoint 的 batch size 和 val seed，可以重算同一个 best 分数。Flow 的随机数分组仍依赖 batch size，所以不要一边更改 batch size，一边要求采样 MSE 逐位一致。

## 3. 新增确定性动作头

```bash
--action-head deterministic
```

三个分支的区别：

| action head | 观测编码器 | 动作生成 | 训练目标 | 是否有 CVAE |
|---|---|---|---|---|
| `transformer` | ACT encoder | action queries + Transformer decoder；推理 z=0 | reconstruction MSE + KL | 是 |
| `deterministic` | 相同 ACT encoder | 相同 action queries、decoder 层及 MSE readout | masked action MSE | 否 |
| `flow` | 相同 ACT encoder | 带时间与噪声动作输入的 velocity decoder，Euler 采样 | masked velocity MSE | 否 |

确定性分支不创建 posterior 或 latent 参数，decoder memory 只有观测 token，训练与推理均不使用真实未来动作作为输入。即使调用方把 target 传给 `forward`，该分支也会忽略它，target 只用于外部 loss。

它并非 `--kl-weight 0`：后者仍保留 CVAE 后验路径。确定性分支与原 ACT 共享结构设计，但没有 latent token，权重尺寸不完全相同，须独立训练；评测和 rollout 会自动读取 checkpoint 的动作头配置。

补充修正：ACT 的 image-only 模式中，posterior 的 state token 现在输入固定零向量，不再读取真实本体状态。这会改变旧 image-only 权重的后验重建指标；纯观测推理权重结构和接口保持兼容。默认 image+state 路径不变。

## 4. Sequence/ACT 数据缓存

二者共用的 dataset 在初始化时逐 demo 读取一次并执行 resize、归一化，缓存 CPU 张量。之后 `__getitem__` 只选择 H 帧历史和 K 步动作，不再打开 HDF5，不把历史窗口整体展开缓存。

历史不足时重复第一帧；未来不足时重复最后一个真实动作，并使用 mask 排除 padding，保持原来的语义。与 BC/Chunk 共用 `prepare_demo_tensors`，预处理一致。

缓存不包含打开的 h5py 对象，可序列化。默认 `--num-workers 0`；双相机约 6,000 帧、84×84 float32 图像本身约 1 GB，多个训练进程或 spawn worker 会增加内存需求。启动阶段的缓存时间不等于每个 epoch 的耗时。以后扩展到多任务大数据集时，应换按窗口读取或受限缓存，不建议无限缓存全部任务。

本机真实 task8 的一个 115 帧 demo、双相机、H5、84×84、CPU 单线程的局部计时：缓存初始化约 0.139 秒，缓存图像/状态/动作约 18.6 MiB；缓存后取窗口平均约 0.155 ms，旧的整 demo 读取并 resize 历史窗口约 51.5 ms。两者重复次数分别为 1,000/10，且受系统文件缓存影响；这是数据取样耗时，不是整轮训练的加速倍数。

## 5. 四类模型统一 rollout

BC、Chunk、Sequence、ACT rollout 均支持：

```bash
--seed 0
--sampling-seed 123
--max-steps 270
--settle-steps 5
--all-init-states
```

其中 sampling seed 默认等于 seed。选场景有三种方式：

- `--num-rollouts 10 --init-state-offset 0`：选择索引 0–9。
- `--init-state-indices 3 7 12`：按显式名单和顺序测试。
- `--all-init-states`：从 offset 开始选全部初始状态；一般保持 offset=0。

显式名单不能与 all/offset 混用。重复、负数或越界的索引会报错，不再取模重复场景；想重复同一场景检查采样方差，应分次运行并记录不同 sampling seed。

每个 episode 的环境 seed 为 `seed + init_index`，策略采样 seed 为 `sampling_seed + init_index`，并在该场景开始前重置。因此某场景的采样结果不取决于前面已经运行了多少个场景。这是新的随机性协议，不能要求与旧版在全程共用 RNG 的 Flow rollout 结果逐项相同。

统一条件包括：默认 270 个策略动作步、5 个零动作稳定步骤、相同初始状态索引、动作反归一化后 clip 到 [-1,1]。稳定步骤不计入策略步数。metrics 保存协议版本、seed、sampling seed、初始状态名单、稳定步数和实际 rollout 数量。ACT 的输入消融入口也透传这些选项。

## 6. 使用示例

在仓库根目录运行，服务器改为实际路径。下面只启动一个训练 seed；完成检查后，分别改为 seed1、seed2，并改输出目录。

```bash
conda activate libero
cd /home/zx/LIBERO
export DATASET="/home/zx/LIBERO/libero/datasets/libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate_demo.hdf5"
export RUN_ROOT="/home/zx/LIBERO/myexperiment/results_v2"

python myexperiment/actionchunk/train_act.py \
  --dataset "$DATASET" \
  --split-file "$RUN_ROOT/splits/task8_split0.json" \
  --split-seed 0 --seed 0 --val-seed 12345 \
  --action-head deterministic \
  --history-length 5 --chunk-size 10 --cameras agentview \
  --visual-representation global_pool --observation-mode both \
  --d-model 128 --num-heads 4 --num-layers 2 --dropout 0.1 \
  --epochs 50 --batch-size 64 --lr 1e-4 --weight-decay 1e-4 \
  --image-size 84 --num-workers 0 --amp \
  --output "$RUN_ROOT/task8/act_deterministic_h5_k10_agentview/split0_seed0"
```

训练 CVAE 时改为 `--action-head transformer --latent-dim 32 --kl-weight 10`，训练 Flow 时改为 `--action-head flow --flow-steps 10`。每种动作头使用独立 output；所有模型复用相同 task8 split 文件。

离线复算验证分数：

```bash
python myexperiment/actionchunk/evaluate_act.py \
  --checkpoint "$RUN_ROOT/task8/act_deterministic_h5_k10_agentview/split0_seed0/best.pt"
```

闭环评测：

```bash
python myexperiment/actionchunk/rollout_act.py \
  --checkpoint "$RUN_ROOT/task8/act_deterministic_h5_k10_agentview/split0_seed0/best.pt" \
  --benchmark LIBERO_SPATIAL --task-id 8 \
  --all-init-states --seed 0 --sampling-seed 123 \
  --max-steps 270 --settle-steps 5 \
  --temporal-aggregation --aggregation-decay 0.2 --history-size 10 \
  --output "$RUN_ROOT/task8/act_deterministic_h5_k10_agentview/split0_seed0/eval_ta02"
```

其他 rollout 脚本也支持以上相同的场景和 seed 参数。调试可换为 `--init-state-indices 0 1 2`，不要同时保留 `--all-init-states`。

## 7. 还需要关注的问题

1. **编码器仍不完全匹配**：BC/Chunk 默认 2×2 pooling，Sequence/ACT global 模式为 1×1，state encoder 也不同。新确定性 ACT / CVAE / Flow 可用于更接近的头部比较，但 BC→Sequence 仍是系统比较；本轮没有统一所有模型骨干。
2. **best MSE 不是 best success rate**：它是无标签辅助、可复现的选权重规则，不保证与闭环成功率完全一致；Flow 的一次随机动作采样 MSE 尤其不能代表生成分布的全部质量。
3. **训练预算仍需显式统一**：旧脚本默认 epochs/lr 和 BC 优化器与其他模型不同。本轮没有擅自改变 BC 的 Adam；主对比用 Chunk K1 可减少这一差异。正式运行按方案显式写 epochs、lr、batch 等参数。
4. **不要用最终 rollout 集调参**：提前确定调试/最终初始状态名单；现在所有模型都能接收同一显式名单。数据 demo 的 train/val 划分与仿真初始状态划分是两件事。
5. **小数据与单任务限制仍在**：用 3 个训练 seed，报告逐 seed 成功率；模型差异接近波动时，优先增加任务而非继续堆结构。

主对比建议在原六组基础上增加 deterministic ACT H5/K10：**7 组 × 3 seeds = 21 次训练**；然后再决定是否补相机和 H/K 消融。确定性头新增不意味着要立即启动全部训练。

## 8. 验证

```bash
python -m unittest discover -s myexperiment/tests -v
```

测试覆盖四相机/零图像、旧权重加载、Flow padding/积分、缓存与原始窗口一致性、缓存序列化、确定性头标签独立性、验证 mask 加权、场景选择和逐场景随机性，以及各训练入口的共享划分和 ACT 三种头的 best/离线指标一致性。微型训练只验证接线，不作为模型效果证据。

另完成 BC、Chunk、Sequence、确定性 ACT、Flow 五路真实仿真的两步冒烟，检查显式索引、seed、稳定步数和输出记录；ACT 距离统计在这次冒烟中用替代数据跳过，未作为验证对象。尚未运行完整成功率对比或服务器长训练。
