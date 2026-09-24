# 最小视觉 BC (`bc.py`) 学习总结

本文档总结 `myexperiment/bc.py` 的核心概念：数据读取、归一化、Dataset、模型结构、张量维度变化与训练机制。

---

## 一、9 维 state 和 7 维 action 包含什么

LIBERO 使用 Franka Panda 机械臂（7 自由度）+ 平行夹爪。

### state = joint_states(7) + gripper_states(2) = 9 维

```python
joint_states   = demo["obs"]["joint_states"]     # (T, 7)
gripper_states = demo["obs"]["gripper_states"]   # (T, 2)
states = np.concatenate([joint_states, gripper_states], axis=-1)  # (T, 9)
```

| 部分 | 维度 | 含义 |
|------|------|------|
| `joint_states` | 7 | Panda 七个关节的角度（弧度） |
| `gripper_states` | 2 | 夹爪两个手指的关节开合量 |
| 合计 | 9 | 机器人当前的本体感知状态（proprioception） |

注意：state 只描述机器人自己，不含物体位置。

### action = 7 维

| 部分 | 维度 | 含义 |
|------|------|------|
| 末端位置增量 | 3 | 末端执行器 x/y/z 位移增量（diff） |
| 末端姿态增量 | 3 | 末端旋转的轴角（axis-angle）增量 |
| 夹爪 | 1 | 夹爪开合指令 |

合计 7 维，经 OSC 控制器约定范围在 `[-1, 1]`（故 `rollout_bc.py` 中有 `np.clip(action, -1, 1)`）。

对比：state 的夹爪是 2 维（两指分开），action 的夹爪合成 1 维下发，所以 action 是 3+3+1=7。

---

## 二、归一化统计量 `compute_stats`

### 目的

神经网络对数据分布敏感。state/action 各维数值范围差异很大，直接训练会慢且不稳。因此对每个维度做 **z-score 标准化**：

```
x_norm = (x - mean) / std
```

把每个维度变成「均值 0、标准差 1」的分布。

### 计算方式

```python
def compute_stats(hdf5_path, demo_names):
    states, actions = [], []
    for demo_name in demo_names:
        _, state, action = read_demo_arrays(hdf5_path, demo_name)
        states.append(state)      # (T_i, 9)
        actions.append(action)    # (T_i, 7)
    state_all  = np.concatenate(states)   # (总帧数, 9)
    action_all = np.concatenate(actions)  # (总帧数, 7)
    return NormalizationStats(
        state_mean=state_all.mean(0),                    # 对"帧"维度取均值 -> (9,)
        state_std=np.maximum(state_all.std(0), 1e-6),    # (9,)
        action_mean=action_all.mean(0),                  # (7,)
        action_std=np.maximum(action_all.std(0), 1e-6),  # (7,)
    )
```

- `.mean(0)` / `.std(0)`：沿第 0 维（帧）计算，为**每个维度**单独算出一个统计量。
  - 例：`state_all` 形状 `(10000, 9)` → `mean(0)` 得到 `(9,)`。
- `.std` 内部流程（numpy 本质）：

  ```python
  mean = state_all.mean(0)     # (9,)
  diff = state_all - mean      # 广播 -> (N,9)
  var  = (diff ** 2).mean(0)   # 方差 (9,)
  std  = np.sqrt(var)          # 标准差 (9,)
  ```

  即「先算偏差平方的平均（方差），再开根号」。标准差描述**离散程度**：某维变化越大 std 越大，越小 std 越小。

### 为什么「只用训练 demo」

避免**数据泄漏**：若验证集统计量也参与计算，模型会间接看到验证集信息，导致验证结果偏乐观。

### 为什么 `np.maximum(std, 1e-6)`

逐元素取 `max(std, 1e-6)`。万一某维 std=0（该维完全不变），防止后面除以 0 得到 inf/nan，用极小值兜底。

> 注意：**图像不做这种统计量归一化**，图像用的是除以 255，两者思路不同。

---

## 三、`LIBEROFrameDataset`：展平成逐帧样本

### 「展平」的含义

原始数据按轨迹（demo）组织，如：

```
demo_0: [帧0, 帧1, ..., 帧99]    # 100 帧
demo_1: [帧0, ..., 帧149]        # 150 帧
```

模型需要的是一个个独立的 `(图像, state, action)` 单步样本。「展平」就是把轨迹拆散成单帧样本排成一长条：

```
样本0   = (demo_0, 帧0)
样本1   = (demo_0, 帧1)
...
样本100 = (demo_1, 帧0)
```

### `__init__` 如何记录

```python
self.index = []
for demo_name in demo_names:
    with h5py.File(self.hdf5_path, "r") as handle:
        length = handle["data"][demo_name]["actions"].shape[0]  # 该 demo 帧数
    self.index.extend((demo_name, t) for t in range(length))
```

- `self.index` 是 list，元素为 `(demo名字, 时间步)`。
- `__len__` 返回 `len(self.index)`（总帧数）。
- 只读取长度 `shape[0]`，不读图像、不保存 `h5py.File` 句柄。

### 为什么不保存 h5py 句柄

PyTorch `DataLoader` 用多进程（`num_workers>0`）时会 pickle dataset 对象，而 h5py 的 `File` **不能被 pickle**（报 `h5py objects cannot be pickled`）。所以每次用 `with` 打开并关闭，不在对象里持有句柄——这就是 `num_workers` 建议保持 0 的原因。

### `__getitem__` 的处理

```python
def __getitem__(self, index):
    demo_name, timestep = self.index[index]
    images, states, actions = read_demo_arrays(self.hdf5_path, demo_name)
    image = torch.from_numpy(images[timestep]).permute(2, 0, 1).float() / 255.0
    image = torch.nn.functional.interpolate(
        image[None], (self.image_size, self.image_size),
        mode="bilinear", align_corners=False)[0]
    state = (states[timestep] - self.stats.state_mean) / self.stats.state_std
    action = (actions[timestep] - self.stats.action_mean) / self.stats.action_std
    return image, torch.from_numpy(state).float(), torch.from_numpy(action).float()
```

- **效率问题**：取一帧却把**整条 demo 重新读一遍**，效率低。可取的做法是缓存到内存或每进程保存句柄。
- **除以 255**：原始像素 uint8 范围 0~255，除以 255 映射到 `[0,1]`，数值小且稳定，避免梯度爆炸。`.float()` 保证是浮点数（卷积需要）。
- **`permute(2,0,1)`**：HDF5 图像是 HWC，PyTorch 卷积要求 CHW（通道在前）。
- **插值到 84×84**：
  - `image[None]`：加 batch 维 → `(1,3,128,128)`，因为 `interpolate` 要求 NCHW；完成后 `[0]` 去掉。
  - `mode="bilinear"`：双线性插值，缩小图像时由周围 2×2 像素加权平均，比最近邻平滑。
  - `align_corners=False`：插值坐标对齐方式，是双线性缩放的推荐默认。
  - 目的：缩小图像省显存、提速，是小视觉模型常见做法。

最终返回：
```
image : (3, 84, 84)   float32，范围 [0,1]
state : (9,)          float32，已标准化
action: (7,)          float32，已标准化
```

### 图像为什么一开始就有 3 个通道

LIBERO 环境/数据集存的原始图像本身就是彩色 RGB（`agentview_rgb`），不是灰度图。MuJoCo 离屏渲染出的相机图像是 `(128,128,3)`：三个通道分别是 R/G/B，每个像素是 0~255 的 uint8。

所以 CNN 第一层 `nn.Conv2d(3, 32, ...)` 的第一个参数 `3` 就是输入通道数。若改成灰度则会是 `Conv2d(1, ...)`，但当前用 RGB，所以是 3。

---

## 四、`SmallVisualBC` 前向计算流程与张量维度变化

> 说明：按「自适应池化 4×4 + `action_head` 首层为 `Linear(2112,128)`」这套**自洽配置**描述维度。若 `action_head` 写 `Linear(192,128)` 则与 2048 维图像特征不匹配，前向会报错。

### 整体结构

```
image (3,84,84) ──► image_encoder (CNN) ──► image_feat ┐
                                                        ├─ cat ─► action_head ─► (7,)
state (9,)      ──► state_encoder (MLP) ──► state_feat  ┘
```

### 逐层维度（batch size = N）

**输入准备（Dataset + DataLoader）**

```
raw image (128,128,3) uint8
   │ permute(2,0,1) → /255 → [None] resize 84×84 → [0]
   ▼
image  (3,84,84)
raw state joint(7)+gripper(2)=(9,)  ── (x-mean)/std ──► state  (9,)
raw action (7,)                     ── (x-mean)/std ──► action (7,)  ← 训练标签

组 batch：
image  (N,3,84,84)
state  (N,9)
action (N,7)
```

**前向传播**

```
                       image (N,3,84,84)
                             │
              ┌──────────────▼───────────────┐
              │        image_encoder (CNN)    │
              │ Conv2d(3→32, k5,s2,p2) +ReLU │  → (N,32,42,42)
              │ Conv2d(32→64,k5,s2,p2) +ReLU │  → (N,64,22,22)
              │ Conv2d(64→128,k3,s2,p1)+ReLU │  → (N,128,11,11)
              │ AdaptiveAvgPool2d((4,))       │  → (N,128,4,4)
              │ Flatten                       │  → (N,2048)   img_feat
              └──────────────┬───────────────┘
                             │
                       state (N,9)
                             │
              ┌──────────────▼───────────────┐
              │        state_encoder (MLP)    │
              │ Linear(9→64)  + ReLU          │  → (N,64)
              │ Linear(64→64) + ReLU          │  → (N,64)   state_feat
              └──────────────┬───────────────┘
                             │
         torch.cat([img_feat, state_feat], dim=-1)
                             ▼
                      (N, 2048+64) = (N, 2112)
                             │
              ┌──────────────▼───────────────┐
              │        action_head (MLP)      │
              │ Linear(2112→128) + ReLU       │  → (N,128)
              │ Linear(128→7)                 │  → (N,7)   pred
              └──────────────┬───────────────┘
                             ▼
                       prediction (N,7)   ← 归一化动作空间
```

卷积输出尺寸公式：`out = floor((in + 2*padding - kernel) / stride) + 1`。

### 关键数字对照表

| 阶段 | 形状 | 说明 |
|------|------|------|
| 原始图像 | (128,128,3) | HWC，RGB |
| 送入 CNN | (N,3,84,84) | CHW，resize 后 |
| 卷积后 | (N,128,11,11) | 三层 stride2 下采样 |
| 池化后 | (N,128,4,4) | AdaptiveAvgPool2d((4,)) |
| 图像特征 | (N,2048) | Flatten |
| 状态特征 | (N,64) | MLP |
| 拼接 | (N,2112) | cat |
| 预测 | (N,7) | 动作 |

### 有无残差连接

代码里**没有残差 / 跳跃连接**。所有层都是顺序串接（`nn.Sequential`），前一层输出直接喂给下一层，没有 `x + f(x)` 这种结构——符合「最小 baseline」定位。

---

## 五、loss 与反向传播

损失和前向在 `train_bc.py` 的 `run_epoch`：

```python
def run_epoch(model, loader, optimizer, device):
    training = optimizer is not None
    model.train(training)
    criterion = torch.nn.MSELoss()          # ① 损失函数
    for image, state, action in loader:
        image, state, action = image.to(device), state.to(device), action.to(device)
        with torch.set_grad_enabled(training):
            loss = criterion(model(image, state), action)   # ② 前向 + 算损失
            if training:
                optimizer.zero_grad(set_to_none=True)       # ③ 清空旧梯度
                loss.backward()                             # ④ 反向传播
                optimizer.step()                            # ⑤ 更新参数
```

### ① 损失函数：MSE

```
loss = mean_over_all( (pred - action)^2 )
```

`pred` 与标签 `action` 均为 `(N,7)`。标签是**归一化后**的动作，所以该 loss 是「归一化动作空间」的 MSE，与 `metrics.json` 中的 `train_loss` 同口径。

### ② 前向 + 算 loss

`model(image, state)` 走完第四节全部流程得到 `(N,7)` 预测，再与真实 `action` 算 MSE，得到标量 loss。

### ③ `zero_grad(set_to_none=True)`

PyTorch 默认**累加**梯度，每步需先清空。`set_to_none=True` 直接把梯度置 None（省内存、略快），是现代推荐写法。

### ④ `loss.backward()` —— 反向传播

- 前向时 PyTorch 已搭建**计算图**（记录张量由哪些操作产生）。
- `backward()` 从标量 loss 出发，按**链式法则**逆推，算出 loss 对每个可训练参数的偏导，存入 `param.grad`：

  ```
  dLoss/dW = dLoss/dOutput · dOutput/dW
  ```
- 只有 `requires_grad=True` 的参数（Linear/Conv 的权重与偏置，默认 True）才计算梯度。

### ⑤ `optimizer.step()` —— 更新参数

用 `Adam`（默认 `lr=1e-3`）根据 `param.grad` 更新参数：

```
Adam: W ← W - lr * (更新后的梯度估计)
```

### 验证阶段

同一 `run_epoch` 在验证时传入 `optimizer=None`：
- `set_grad_enabled(False)` → 不建计算图、不算梯度（省内存提速）；
- 只前向算 loss，不 backward、不 step；
- 用于监控过拟合、挑选 `best.pt`。

---

## 六、关于 `AdaptiveAvgPool2d((1,1))` vs `(4,)` 的取舍

直觉上，`(4,4)` 比 `(1,1)` 保留了更多空间信息，但实测常常**loss 更高、成功率更低**。原因是「信息更多」不等于「效果更好」：

| 原因 | 说明 |
|------|------|
| **过拟合（主因）** | 池化从 1×1→4×4，图像特征 128→2048，`action_head` 首层参数从 ~2.5 万涨到 ~27 万（约 10 倍），而数据量不变，模型更容易"背题"而非学规律，导致验证 loss 更高、闭环成功率更低 |
| 特征冗余 | 84×84 小图 + 简单抓放任务，2048 维里大量是噪声/背景维，被当成信号拟合 |
| 表示不友好 | (1,1) 给的是整张图全局摘要；(4,4) 把空间打散成 16 块，MLP 缺乏空间邻接的归纳偏置，需自行重组全局关系，更难学 |
| 优化更难 | 参数多 10 倍 → 收敛更慢/更不稳，有限 epoch 下可能未充分收敛 |
| 评估噪声 | 少量 rollout 的成功率本身方差大，不能只凭几次比较 |

**核心口诀**：更多信息 + 不变的数据 = 更多「可以拿来过拟合」的自由度。

在数据少、任务简单时，**适度丢空间信息（全局池化）反而是一种正则化**，帮助模型学泛化而非记忆。这正是原版 baseline 用 `(1,1)` 的原因。

**验证建议**：控制变量（仅改 pooling，固定 seed/epoch/lr/划分），对比 train/val loss 曲线；4×4 版 val loss 明显更高即过拟合；也可给 4×4 版加 dropout/weight decay、减小中间层，或增加数据后再试。
