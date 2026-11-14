# 步骤七：按 MatterGen 定义联合损失并接通优化器

本步采用方案 A：独立定义损失，扩散模块组织一次训练前向，训练脚本执行参数更新。
步骤三至五的 corruption、步骤六的动态建图及输出参数化继续使用原有实现。

## 1. 先分清三个位置的职责

| 文件 | 本步职责 |
|---|---|
| `src/loss.py` | 后验与 KL 的参照计算、三路损失、按晶体聚合、加权求和 |
| `src/diffusion.py` | `JointDiffusion.compute_loss()`：采样时间、加噪、预测、调用损失对象 |
| `scripts/train_mask.py` | 加载标准化数据、统计训练集常数、优化器、反向传播、日志及可选 checkpoint |
| `scripts/step_07_demo.py` | 不依赖数据文件的两晶体单步训练示例 |
| `test/test_joint_loss.py` | 独立数学参照、数值边界、实际网络及优化器检查 |

虽然训练脚本沿用 `train_mask.py` 的文件名，现在执行的是元素、坐标、晶格三路联合训练。

MatterGen 的对应组织是：`MaterialsLoss` 组合各字段损失，`DiffusionModule.calc_loss()`
组织加噪、预测和损失计算，外层 Lightning 模块管理优化器。
本项目用一个独立的 `JointDiffusionLoss` 保留这种职责划分，没有引入完整的 Lightning/Hydra 框架。

参考：[MaterialsLoss](https://github.com/microsoft/mattergen/blob/main/mattergen/common/loss.py)、
[DiffusionModule](https://github.com/microsoft/mattergen/blob/main/mattergen/diffusion/diffusion_module.py)。

## 2. 损失函数需要接收什么

先手写 `JointDiffusionLoss.forward(prediction, info, batch, num_atoms)` 的接口。
其输入全部来自步骤六，不需要重新采样噪声，也不需要再次调用 Denoiser。

| 来源 | 字段 | 形状 | 含义 |
|---|---|---|---|
| `prediction` | `atom_logits` | `[N,118]` | 预测干净元素分布的 logits |
| `prediction` | `coord_score_times_std` | `[N,3]` | 已用当前晶格转换到分数坐标的 scaled score |
| `prediction` | `lattice_score_times_std` | `[B,3,3]` | 晶格 scaled score |
| `info` | `atom_target` | `[N]` | 干净原子序数减一，即 CE 使用的 0～117 标签 |
| `info` | `masked_atom` | `[N]` | 本次前向过程实际产生的 mask |
| `info` | `k_graph` | `[B]` | 整数扩散步数 1～1000 |
| `info` | `coord_target` | `[N,3]` | 周期分布的真实 score 乘标准差 |
| `info` | `lattice_target` | `[B,3,3]` | 负的对称标准高斯噪声 |
| 状态 | `batch`、`num_atoms` | `[N]`、`[B]` | 原子归属及每个晶体的原子数 |

这里 `N` 指拼接后的原子总数。损失计算不把 `info` 输入网络，避免干净目标泄漏。
`q_sample()` 在 `no_grad` 下构造目标；`predict()` 和损失计算保留计算图。

## 3. 先用后验明确元素变分项

先实现 `categorical_posterior()`，暂时不急于写简化 CE。
约定转移矩阵按行表示概率：

\[
Q_k[i,j]=q(A_k=j\mid A_{k-1}=i),\qquad
\bar Q_{k-1}=Q_1Q_2\cdots Q_{k-1}.
\]

给定某个干净分布 \(p_0\) 和观察到的当前类别 \(j\)，先计算：

\[
r_i=Q_k[i,j]\,[p_0\bar Q_{k-1}]_i,
\qquad
p(A_{k-1}=i\mid A_k=j)=\frac{r_i}{\sum_h r_h}.
\]

手写时对应三个操作：`p0 @ Q_bar_previous`、取 `Q_k` 的第 `j` 列、逐元素相乘后归一化。
通用函数允许每个原子使用不同的转移矩阵，形状为 `[N,C,C]`，也支持共享 `[C,C]` 矩阵。

把真实干净类别的 one-hot 代入，得到真实后验 `q`；把网络预测的干净概率代入，得到模型后验 `p`。
`k > 1` 比较 `KL(q || p)`，`k = 1` 使用真实干净类别在模型反向后验中的负对数概率。
首步用的是后验概率；未被 mask 的原子，其后验已经确定，不会再收取变分 CE。

小类别测试采用“三个真实元素 + 一个 MASK”，逐次相乘构造 \(\bar Q\)，遍历 `k=1..7`。
这样数学参照不会依赖待验证的 `1/k` 代码。另用非对称、非吸收的转移矩阵检查矩阵方向。

**类别编号要分清。** 晶体输入以原子序数表示，MASK 为 0；后验矩阵的类别顺序是
`[Z=1,...,118,MASK]`，因此后验 MASK 索引是 118。真实元素的类别索引是 `Z-1`。
网络的 118 类 `softmax` 后，附加的是概率零，不是 logit 零。

参考：[官方 D3PM 后验与变分损失](https://github.com/microsoft/mattergen/blob/main/mattergen/diffusion/d3pm/d3pm.py)。

## 4. 推导并实现高效元素损失

当前 standard schedule 给出：

\[
\beta_k=\frac1{K-k+1},\qquad
\bar\alpha_{k-1}=\frac{K-k+1}{K},\qquad
1-\bar\alpha_k=\frac{k}{K}.
\]

如果当前原子是 MASK，它在这一步刚变为 MASK 的后验概率为：

\[
\frac{\bar\alpha_{k-1}\beta_k}{1-\bar\alpha_k}
=\frac{(K-k+1)/K\cdot1/(K-k+1)}{k/K}
=\frac1k.
\]

设真实干净元素为 `a`。真实后验在 `a` 上分配 `1/k`，在 MASK 上分配 `1-1/k`；
模型后验在各真实元素 `z` 上分配 `p_theta(z)/k`，在 MASK 上分配相同的 `1-1/k`。
因此 MASK 项的对数比为零，剩下：

\[
\mathrm{KL}(q\Vert p)=\frac1k\log\frac{1/k}{p_\theta(a)/k}
=\frac1k[-\log p_\theta(a)].
\]

如果当前原子仍然是元素，真实和模型反向后验都确定为该元素，变分项为零。
当 `k=1` 时，被 mask 原子的重建 NLL 就是 CE，因此同一个简式覆盖首步：

\[
\ell_{\mathrm{VB},i}=\mathbf1[\text{masked}_i]\,\mathrm{CE}_i/k_i,
\qquad
\ell_{A,i}=\ell_{\mathrm{VB},i}+0.01\,\mathrm{CE}_i.
\]

正式训练代码的核心只有：

```python
ce = F.cross_entropy(atom_logits, atom_target, reduction="none")
k_atom = k_graph[batch]
vb = torch.where(masked_atom, ce / k_atom.to(ce), torch.zeros_like(ce))
atom_per_atom = vb + 0.01 * ce
```

注意四件事：

1. 用 `k_graph[batch]`，不用连续时间 `t_graph`，也不用 `k-1`。
2. `ce` 保留 `[N]`，不能提前对所有原子取平均。
3. 辅助 CE 覆盖所有原子。没有 mask 时，`VB=0`，辅助 CE 仍提供元素头梯度。
4. 不额外乘 `K`。这里匹配的是官方训练时抽取单个时间项的尺度；改成整个时间求和的 ELBO
   估计会改变它与连续项、辅助 CE 的相对权重。

该简式依赖 standard absorbing schedule 和干净 MASK 概率固定为零，不能用于任意离散扩散。
在这些约定下，终点与 MASK 先验之间的 KL 为零，不需要添加一个非零先验损失。

**零概率处理：** 参照 KL 只在 `q>0` 的支持集计算对数比，`q=0` 的项直接贡献零。
若 `q>0,p=0`，数学结果应为无穷；若观察事件本身概率为零，后验函数明确报错。
正式训练直接对 logits 使用稳定的 `cross_entropy`，不经过 `softmax().log()`。
这既保留精确的 MASK 零概率，也避免极端 logits 的 softmax 下溢。

## 5. 连续损失直接比较 scaled score

坐标和晶格的目标在步骤六已经算好：

\[
y_X^\star=\sigma_X\nabla_{X_t}\log q(X_t\mid X_0),
\qquad y_L^\star=-\epsilon_L.
\]

所以先逐元素计算：

```python
coord_error = (prediction["coord_score_times_std"] - info["coord_target"]).square()
lattice_error = (prediction["lattice_score_times_std"] - info["lattice_target"]).square()
```

晶格表达式就是 \((y_L+\epsilon_L)^2\)。坐标使用 wrapped-normal 目标，不能直接替换成采样噪声。
两边都已经乘过标准差，损失里不再除以 `sigma`，也不重复做逆晶格转换。
当前 wrapped-normal 镜像截断继续使用步骤六保留的 `M=3`；本步没有更改已验证的前向过程。

参考：[官方逐字段损失](https://github.com/microsoft/mattergen/blob/main/mattergen/diffusion/training/field_loss.py)、
[周期坐标损失](https://github.com/microsoft/mattergen/blob/main/mattergen/diffusion/wrapped/wrapped_normal_loss.py)。

## 6. 聚合顺序也是损失定义的一部分

对第 `b` 个晶体：

\[
L_X^{(b)}=\sum_{i\in b}\frac13\sum_{d=1}^3(y_{X,id}-y_{X,id}^\star)^2,
\]

\[
L_L^{(b)}=\frac19\sum_{r,c=1}^3(y_{L,brc}-y_{L,brc}^\star)^2,
\qquad
L_A^{(b)}=\sum_{i\in b}\ell_{A,i}.
\]

最后才对晶体取平均：

\[
L=\frac1B\sum_b\left(0.1L_X^{(b)}+L_L^{(b)}+L_A^{(b)}\right).
\]

手写 `aggregate_per_crystal()` 时，先把非首维展平并 `mean(-1)`；原子字段通过 `index_add`
按 `batch` 求和，晶格字段的首维已经是晶体维，不需要再 scatter。
最后三路都是 `[B]`，乘权重相加，再 `.mean()`。

例如两晶体分别有 2 和 3 个原子，每个原子的坐标分量均方误差都是 1：
晶体损失为 `[2,3]`，最终坐标损失为 `2.5`，并不是全体原子的均值 `1`。
因此不能直接用默认 `F.mse_loss()` 对整个拼接张量求均值。

晶格虽然对称，仍按完整的九个矩阵元素求平均；不能擅自改成六个独立元素的平均。
两项对称的非对角误差都会进入损失，与官方聚合一致。

参考：[默认权重与 reduce=sum](https://github.com/microsoft/mattergen/blob/main/mattergen/conf/lightning_module/diffusion_module/default.yaml)、
[字段组合及晶体平均](https://github.com/microsoft/mattergen/blob/main/mattergen/diffusion/losses.py)。

## 7. 接回 JointDiffusion

构造函数新增可选参数 `loss_fn`。默认构造 `JointDiffusionLoss()`，也可以显式传入配置后的对象。
损失文件只依赖 PyTorch，不反向导入 diffusion、scheduler 或 denoiser。

```python
def compute_loss(self, clean, k=None, noise=None):
    if k is None:
        k = torch.randint(1, self.timesteps + 1, clean["num_atoms"].shape,
                          device=clean["num_atoms"].device)
    noisy, info = self.q_sample(clean, k, noise=noise)
    prediction = self.predict(noisy, info["t_graph"])
    return self.loss_fn(prediction, info, clean["batch"], clean["num_atoms"])
```

`k=None` 是日常训练；传入固定 `k` 和 `noise` 可以复现某个训练任务。
每个晶体独立抽一个 `k`，同一晶体的三个分量共享它。`noise=None` 也不会复用上一轮噪声。

返回 `(loss, metrics)`：`loss` 保留梯度，用于 `backward()`；`metrics` 中的标量已 detach，适合日志。

| 指标 | 定义 |
|---|---|
| `loss_coord` | 已按晶体平均、尚未乘 0.1 的坐标损失 |
| `loss_lattice` | 已按晶体平均的晶格损失 |
| `loss_atom_vb` | 已按晶体平均的元素变分项 |
| `loss_atom_ce` | 已按晶体平均、尚未乘 0.01 的辅助 CE |
| `loss_atom` | `loss_atom_vb + 0.01 * loss_atom_ce` |

默认权重下：`loss = 0.1 * loss_coord + loss_lattice + loss_atom`。
detach 的是日志副本，总损失的计算图完整保留。

## 8. 最后才接优化器

一次更新按下面顺序进行：

```python
model.train()
optimizer.zero_grad(set_to_none=True)
loss, metrics = model.compute_loss(clean)
loss.backward()
# 此处读取各输出头的梯度范数。
optimizer.step()
```

`train_step()` 对 loss 和梯度做有限性检查，遇到异常明确报错，不静默跳过问题 batch。
它不会由于没有 mask 而跳过更新，也不会直接使用数据库中的干净结构边。

优化器采用官方 YAML 的 `Adam(lr=1e-4)`；学习率调度为
`ReduceLROnPlateau(factor=0.6, patience=100, min_lr=1e-6)`，监控完整训练 epoch 的平均总损失。
这与官方 Lightning 类在没有配置时的 fallback 优化器不同，此处以正式 YAML 配置为依据。
当前 PyTorch 接口不再需要旧配置里的 `verbose` 参数。

参考：[官方训练优化配置](https://github.com/microsoft/mattergen/blob/main/mattergen/conf/lightning_module/default.yaml)。

`grad_atom`、`grad_coord`、`grad_lattice` 分别记录三个输出分支在总损失 backward 后的梯度 L2 范数。
它们是参数梯度的诊断量，不是另一组损失或额外梯度权重。
完整 epoch 日志对各 batch 的损失按晶体数加权，避免最后一个小 batch 得到过大权重。

训练入口现在加载 `data.CrystalDataset` 的标准化 NPY 缓存。原 `--database` 参数改为 `--data-dir`，
不再将旧 SQLite 数据及旧 Denoiser 调用混入本流程。标准化与缓存实现没有修改。

开始训练前，遍历整个训练 split 一次，固定：

- 平均密度 `rho = mean_i(N_i / det(L_i))`；不是每个 minibatch 的密度，也不是总原子数除以总体积。
- 训练集出现的元素集合。
- 原子数的经验分布。

这些统计不读取验证/测试样本。训练 loss 仍使用完整 118 类 logits；元素集合保存供后续采样阶段使用。
传入 `--checkpoint` 时，保存模型、优化器、学习率调度器状态，以及 scheduler 参数、损失权重、
输出参数化、时间网格、类别约定、训练集统计和标准化元数据。checkpoint 使用新路径，不覆盖已有文件。
本步只建立训练接口，尚未实现步骤八以后的先验采样或反向生成。

## 9. 已完成的检查

`test/test_joint_loss.py` 提供 15 项测试；与步骤六的 8 项联合测试一起运行，共 23 项通过。

| 检查 | 参照及结果 |
|---|---|
| 通用后验方向 | 非对称转移矩阵，逐项枚举 Bayes 公式，与矩阵实现一致 |
| 变分项简式 | 小类别数逐次乘矩阵，所有时间步的 KL/首步 NLL 与 `masked * CE / k` 一致 |
| 元素梯度 | 通用后验路径与简式路径的 logits 梯度一致 |
| 零概率 | `q=0` 项贡献零；正质量对零模型概率为无穷；无效条件事件明确报错 |
| 数值稳定性 | logits 为正负 1000 时，CE 和梯度仍有限 |
| 聚合 | 原子数为 2、3 的非连续 batch 索引，与手算数值一致 |
| 晶格维度 | 两个镜像非对角误差都进入九元素平均 |
| 正确连续目标 | 坐标及晶格损失和对应输出梯度均为零 |
| 无 mask | 元素变分项为零，辅助 CE 仍能反向传播 |
| batch 重复 | 完整重复一份 batch 后，最终损失不变 |
| 训练前向 | `compute_loss()` 与分别调用三个阶段的结果一致，干净输入不变 |
| 时间采样 | 每个晶体独立采样正整数步，`k=1` 和 `k=1000` 可用 |
| 实际更新 | CPU float64 和 CUDA float32 均完成三路输出头的有限、非零梯度及 Adam 更新 |
| 训练集统计 | 手算密度均值、元素集合、原子数经验分布一致 |

两个实际运行的例子：

1. 独立合成示例：总损失 `1.23301363`，坐标 `1.90810716`，晶格 `0.86218405`，
   元素 `0.18001875`；完成一次 Adam 更新。
2. 标准化 MP-20：训练 split 为 27,136 个结构，平均密度约 `0.05771452`；
   两晶体 batch 的总损失约 `1.48326`，三路输出头梯度均有限且非零，完成一次更新。

这些结果验证损失与训练闭环。单步日志不能说明模型已经收敛，也没有在本步运行完整训练或评估生成质量。

## 10. 自己运行

以下命令从项目根目录运行；每次新开 PowerShell，都先激活并检查 diffusion 环境。

```powershell
Set-Location 'E:\THU\AI2DFT\Diffusion-Material-Generation\diffusion_mattergen'
. './scripts/activate_diffusion.ps1'
if ($env:CONDA_DEFAULT_ENV -ne 'diffusion' -or (Get-Command python).Source -ne 'D:\Anaconda\anaconda3\envs\diffusion\python.exe') {
    throw 'Expected the diffusion Python environment.'
}

# 无需数据文件，按代码展开一次训练更新。
python -B -m scripts.step_07_demo
if ($LASTEXITCODE -ne 0) { throw 'Demo failed.' }

# 损失数学、实际更新及步骤六回归。
python -B -m unittest discover -s test -p 'test_joint*.py' -v
if ($LASTEXITCODE -ne 0) { throw 'Tests failed.' }

# 使用现有标准化 MP-20，只运行一个真实数据 batch。
python -B -m scripts.train_mask --device cpu --batch-size 2 --epochs 1 --max-steps 1 --hidden-dim 16 --num-radial 16 --num-layers 1
```

最后一条命令的 `partial=True` 表示在完整 epoch 结束前达到步数上限；它仍用完整训练 split 统计密度。
去掉 `--max-steps` 才会遍历完整 epoch；训练步数、模型规模与论文规模的复现实验应另行设定。

建议按以下顺序手写源码：`categorical_posterior` → `categorical_kl` → `absorbing_d3pm_terms`
→ `aggregate_per_crystal` → `JointDiffusionLoss.forward` → `JointDiffusion.compute_loss`
→ `train_step`。每写完一个数学部分，先跑对应测试，再连接到网络。

完整可执行代码在 `src/loss.py` 和 `scripts/step_07_demo.py`。若已有步骤六的 `model`、`clean`，
可以直接执行以下最小训练代码：

```python
import torch
from src.loss import JointDiffusionLoss

model.loss_fn = JointDiffusionLoss(
    coord_weight=0.1,
    lattice_weight=1.0,
    atom_weight=1.0,
    d3pm_hybrid_lambda=0.01,
)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

model.train()
optimizer.zero_grad(set_to_none=True)
loss, metrics = model.compute_loss(clean)  # 每个晶体随机抽 k，再独立加噪
if not torch.isfinite(loss):
    raise RuntimeError("Non-finite loss")
loss.backward()
optimizer.step()
print("loss:", loss.item())
print({name: value.item() for name, value in metrics.items()})
```
