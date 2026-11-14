# 步骤五核查与步骤六实现讲解

本次只核查步骤五的晶格过程；步骤六复用已验收的元素、坐标、scheduler、周期图和 Denoiser。实现位于 `src/diffusion.py` 的 `JointDiffusion`；可执行示例为 `scripts/step_06_demo.py`。

## 1. 步骤五的结论

`LatticeCorruption.marginal_prob`、`symmetric_noise`、`q_sample` 的核心公式正确。

令原始矩阵 `Z` 的九个元素相互独立，且服从标准高斯。当前代码产生

\[
\epsilon_{ii}=Z_{ii},\qquad
\epsilon_{ij}=\epsilon_{ji}=(Z_{ij}+Z_{ji})/\sqrt2.
\]

非对角项的方差为 `(1+1)/2=1`，对角项也为 1。上三角的六个变量使用互不重叠的原始随机变量，因此相互独立。这与计划中“采样六个独立高斯再填入对称矩阵”同分布，也与官方 `make_noise_symmetric_preserve_variance` 一致。镜像的两个矩阵元素相同，不能把全部九项误认为相互独立。

`mean_coeff` 对应 `a(t)`；`limit_mean` 对应 `mu_N`；`std` 对应标准差 `sigma_L`。所以

\[
\underbrace{aL_0+(1-a)\mu_N}_{m_t}
=\mu_N+a(L_0-\mu_N),\qquad L_t=m_t+\sigma_L\epsilon_L.
\]

`score_target = -(noisy_lattice-mean)/std` 返回的是官方约定的 **score_times_std**，即 `-epsilon_L`，不是未缩放的 score。此处除以标准差一次是正确的。

修正了两个接口问题：

1. `prior_sample` 已计算张量，却缺少 `return prior_lattice`，原调用结果为 `None`。已补上返回值。
2. `t=0` 时 `std=0`。前向采样应恢复 `L0`，但训练目标此时不能用 `0/0` 计算。现在 `score_target` 明确报错，联合训练入口只接受正时间。没有通过给分母加 epsilon 来修改分布。

另外去掉一个未使用的 `limit_var` 局部变量，将说明中的“std-variance”改为“standard deviation”，并统一装饰器写法为 `@torch.no_grad()`。原来不带括号的装饰器在当前 PyTorch 下能工作，不应将它误判为主要错误。

两个使用边界：传入 `q_sample(noise=...)` 的应当是**原始 IID 矩阵噪声**，不能先对称化后再传入；对称化两次会放大非对角项的方差。加噪晶格只需保持对称，不执行正定投影。`a(1)=exp(-5.025)` 仍略大于零，因此 `t=1` 的条件边缘与极限先验接近，并非严格相等。

依据：[官方晶格过程](https://github.com/microsoft/mattergen/blob/main/mattergen/common/diffusion/corruption.py)、[官方预测目标定义](https://github.com/microsoft/mattergen/blob/main/mattergen/diffusion/model_target.py)。

## 2. 步骤六的职责

本步建立两条数据流：

```text
clean + k + 随机数 -> q_sample -> noisy + info
noisy + t         -> predict  -> atom_logits + y_X + y_L
```

`info` 保存训练目标，供下一步的 loss 使用；它不能作为模型输入。`predict` 的参数中没有 `clean`，这样后续从随机初态生成时也能调用同一个接口。

计划列出的 `compute_loss`、`sample_prior`、`reverse_step`、`sample` 分别由后续步骤填入。本步提供它们要用到的状态、输出与尺度约定。官方也是分别负责联合加噪、网络调用、损失和采样，并没有要求在一个函数中同时完成这些工作。

依据：[MultiCorruption](https://github.com/microsoft/mattergen/blob/main/mattergen/diffusion/corruption/multi_corruption.py)、[DiffusionModule](https://github.com/microsoft/mattergen/blob/main/mattergen/diffusion/diffusion_module.py)。

## 3. 先把输入、时间和输出写清楚

保留五个状态字段：

| 字段 | 形状 | 用途 |
|---|---|---|
| `atomic_numbers` | `[N_total]`，int64 | 真实元素 1～118，MASK=0 |
| `frac_coords` | `[N_total,3]` | 当前分数坐标 |
| `lattice` | `[B,3,3]` | 当前晶格，行向量约定 |
| `num_atoms` | `[B]`，int64 | 每个晶体的原子数 |
| `batch` | `[N_total]`，int64 | 每个原子的晶体编号 |

`k` 接受一个整数，或者 `[B]` 整数张量。标量会扩展为每个晶体相同的步数。计算一次 `t = k / K`，随后三个 corruption 和网络都使用这份时间。训练时可以由下一步的 `compute_loss` 用 `torch.randint(1, K+1, (B,))` 采样 `k`。

例如 `num_atoms=[2,3]`、`k=[100,700]` 时，图时间为 `[0.1,0.7]`，原子时间为 `[0.1,0.1,0.7,0.7,0.7]`。时间属于晶体，不应分别给三个分量重新采样。

保留计划的整数网格；它是本项目的离散化选择。官方直接采样连续时间并转换 D3PM 索引。当前 `MaskScheduler.step_from_time(k/K)` 在 `k=1..1000` 上会返回相同的 `k`，已做完整网格检查。这个入口不接受 `k=0`，因为它会同时生成训练目标。

## 4. 组装模块时使用已有实例

构造 `JointDiffusion` 时传入 `denoiser` 和三个 corruption。将它们赋值到 `nn.Module` 的成员上，会自动注册为子模块。之后对外层调用 `.to(device=..., dtype=...)`，网络、径向基及 scheduler 的 buffer 都一起移动。

这里不在 `diffusion.py` 顶部导入 `scheduler.py`：当前 scheduler 中仍有旧工具函数对 diffusion 的导入，反向导入会形成循环。参数注入可直接绕开这个依赖问题，无需改写已验收模块。

`RadialBasis` 的通道数使用 `denoiser.num_radial`，cutoff 与建图函数使用同一个值，避免建图范围和距离编码范围不一致。模型和 batch 的浮点 dtype/device 必须一致；整数编号保持 int64。

## 5. 按字段完成联合加噪

调用顺序为：

```python
atoms_t, masked = atom_corruption.q_sample(A0, t, batch, uniform=u)
frac_t = position_corruption.q_sample(X0, t, num_atoms, batch, noise=z_x)
lattice_t = lattice_corruption.q_sample(L0, t, num_atoms, noise=z_l)
```

三条前向分布给定干净状态和时间后独立采样；“联合”体现在随后的同一个网络同时看到 `(A_t, X_t, L_t)`。共享时间不等于共享随机数。

为便于逐步核对，`noise` 支持三个键：

| 键 | 形状 | 随机分布 |
|---|---|---|
| `atom_uniform` | `[N_total]` | `[0,1)` 上的均匀随机数 |
| `position` | `[N_total,3]` | IID 标准高斯 |
| `lattice` | `[B,3,3]` | **对称化之前**的 IID 标准高斯 |

未提供的项自动采样。固定 `clean`、`k` 和这三个张量，就能重现相同的加噪结果。三个随机张量的 shape、dtype、device 会被检查，避免隐式广播将所有原子变成同一份噪声。

使用新的五字段字典保存 `noisy`，不对 `clean` 做原地赋值，也不携带缓存的干净边。`num_atoms` 和 `batch` 的值保持不变，输出中复制它们，防止两个字典共用可修改的张量。

`q_sample` 带 `@torch.no_grad()`，因为此时还没有用到模型参数。`predict` 不带该装饰器，保证随后 loss 能反向传播。

## 6. 返回训练目标信息

`q_sample` 返回 `(noisy, info)`；其中：

| `info` 键 | 含义 |
|---|---|
| `k_graph`, `t_graph` | 同一份离散步数和连续时间 |
| `masked_atom` | 本次实际被 MASK 的原子 |
| `atom_target` | `A0-1`，用于 118 维 logits 的类别索引 |
| `coord_target` | 已验收坐标模块给出的 `sigma_X * wrapped_score` |
| `lattice_target` | `-epsilon_L_symmetric` |
| `sigma_x` | `[N_total,1]` 的坐标标准差 |
| `sigma_l` | `[B,1,1]` 的晶格标准差 |

坐标目标调用现有 `PositionCorruption.score_target`，不替换为 `-z_x`；晶格目标调用现有 `LatticeCorruption.score_target`，不使用对称化之前的 `-z_l`。

本步保留当前已验收坐标代码的默认 `max_offset_integer=3`，也就是官方 wrapped-normal 实现的默认值。原始计划中的 `M=max(13, ceil(8*sigma_max_effective)+1)` 是扩大截断的数值选择，两者不能称作完全相同的截断配置。外层构造器暴露了 `max_offset_integer`；需要采用原计划的大截断时，可以显式传入所需的 M。对于计划默认 `sigma_max=5` 且 `N>=1`，固定 `M=41` 覆盖该规则的上界，但当前三维枚举实现的内存开销也会变大。本次不重新实现或重新验收步骤四。

依据：[官方 wrapped-normal 目标及默认截断](https://github.com/microsoft/mattergen/blob/main/mattergen/diffusion/wrapped/wrapped_normal_loss.py)。

## 7. 每次 predict 都从当前状态建图

必须使用 `build_periodic_graph`，保留周期镜像邻居；不要退回只使用最近镜像的旧 `build_neighbor_graph`。

对当前边 `(i,j,m)`，边向量来自

\[
\mathbf r_{ijm}=(X_{t,j}+m-X_{t,i})L_{t,b(i)},\quad
d_{ijm}=\|\mathbf r_{ijm}\|.
\]

因此坐标或晶格任一变化，边是否存在、边向量、距离和径向特征都可能改变。每次调用按下列顺序执行：

1. `build_periodic_graph(Xt, Lt, batch=...)` 返回边、周期镜像及图归属。
2. 将本次距离传入 `RadialBasis`。
3. 将 `At`、新图、径向特征、图时间和原子归属传给 Denoiser。
4. 从这一轮 Denoiser 调用同时获得三个头的结果。

对接现有 Denoiser 时，特别注意关键字名为 `edge_vectors`、`edge_distance`、`t_graph`、`num_edges`、`edge_ptr` 和 `edge_batch`。`edge_image` 用于调试，不是当前 Denoiser 的参数。

设置 `return_graph=True` 可在预测结果的 `graph` 字段查看本次生成的图。它仅用于检查，不会缓存下来供下一时刻复用。奇异晶格由建图函数报错；没有用伪逆或正定投影掩盖问题。

依据：[官方 Denoiser 中 on-the-fly 图构造入口](https://github.com/microsoft/mattergen/blob/main/mattergen/denoiser.py)。

## 8. 明确三路输出的参数化

元素头：保留 `atom_logits`，形状 `[N_total,118]`，第 `c` 列对应原子序数 `c+1`。这表示干净元素分布，训练时不先 argmax，也不在这里把 logits 变成离散元素。

坐标头：将原始 `coord_cart` 转换成

\[
y_X=\texttt{coord\_cart}\,L_t^{-1}.
\]

直接复用 `geometry.cart_to_frac(coord_cart, noisy['lattice'], noisy['batch'])`。索引晶格时用原子的 `batch`，所以不同晶体使用各自的当前噪声晶格。非立方晶格下不能仅除以三个晶格长度。

这是网络输出参数化；不要用真实标量函数梯度的协变换公式替代它。例如行向量约定下 `grad_X f = grad_R f * L.T` 是另一个问题，不能据此把这里的逆晶格改为转置晶格。官方输出适配采用的列向量表达是 `L^{-T} u_column`，转回行向量正是 `u_row L^{-1}`。

晶格头：原始 `lattice_pred` 直接作为 `y_L`，形状 `[B,3,3]`；此处不再次取负号，不再对称化，不进行归一化。现有 Denoiser 已用边方向的外积构造对称矩阵输出。

因此 `predict` 的三个核心键是 `atom_logits`、`coord_score_times_std`、`lattice_score_times_std`。它们分别对应后续的元素概率目标、`coord_target` 和 `lattice_target`。

依据：[官方 get_chemgraph_from_denoiser_output](https://github.com/microsoft/mattergen/blob/main/mattergen/denoiser.py)。本轮仍沿用计划中的 MPBlock/Denoiser，完整 GemNet 架构和晶格夹角输入的对齐仍是既定后续工作。

## 9. 给后续采样准备无干净数据依赖的转换

`scores_from_prediction(prediction, noisy, t)` 用 scheduler 及 `num_atoms` 重新得到标准差，然后计算

\[
s_X=y_X/\sigma_X,\qquad s_L=y_L/\sigma_L.
\]

这一步不需要 `clean`、训练目标或前向采样的噪声，也不再次调用网络。以后 corrector 或 predictor 可以使用已经计算过的一份预测。不要多加一次负号；负号已属于 `y_L` 的训练目标定义。

`atom_probs_with_mask` 仅在需要完整后验类别空间时调用：对 118 维 logits 做 softmax，再在最后追加一个**零概率** MASK 列。追加零 logit 会得到非零 MASK 概率，是错误的。

状态编号和后验类别编号分别是：

| 内容 | 元素 Z | MASK |
|---|---|---|
| 状态 `atomic_numbers` | `Z`，1～118 | 0 |
| 后验分布的列索引 | `Z-1`，0～117 | 118 |

这保留原计划的 118 维输出约定，是本项目的接口选择，并非声称官方默认类别数量与该接口完全相同。

## 10. 验收与运行

新增的 13 项针对性检查全部通过，包括：

- 步骤五：两个原子数分别重复采样 80,000 次，检查边缘与先验的均值、逐元素方差；检查六个独立分量的协方差、对称性、正时间目标、零时间恢复，以及未执行正定投影。
- 步骤六：固定随机数重现、三路共享时间、整个正整数时间网格、状态不被原地修改、非立方及非对称矩阵的变换、坐标/晶格变化后的新图、零边图、MASK 概率、奇异晶格报错和真实 Denoiser 三头的有限非零梯度。
- CPU float64 及 CUDA float32 的联合路径。

梯度测试使用简单的数值探针目标，只检查计算图连通性；它不是步骤七的训练损失。本步没有实现或声称验证完整训练与反向生成。

从项目根目录运行：

```powershell
. './scripts/activate_diffusion.ps1'
if ($env:CONDA_DEFAULT_ENV -ne 'diffusion') {
    throw 'The diffusion conda environment is not active.'
}
python -B -m scripts.step_06_demo
python -B -m unittest discover -s test -p 'test_lattice_corruption.py' -v
python -B -m unittest discover -s test -p 'test_joint_diffusion.py' -v
```

建议手写时依次实现：构造器与时间约定 → `q_sample` 三次调用 → 目标信息 → 动态建图 → Denoiser 参数对接 → 三头适配 → score/概率转换。最后对照 `src/diffusion.py` 中的完整实现，并运行 `scripts/step_06_demo.py`。
