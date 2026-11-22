# 本地 src 与 MatterGen 的方法对照

本地已经实现元素、分数坐标和晶格的联合扩散训练，也有完整的无条件反向采样循环。去噪器可用的几何信息和元素生成空间仍是影响复现结果的主要差异。这些差异在训练损失公式完全相同时也会存在。

核对覆盖 `src` 全部 8 个 Python 文件，并沿 `scripts/train.py`、`scripts/generate.py` 检查实际调用。官方基准为微软公开仓库提交 `92423660a8bd70e83679086e88f88596d484dc16`。当前工作区的训练缓存是 MP-20，所以同时比对了 `mp_20_base` 和基于 Alex-MP-20 的 `mattergen_base`。微软说明公开权重由发布仓库重新训练，论文使用了单独的权重。[本地训练缓存][lcache] · [官方模型说明][opublic] · [官方基础模型配置][obase]

## 无条件基础模型的差异

| 项目 | 本地实现、官方实现及影响 |
|---|---|
| 1. 三体角与定向消息 | 本地 `MPBlock` 用边的两个端点、旧边特征和距离基更新边，再向原子求和。官方 GemNet-T 为共享中心的两条有向边构造三元组，显式计算夹角，将角基与径向基送入双线性交互，并交换正反向边的信息。多层本地网络可以间接利用邻接关系，但单次消息没有键角输入，圆形基和三体边到边更新均缺失。[本地 MPBlock][lblock] · [官方三元组][otrip] · [角基][obasis] · [双线性交互][obil] |
| 2. 键相对晶格轴的方向 | 官方还把每条边与当前带噪晶格三个基矢的夹角余弦拼入边嵌入。这三个量独立于相邻两条键的夹角。本地晶格参与周期构图，边方向参与最终向量和张量读出，但消息层与元素分类头没有得到这三个显式特征。只补三体角仍会留下这一差异。[本地边初始化与读出][linit] · [本地构图与模型输入][lpred] · [官方晶格方向编码][obasis] |
| 3. 主干容量与读出网络 | 本地为 3 个交互块、节点和边宽度 64、32 个径向基。两个官方基础模型使用 4 个 GemNet 交互块、宽度 512，配合代码默认的 128 个径向基和 7 个角基通道。官方还有投影、分支方差缩放和残差归一化。本地用较小的残差 MLP 和径向门控读出，元素头是两层 MLP，官方是最终节点嵌入上的单层线性映射。距离基同为高斯加五次起始的多项式包络。即便把宽度改成 512，两边仍是不同的模型，也无法直接互载权重。[本地去噪器][lmodel] · [本地径向读出][lread] · [官方发布配置][obase] · [GemNet 基参数][ogemparam] · [残差与缩放][oint] · [官方元素头][oatomhead] |
| 4. 时间编码与注入 | 本地将标量时间送入可学习 MLP，随后加到每个原子的元素嵌入上。官方对连续时间做多频率正弦/余弦编码，将其与元素嵌入拼接后再线性投影。两边都为每个晶体采一个时间，但去噪器获得的时间特征不同。[本地时间编码][ltime] · [本地相加][linit] · [官方时间编码][otime] · [官方拼接与投影][obasis] |
| 5. 周期近邻图的边集合（已实现） | 本地已采用官方规则。默认在每轴最多 5 层周期镜像中枚举 `0.01 Å < d ≤ 7 Å` 的候选边，每个中心按距离最多取 50 条，包括第 50 条落在等距壳层的情况。随后保留邻居编号小于中心编号的边。同一原子连接自身的周期镜像时，保留平移向量首个非零分量为负的方向。最后对这些边逐条补反向边并取反平移，最终每个中心的邻居数可以超过 50。[本地距离过滤][lcut] · [本地近邻截断][lgraph] · [本地双向化][lmirror] · [官方近零过滤][oshort] · [官方最近邻][otop] · [官方双向化][osym] |
| 6. 元素训练空间 | 本地学习 118 个真实元素和 MASK，共 119 个 logits。官方学习 100 个真实元素和 MASK，共 101 个 logits。本地把 MASK 放在首列，官方放在末列并使用偏移量 1。列的位置可以通过重排对应，多出的 18 类会参与每次训练的 softmax 归一化。[本地输出维度][lmodel] · [本地损失维度][lloss] · [官方类别与偏移][obase] · [官方元素上限][owlist] |
| 7. 生成时的元素约束 | 官方生成命令会在计算反向后验前屏蔽干净元素 logits，仅开放预选的 76 种元素。有效化学体系条件还会进一步限制集合。中间的噪声状态仍可保留 MASK。本地对所有 logits 直接求后验，若最终预测出 MASK，结构导出会抛错。若预测出白名单外但在 1–118 内的元素，导出则会接受。即使无条件生成，这条差异也成立。[本地后验][latomp] · [本地导出][lexport] · [官方生成命令][oclimask] · [官方屏蔽函数][omask] · [76 种元素][owlist] |
| 8. 晶格反向 predictor 的噪声协方差（已实现） | 本地与官方都用 `randn_like(x_coeff)` 从形状为 `[B, 1, 1]` 的系数张量抽样，每个晶体在每个预测步独立抽一个标准高斯数 g。对称化后的 3×3 噪声矩阵中，三个对角元都是 g，六个非对角元都是 √2·g。采样时将矩阵乘以反向标准差，再加到预测均值上。晶格前向加噪、先验和 Langevin 校正步均从完整的 3×3 高斯矩阵构造对称噪声。[本地采样][lvpp] · [本地逐晶体抽样用例][ltestvp] · [官方系数形状][oalphashape] · [维度扩展][obroadcast] · [反向系数][ocoef] · [官方抽样][orand] · [官方对称化][osymnoise] |
| 9. 生成晶体的原子数先验 | 两边都先抽原子数 N，再生成元素、位置和晶格。官方生成器默认选用内置的 `ALEX_MP_20` 分布，本地从所选训练集统计 N，生成时读取 `statistic.json`。按当前 MP-20 缓存重算，P(N=20) 为 **8.5164%**，官方默认是 **4.3457%**。即使给定 N 的去噪器完全相同，无条件样品的大小分布仍会不同。[本地统计][lstats] · [本地抽样][lsampleN] · [官方生成器默认值][ogendef] · [官方概率表][on] |
| 10. 生成批量会改变校正步长 | 本地命令行默认每批 1 个晶体，官方生成命令默认 64 个。两边的 Langevin corrector 都先算每个晶体的噪声范数和 score 范数，再分别对整批取平均以决定步长。因此单样本生成与批量生成也会得到不同的自适应步长。两边的批大小都可指定。[本地生成参数][lgenCLI] · [本地校正器][lcorr] · [官方生成参数][ogcli] · [官方位置校正器][ocorr] · [官方晶格校正器][olcorr] |

**第 5 条修改前的对照记录。** 当时用本地 MP-20 的 `mp-1078776`（InBiS₃，原胞 10 个原子）按两套规则独立枚举。默认截断得到 632 条候选有向边，官方规则逐中心截取后剩 500 条，双向化后为 **496** 条。旧版本地规则保留整个等距壳层并取两端所选边的并集，得到 **504** 条。该例第 50 与第 51 邻居的最小距离差约 0.001 Å。本次修改仅做静态检查，新版在该晶体上的实际边数尚未复测。[缓存记录][lexample] · [本地双向化][lmirror] · [官方截断][otop] · [官方双向化][osym]

**最终 MASK 的影响。** 对最后一步仍为 MASK 的位置，忽略数值平滑项，吸收态后验给“继续是 MASK”的权重为模型的干净 MASK 概率，给真实元素 z 的权重为模型的干净元素概率乘 1/1000。即使干净 MASK 的预测概率很小，它也可能成为该位置最后的后验众数。本地导出使用最后的众数，官方生成入口通过屏蔽干净 MASK 的 logit 消除了这条实际失败路径。[本地后验公式][ld3] · [本地生成入口][lgen] · [官方屏蔽函数][omask] · [官方 D3PM predictor][oposterior]

## 官方已支持、本地尚未接入的任务

| 能力 | 当前缺口和适用范围 |
|---|---|
| 条件训练与属性表示 | 本地数据加载返回元素、坐标、晶格和标识，`q_sample` 显式重建只含结构字段的带噪状态。原始记录中保存了带隙等信息，训练网络收不到这些标签。官方支持连续属性的标准化与嵌入、空间群嵌入、化学体系 multi-hot、缺失标签和训练时随机丢弃条件。基础模型的属性嵌入本身为空，这一缺口影响带隙、磁密度、体模量、空间群、指定化学体系等条件任务。[本地数据加载][lload] · [本地带噪状态][lcond] · [官方属性实现][opropfit] · [化学体系嵌入][oprop] · [条件丢弃][odrop] · [官方已发布任务][opublic] |
| 预训练权重与条件微调 | 本地加载器要求本项目的 `.pt` 结构并实例化本地 GNN。官方可加载发布的基础模型，用逐层 adapter 注入新属性，混入层零初始化，还能选择冻结原权重或全量微调。当前缺少对应的 GemNet 权重结构、adapter 和微调训练入口，单属性及多属性微调都受影响。[本地加载器][lckpt] · [官方参数初始化][oadaptzero] · [逐层注入][oadapter] · [微调与冻结][ofine] |
| 无分类器引导（CFG） | 官方在采样时组合无条件和条件输出，连续字段组合 score，元素字段组合 logits。引导系数 γ 对应 `(1−γ) × 无条件输出 + γ × 条件输出`。本地校正前后分别调用模型以更新几何，两次调用的目的都是无条件去噪，尚无属性引导分支。官方无条件基础采样的默认 γ 为 0。[本地采样步][lpc] · [官方 CFG][oguided] · [官方默认采样配置][ocfg] |
| 固定化学计量的晶体结构预测（CSP） | 官方 CSP 给定各元素及其整数个数，固定元素种类，只训练和采样位置、晶格。它与只限定“允许出现哪些元素”的化学体系条件不同。本地训练固定包含元素加噪和元素损失，默认生成从全 MASK 开始，并在每一步更新元素。把元素损失权重调为零仍会执行加噪和采样。[本地联合加噪][lq] · [本地先验][lprior] · [本地采样步][lpc] · [官方 CSP 训练配置][ocsp] · [官方配方输入][ocompose] |
| 局部固定与采样诊断 | 官方通用采样器支持用字段掩码保留已知的连续自由度、按字段选择 predictor/corrector、配置校正次数、记录中间轨迹。本地能从调用者传来的初态开始，但现有循环固定每步一次校正，三个字段都会更新，只返回最后的样本与均值。默认无条件采样的校正次数已经一致，差距在局部补全实验及轨迹对照。[本地采样循环][lgrid] · [官方字段配置][osampleriface] · [固定字段][oinpaint] · [官方轨迹记录][opc] |

## 训练与论文指标的可比性

| 项目 | 核对结果 |
|---|---|
| 数据与训练预算 | 当前可见 MP-20 缓存共 45,229 个结构，训练集 27,136 个，对应的公开对照组是 `mp_20_base`。`mattergen_base` 使用约 60 万个结构的 Alex-MP-20。本地训练命令默认单设备、batch 2、1 个 epoch。两份官方发布配置的全局训练 batch 均为 512，MP-20 和 Alex-MP-20 的最大 epoch 分别为 900 和 2200。本地可改 batch 和 epoch，训练入口没有对应的分布式和续训流程。默认只跑 1 个 epoch 时，也不会触发每 5 个 epoch 一次的验证或保存验证最优权重。这里比较的是配置上限和默认值，实际训练量以运行记录为准。[本地缓存][lcache] · [本地训练参数][ltrain] · [本地验证与保存][lval] · [官方 MP-20 训练配置][otrainmp] · [官方 Alex-MP-20 训练配置][otrainbase] · [官方数据说明][odata] · [官方续训][oresume] |
| 晶格密度先验 | 本地从训练集逐晶体取 N/V 的算术平均，官方发布配置直接给定 0.05771451654022283 atoms/Å³。按本地数据加载时的浮点精度复算当前缓存，得到 **0.05771451651 atoms/Å³**。对这份 MP-20 缓存，该参数已经数值对齐。换训练集时本地均值会随之改变。[本地密度估计][lstats] · [本地读取精度][lload] · [官方密度与晶格配置][obase] |
| 原始 CIF 的接收范围 | 两边都做原胞化、Niggli 约化和对称晶格表示。本地要求有序、满占位的输入，严格解析后再做原胞化。官方 CSV 入口用 CIF parser 的原胞解析并取第一个返回结构，再做约化。特殊占位或有歧义的 CIF 可能获得不同的接受结果。这份本地缓存记录的跳过数为零。[本地原胞与约化][lprep] · [本地 CIF 解析][lparse] · [官方 CIF 读取][ocif] · [官方约化][odataprep] · [缓存统计][lcache] |
| 生成质量评估 | 本地验证入口计算去噪损失，生成入口输出 CIF。当前项目缺少生成后弛豫、基于参考能量凸包的稳定性、基于结构匹配的唯一性和新颖性，以及三者交集 SUN 比例的评估流程。官方还支持结构/组分有效性、弛豫前后 RMSD 及带目标属性的联合指标。公开工具默认用 MatterSim 弛豫和估能，支持导入外部 DFT 能量、选择能量校正和有序/无序结构匹配。默认稳定性阈值为高于凸包不超过 0.1 eV/atom。论文基准使用 DFT 弛豫与能量，两种口径应分别对照。[本地验证][lval] · [本地生成][lgenCLI] · [官方评估入口][oeval] · [稳定性阈值][othr] · [能量与 SUN 指标][ohull] · [属性联合指标][opropeval] · [官方对论文评估的说明][oevalread] |

## 八个源文件中已经对齐的部分

| 本地文件 | 已核对的主路径 |
|---|---|
| `scheduler.py` | 标准吸收态离散时间映射、1000 步，位置 VE 的 σ 从 0.01 到 5 并除以 ∛N，晶格 VP 的线性 β 从 0.1 到 20。[本地调度][lsc] · [官方配置][obase] · [官方位置 SDE][ove] |
| `corruption.py` | 对真实元素做吸收态遮蔽，分数坐标加高斯噪声再包回单位晶胞。位置目标为乘 σ 的 wrapped-normal score，每轴取 −3 到 3 的镜像。晶格向边长 ∛(N/ρ) 的立方晶胞回归，极限方差为 0.25 × (∛N)²，使用保方差的对称高斯噪声。[本地位置目标][lpos] · [本地晶格噪声][llattice] · [官方位置目标][owrapped] · [官方晶格 SDE][ovp] · [官方晶格方差和采样][ovpcorr] |
| `geometry.py` | 行向量晶格约定、分数/笛卡尔转换、坐标取模、左极分解，以及周期多镜像和非零平移自邻居都已实现。训练和生成走 `build_periodic_graph`。[本地几何][lgeo] · [本地实际调用][lpred] · [官方坐标输出转换][ocart] |
| `gnn.py` | 距离用高斯径向基展开并在 cutoff 平滑衰减。消息有径向门控，节点按入边求和。差距主要在三体和定向交互。[本地径向基][lrbf] · [本地消息聚合][lblock] · [官方基与交互][obasis] |
| `denoiser.py` | 已有三个头。坐标头将各层边标量乘单位边向量后按原子求和。晶格头用单位边向量外积形成对称张量，按晶体边数归一化并累计各层贡献。元素头取最终节点表示。这两类几何读出与官方同构，决定边系数的特征提取网络有前述差异。[本地三个头][lhead] · [官方坐标输出][oforce] · [官方晶格归一化][olread] · [官方外积][olscore] |
| `diffusion.py` | 每个晶体采一个连续时间，三个字段共享该时间、各自加噪，随后在带噪结构上重建图。预测分数坐标的 score×σ 时，用当前晶格的逆将笛卡尔输出换回分数坐标。采样时除以对应标准差。[本地加噪][lq] · [本地预测][lpred] · [本地训练与还原 score][ltrainloss] · [官方联合训练][odiff] · [官方坐标转换][ocart] |
| `loss.py` | 位置 wrapped-normal score MSE、晶格对称噪声目标 MSE、元素 D3PM 后验 KL（首步重建 NLL）加 0.01 倍干净类别交叉熵。分量取均值、节点按晶体求和、晶体间取均值。位置/晶格/元素权重为 0.1/1/1，与发布配置一致。[本地损失][lloss] · [本地归约][lred] · [官方 KL 与首步][od3loss] · [官方归约][ofield] · [官方权重][obase] |
| `sampling.py` | 先验为全 MASK、按 N 缩放的包裹高斯坐标和密度相关对称晶格。默认 1000 步，每步先做位置与晶格 Langevin 校正，再在校正后的状态重算 score，三个 predictor 联合更新。位置/晶格 SNR 分别 0.4/0.2，最大校正步长 10⁶。生成取最后一步连续均值和元素后验众数。晶格 predictor 的随机项见第 8 项。[本地先验][lprior] · [本地循环][lpc] · [官方采样配置][ocfg] · [官方更新顺序][opc] · [官方 D3PM 后验][oposterior] |

本地还与两个发布配置共享 Adam、初始学习率 10⁻⁴、按训练损失工作的 ReduceLROnPlateau（系数 0.6、耐心 100、最低 10⁻⁶）、0.5 的逐值梯度裁剪和每 5 个 epoch 的验证频率。[本地优化][lopt] · [本地验证][lval] · [官方优化][obase] · [官方训练器][otrainbase]

## 非默认调用和验证上的缺口

- **自定义元素数和步数时，损失参数可能脱节。** 直接构造 `JointDiffusion` 并省略 `loss_fn` 时，损失会自行取 118 个元素和 1000 步。若只把去噪器和 corruption 改为官方的 100 种元素，就会触发 logits 形状检查。若只改扩散步数，则加噪与损失可能套用不同的离散时刻。官方从实际 corruption 读取 N。当前本地训练脚本显式传了匹配的损失参数。[本地默认构造][ldefloss] · [本地损失映射][lred] · [本地脚本的显式配置][lconf] · [官方按 corruption 取 N][ofield]
- **直接调用先验时，晶格密度取自调度器。** 本地 `sample_prior` 的形参 `limit_density` 没有被读取，晶格实际使用传入的 `lat_cor.scheduler.limit_density`。正常生成传入同一个配置值，结果一致。单独做密度消融时，只改函数参数不会改变晶格先验。[本地先验][lprior] · [本地生成入口][lgen] · [官方晶格先验][ovpcorr]
- **独立调用元素边际分布时，MASK 的概率行会被覆盖。** `AtomTypeCorruption.marginal_prob` 先给 MASK 列赋遮蔽概率，再向输入元素列写入保留概率。若输入本身就是 MASK，两次写入落在同一列，行和变为 1−p。吸收态的正确行和及其 MASK 概率都应为 1。正常联合训练在入口拒绝干净 MASK，因此这项只影响单独复用该接口。[本地概率构造][latom] · [本地联合入口][lq] · [官方边际分布][oabs]
- **部分辅助函数和旧测试采用另一套约定。** `mask_atomtype/atomtype_ce` 提供直接遮蔽和仅对遮蔽位置算 CE 的路径，`coordinate_target` 用一维镜像和另一套符号、归一化，`build_neighbor_graph` 逐分量取整且每对原子只取一个镜像。训练入口实际调用的是联合 D3PM、三维 wrapped-normal 目标和多镜像周期图。部分旧集成测试仍传入 250、1000 之类的整数时间，或断言末列 MASK 概率恒为零，与当前连续时间和训练时可学习的首列 MASK 不相容。仓库 TODO 中的周期表表示或元素 VAE 属于额外研究方向，官方基础模型采用吸收态 D3PM。[本地辅助函数][lold] · [简化邻居函数][loldgraph] · [训练入口][ltrainloss] · [旧时间断言][ltest] · [旧 MASK 断言][ltestmask] · [项目 TODO][ltodo] · [官方基础配置][obase]

以上结论来自固定版本的源码与配置、当前工作树以及本地缓存的独立数值复算。模型质量差距的大小仍须在同一数据集、样本数和评估口径下通过训练与生成实验测量。

[lcache]: <E:/THU/AI2DFT/Diffusion-Material-Generation/diffusion_mattergen/data_cache/mp_20/metadata.json:43>
[opublic]: https://github.com/microsoft/mattergen/blob/92423660a8bd70e83679086e88f88596d484dc16/README.md#L67-L85
[obase]: https://github.com/microsoft/mattergen/blob/92423660a8bd70e83679086e88f88596d484dc16/checkpoints/mattergen_base/config.yaml#L64-L143
[lblock]: <E:/THU/AI2DFT/Diffusion-Material-Generation/diffusion_mattergen/src/gnn.py:120>
[otrip]: https://github.com/microsoft/mattergen/blob/92423660a8bd70e83679086e88f88596d484dc16/mattergen/common/gemnet/gemnet.py#L383-L428
[obasis]: https://github.com/microsoft/mattergen/blob/92423660a8bd70e83679086e88f88596d484dc16/mattergen/common/gemnet/gemnet.py#L687-L715
[obil]: https://github.com/microsoft/mattergen/blob/92423660a8bd70e83679086e88f88596d484dc16/mattergen/common/gemnet/layers/interaction_block.py#L261-L290
[linit]: <E:/THU/AI2DFT/Diffusion-Material-Generation/diffusion_mattergen/src/denoiser.py:66>
[lpred]: <E:/THU/AI2DFT/Diffusion-Material-Generation/diffusion_mattergen/src/diffusion.py:184>
[lmodel]: <E:/THU/AI2DFT/Diffusion-Material-Generation/diffusion_mattergen/src/denoiser.py:21>
[lread]: <E:/THU/AI2DFT/Diffusion-Material-Generation/diffusion_mattergen/src/gnn.py:56>
[ogemparam]: https://github.com/microsoft/mattergen/blob/92423660a8bd70e83679086e88f88596d484dc16/mattergen/common/gemnet/gemnet.py#L201-L228
[oint]: https://github.com/microsoft/mattergen/blob/92423660a8bd70e83679086e88f88596d484dc16/mattergen/common/gemnet/layers/interaction_block.py#L165-L211
[oatomhead]: https://github.com/microsoft/mattergen/blob/92423660a8bd70e83679086e88f88596d484dc16/mattergen/denoiser.py#L196-L210
[ltime]: <E:/THU/AI2DFT/Diffusion-Material-Generation/diffusion_mattergen/src/denoiser.py:7>
[otime]: https://github.com/microsoft/mattergen/blob/92423660a8bd70e83679086e88f88596d484dc16/mattergen/diffusion/model_utils.py#L54-L74
[lgraph]: <E:/THU/AI2DFT/Diffusion-Material-Generation/diffusion_mattergen/src/geometry.py:117>
[lcut]: <E:/THU/AI2DFT/Diffusion-Material-Generation/diffusion_mattergen/src/geometry.py:174>
[lmirror]: <E:/THU/AI2DFT/Diffusion-Material-Generation/diffusion_mattergen/src/geometry.py:234>
[oshort]: https://github.com/microsoft/mattergen/blob/92423660a8bd70e83679086e88f88596d484dc16/mattergen/common/utils/ocp_graph_utils.py#L214-L235
[otop]: https://github.com/microsoft/mattergen/blob/92423660a8bd70e83679086e88f88596d484dc16/mattergen/common/utils/ocp_graph_utils.py#L313-L332
[osym]: https://github.com/microsoft/mattergen/blob/92423660a8bd70e83679086e88f88596d484dc16/mattergen/common/gemnet/gemnet.py#L440-L482
[lloss]: <E:/THU/AI2DFT/Diffusion-Material-Generation/diffusion_mattergen/src/loss.py:176>
[owlist]: https://github.com/microsoft/mattergen/blob/92423660a8bd70e83679086e88f88596d484dc16/mattergen/common/utils/globals.py#L55-L133
[latomp]: <E:/THU/AI2DFT/Diffusion-Material-Generation/diffusion_mattergen/src/sampling.py:161>
[lexport]: <E:/THU/AI2DFT/Diffusion-Material-Generation/diffusion_mattergen/src/sampling.py:419>
[oclimask]: https://github.com/microsoft/mattergen/blob/92423660a8bd70e83679086e88f88596d484dc16/mattergen/scripts/generate.py#L65-L78
[omask]: https://github.com/microsoft/mattergen/blob/92423660a8bd70e83679086e88f88596d484dc16/mattergen/denoiser.py#L49-L125
[lvpp]: <E:/THU/AI2DFT/Diffusion-Material-Generation/diffusion_mattergen/src/sampling.py:306>
[ltestvp]: <E:/THU/AI2DFT/Diffusion-Material-Generation/diffusion_mattergen/test/test_lattice_predictor.py:112>
[oalphashape]: https://github.com/microsoft/mattergen/blob/92423660a8bd70e83679086e88f88596d484dc16/mattergen/common/diffusion/corruption.py#L98-L108
[obroadcast]: https://github.com/microsoft/mattergen/blob/92423660a8bd70e83679086e88f88596d484dc16/mattergen/diffusion/corruption/corruption.py#L27-L55
[ocoef]: https://github.com/microsoft/mattergen/blob/92423660a8bd70e83679086e88f88596d484dc16/mattergen/diffusion/sampling/predictors.py#L130-L165
[orand]: https://github.com/microsoft/mattergen/blob/92423660a8bd70e83679086e88f88596d484dc16/mattergen/common/diffusion/predictors_correctors.py#L33-L53
[osymnoise]: https://github.com/microsoft/mattergen/blob/92423660a8bd70e83679086e88f88596d484dc16/mattergen/common/diffusion/corruption.py#L22-L38
[lstats]: <E:/THU/AI2DFT/Diffusion-Material-Generation/diffusion_mattergen/scripts/train.py:27>
[ogendef]: https://github.com/microsoft/mattergen/blob/92423660a8bd70e83679086e88f88596d484dc16/mattergen/generator.py#L186-L209
[on]: https://github.com/microsoft/mattergen/blob/92423660a8bd70e83679086e88f88596d484dc16/mattergen/common/data/num_atoms_distribution.py#L4-L26
[lcorr]: <E:/THU/AI2DFT/Diffusion-Material-Generation/diffusion_mattergen/src/sampling.py:62>
[ogcli]: https://github.com/microsoft/mattergen/blob/92423660a8bd70e83679086e88f88596d484dc16/mattergen/scripts/generate.py#L16-L33
[ocorr]: https://github.com/microsoft/mattergen/blob/92423660a8bd70e83679086e88f88596d484dc16/mattergen/diffusion/sampling/predictors_correctors.py#L94-L122
[olcorr]: https://github.com/microsoft/mattergen/blob/92423660a8bd70e83679086e88f88596d484dc16/mattergen/common/diffusion/predictors_correctors.py#L64-L98
[lexample]: <E:/THU/AI2DFT/Diffusion-Material-Generation/diffusion_mattergen/data_cache/mp_20/train/records.jsonl:9>
[ld3]: <E:/THU/AI2DFT/Diffusion-Material-Generation/diffusion_mattergen/src/loss.py:99>
[lgen]: <E:/THU/AI2DFT/Diffusion-Material-Generation/diffusion_mattergen/scripts/generate.py:90>
[oposterior]: https://github.com/microsoft/mattergen/blob/92423660a8bd70e83679086e88f88596d484dc16/mattergen/diffusion/d3pm/d3pm_predictors_correctors.py#L82-L105
[lload]: <E:/THU/AI2DFT/Diffusion-Material-Generation/diffusion_mattergen/data/dataset.py:76>
[lcond]: <E:/THU/AI2DFT/Diffusion-Material-Generation/diffusion_mattergen/src/diffusion.py:159>
[opropfit]: https://github.com/microsoft/mattergen/blob/92423660a8bd70e83679086e88f88596d484dc16/mattergen/property_embeddings.py#L452-L526
[oprop]: https://github.com/microsoft/mattergen/blob/92423660a8bd70e83679086e88f88596d484dc16/mattergen/property_embeddings.py#L375-L452
[odrop]: https://github.com/microsoft/mattergen/blob/92423660a8bd70e83679086e88f88596d484dc16/mattergen/property_embeddings.py#L125-L226
[lckpt]: <E:/THU/AI2DFT/Diffusion-Material-Generation/diffusion_mattergen/scripts/generate.py:32>
[oadaptzero]: https://github.com/microsoft/mattergen/blob/92423660a8bd70e83679086e88f88596d484dc16/mattergen/common/gemnet/gemnet_ctrl.py#L53-L67
[oadapter]: https://github.com/microsoft/mattergen/blob/92423660a8bd70e83679086e88f88596d484dc16/mattergen/common/gemnet/gemnet_ctrl.py#L188-L205
[ofine]: https://github.com/microsoft/mattergen/blob/92423660a8bd70e83679086e88f88596d484dc16/mattergen/scripts/finetune.py#L77-L106
[lpc]: <E:/THU/AI2DFT/Diffusion-Material-Generation/diffusion_mattergen/src/sampling.py:345>
[oguided]: https://github.com/microsoft/mattergen/blob/92423660a8bd70e83679086e88f88596d484dc16/mattergen/diffusion/sampling/classifier_free_guidance.py#L46-L99
[ocfg]: https://github.com/microsoft/mattergen/blob/92423660a8bd70e83679086e88f88596d484dc16/mattergen/sampling_conf/default.yaml#L1-L40
[lq]: <E:/THU/AI2DFT/Diffusion-Material-Generation/diffusion_mattergen/src/diffusion.py:138>
[lprior]: <E:/THU/AI2DFT/Diffusion-Material-Generation/diffusion_mattergen/src/sampling.py:43>
[ocsp]: https://github.com/microsoft/mattergen/blob/92423660a8bd70e83679086e88f88596d484dc16/mattergen/conf/lightning_module/diffusion_module/csp.yaml#L2-L13
[ocompose]: https://github.com/microsoft/mattergen/blob/92423660a8bd70e83679086e88f88596d484dc16/mattergen/common/data/condition_factory.py#L57-L77
[lgrid]: <E:/THU/AI2DFT/Diffusion-Material-Generation/diffusion_mattergen/src/sampling.py:365>
[osampleriface]: https://github.com/microsoft/mattergen/blob/92423660a8bd70e83679086e88f88596d484dc16/mattergen/diffusion/sampling/pc_sampler.py#L28-L82
[oinpaint]: https://github.com/microsoft/mattergen/blob/92423660a8bd70e83679086e88f88596d484dc16/mattergen/diffusion/sampling/pc_sampler.py#L225-L279
[opc]: https://github.com/microsoft/mattergen/blob/92423660a8bd70e83679086e88f88596d484dc16/mattergen/diffusion/sampling/pc_sampler.py#L174-L222
[ltrain]: <E:/THU/AI2DFT/Diffusion-Material-Generation/diffusion_mattergen/scripts/train.py:182>
[lval]: <E:/THU/AI2DFT/Diffusion-Material-Generation/diffusion_mattergen/scripts/train.py:283>
[otrainmp]: https://github.com/microsoft/mattergen/blob/92423660a8bd70e83679086e88f88596d484dc16/checkpoints/mp_20_base/config.yaml#L147-L184
[otrainbase]: https://github.com/microsoft/mattergen/blob/92423660a8bd70e83679086e88f88596d484dc16/checkpoints/mattergen_base/config.yaml#L147-L183
[odata]: https://github.com/microsoft/mattergen/blob/92423660a8bd70e83679086e88f88596d484dc16/README.md#L348-L355
[oresume]: https://github.com/microsoft/mattergen/blob/92423660a8bd70e83679086e88f88596d484dc16/mattergen/diffusion/run.py#L131-L155
[lprep]: <E:/THU/AI2DFT/Diffusion-Material-Generation/diffusion_mattergen/data/standardize.py:49>
[lparse]: <E:/THU/AI2DFT/Diffusion-Material-Generation/diffusion_mattergen/data/standardize.py:132>
[ocif]: https://github.com/microsoft/mattergen/blob/92423660a8bd70e83679086e88f88596d484dc16/mattergen/common/data/dataset.py#L528-L548
[odataprep]: https://github.com/microsoft/mattergen/blob/92423660a8bd70e83679086e88f88596d484dc16/mattergen/common/data/dataset.py#L351-L370
[oeval]: https://github.com/microsoft/mattergen/blob/92423660a8bd70e83679086e88f88596d484dc16/mattergen/scripts/evaluate.py#L23-L65
[othr]: https://github.com/microsoft/mattergen/blob/92423660a8bd70e83679086e88f88596d484dc16/mattergen/evaluation/utils/globals.py#L4-L5
[ohull]: https://github.com/microsoft/mattergen/blob/92423660a8bd70e83679086e88f88596d484dc16/mattergen/evaluation/metrics/energy.py#L299-L355
[opropeval]: https://github.com/microsoft/mattergen/blob/92423660a8bd70e83679086e88f88596d484dc16/mattergen/evaluation/metrics/property.py#L140-L160
[oevalread]: https://github.com/microsoft/mattergen/blob/92423660a8bd70e83679086e88f88596d484dc16/README.md#L130-L155
[lsc]: <E:/THU/AI2DFT/Diffusion-Material-Generation/diffusion_mattergen/src/scheduler.py:8>
[ove]: https://github.com/microsoft/mattergen/blob/92423660a8bd70e83679086e88f88596d484dc16/mattergen/common/diffusion/corruption.py#L215-L270
[lpos]: <E:/THU/AI2DFT/Diffusion-Material-Generation/diffusion_mattergen/src/corruption.py:52>
[llattice]: <E:/THU/AI2DFT/Diffusion-Material-Generation/diffusion_mattergen/src/corruption.py:94>
[owrapped]: https://github.com/microsoft/mattergen/blob/92423660a8bd70e83679086e88f88596d484dc16/mattergen/diffusion/wrapped/wrapped_normal_loss.py#L98-L119
[ovp]: https://github.com/microsoft/mattergen/blob/92423660a8bd70e83679086e88f88596d484dc16/mattergen/common/diffusion/corruption.py#L73-L108
[ovpcorr]: https://github.com/microsoft/mattergen/blob/92423660a8bd70e83679086e88f88596d484dc16/mattergen/common/diffusion/corruption.py#L148-L182
[lgeo]: <E:/THU/AI2DFT/Diffusion-Material-Generation/diffusion_mattergen/src/geometry.py:32>
[ocart]: https://github.com/microsoft/mattergen/blob/92423660a8bd70e83679086e88f88596d484dc16/mattergen/denoiser.py#L148-L165
[lrbf]: <E:/THU/AI2DFT/Diffusion-Material-Generation/diffusion_mattergen/src/gnn.py:18>
[lhead]: <E:/THU/AI2DFT/Diffusion-Material-Generation/diffusion_mattergen/src/denoiser.py:72>
[oforce]: https://github.com/microsoft/mattergen/blob/92423660a8bd70e83679086e88f88596d484dc16/mattergen/common/gemnet/gemnet.py#L748-L791
[olread]: https://github.com/microsoft/mattergen/blob/92423660a8bd70e83679086e88f88596d484dc16/mattergen/common/gemnet/gemnet.py#L125-L139
[olscore]: https://github.com/microsoft/mattergen/blob/92423660a8bd70e83679086e88f88596d484dc16/mattergen/common/utils/lattice_score.py#L28-L37
[ltrainloss]: <E:/THU/AI2DFT/Diffusion-Material-Generation/diffusion_mattergen/src/diffusion.py:214>
[odiff]: https://github.com/microsoft/mattergen/blob/92423660a8bd70e83679086e88f88596d484dc16/mattergen/diffusion/diffusion_module.py#L73-L116
[lred]: <E:/THU/AI2DFT/Diffusion-Material-Generation/diffusion_mattergen/src/loss.py:223>
[od3loss]: https://github.com/microsoft/mattergen/blob/92423660a8bd70e83679086e88f88596d484dc16/mattergen/diffusion/d3pm/d3pm.py#L795-L859
[ofield]: https://github.com/microsoft/mattergen/blob/92423660a8bd70e83679086e88f88596d484dc16/mattergen/diffusion/training/field_loss.py#L128-L197
[lopt]: <E:/THU/AI2DFT/Diffusion-Material-Generation/diffusion_mattergen/scripts/train.py:233>
[ldefloss]: <E:/THU/AI2DFT/Diffusion-Material-Generation/diffusion_mattergen/src/diffusion.py:78>
[lconf]: <E:/THU/AI2DFT/Diffusion-Material-Generation/diffusion_mattergen/scripts/train.py:211>
[latom]: <E:/THU/AI2DFT/Diffusion-Material-Generation/diffusion_mattergen/src/corruption.py:12>
[oabs]: https://github.com/microsoft/mattergen/blob/92423660a8bd70e83679086e88f88596d484dc16/mattergen/diffusion/d3pm/d3pm.py#L498-L542
[lold]: <E:/THU/AI2DFT/Diffusion-Material-Generation/diffusion_mattergen/src/diffusion.py:10>
[loldgraph]: <E:/THU/AI2DFT/Diffusion-Material-Generation/diffusion_mattergen/src/geometry.py:62>
[ltest]: <E:/THU/AI2DFT/Diffusion-Material-Generation/diffusion_mattergen/test/test_joint_diffusion.py:72>
[ltestmask]: <E:/THU/AI2DFT/Diffusion-Material-Generation/diffusion_mattergen/test/test_joint_diffusion.py:146>
[ltodo]: <E:/THU/AI2DFT/Diffusion-Material-Generation/diffusion_mattergen/TODO.md:1>
