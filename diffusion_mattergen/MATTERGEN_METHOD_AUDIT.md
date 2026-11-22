# 本地 src 与 MatterGen 的方法对照

本地已经实现元素、分数坐标和晶格的联合扩散训练，也有完整的无条件反向采样循环。去噪器采用 GemNet-T 的三体消息、晶格方向编码、交互块和三个去噪读出。

核对覆盖 `src` 全部 8 个 Python 文件，并沿 `scripts/train.py`、`scripts/generate.py` 检查实际调用。官方基准为微软公开仓库提交 `92423660a8bd70e83679086e88f88596d484dc16`。当前工作区的训练缓存是 MP-20，所以同时比对了 `mp_20_base` 和基于 Alex-MP-20 的 `mattergen_base`。微软说明公开权重由发布仓库重新训练，论文使用了单独的权重。[本地训练缓存][lcache] · [官方模型说明][opublic] · [官方基础模型配置][obase]

## 无条件基础模型的差异

第 1–3 项按 2026-11-22 的工作树和默认训练入口复核，具体计算见下方“第 1–3 项复核明细”。其余编号按各自上次核对时的代码与配置记录。

| 项目 | 本地实现、官方实现及影响 |
|---|---|
| 1. 三体角与定向消息（核心计算已对齐） | 本地和官方都在带噪周期图的全部邻边中配对。发送边距离为边消息提供径向门控，接收边距离与球谐角基联合投影，再用双线性权重计算接收边的更新。每条边合并自身结果的本向投影与配对边结果的反向投影。[本地构建三元组][ltripselect] · [本地基投影][ltripbasis] · [本地双线性交互][ltripmsg] · [本地反向边索引][lrevedge] · [官方交互与反向边][otripupdate] |
| 2. 键相对晶格轴的方向（已实现） | 两边都对每条有向周期边计算其与当前带噪晶格三个基矢的夹角余弦，把三个有符号的量拼入初始边特征。线性层、ReLU 和第二个线性层将结果投影回边宽度，供第一次读出和全部交互块使用。[本地晶格方向余弦][laxis] · [本地边特征融合][laxisembed] · [官方晶格方向编码][obasis] |
| 3. 主干容量与读出（已对齐三个去噪输出） | 默认 Hydra 配置与官方基础模型共用 4 个交互块、512 维节点和边、128 个径向基和 7 个角基。普通边与节点更新、残差缩放、共享投影及三个去噪头按官方实现对齐。本地包含三个去噪输出，官方模型另含独立的能量分支。[本地训练配置][larchcfg] · [本地交互块][lblock] · [官方 MP-20 配置][ompmodel] · [GemNet 参数][ogemparam] |
| 4. 时间编码与注入 | 本地将标量时间送入可学习 MLP，随后加到每个原子的元素嵌入上。官方对连续时间做多频率正弦/余弦编码，将其与元素嵌入拼接后再线性投影。两边都为每个晶体采一个时间，但去噪器获得的时间特征不同。[本地时间编码][ltime] · [本地相加][linit] · [官方时间编码][otime] · [官方拼接与投影][obasis] |
| 5. 周期近邻图的边集合（已实现） | 本地已采用官方规则。默认在每轴最多 5 层周期镜像中枚举 `0.01 Å < d ≤ 7 Å` 的候选边，每个中心按距离最多取 50 条，包括第 50 条落在等距壳层的情况。随后保留邻居编号小于中心编号的边。同一原子连接自身的周期镜像时，保留平移向量首个非零分量为负的方向。最后对这些边逐条补反向边并取反平移，最终每个中心的邻居数可以超过 50。[本地距离过滤][lcut] · [本地近邻截断][lgraph] · [本地双向化][lmirror] · [官方近零过滤][oshort] · [官方最近邻][otop] · [官方双向化][osym] |
| 6. 元素训练空间 | 本地默认 Hydra 配置与官方基础模型均为 100 种真实元素加 MASK。去噪器的嵌入与最终线性层采用官方 MASK 末列顺序，进入本地联合扩散时将 MASK 移到首列，以原子编号作为 D3PM 类索引。[本地模型配置][larchcfg] · [本地元素头][lclassifier] · [本地列重排][latomorder] · [官方元素空间][obase] |
| 7. 生成时的元素约束 | 官方生成命令会在计算反向后验前屏蔽干净元素 logits，仅开放预选的 76 种元素。有效化学体系条件还会进一步限制集合。中间的噪声状态仍可保留 MASK。本地对所有 logits 直接求后验，若最终预测出 MASK，结构导出会抛错。若预测出白名单外但在 1–118 内的元素，导出则会接受。即使无条件生成，这条差异也成立。[本地后验][latomp] · [本地导出][lexport] · [官方生成命令][oclimask] · [官方屏蔽函数][omask] · [76 种元素][owlist] |
| 8. 晶格反向 predictor 的噪声协方差（已实现） | 本地与官方都用 `randn_like(x_coeff)` 从形状为 `[B, 1, 1]` 的系数张量抽样，每个晶体在每个预测步独立抽一个标准高斯数 g。对称化后的 3×3 噪声矩阵中，三个对角元都是 g，六个非对角元都是 √2·g。采样时将矩阵乘以反向标准差，再加到预测均值上。晶格前向加噪、先验和 Langevin 校正步均从完整的 3×3 高斯矩阵构造对称噪声。[本地采样][lvpp] · [本地逐晶体抽样用例][ltestvp] · [官方系数形状][oalphashape] · [维度扩展][obroadcast] · [反向系数][ocoef] · [官方抽样][orand] · [官方对称化][osymnoise] |
| 9. 生成晶体的原子数先验 | 两边都先抽原子数 N，再生成元素、位置和晶格。官方生成器默认选用内置的 `ALEX_MP_20` 分布，本地从所选训练集统计 N，生成时读取 `statistic.json`。按当前 MP-20 缓存重算，P(N=20) 为 **8.5164%**，官方默认是 **4.3457%**。即使给定 N 的去噪器完全相同，无条件样品的大小分布仍会不同。[本地统计][lstats] · [本地抽样][lsampleN] · [官方生成器默认值][ogendef] · [官方概率表][on] |
| 10. 生成批量会改变校正步长 | 本地命令行默认每批 1 个晶体，官方生成命令默认 64 个。两边的 Langevin corrector 都先算每个晶体的噪声范数和 score 范数，再分别对整批取平均以决定步长。因此单样本生成与批量生成也会得到不同的自适应步长。两边的批大小都可指定。[本地生成参数][lgenCLI] · [本地校正器][lcorr] · [官方生成参数][ogcli] · [官方位置校正器][ocorr] · [官方晶格校正器][olcorr] |

**第 5 条修改前的对照记录。** 当时用本地 MP-20 的 `mp-1078776`（InBiS₃，原胞 10 个原子）按两套规则独立枚举。默认截断得到 632 条候选有向边，官方规则逐中心截取后剩 500 条，双向化后为 **496** 条。旧版本地规则保留整个等距壳层并取两端所选边的并集，得到 **504** 条。该例第 50 与第 51 邻居的最小距离差约 0.001 Å。本次修改仅做静态检查，新版在该晶体上的实际边数尚未复测。[缓存记录][lexample] · [本地双向化][lmirror] · [官方截断][otop] · [官方双向化][osym]

**最终 MASK 的影响。** 对最后一步仍为 MASK 的位置，忽略数值平滑项，吸收态后验给“继续是 MASK”的权重为模型的干净 MASK 概率，给真实元素 z 的权重为模型的干净元素概率乘 1/1000。即使干净 MASK 的预测概率很小，它也可能成为该位置最后的后验众数。本地导出使用最后的众数，官方生成入口通过屏蔽干净 MASK 的 logit 消除了这条实际失败路径。[本地后验公式][ld3] · [本地生成入口][lgen] · [官方屏蔽函数][omask] · [官方 D3PM predictor][oposterior]

### 第 1–3 项复核明细

#### 三体消息和键与晶格的夹角

1. **三元组覆盖已对齐。** 两边都对同一中心的全部周期邻边做有序配对，按边 ID 排除与自身配对，保留同一原子的不同周期镜像。最终周期图上有 `d` 条邻边的中心会产生 `d(d-1)` 个三元组。以 50 条邻边为例，两边都是 50 × 49 = 2,450 个。[本地配对][ltripselect] · [本地预测入口][lpred] · [官方全部配对][otrip]
2. **距离与角度的结合已对齐。** 发送边的径向基先投影到 16 维，再门控边特征并降到 64 维。接收边的径向基和归一化的实球谐角基（m = 0）构成 16 维圆形基。接收边聚合邻边时，双线性权重为圆形基的 16 个通道和发送消息的 64 个通道学习交叉项，输出 64 维再投影到边宽度。两套基投影在所有交互块间共享。本地先按接收边对每项球谐函数的加权消息求和，与官方的填充矩阵乘法计算相同的收缩。[本地角基][langbasis] · [共享基投影][ltripbasis] · [本地双线性计算][ltripmsg] · [官方角基][ocircular] · [官方双线性计算][obilcompute]
3. **正反向边融合已对齐。** 每条边的双线性结果经两套独立的线性层和 ScaledSiLU 投影。边 `e` 取自身的本向投影和 `id_swap[e]` 对应的反向投影，相加后乘 1/√2。本地根据晶体内两个半段的对应位置构造 `id_swap`，可区分原子编号相同的不同周期镜像。中心没有三元组的边仍可接收配对边的消息。[本地反向边索引][lrevedge] · [本地双向投影][ltripexchange] · [官方反向边交换][otripupdate]
4. **键与晶格三个基矢的夹角已对齐。** 每条边使用当前带噪晶格的三行基矢，按从邻居镜像指向中心的方向计算 `edge_lattice_cosines`，形状为 `[E, 3]`。反向边的三个余弦换号。512 维初始边特征拼成 515 维，再经 `515 → 512 → 512` 投影进入第一次读出与全部交互。[本地余弦计算][laxis] · [本地注入位置][laxisembed] · [官方方向投影][oaxismodule] · [官方注入位置][obasis]

#### 主干结构和三个读出头

| 部位 | 当前本地与官方的具体差别 |
|---|---|
| 交互块 | 初始边用无偏置的单层投影处理发送端、接收端与距离。每个块将旧边投影和三元组消息相加，主跳连前 1 个、后 2 个残差块。节点更新以共享的 16 维径向特征门控入边求和，再用一层 Dense 和 3 个残差块。节点反馈由边嵌入和 1 个残差块完成。Dense 使用 He 正交初始化与 ScaledSiLU，残差和分支合并采用 `1/√2`。额外分支缩放取 MatterGen 发布的标定值，缺省值为 1。[本地初始边][linit] · [本地块][lblock] · [官方主干更新][oint] · [基础层][obaselayers] · [官方标定值][oscalefile] |
| 坐标读出 | 初始边及每个交互块后采用一层 Dense 和 3 个残差块。各读出共用 `128 → 16` 径向投影，每次做独立的 `16 → 512` 门控，并在输出标量前应用标定系数。边系数乘从邻居镜像指向中心的单位向量，按中心原子求和并转回分数坐标。[本地读出层][lread] · [本地向量聚合][lhead] · [官方力头][oforcehead] · [官方坐标求和][oforce] |
| 晶格读出 | 使用两层边变换与另一套跨阶段共享的径向投影。各阶段标量之和乘单位边向量的外积，再按晶体边数归一化。[本地晶格读出][lhead] · [本地径向门控][lread] · [官方晶格读出][olatticehead] · [官方外积][olscore] |
| 元素读出 | 最终节点特征上的 `512 → 101` 线性层按官方顺序输出。三个去噪任务从主干取坐标、晶格和元素结果。[本地元素头][lclassifier] · [官方元素头][oatomhead] · [官方三项去噪输出][odenoiseout] |

**容量的静态计数。** 下表按默认无条件训练配置中的层尺寸统计可训练权重和偏置。共享参数计一次，缓冲区和冻结的缩放因子不计。能量分支单列，以便比较实际参与三个去噪输出计算的参数。[本地网络][lmodel] · [本地交互与读出][lread] · [官方网络组装][oshared] · [官方输出块][oforcehead]

| 参数范围 | 本地 | 官方基础模型 |
|---|---:|---:|
| 输入嵌入、初始边与共享投影 | 1,715,712 | 1,715,712 |
| 4 个交互块 | 21,692,416 | 21,692,416 |
| 5 组坐标与晶格读出 | 11,883,520 | 11,883,520 |
| 元素头 | 51,813 | 51,813 |
| 三个去噪输出合计 | **35,343,461** | **35,343,461** |
| 额外能量分支 | 0 | 9,218,560 |
| 去噪器全部可训练参数 | **35,343,461** | **44,562,021** |

**检查点的模型结构。** 检查点保存去噪器的类别顺序与完整权重，包括作为缓冲区的分支缩放值。旧模型的交互块与两层分类头权重结构不同，生成加载器会提示使用训练时的版本。[本地保存字段][larchsave] · [本地生成器重建][larchload]

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
| `gnn.py` | 包含高斯径向基、三体双线性交互、GemNet-T 残差块和两个几何读出层。[本地径向基][lrbf] · [本地角消息][ltripmsg] · [本地交互块][lblock] · [官方基与交互][obasis] |
| `denoiser.py` | 坐标头将边标量乘单位向量后按原子求和，晶格头对单位向量外积加权并按晶体边数归一化，元素头对最终节点表示做线性投影。[本地三个头][lhead] · [官方坐标输出][oforce] · [官方晶格归一化][olread] · [官方外积][olscore] |
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
[lblock]: <E:/THU/AI2DFT/Diffusion-Material-Generation/diffusion_mattergen/src/gnn.py:293>
[ltripselect]: <E:/THU/AI2DFT/Diffusion-Material-Generation/diffusion_mattergen/src/geometry.py:264>
[lrevedge]: <E:/THU/AI2DFT/Diffusion-Material-Generation/diffusion_mattergen/src/geometry.py:255>
[ltripmsg]: <E:/THU/AI2DFT/Diffusion-Material-Generation/diffusion_mattergen/src/gnn.py:227>
[ltripexchange]: <E:/THU/AI2DFT/Diffusion-Material-Generation/diffusion_mattergen/src/gnn.py:250>
[ltripbasis]: <E:/THU/AI2DFT/Diffusion-Material-Generation/diffusion_mattergen/src/gnn.py:174>
[langbasis]: <E:/THU/AI2DFT/Diffusion-Material-Generation/diffusion_mattergen/src/gnn.py:157>
[ledgevec]: <E:/THU/AI2DFT/Diffusion-Material-Generation/diffusion_mattergen/src/geometry.py:240>
[larchcfg]: <E:/THU/AI2DFT/Diffusion-Material-Generation/diffusion_mattergen/scripts/conf/model/joint_diffusion.yaml:4>
[lclassifier]: <E:/THU/AI2DFT/Diffusion-Material-Generation/diffusion_mattergen/src/denoiser.py:66>
[latomorder]: <E:/THU/AI2DFT/Diffusion-Material-Generation/diffusion_mattergen/src/diffusion.py:225>
[oscalefile]: https://github.com/microsoft/mattergen/blob/92423660a8bd70e83679086e88f88596d484dc16/mattergen/common/gemnet/gemnet-dT.json
[larchsave]: <E:/THU/AI2DFT/Diffusion-Material-Generation/diffusion_mattergen/scripts/train.py:150>
[larchload]: <E:/THU/AI2DFT/Diffusion-Material-Generation/diffusion_mattergen/scripts/generate.py:40>
[ompmodel]: https://github.com/microsoft/mattergen/blob/92423660a8bd70e83679086e88f88596d484dc16/checkpoints/mp_20_base/config.yaml#L103-L125
[ocircular]: https://github.com/microsoft/mattergen/blob/92423660a8bd70e83679086e88f88596d484dc16/mattergen/common/gemnet/layers/spherical_basis.py#L48-L82
[obilcompute]: https://github.com/microsoft/mattergen/blob/92423660a8bd70e83679086e88f88596d484dc16/mattergen/common/gemnet/layers/efficient.py#L125-L176
[otripupdate]: https://github.com/microsoft/mattergen/blob/92423660a8bd70e83679086e88f88596d484dc16/mattergen/common/gemnet/layers/interaction_block.py#L314-L344
[oaxismodule]: https://github.com/microsoft/mattergen/blob/92423660a8bd70e83679086e88f88596d484dc16/mattergen/common/gemnet/gemnet.py#L250-L256
[oshared]: https://github.com/microsoft/mattergen/blob/92423660a8bd70e83679086e88f88596d484dc16/mattergen/common/gemnet/gemnet.py#L282-L374
[obaselayers]: https://github.com/microsoft/mattergen/blob/92423660a8bd70e83679086e88f88596d484dc16/mattergen/common/gemnet/layers/base_layers.py#L31-L72
[ovector]: https://github.com/microsoft/mattergen/blob/92423660a8bd70e83679086e88f88596d484dc16/mattergen/common/gemnet/gemnet.py#L565-L589
[odistancevec]: https://github.com/microsoft/mattergen/blob/92423660a8bd70e83679086e88f88596d484dc16/mattergen/common/utils/data_utils.py#L199-L223
[oforcehead]: https://github.com/microsoft/mattergen/blob/92423660a8bd70e83679086e88f88596d484dc16/mattergen/common/gemnet/layers/atom_update_block.py#L139-L198
[olatticehead]: https://github.com/microsoft/mattergen/blob/92423660a8bd70e83679086e88f88596d484dc16/mattergen/common/gemnet/gemnet.py#L68-L140
[odenoiseout]: https://github.com/microsoft/mattergen/blob/92423660a8bd70e83679086e88f88596d484dc16/mattergen/denoiser.py#L248-L271
[otrip]: https://github.com/microsoft/mattergen/blob/92423660a8bd70e83679086e88f88596d484dc16/mattergen/common/gemnet/gemnet.py#L383-L428
[obasis]: https://github.com/microsoft/mattergen/blob/92423660a8bd70e83679086e88f88596d484dc16/mattergen/common/gemnet/gemnet.py#L687-L715
[laxis]: <E:/THU/AI2DFT/Diffusion-Material-Generation/diffusion_mattergen/src/diffusion.py:210>
[laxisembed]: <E:/THU/AI2DFT/Diffusion-Material-Generation/diffusion_mattergen/src/denoiser.py:91>
[linit]: <E:/THU/AI2DFT/Diffusion-Material-Generation/diffusion_mattergen/src/gnn.py:64>
[lpred]: <E:/THU/AI2DFT/Diffusion-Material-Generation/diffusion_mattergen/src/diffusion.py:199>
[lmodel]: <E:/THU/AI2DFT/Diffusion-Material-Generation/diffusion_mattergen/src/denoiser.py:23>
[lread]: <E:/THU/AI2DFT/Diffusion-Material-Generation/diffusion_mattergen/src/gnn.py:115>
[ogemparam]: https://github.com/microsoft/mattergen/blob/92423660a8bd70e83679086e88f88596d484dc16/mattergen/common/gemnet/gemnet.py#L201-L228
[oint]: https://github.com/microsoft/mattergen/blob/92423660a8bd70e83679086e88f88596d484dc16/mattergen/common/gemnet/layers/interaction_block.py#L165-L211
[oatomhead]: https://github.com/microsoft/mattergen/blob/92423660a8bd70e83679086e88f88596d484dc16/mattergen/denoiser.py#L196-L210
[ltime]: <E:/THU/AI2DFT/Diffusion-Material-Generation/diffusion_mattergen/src/denoiser.py:10>
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
[lsampleN]: <E:/THU/AI2DFT/Diffusion-Material-Generation/diffusion_mattergen/src/sampling.py:18>
[lgenCLI]: <E:/THU/AI2DFT/Diffusion-Material-Generation/diffusion_mattergen/scripts/generate.py:114>
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
[lcond]: <E:/THU/AI2DFT/Diffusion-Material-Generation/diffusion_mattergen/src/diffusion.py:160>
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
[lq]: <E:/THU/AI2DFT/Diffusion-Material-Generation/diffusion_mattergen/src/diffusion.py:130>
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
[lrbf]: <E:/THU/AI2DFT/Diffusion-Material-Generation/diffusion_mattergen/src/gnn.py:76>
[lhead]: <E:/THU/AI2DFT/Diffusion-Material-Generation/diffusion_mattergen/src/denoiser.py:125>
[oforce]: https://github.com/microsoft/mattergen/blob/92423660a8bd70e83679086e88f88596d484dc16/mattergen/common/gemnet/gemnet.py#L748-L791
[olread]: https://github.com/microsoft/mattergen/blob/92423660a8bd70e83679086e88f88596d484dc16/mattergen/common/gemnet/gemnet.py#L125-L139
[olscore]: https://github.com/microsoft/mattergen/blob/92423660a8bd70e83679086e88f88596d484dc16/mattergen/common/utils/lattice_score.py#L28-L37
[ltrainloss]: <E:/THU/AI2DFT/Diffusion-Material-Generation/diffusion_mattergen/src/diffusion.py:252>
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
