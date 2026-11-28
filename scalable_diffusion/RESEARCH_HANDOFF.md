# 大晶胞扩散生成研究交接文档

整理日期：2026-11-28，Asia/Shanghai。

本文供其他 Codex 接续本研究，保留本次对话的研究目标、用户约束、文献调研结果、证据边界和待研究问题。它是语义交接摘要，不是逐字聊天记录。文献内容来自此前对话中已执行的检索及三个 Subagent 的原文核查；本次文件整理没有重新检索，也没有开展实验。

**核心目标：仅以 MP20 小晶胞训练生成器，外推生成弛豫后最小平移晶胞仍含 50–100 个原子的材料，保持较高 SUN，并进一步考虑有物理意义的缺陷结构。**

**当前阶段：已完成初步问题分析及简要文献调研，尚未选定模型方案、确认创新点或启动实现。**本轮核查的论文没有充分验证上述全部条件的组合；这仅描述已核查范围，不是穷尽文献后的优先权结论。

## 用户意图和工作约束

用户要求 Codex 担任 Research Director，协助推进科研。研究对象是无机晶体材料的扩散生成，关注小晶胞训练到大晶胞生成的尺寸外推。

用户提出的要求：

- 在 MP20 上训练后，能够生成原子数达到 50–100 的晶胞，并保持 SUN rate。用户没有指定必须达到的数值阈值。
- 生成的大体系不能只是小原胞的平庸平移复制，必须具有实际物理意义和不可约性。
- 大体系中可进一步考虑缺陷等结构；缺陷尚未被指定为第一阶段的必做任务。
- 初始阶段先理解主题、总结已有方法、分析优点和局限、提取关键科学问题，不展开大量创新点。
- 第二阶段只进行简要文献调研，但应分析方法，而不止摘录摘要。
- 每篇重要论文提取研究问题、核心方法、理论基础、关键创新、实验结果和局限性。
- 尽可能追溯原始论文，给 DOI 或可核查链接；禁止编造论文、作者、结果或引用。
- 明确区分文献观察、作者解释、助手推断和待验证问题。

第二阶段用户明确要求创建三个独立 Subagent，使用 GPT-6.1 Sol，并尽可能并行执行。该要求已落实。它是此次调研的模型与执行要求，不能自动理解为今后所有任务都必须使用该模型或继续创建 Agent。

当前用户的最新请求是把对话整理成文件，供其他 Codex 理解和接续。

## 研究对象的准确界定

### 数据和尺寸

MP20 包含 1–20 个原子的晶胞，包括 20；不是严格小于 20。它是常用基准，而非所有晶体生成工作的唯一训练数据。[CDVAE][R1]

以下四种能力必须区分：

1. 程序支持输入更多原子。
2. 生成已知小原胞的超胞。
3. 用大晶胞数据训练后生成大晶胞。
4. 用 MP20 训练后生成稳定、非平凡的大原胞。

本研究主要追求第 4 项。其余能力可以作为诊断或对照，不能直接充当目标已经实现的证据。

还应分别记录总原子数、原胞原子数、不等价位点数及结构自由度。少数空间群轨道代表可以展开出很多原子，因此总原子数大并不自动意味着独立结构复杂度高。

### 不可约性

本对话中将不可约化理解为：不能通过纯平移约化成更小晶胞。它不是要求所有原子都在空间群下不等价，也不是要求结构必须低对称。空间群展开与平移复制不能混为一谈。[Spglib][R14]

需要关注弛豫后的最小平移晶胞，并检查对称识别容差敏感性。小原胞超胞叠加微小随机畸变可能在严格容差下被误识别为大原胞；过度对称化也可能抹去真实的占据有序、重构或缺陷。

已有小宿主经过有意义的占据有序或缺陷重构，可能成为非平凡大周期结构。是否把这种结构与全新大原胞体相作为同一研究目标，尚未由用户定案。

### 训练信息边界

应清楚区分：生成器仅用 MP20、MP20 派生超胞增强、大晶胞微调、宿主模板条件，以及使用外部基础势函数引导采样或弛豫。

用户尚未禁止超胞增强、外部势函数或后处理。不能擅自把“生成器仅用 MP20 训练”扩大成“全流程不能接触任何其他物理信息”。但如果使用外部信息，应报告其作用，避免将辅助流程的收益全部归因于纯生成器外推。

## 用户的观察及其证据状态

### 周期镜像与噪声相关性

用户认为，小胞中的原子及其周期镜像更接近，而且加噪和去噪时镜像会随原子一起运动，因此存在可利用的确定性。大胞中镜像更远，模型需要从其他原子获得信息。

已有直接相关研究是 Kim 等提出的 curse of periodicity：所测模型超出训练尺寸后性能下降，作者将其与周期边界造成的加噪统计变化联系起来。[Kim 等][R4]

助手对机制的形式化分析如下，属于基于给定假设的推导，不是实验结论：在固定晶格、每个显式原子独立加噪时，

```text
先复制成超胞，再对每个显式原子独立加噪
与
先对原胞加噪，再复制其周期镜像
通常不具有相同分布。
```

前者的复制位点可以获得独立噪声，后者对应镜像共享噪声。这说明同一干净物理晶体的不同表示可能产生不同的加噪中间态。

需要保留的限定：

- 独立噪声不等于原子之间没有结构相关性，不能直接沿用“其他原子毫无关联”的字面表述。
- 镜像共享位移并不直接给出正确去噪方向，其帮助程度需要隔离验证。
- 原子数不能单独替代实际空间尺度，密度、晶胞形状、最短晶格平移长度也会影响镜像距离。
- Kim 等的理论分析并没有证明所有扩散架构必然无法尺寸外推。

### 局域合理与整体失序

用户引用 CHGGen，关注局域键长、键角或配位正确，但更大尺度排列错误的问题。

原文支持的较窄结论是：生成结构的 Li–S、P–S RDF 主要峰和局部配位统计与数据相近，但整体呈非晶式排列。它不能直接证明“每个原胞均完整正确，仅原胞之间配准错误”。论文的主要证据见 §3.1 / Fig. 2。[CHGGen][R8]

作者把有限感受野和反向 SDE 随机性列为解释假说，使用了 hypothesize；没有通过独立因果消融确认两者各自贡献。

### 两个必须保留的纠正

- DiffCSP 原版使用全连接图，不能把其失效统一解释为固定空间截断带来的有限感受野。全连接本身也不证明模型已经学到正确的全局相关。[DiffCSP §4.3][R2]
- MatterGen 已进行尺寸相关的噪声缩放和密度相关晶格先验设计，不能把“完全未处理物理噪声尺度”作为已有方法的共同缺点。[MatterGen 方法 A.6–A.7][R3a]

## 并行调研记录

三个 Agent 均使用 GPT-6.1 Sol，独立获取任务上下文并并行研究，均已完成。

| Agent | 文献方向 | 主要论文 |
| --- | --- | --- |
| `/root/periodic_size_mechanisms` | 周期扩散与尺寸外推机制 | DiffCSP、MatterGen、Kim 等尺寸外推研究 |
| `/root/symmetry_large_cells` | 显式对称性与大晶胞生成 | DiffCSP++、SymmCD、GenLCS |
| `/root/inpainting_defects_disorder` | 宿主补全与无序复杂体系 | CHGGen、Dismai-Bench、DMFlow |

这些 Agent 名称用于保存对话来源；在新 Codex 会话中不保证仍能调用，不应依赖其运行状态。

## 九篇核心论文的方法记录

下列结果按照原论文的任务、条件和评价口径保留。CSP 匹配率、SUN 和对称成功率不能直接作为同一种成功率比较。

### DiffCSP

来源：Jiao 等，Crystal Structure Prediction by Joint Equivariant Diffusion，NeurIPS 2023。[原文][R2]

- **研究问题：**给定组成，预测稳定的周期晶体结构。
- **核心方法：**晶格采用 DDPM，分数坐标采用 wrapped-normal 分数匹配，联合去噪；网络使用晶格内积和相对分数坐标 Fourier 特征。
- **理论基础：**周期坐标空间、旋转与周期平移对称性。理论命题证明的是相应生成分布的对称性，不是跨尺寸泛化。
- **关键创新：**联合生成晶格和坐标，显式处理周期几何。
- **实验结果：**MP20 上每个组成生成 20 个候选时，CSP 匹配率 77.93%。单候选匹配率 51.49%，后者也是 DiffCSP++ 同表基线。
- **局限性：**§4.3 明确使用全连接图；原版无条件生成按训练集原子数分布采样。晶格高噪声端采用尺寸无关的标准高斯，未建立原胞与超胞的表示一致性或大原胞外推保证。MPTS-52 上训练测试不等于 MP20 外推。

### MatterGen

来源：Zeni 等，A generative model for inorganic materials design，Nature 2025。DOI：10.1038/s41586-025-08628-5。[正式论文][R3]，[方法版本][R3a]

- **研究问题：**联合生成组成和结构，并按目标性质控制生成。
- **核心方法：**元素掩码扩散、周期坐标扩散和晶格扩散联合建模；GemNet 去噪，适配器微调及条件引导。
- **理论基础：**离散与连续扩散、周期坐标几何、等变网络；用物理尺度和密度组织噪声先验。
- **关键创新：**方法 A.6 中分数坐标噪声标准差按 N 的负三分之一次方缩放；A.7 中晶格先验尺度随 N 的三分之一次方变化，在近似定密度下协调尺度。
- **实验结果：**正式论文 Fig. 2 以 DFT 弛豫评价 SUN，报告相对早期基线的改善；扩大到 Alex-MP20 进一步改善。当前交接不保存跨版本可能变化的具体 SUN 数字。
- **局限性：**Alex-MP20 仍限制最多 20 个晶胞原子。尺度归一化并不等于消除了镜像相关性或学会了大胞整体排列。GemNet 有空间截断，但晶格分支含全局聚合，不能将整个模型概括为纯局域。

### Kim 等的尺寸外推研究

来源：Are diffusion models ready for materials discovery in unexplored chemical space?，Patterns 2026；预印本首发于 2025。DOI：10.1016/j.patter.2026.101537。[正式论文][R4]，[arXiv v3][R4a]

- **研究问题：**扩散模型在欠采样化学空间及训练尺寸之外是否可靠。
- **核心方法：**比较 MatterGen、DiffCSP，并在尺寸测试中加入 EquiCSP；分析能量、弛豫位移和固定组成案例。
- **理论基础：**用简化分布假设分析噪声与数据分布的差异，推导 KL 散度的尺寸依赖；将周期边界相关问题称为 curse of periodicity。
- **关键贡献：**提供直接尺寸外推反例，并尝试区分化学覆盖、晶格表示和周期噪声统计的作用。
- **实验结果：**MatterGen、EquiCSP 的能量与弛豫位移随原子数增大恶化；MPTS-52 训练的 DiffCSP 在测试的 28 原子电解质案例成功，在 76、84、96 原子案例未复现基态。arXiv v3 对应 Fig. 5，正式发表版本的图号可能不同。
- **局限性：**上述结论适用于所测模型与材料；KL 随维度增长不是泛化误差下界，也不是所有扩散模型必然失败的证明。文章对逐原子模型的局域性解释，应结合 DiffCSP 全连接实现谨慎解读。

### DiffCSP++

来源：Jiao 等，Space Group Constrained Crystal Generation，ICLR 2024。[原文 §4–5][R5]

- **研究问题：**如何在生成中显式保持空间群。
- **核心方法：**以晶格极分解和矩阵对数空间限制晶格自由度；将坐标限制在指定 Wyckoff 位置，对同一轨道的去噪结果投影与协调。
- **理论基础：**晶格极分解、空间群轨道和 Wyckoff 子空间约束。
- **关键创新：**将抽象空间群要求转化为扩散过程可执行的参数约束。
- **实验结果：**§5.2 表2的 MP20 单次采样匹配率为 70.58%（CSPML 模板条件）、80.27%（真实对称条件）；同表 DiffCSP 为 51.49%。三个数字都是单候选口径。
- **局限性：**需要空间群和 Wyckoff 条件；真实条件版本使用未知结构发现时未必可得的信息。模板检索限制可探索类型。MPTS-52 实验重新使用相应数据训练。

### SymmCD

来源：Levy 等，SymmCD: Symmetry-Preserving Crystal Generation with Diffusion Models，ICLR 2025。[原文 §4–5][R6]

- **研究问题：**能否生成对称信息，减少对固定模板的依赖。
- **核心方法：**生成非对称单元的轨道代表、元素、位点对称性及晶格，再投影到相容位置并展开；离散扩散处理元素和对称性，连续扩散处理几何。
- **理论基础：**空间群轨道与稳定子的关系，跨空间群共享的位点对称性表示。
- **关键创新：**联合学习对称信息及结构，允许生成新的对称模板组合。
- **实验结果：**§5.2 表2中，作者 CHGNet 与 MP 凸包评价口径下，全部空间群模型 SUN 为 6.86%，常见十群版本为 7.59%；分母为最初生成的 10,000 个样本。同表 DiffCSP 为 8.92%。这些不是 DFT 验证的 SUN。
- **局限性：**正文未直接明确对应能量阈值，不能擅自写成 0.1 eV/atom 或 0 eV/atom。空间群及轨道数仍按数据分布采样，使用惯用胞表示；没有展示 MP20 到目标大原胞的受控外推。主要优势不能概括为 SUN 全面领先。

### GenLCS

来源：Kim 等，Generative model for large-cell crystal structure，Research Square v1，2025。DOI：10.21203/rs.3.rs-7472776/v1。[论文入口][R7]，[本轮读取的原始 PDF][R7a]

- **研究问题：**如何生成含较多原子的对称晶体。
- **核心方法：**先以 VQ 离散扩散生成空间群、晶格、basis 元素和重数，再用 VE-SDE 与等变 GNN 生成 basis 坐标；对称展开后进行保持重数的采样，最后合并等价原子。
- **理论基础：**空间群对称展开、离散扩散及连续分数匹配。basis 在这里指空间群轨道代表，不是原胞中的全部原子。
- **关键创新：**把结构属性和坐标分阶段生成，在较少独立位点上联系远距离对称关系。
- **实验结果：**MOA-over20 包含 127,225 个结构，晶胞原子数 20–120，basis 原子数少于 20，来源为 MP、OQMD、Alexandria。给定空间群、晶格、basis 元素及重数的坐标测试中，160/126/126 原子参考结构各 50 次尝试，分别匹配 6/8/21 次；264 原子 Ca3Al2O6 案例匹配 1/50。
- **局限性：**使用大晶胞训练数据；上述大尺寸坐标测试包含较强结构条件。160 原子 Ia-3d、126 原子 R3c/R-3c 案例涉及中心化惯用胞，不能直接当成同样大小的原胞。原文未系统报告弛豫后原胞约化检验。第二阶段难以纠正第一阶段不合理的组成和晶格。

核查边界：本轮确认的是 Research Square v1；网页访问受限，但 Agent 读到了原始资产 PDF。264 原子 1/50 是正文对 Supplementary Fig. 7 的报告，未独立取得该补图。论文所列代码地址 `https://github.com/sunilegna/GenLCS` 在当次访问中返回 404；不应将这一临时状态写成永久不可用。

### CHGGen

来源：Zhong 等，Crystal structure prediction with host-guided inpainting generation and foundation potentials，Materials Horizons 2025。DOI：10.1039/D5MH00774G。[正式论文][R8]，[arXiv 原文][R8a]

- **研究问题：**局域成键合理但整体失序时，如何改善晶体生成，并完成宿主内的客体放置。
- **核心方法：**无条件 NequIP 扩散、宿主对称化、RePaint 补全与 CHGNet 弛豫组成流程。每步由原始宿主加入相应噪声后覆盖宿主部分，客体部分保留生成结果，通过重采样协调二者。
- **理论基础：**去噪分数匹配、VE-SDE、带 mask 的条件补全与重采样。
- **关键创新：**将经过对称化的框架作为整体结构条件，支持局部结构修改。
- **实验结果：**Fig. 2 展示局域 RDF、配位统计合理而整体失序；Zn 系无条件对称成功率低于 5%，补全提高该成功率。两个代表候选 DFT 分解能为 15、46 meV/atom，属于亚稳结果。Agent 核查的方法文本给出训练集为 109,805 个 MP 结构，筛选 E_hull <0.1 eV，8:2 划分，并非 MP20；训练尺寸上限未明确。
- **局限性：**依赖可对称化的宿主，尚不能证明从小晶胞数据自由生成新的大骨架。局域偏置与 SDE 随机性导致失序属于作者假说，未隔离因果。对称成功率不能代替 SUN 或不可约性。

### Dismai-Bench

来源：Yong、Su、Ertekin，Dismai-Bench: benchmarking and designing generative models using disordered materials and interfaces，Digital Discovery 2024。DOI：10.1039/D4DD00100A。[正式论文][R9]，[方法与实验原文][R9a]

- **研究问题：**如何在复杂大体系中判断模型是否学到正确结构分布。
- **核心方法：**在固定组成的合金、非晶和界面数据上分别训练及评价；按任务冻结部分元素或晶格变量，比较图扩散与 U-Net 等方法。
- **理论基础：**原子表示的表达力，以及 RDF、键角、配位、短程序等结构统计。该工作不是尺寸外推理论。
- **关键创新：**以固定体系的可比结构分布评价生成质量，并揭示有效率等简单指标遗漏的问题。
- **实验结果：**六套数据各 1,500 个结构；合金和非晶 Si 为 256 原子，界面为 264 原子。图扩散模型总体优于两类 U-Net 基线，但仍可能有占位重叠、空位等错误。
- **局限性：**各数据集单独训练并生成同尺寸、固定体系结构；不能把其 256 原子结果当成 MP20 外推证据。不同模型还经过任务适配，例如 DiffCSP 关闭晶格扩散。

### DMFlow

来源：Wu 等，DMFlow: Disordered Materials Generation by Flow Matching，2026。[原文 §4–5][R10]

- **研究问题：**如何生成晶体学平均意义下的替位无序和位置无序。
- **核心方法：**以位点元素概率、主次位置与位置概率描述无序，联合生成晶格、坐标和概率；采用加权多位置消息传递及两阶段离散化。
- **理论基础：**用平方根映射将概率单纯形上的 Fisher–Rao 几何转成球面流匹配。
- **关键创新：**统一有序、替位无序和位置无序表示，并在概率几何上生成合法权重。
- **实验结果：**COD-SPD-50 上匹配率 45.87%，同表 FlowMM-Prob 为 44.00%；评价还包括无碰撞、电中性、覆盖率及密度和元素数分布。
- **局限性：**COD-20 和 COD-50 分别构建训练集，非 20→50 外推；未进行能量弛豫、凸包稳定性或 SUN 验证。输出概率占据及 multi-hot 元素集合，尚未确定具体显式缺陷构型、跨位点占据关联和缺陷重构。

## 补充线索

以下论文在第一轮中被检索，用于定位相关路线，第二轮未作为九篇重点论文完整展开。

- **CDVAE，ICLR 2022：**VAE 加扩散解码，提供 MP20 等基准，是早期方法与数据来源的重要入口。[R1]
- **FlowMM，ICML 2024：**在适合晶体几何的流形上进行流匹配，可作为研究失败是否依赖具体扩散采样过程的相关对照。[R11]
- **ADiT，ICML 2025：**用 Transformer 自编码器和潜空间扩散统一分子、材料生成；其晶体实验不构成目标尺寸外推证据。[R12]
- **Fourier Transformers，2026 预印本：**以元素分辨密度的傅里叶系数表示晶体；报告最高每元素 108 原子的表示重建，而无条件生成比较主要在最多 16 原子的范围。重建能力不能替代从噪声生成能力。[R13]
- **DynaCrys，2026 预印本：**让空间群、Wyckoff 占据与元素共同演化，采用群与子群关系；已检查的实验主要为 MP20 和 MPTS-52，尚不能直接证明本课题联合目标。[R15]
- **PhononBench，2026：**补充动力学稳定性评价。大规模计算主要依赖 MatterSim，不能描述为全部经过 DFT 声子验证。[R16]
- **Systematic softening，2025：**在所测 M3GNet、CHGNet、MACE-MP-0 等模型和任务中报告非平衡环境、缺陷能等偏差；不能将结果无条件推广到所有基础势函数。[R17]

## 方法发展脉络和路线比较

目前整理的方法脉络是：联合学习周期几何，进一步显式处理对称自由度，再通过宿主条件或概率表示研究复杂结构。与方法发展并行，尺寸外推和复杂体系评测开始检查失效机制。

| 路线 | 整体结构信息来源 | 对本课题的价值 | 未解决的主要问题 |
| --- | --- | --- | --- |
| 联合坐标与晶格生成 | 训练分布与去噪网络 | 自由生成的基础对照 | 噪声中间态统计随尺寸变化，整体排列如何迁移 |
| 空间群与轨道生成 | 群关系和位点条件 | 远距离等价位点的一致性，减少自由度 | 总原子数扩展能否覆盖独立位点和新拓扑扩展 |
| 宿主引导补全 | 已有或精修的宿主框架 | 插入、局部重构和潜在缺陷任务 | 宿主之外的新骨架生成 |
| 无序表示与大体系同尺寸训练 | 概率占据或专门大体系数据 | 无序表示、结构统计和评价方法 | 具体显式构型、能量稳定性及跨尺寸迁移 |

以上比较是研究判断，不是论文间统一条件的性能排名。

## 关键科学问题

| 编号 | 问题 | 需要回答的内容 |
| --- | --- | --- |
| Q1 | 尺寸失效的主导变量 | 原子数、密度、最短晶格平移、形状、噪声幅度和网络信息范围，谁更能解释失败 |
| Q2 | 周期镜像相关性的实际贡献 | 固定干净物理结构，仅改变原胞或超胞表示和噪声相关性，去噪难度如何变化 |
| Q3 | 小胞数据的可辨识性 | 局域环境相近但拓扑或长程排列不同的大结构，是否能由 MP20 信息区分 |
| Q4 | 有序性的生成时间 | 中长程一致性是在高噪声阶段未建立、中间阶段破坏，还是低噪声阶段未修复 |
| Q5 | 外推的复杂度是什么 | 总原子数、原胞原子数、不等价位点数、连接网络和相关长度分别如何变化 |
| Q6 | 不可约性是否有物理支撑 | 弛豫后大周期是否保留，是否只是微小畸变造成的数值不可约 |
| Q7 | 新体相与缺陷的成功标准 | 体相稳定性与缺陷形成代价、浓度及周期镜像作用应如何分别评价 |

主要技术瓶颈包括：

- 多种分布变化同时发生，难以隔离周期表示、噪声、组成和晶格因素。
- 大胞整体结构规律是否包含在小胞数据中，尚未建立可检验的边界。
- 局域合理、对称性正确、整体连接合理和能量稳定分别需要证据。
- 固定高对称约束与缺陷造成的真实对称性破缺需要协调。
- 势函数引导和弛豫可能掩盖生成器自身质量，独立物理验证成本高。

助手建议的优先顺序是 Q2、Q3 和 Q6：先确定周期表示效应，再界定信息可迁移范围，并建立不可约大原胞评价。这只是当前建议，用户尚未批准为具体实验计划或模型方案。

## SUN 和物理评价注意事项

SUN 表示 Stable、Unique、Novel。必须说明稳定阈值、能量方法、参考凸包、去重匹配条件、新颖性参考库和分母。

### 稳定性

MatterGen 使用弛豫后能量不超过参考凸包 0.1 eV/atom 的操作性标准。它允许一定亚稳性，也不包含完整动力学稳定性。[MatterGen][R3]

低凸包能、局部力收敛、无明显碰撞和声子稳定性是不同层面的检查。PhononBench 提供了动力学评价的相关证据，但其规模化结果主要基于机器学习势。[PhononBench][R16]

### 唯一性和新颖性

以下是助手由任务定义得到的分析：若真正原胞超过 20 原子，只与 MP20 比较新颖性几乎会天然计为 novel，不能证明发现了数据库之外的新材料。应使用独立、较完整的参考库，明确数据库版本，并区分新结构实例和新原型。

### 针对本课题的联合成功条件

此前提出的评价草案为：

```text
成功样本数 = 同时满足 S、U、N，且弛豫后原胞有 50–100 个原子的样本数
联合成功率 = 成功样本数 / 全部生成尝试数
```

该定义是助手建议，不是通行标准，也未由用户确认。应另报各项通过率和原胞尺寸分布；不能只在筛选后的大胞子集中计算 SUN 而隐藏总产率。

### 缺陷

不可平移约化不自动证明缺陷有物理意义。周期超胞会带来缺陷镜像相互作用，带电缺陷尤其需要有限尺寸处理。[Freysoldt 等][R18]

助手的量纲分析：若单个局域错误的额外能量近似固定，它对每原子能量的贡献随 1/N 减小。因此，大体系的低平均能量可能掩盖局域缺陷或结构错误，不能仅凭平均能量认定缺陷合理。

## 接续工作与尚未决定事项

当前已完成：

- 理解研究主题，初步区分训练尺寸外推、整体有序与不可约性。
- 总结关键方法、优缺点与科学问题。
- 按用户要求执行三个 GPT-6.1 Sol Subagent 的并行简要调研，并整合九篇论文。
- 保存这份交接文档。

当前未完成，也没有声称完成：

- 没有下载或清洗 MP20、大晶胞测试集。
- 没有复现论文、训练模型、运行生成、弛豫或 DFT。
- 没有选定 backbone、噪声过程、模型创新或正式实验矩阵。
- 没有确定首个目标化学体系、缺陷类型或期望 SUN 阈值。
- 没有确认超胞增强、外部势函数引导或宿主条件在最终主任务中的具体角色。
- 没有形成穷尽性文献综述或研究优先权声明。

如果下一步仍是科研论证，优先围绕上述未决问题开展针对性分析，避免再次无差别扩充大量论文或直接堆砌架构创新。若要把数值用于正式论文或实验对照，应锁定原文版本并复查评价实现，尤其是 SymmCD 的稳定阈值、GenLCS 的条件输入和原胞计数、CHGGen 的训练尺寸范围。

如果下一步是实现，应先检查当时的仓库、AGENTS.md 和用户新指令。交接时项目目录为 `E:\THU\AI2DFT\Diffusion-Material-Generation\scalable_diffusion`，环境为 Windows PowerShell；整理文件前目录未发现其他项目文件，也未发现适用的 AGENTS.md。本次整理仅新增此文档。

## 原始文献与核查入口

| 编号 | 文献 | 可核查出处 |
| --- | --- | --- |
| R1 | Xie 等，Crystal Diffusion Variational Autoencoder for Periodic Material Generation，ICLR 2022 | [arXiv 2110.06197](https://arxiv.org/abs/2110.06197) |
| R2 | Jiao 等，Crystal Structure Prediction by Joint Equivariant Diffusion，NeurIPS 2023 | [原文 v2](https://arxiv.org/html/2309.04475v2) |
| R3 | Zeni 等，A generative model for inorganic materials design，Nature 2025 | [DOI 10.1038/s41586-025-08628-5](https://doi.org/10.1038/s41586-025-08628-5)；[预印本方法 A.6–A.7](https://arxiv.org/html/2312.03687v2) |
| R4 | Kim 等，Are diffusion models ready for materials discovery in unexplored chemical space?，Patterns 2026 | [DOI 10.1016/j.patter.2026.101537](https://doi.org/10.1016/j.patter.2026.101537)；[arXiv v3](https://arxiv.org/pdf/2510.09406v3) |
| R5 | Jiao 等，Space Group Constrained Crystal Generation，ICLR 2024 | [arXiv 2402.03992](https://arxiv.org/html/2402.03992) |
| R6 | Levy 等，SymmCD: Symmetry-Preserving Crystal Generation with Diffusion Models，ICLR 2025 | [arXiv 2502.03638](https://arxiv.org/html/2502.03638) |
| R7 | Kim 等，Generative model for large-cell crystal structure，Research Square v1，2025 | [DOI 10.21203/rs.3.rs-7472776/v1](https://doi.org/10.21203/rs.3.rs-7472776/v1)；[已读 PDF](https://assets-eu.researchsquare.com/files/rs-7472776/v1_covered_fd86466a-8b8b-494f-96a8-980aa034a83d.pdf?c=1763536544) |
| R8 | Zhong 等，Crystal structure prediction with host-guided inpainting generation and foundation potentials，Materials Horizons 2025 | [DOI 10.1039/D5MH00774G](https://doi.org/10.1039/D5MH00774G)；[arXiv](https://arxiv.org/pdf/2504.16893) |
| R9 | Yong 等，Dismai-Bench: benchmarking and designing generative models using disordered materials and interfaces，Digital Discovery 2024 | [DOI 10.1039/D4DD00100A](https://doi.org/10.1039/D4DD00100A)；[arXiv v2](https://arxiv.org/html/2404.06734v2) |
| R10 | Wu 等，DMFlow: Disordered Materials Generation by Flow Matching，2026 | [arXiv v1](https://arxiv.org/html/2602.04734v1) |
| R11 | Miller 等，FlowMM: Generating Materials with Riemannian Flow Matching，ICML 2024 | [PMLR 正式论文](https://proceedings.mlr.press/v235/miller24a.html) |
| R12 | Joshi 等，All-atom Diffusion Transformers: Unified generative modelling of molecules and materials，ICML 2025 | [arXiv 2503.03965](https://arxiv.org/abs/2503.03965) |
| R13 | Duersch 等，Fourier Transformers for Latent Crystallographic Diffusion and Generative Modeling，2026 预印本 | [arXiv 2602.12045](https://arxiv.org/abs/2602.12045) |
| R14 | Togo 等，Spglib: a software library for crystal symmetry search，2024 | [DOI 10.1080/27660400.2024.2384822](https://doi.org/10.1080/27660400.2024.2384822) |
| R15 | Jin 等，DynaCrys: Crystal Generation with Dynamic Space-Group Diffusion，2026 预印本 | [arXiv 2608.07401](https://arxiv.org/abs/2608.07401) |
| R16 | Han 等，PhononBench: A Large-Scale Phonon-Based Benchmark for Dynamical Stability in Crystal Generation，2026 | [arXiv 2512.21227](https://arxiv.org/abs/2512.21227)；[DOI 10.1088/3050-287X/ae9ee4](https://doi.org/10.1088/3050-287X/ae9ee4) |
| R17 | Deng 等，Systematic softening in universal machine learning interatomic potentials，npj Computational Materials 2025 | [DOI 10.1038/s41524-024-01500-6](https://doi.org/10.1038/s41524-024-01500-6) |
| R18 | Freysoldt、Neugebauer、Van de Walle，Fully Ab Initio Finite-Size Corrections for Charged-Defect Supercell Calculations，PRL 2009 | [DOI 10.1103/PhysRevLett.102.016402](https://doi.org/10.1103/PhysRevLett.102.016402) |

[R1]: https://arxiv.org/html/2110.06197v3
[R2]: https://arxiv.org/html/2309.04475v2
[R3]: https://doi.org/10.1038/s41586-025-08628-5
[R3a]: https://arxiv.org/html/2312.03687v2
[R4]: https://doi.org/10.1016/j.patter.2026.101537
[R4a]: https://arxiv.org/pdf/2510.09406v3
[R5]: https://arxiv.org/html/2402.03992
[R6]: https://arxiv.org/html/2502.03638
[R7]: https://doi.org/10.21203/rs.3.rs-7472776/v1
[R7a]: https://assets-eu.researchsquare.com/files/rs-7472776/v1_covered_fd86466a-8b8b-494f-96a8-980aa034a83d.pdf?c=1763536544
[R8]: https://pubs.rsc.org/en/content/articlehtml/2025/mh/d5mh00774g
[R8a]: https://arxiv.org/pdf/2504.16893
[R9]: https://doi.org/10.1039/D4DD00100A
[R9a]: https://arxiv.org/html/2404.06734v2
[R10]: https://arxiv.org/html/2602.04734v1
[R11]: https://proceedings.mlr.press/v235/miller24a.html
[R12]: https://arxiv.org/abs/2503.03965
[R13]: https://arxiv.org/abs/2602.12045
[R14]: https://doi.org/10.1080/27660400.2024.2384822
[R15]: https://arxiv.org/abs/2608.07401
[R16]: https://arxiv.org/abs/2512.21227
[R17]: https://doi.org/10.1038/s41524-024-01500-6
[R18]: https://doi.org/10.1103/PhysRevLett.102.016402
