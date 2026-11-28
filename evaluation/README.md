# 晶体结构的唯一性 U 与新颖性 N

给定一批生成晶体，这个脚本统计其中有多少种不同结构（U），以及有多少个样本在指定参考集中找不到匹配（N）。默认使用 MatterGen 的 `DefaultDisorderedStructureMatcher`。只下载 `evaluation` 文件夹即可运行，计算使用 CPU。

## 跑通第一个例子

进入下载的 `evaluation` 目录，在 Python 3.10 的独立环境中执行以下两条命令。

```bash
python -m pip install -r requirements.txt
python evaluate_u_n.py --dataset demo=examples/candidates.extxyz --reference structures --reference-path examples/reference.extxyz --output results/demo
```

示例有三个候选结构。`salt` 在参考集中有匹配，`silicon` 是新结构，`silicon_shifted` 只是把 `silicon` 的原点平移了。打开 `results/demo/per_structure.csv` 可以逐个核对。

| 候选结构 | 计入 U | 计入 N | 同时计入 U 和 N |
| --- | --- | --- | --- |
| `salt` | 是 | 否 | 否 |
| `silicon` | 是 | 是 | 是 |
| `silicon_shifted` | 否 | 是 | 否 |

`results/demo/summary.json` 中应得到 **U = 2/3，N = 2/3，U∩N = 1/3**。示例参考集只有一个晶体，适合检查安装和理解指标。

## 换成自己的结构

`--dataset` 写成 `标签=路径`，路径支持单个 CIF、含 `*.cif` 的目录（按文件名自然排序），或带三维晶胞的多帧 `.extxyz` / `.xyz` 文件（按帧顺序）。可以把候选结构和参考结构分别放进 `generated/` 和 `known/`，然后运行：

```bash
python evaluate_u_n.py --dataset run1=generated/ --reference structures --reference-path known/ --output results/run1
```

如果手头只有候选结构，可以先算 U。下面的命令分别统计弛豫前、后的两批晶体，每一批独立去重。加 `--limit 10` 可以只试跑每批的前十个候选结构。

```bash
python evaluate_u_n.py --dataset pre=generated_crystals.extxyz --dataset relaxed=relaxed.extxyz --reference none --output results/u_only
```

## 选择新颖性的参考集

**MP-20 训练集**适合回答“生成结果里有多少结构没有在训练集中出现”。如果下载了整个 `ljc` 分支，从 `evaluation` 目录可以直接使用仓库里的缓存：

```bash
python evaluate_u_n.py --dataset run1=generated/ --reference mp20-cache --mp20-cache ../diffusion_mattergen/data_cache/mp_20 --split train --output results/mp20_train
```

单独下载本文件夹时，将 `--mp20-cache` 改为你自己的缓存路径。其下应有相应的 `train/`、`val/` 或 `test/` 子目录，每个 split 包含 `atomic_numbers.npy`、`pos.npy`（分数坐标）、`cell.npy`、`num_atoms.npy` 和 `structure_id.npy`。`--split all` 合并三个 split。默认参考模式为 `mp20-cache`，默认 split 为 `train`。

**MatterGen 官方 Alex-MP 参考集**适合评估相对于 Materials Project 和 Alexandria 的 N。首次运行从 Hugging Face 下载约 873 MB 的压缩库，后续使用缓存。请为临时解压文件预留数 GB 磁盘空间。

```bash
python -m pip install -r requirements-official.txt
python evaluate_u_n.py --dataset run1=generated/ --reference official-mp2020 --output results/alex_mp
```

离线使用时，在第二条命令后加 `--reference-gz /path/to/reference_MP2020correction.gz`。该参数读取含 pickle 数据的官方 LMDB 压缩包，请使用可信来源的实际数据文件。研究结果中请注明所用参考集和 split。[官方参考集](https://huggingface.co/microsoft/mattergen/tree/main/data-release/alex-mp)

## 读懂输出

- `summary.json` 给出每一批的结构数、计数和比例。`unique_fraction`、`novel_fraction`、`novel_unique_fraction` 的分母都是该批实际参与计算的全部候选数。使用 `--reference none` 时，新颖性相关字段为 `null`。
- `per_structure.csv` 每行一个候选。`duplicate_of` 指向该批中最先保留且匹配的结构 ID，`reference_matches` 列出匹配的参考结构 ID。extxyz 优先使用帧内 `structure_id`，CIF 使用文件名（去掉扩展名）。
- 脚本沿用原评估程序的口径，只在**约化化学式完全相同**的结构间调用匹配器。U 按输入顺序保留代表，N 会检查每个候选，包括重复样本。输入可以是弛豫前或弛豫后的结构；评估本身只比较几何结构。`--ordered-matcher` 可切换到官方有序匹配器。

`structure_matcher.py` 包含 U/N 用到的 MatterGen 官方匹配器和辅助函数，微软的 MIT 许可随目录提供。[MatterGen 源码](https://github.com/microsoft/mattergen/blob/main/mattergen/evaluation/utils/structure_matcher.py) · [许可证](LICENSE-MatterGen)
