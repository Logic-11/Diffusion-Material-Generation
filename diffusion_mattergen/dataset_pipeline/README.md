# 晶体数据集处理项目

创建 SQLite 文件和在其他 Python 脚本中读取样本，见[简明使用指南](USAGE.md)。

本项目把一个或多个 CIF 文件转成可直接供 PyTorch 训练读取的晶体图数据集。
代码、测试和示例全部位于 `diffusion_mattergen/dataset_pipeline/`，与现有
`scripts/`、`src/`、`test/`、`data/` 平级。核心包 `crystal_dataset`
不导入这些目录中的代码；只有训练示例会调用现有 `src.denoiser` 和 `src.diffusion`。

流程：`CIF 文件/目录 → 解析、检查、周期邻接图 → SQLite 文件 → Dataset → DataLoader → 模型`。

## 1. 快速开始

### 环境

使用现有 `diffusion` Conda 环境。PowerShell 7 中需要在运行 Python 的同一 shell
内完成初始化和激活；新开的终端或工具调用需要重新执行：

```powershell
. 'D:\Anaconda\anaconda3\shell\condabin\conda-hook.ps1'
conda activate diffusion
if ($env:CONDA_DEFAULT_ENV -ne 'diffusion') {
    throw 'The diffusion conda environment is not active.'
}
if ((Get-Command python).Source -ne 'D:\Anaconda\anaconda3\envs\diffusion\python.exe') {
    throw 'Unexpected Python executable.'
}
```

以下命令均从 **`diffusion_mattergen/dataset_pipeline`** 目录执行。依赖已存在时无需安装本项目：

```powershell
# 如果当前位于 diffusion_mattergen
Set-Location ./dataset_pipeline

# 递归读取 ../data 中的所有 .cif / .CIF，构建半径为 5 Å 的周期图
python -B -m crystal_dataset build ../data ./artifacts/crystals.sqlite --cutoff 5.0

# 查看结构数、原子数、边数、划分数量、参数和解析异常记录
python -B -m crystal_dataset inspect ./artifacts/crystals.sqlite

# 调用现有 Denoiser，完成一个训练 epoch
python -B ./examples/train_mask.py ./artifacts/crystals.sqlite --epochs 1 --batch-size 8 --device cpu
```

当前 `Ba2PrCu3O7.cif` 会生成一个含 13 个原子的样本。只有一个 CIF 分组时全部分到
`train`，`val`、`test` 为空，不会复制同一个结构来凑验证集。
默认不跳过无效文件，也不覆盖已有数据库；重复构建请使用新路径，例如
`./artifacts/crystals_v2.sqlite`。添加 CIF、改变 cutoff 或划分参数后，需要重新构建。

如需从其他目录或其他项目导入，可在激活环境后执行一次可编辑安装：

```powershell
python -B -m pip install -e . --no-deps
```

`pyproject.toml` 声明了 NumPy、pymatgen 和 PyTorch 依赖；`--no-deps` 使用环境中已有的
版本。数据模块和模型均使用 PyTorch 原生算子，无需 `torch_scatter`。
构建参数中记录实际使用的 NumPy/pymatgen 版本，便于追溯解析行为。

## 2. 文件布局

```text
dataset_pipeline/
├── pyproject.toml
├── README.md
├── crystal_dataset/
│   ├── __init__.py       # 对外 API
│   ├── __main__.py       # build / inspect 命令
│   ├── build.py          # CIF 解析、周期图、数据划分
│   ├── storage.py        # SQLite 格式、数值数组编码
│   └── dataset.py        # Dataset 与 collate_crystals
├── examples/
│   └── train_mask.py     # 现有 mask diffusion 模型的最小训练示例
├── tests/
│   └── test_dataset.py   # 几何、存储、批处理、多进程测试
└── artifacts/            # 本地生成的数据库，已被本项目 .gitignore 排除
```

## 3. 多结构导入和数据划分

```powershell
# 单个 CIF 也可以作为输入
python -B -m crystal_dataset build ../data/Ba2PrCu3O7.cif ./artifacts/single.sqlite

# 自定义划分与随机种子
python -B -m crystal_dataset build ../data ./artifacts/v2.sqlite --cutoff 6.0 --split-ratios 0.8 0.1 0.1 --seed 42

# 全部用于训练
python -B -m crystal_dataset build ../data ./artifacts/all_train.sqlite --split-ratios 1 0 0

# 显式允许跳过无效 CIF；原因保存在 metadata.skipped_files
python -B -m crystal_dataset build ../data ./artifacts/valid_only.sqlite --skip-invalid
```

- 递归扫描目录，按相对路径排序；不同子目录下的同名文件不会冲突。
- 一个 CIF 中的多个可解析结构分别生成样本，ID 为 `相对路径::解析结构序号`，序号从 0 开始。
- 使用 pymatgen 展开对称性，显式设置 `primitive=False`，不做原胞约化。
  pymatgen 可能调整原子顺序；图中索引对应解析后的顺序，不对应 CIF 原始行号。
- 只接受有序、完全占位、有效晶格和有限坐标的结构。混合占位、缺位结构会报错，
  不把分数占位强制转换成某一种元素。氧化态不作为模型输入，元素映射为原子序数。
- 默认任意文件失败都会终止本次构建。`--skip-invalid` 按整个 CIF 跳过；同一文件中有
  一个结构无效时，不保留该文件中先前已解析的结构。I/O、数据库和内存错误仍会终止构建。
- 按**原始文件 SHA256** 分组：同一文件内的所有结构、字节内容相同的文件副本均在同一划分。
  副本仍是独立样本，不自动去重。不同文本表达的等价晶体不在本项目的去重范围内，
  正式评估前如有此类数据，需另外按材料来源或结构等价性分组。
- 固定种子打乱排序后的分组，按最大余数法分配整数分组数，保证至少一个训练分组。
  小数据集的验证/测试集可以为空；实际结构比例也会因多结构 CIF 的大小而偏离目标比例。
  相同输入、参数和依赖版本下，划分及数值数组可复现；构建时间等元数据使数据库文件
  本身不保证字节相同。增加或修改输入后重新构建，旧结构的划分也可能改变。

## 4. 存储格式和周期图约定

一个数据集对应一个可搬移的 `.sqlite` 文件，训练时不再依赖原始 CIF 所在目录。
数据库格式版本为 `PRAGMA user_version = 1`，包含：

| 表 | 内容 |
| --- | --- |
| `metadata` | 构建配置、总数、各划分数量、依赖版本、跳过文件及原因 |
| `sources` | CIF 相对路径、SHA256、原始 CIF 文本、解析警告列表 |
| `samples` | 样本 ID、来源、结构序号、化学式、原子/边数量、划分、数值数组 |

每个样本的数组以压缩 NPZ BLOB 保存，读取时 `allow_pickle=False`。存储浮点精度为
`float64`，整数为 `int64`；Dataset 默认转换为训练常用的 `float32`。
`CrystalDataset(path, dtype=torch.float64)` 可保留双精度。

设一个样本含 `N` 个原子、`E` 条有向边：

| 字段 | 单样本形状 | 含义 |
| --- | --- | --- |
| `atomic_numbers` | `[N]` | 元素原子序数 1–118，0 留给训练时的 mask |
| `lattice` | `[3, 3]` | 行向量为 a/b/c 晶格基矢，单位 Å |
| `frac_coords` | `[N, 3]` | 包裹到 `[0, 1)` 的分数坐标 |
| `positions` | `[N, 3]` | `frac_coords @ lattice`，单位 Å |
| `edge_index` | `[2, E]` | 第 0 行中心原子 i，第 1 行邻居 j |
| `edge_images` | `[E, 3]` | 邻居 j 的整数周期平移 S |
| `edge_vectors` | `[E, 3]` | `(frac[j] + S - frac[i]) @ lattice`，单位 Å |
| `edge_distances` | `[E]` | 边向量的模长，单位 Å |

邻接图包含距离 `<= cutoff` 的所有周期镜像，适用于非正交晶胞和大于半晶胞尺寸的
截断半径，不仅保留最小镜像。排除 `i == j 且 S == 0` 的自身边，保留自身原子在其他
晶胞中的合法邻居。因此单原子晶胞也可能有边，同一对原子也可能有多个不同平移的边。
边按中心、邻居、平移排序；无近邻时保留形状正确的空边张量。没有人为限制每个原子的
邻居数，较大 cutoff 会增加存储和训练开销。重合原子或近零体积晶格会被拒绝。

构建逐文件处理，每个样本写入后不累计保存在 Python 列表中；单个多结构 CIF
仍会一次解析出该文件内的结构。构建完成后通过同目录硬链接发布最终数据库，失败不留下
可被误认为完成的目标文件，并拒绝已有目标路径。此发布方式需要支持硬链接的文件系统
（当前 Windows 本地 NTFS 支持）；建议在本地磁盘构建，再复制完整数据库到其他位置。
不要在训练时编辑或替换正在读取的数据库；使用新的版本路径。

可以直接用 SQLite 查询溯源信息，例如：

```sql
SELECT sample_id, formula, num_atoms, num_edges, split FROM samples ORDER BY id;
SELECT source_path, parser_warnings FROM sources WHERE parser_warnings != '[]';
SELECT cif_text FROM sources WHERE source_path = 'Ba2PrCu3O7.cif';
```

## 5. 给训练供数

```python
import torch
from torch.utils.data import DataLoader
from crystal_dataset import CrystalDataset, collate_crystals

def main():
    with CrystalDataset("artifacts/crystals.sqlite", split="train") as dataset:
        loader = DataLoader(
            dataset,
            batch_size=8,
            shuffle=True,
            num_workers=2,
            collate_fn=collate_crystals,
            generator=torch.Generator().manual_seed(42),
        )
        for batch in loader:
            print(batch["atomic_numbers"].shape, batch["lattice"].shape)

if __name__ == "__main__":  # Windows 多进程必须放在此保护之下
    main()
```

Dataset 初始化只加载样本行 ID 和元数据；每次 `__getitem__` 按需解压一个结构。
每个进程独立打开只读 SQLite 连接，支持 Windows `spawn` 和 Unix `fork`。
训练进程应使用上下文管理器或调用 `dataset.close()` 释放连接。
使用 `split=None`（默认）读取全部结构，`split="val"` / `"test"` 读取对应集合。
空划分是合法 Dataset；创建 `shuffle=True` 的 DataLoader 前应先检查其长度。

不同结构原子数量可以不同。`collate_crystals` 按样本顺序拼接原子和边，
以 `N_total=sum(num_atoms)` 和 `E_total=sum(num_edges)` 表示总原子数和总边数。
每个晶体保留自己的晶格，边索引按累计原子数偏移，因此各晶体仍为互不连通的子图。

| 批次字段 | 形状 / 用途 |
| --- | --- |
| `atomic_numbers` | `[N_total]`，原子序数 1–118；0 用于训练 mask token |
| `frac_coords`、`positions` | `[N_total, 3]` |
| `edge_index` | `[2, E_total]`，第 0 行中心、第 1 行邻居，使用批次内原子编号 |
| `edge_vectors`、`edge_images` | `[E_total, 3]` |
| `edge_distances` | `[E_total]` |
| `lattice` | `[B, 3, 3]`，每个结构保留自己的晶格 |
| `batch` | `[N_total]`，每个原子所属晶体的编号 |
| `ptr` / `edge_ptr` | `[B+1]`，各结构的原子/边切片边界 |
| `num_atoms` / `num_edges` | `[B]`，每个结构的原子/边数量 |
| `num_graphs` | Python 整数 `B` |
| `sample_id`、来源、化学式等 | 长度为 `B` 的 Python 列表 |

SQLite 中的单结构数组格式与此兼容，已有数据库可继续使用。
单样本经 `collate_crystals([dataset[i]])` 后，坐标仍是 `[N_i, 3]`，
晶格为 `[1, 3, 3]`，`batch` 全为 0，`ptr` 为 `[0, N_i]`。

`Denoiser` 接收原来的四个参数，时间参数为每个原子的 `[N_total, 1]`，
输出 logits 为 `[N_total, 118]`。先按晶体采样时间，再通过 `batch` 展开：

```python
# batch 中的张量已移动到与 model 相同的设备
graph_t = torch.rand(batch["num_graphs"], device=batch["atomic_numbers"].device)
atom_t = graph_t[batch["batch"]]
masked_atoms, mask = mask_atomtype(batch["atomic_numbers"], atom_t)
if mask.any():
    logits = model(masked_atoms, batch["edge_index"], batch["edge_vectors"], atom_t[:, None])
    loss = atomtype_ce(logits, batch["atomic_numbers"], mask)
```

坐标扩散使用同一布局。每个晶体的尺度通过归属索引展开为 `[N_total, 1]`：

```python
from src.diffusion import add_coord_noise, coordinate_target

steps = torch.randint(1, len(schedule.sigmas), (batch["num_graphs"],), device=batch["frac_coords"].device)
sigmas = schedule.sigmas[steps][batch["batch"], None]  # schedule 已移动到相同设备
norms = schedule.sigmas_norm[steps][batch["batch"], None]
frac_t, noise = add_coord_noise(batch["frac_coords"], sigmas)
target = coordinate_target(noise, sigmas, norms)
```

几何转换 `frac_to_cart(frac_coords, lattice, batch)` 和 `cart_to_frac` 根据
每个原子的晶体归属选择晶格。单晶体也可以直接传 `[N, 3]` 坐标和 `[3, 3]` 晶格。
新增的 `build_graph_fc(frac_coords, lattice, batch, num_atoms)` 及
`build_neighbor_graph(frac_coords, lattice, cutoff, batch=batch)` 返回
`[2, E]` 边索引、`[E, 3]` 边向量和 `[E]` 距离，并保持晶体之间互不连接。
这些辅助函数沿用分数坐标逐分量取整的周期镜像约定；完整的多镜像周期半径图由数据集预处理构建。

完整导入、设备迁移、优化器和训练循环见 `examples/train_mask.py`。
示例沿用现有原子类型去噪目标，不提供能量、力或其他未出现在 CIF 中的标签，
也不保存 checkpoint 或实现完整验证流程。存储中保留未加噪的结构，每次迭代动态采样 mask。
如果后续训练会扰动坐标或晶格，需要在扰动后重新计算邻接关系和边向量，不能继续使用
预处理时固定的几何信息。

## 6. 验证

激活 `diffusion` 后，在本项目目录运行：

```powershell
python -B -m unittest discover -s tests -v
```

测试覆盖真实 CIF 往返与来源保存、单原子周期镜像、多个镜像、非正交晶胞与显式枚举
对照、递归目录和多结构 CIF、变长图批处理、空边、可复现且不交叉的划分、无效输入回滚、
数据库搬移以及父进程已读取样本后的 Windows `spawn` 多进程加载。
还包含真实 13 原子 CIF 的单样本前向、反向传播和优化器更新检查，
多结构拼批/逐结构输出一致性检查，以及新增 `data.standardize_cif` 输出与
拼接批处理、几何转换和 Denoiser 的衔接检查。

若 Codex 沙箱启动 Conda 出现“无效的系统 DLL 重定位”，按 `diffusion_mattergen/AGENTS.md`
的规定申请在沙箱外运行必要的激活与测试命令；这不表示 `diffusion` 环境损坏。
