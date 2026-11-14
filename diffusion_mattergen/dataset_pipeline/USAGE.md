# crystal_dataset 使用指南

## 1. 从 CIF 数据集创建 SQLite 文件

在 `diffusion_mattergen` 目录打开 PowerShell，激活 `diffusion` 环境并安装本地包。安装只需执行一次，之后同一环境中的其他 Python 文件也能导入 `crystal_dataset`。

```powershell
python -m pip install --no-build-isolation --no-deps -e ./dataset_pipeline
```

将 `./data` 中的 CIF 转成一个数据库：

```powershell
python -m crystal_dataset build ./data ./dataset_pipeline/artifacts/training.sqlite --cutoff 5.0
```

第一个路径是输入，可以换成自己的数据集目录或单个 CIF 文件。输入为目录时会递归读取其中的 CIF，每个解析出的结构保存为一个样本。第二个路径是输出文件，请使用新的文件名。`--cutoff` 是周期邻居的截断半径，单位为 Å，默认值为 5.0。

## 2. 在其他 Python 文件中读取和使用

使用同一个 `diffusion` 环境运行以下代码。把 `db_path` 设为上一步生成的文件的绝对路径，这样脚本放在其他目录也能读取它。

```python
from pathlib import Path

from torch.utils.data import DataLoader
from crystal_dataset import CrystalDataset, collate_crystals

db_path = Path(
    "E:/THU/AI2DFT/Diffusion-Material-Generation/diffusion_mattergen/"
    "dataset_pipeline/artifacts/training.sqlite"
)

with CrystalDataset(db_path, split="train") as dataset:
    # 按下标读取一个结构
    sample = dataset[0]
    print(sample["sample_id"], sample["formula"])
    print(sample["atomic_numbers"])  # [N]，每个原子的原子序数

    # 原子数不同的晶体按原子维拼接成一个批次
    loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_crystals)
    for batch in loader:
        atoms = batch["atomic_numbers"]  # [N_total]
        edges = batch["edge_index"]      # [2, E_total]
        vectors = batch["edge_vectors"]  # [E_total, 3]
        print(atoms.shape, edges.shape, vectors.shape)
```

`split="train"` 读取训练集，也可以指定 `"val"` 或 `"test"`。省略 `split` 会读取全部结构。当前只有一个 CIF，默认划分下该样本在训练集。

## 3. 读取 batch 中的字段

`B` 为当前批次的晶体数，`N_total` 为所有晶体的原子总数，
`E_total` 为所有晶体的有向周期边总数。原子和边按样本顺序拼接，
晶体之间不连边，各自保留晶格矩阵。

| 字段 | 形状或类型 | 含义 |
| --- | --- | --- |
| `atomic_numbers` | `[N_total]` | 原子序数 1–118；0 用于训练 mask token |
| `frac_coords` | `[N_total, 3]` | 原子在所属晶胞中的分数坐标 |
| `positions` | `[N_total, 3]` | 笛卡尔坐标，单位 Å |
| `lattice` | `[B, 3, 3]` | 每个晶体的行向量晶格矩阵，单位 Å |
| `edge_index` | `[2, E_total]` | 第 0 行是中心、第 1 行是邻居，使用批次内原子编号 |
| `edge_images` | `[E_total, 3]` | 邻居镜像沿晶格基矢的整数平移量 |
| `edge_vectors` | `[E_total, 3]` | 中心到邻居镜像的笛卡尔位移，单位 Å |
| `edge_distances` | `[E_total]` | 边向量长度，单位 Å |
| `batch` | `[N_total]` | 每个原子所属晶体在批次中的编号，从 0 开始 |
| `ptr`、`edge_ptr` | 各为 `[B+1]` | 各晶体的原子、边切片边界 |
| `num_graphs` | Python `int` | 当前批次晶体数 `B` |
| `num_atoms`、`num_edges` | 各为 `[B]` | 每个晶体的实际原子数和边数 |
| `sample_id`、`source_path`、`structure_index` | 各为长度 `B` 的列表 | 样本 ID、来源和结构序号 |
| `formula`、`split` | 各为长度 `B` 的列表 | 化学式和数据划分 |

几何量默认是 `torch.float32`，整数为 `torch.int64`。
例如两个晶体分别有 2 和 3 个原子，则 `batch=[0,0,1,1,1]`，
`ptr=[0,2,5]`。第二个晶体的局部边 `0 → 2` 在批次中变为 `2 → 4`。

取回批次中某个晶体：

```python
g = 0
n0, n1 = batch["ptr"][g:g + 2].tolist()
e0, e1 = batch["edge_ptr"][g:g + 2].tolist()
atoms = batch["atomic_numbers"][n0:n1]
frac = batch["frac_coords"][n0:n1]
lattice = batch["lattice"][g]
local_edges = batch["edge_index"][:, e0:e1] - n0
vectors = batch["edge_vectors"][e0:e1]
```

单样本使用 `collate_crystals([dataset[0]])`。当前 13 原子 CIF 的坐标为
`[13, 3]`、原子类型为 `[13]`、晶格为 `[1, 3, 3]`，`batch` 是 13 个 0。
已有 SQLite 文件可以直接使用。原子数和边数均按实际样本大小保存。

两个训练入口均支持 `--batch-size`。在 `diffusion_mattergen` 目录激活环境后，
可以完成一次单样本训练：

```powershell
python -B ./scripts/train_mask.py --epochs 1 --batch-size 1
```

模型的时间参数要先展开到原子，再传入 `[N_total, 1]`：

```python
graph_t = torch.rand(batch["num_graphs"], device=batch["atomic_numbers"].device)
atom_t = graph_t[batch["batch"]]
logits = model(batch["atomic_numbers"], batch["edge_index"], batch["edge_vectors"], atom_t[:, None])
# logits: [N_total, 118]
```

新增的单晶体标准化接口输出 `atomic_numbers [N]`、`frac_coords [N,3]`、
`lattice [3,3]` 和标量 `num_atoms`，同样采用按实际原子数保存的约定。
见 [单晶体标准化说明](../data/README.md)。
