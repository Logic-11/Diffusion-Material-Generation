# 晶体标准化与数据集处理

将有序晶体转换为模型使用的原胞表示。支持单个 CIF、内存中的 `pymatgen.Structure`，以及 MP-20 这样的 CSV 数据集。每个结构都经过原胞提取、Niggli 约化和对称正定晶格变换。

单晶体接口返回四个 NumPy 数组或保存 NPZ；批量接口保存按划分组织的 NPY 缓存。`CrystalDataset` 和 `collate_crystals` 提供可直接传给 `src.geometry.build_periodic_graph` 的批次。

## 批量处理 MP-20

从项目根目录运行以下 PowerShell 命令。输入路径中的 `mp_20` 实际包含三个 CSV；每行的 `cif` 列是一个完整、多行的 CIF 字符串，`material_id` 标识材料。代码按 CSV 记录读取，不能把物理文本行当成结构数量。

```powershell
Set-Location 'E:\THU\AI2DFT\Diffusion-Material-Generation\diffusion_mattergen'
. './scripts/activate_diffusion.ps1'
if ($env:CONDA_DEFAULT_ENV -ne 'diffusion' -or (Get-Command python).Source -ne 'D:\Anaconda\anaconda3\envs\diffusion\python.exe') {
    throw 'Expected the diffusion Python environment.'
}
$source = 'E:\THU\AI2DFT\Diffusion-Material-Generation\data\mp_20'
$cache = 'E:\THU\AI2DFT\Diffusion-Material-Generation\diffusion_mattergen\data\processed\mp_20'
python -B -m data build $source --output $cache
if ($LASTEXITCODE -ne 0) { throw 'Dataset build failed.' }
python -B -m data inspect $cache
```

本地输入的实测数量是 `train=27136`、`val=9047`、`test=9046`，共 **45229** 个结构；以实际 CSV 为准。MP-20 中的 20 指晶胞原子数上限。保留输入划分、材料 ID 和划分内的原始行顺序，不随机重新划分，不按结构等价性删除样本。

批量命令接受以下输入：

| 输入 | 划分方式 |
|---|---|
| 含 `train.csv`、`val.csv`、`test.csv` 的目录 | 保留已有划分；允许只提供其中部分文件。 |
| 单个 CSV | 文件名为 train/val/test 时使用对应划分，其他文件名使用 train。 |
| 单个 CIF 或递归包含 CIF 的目录 | 全部进入 train；用相对文件路径作为结构 ID。 |

CSV 必须包含 `material_id` 和 `cif` 列。每个 CIF 文件或 CIF 字符串仍只能描述一个结构。`standardize_cif_text(text)` 也是公开接口，可直接用于内存中的 CIF 文本。

默认遇到无效结构就中止，错误包含来源文件、CSV 记录序号和材料 ID；失败不会留下可误认为完整缓存的目标目录。可显式传 `--skip-invalid`，将无效结构记录到 `errors.jsonl`。重复材料 ID（包括跨划分）、CSV 格式错误和磁盘错误始终中止。原始 CSV 不会被修改。

输出目录必须不存在；重建时使用新路径，例如 `mp_20_v2`。批量构建不提供覆盖选项。`--primitive-tolerance` 默认仍为 `0.25` Å，所有结构与单晶体接口使用同一套标准化方法。

## 数据集存储格式

核心数组采用 [MatterGen 数据集的字段和拼接布局](https://github.com/microsoft/mattergen/blob/main/mattergen/common/data/dataset.py)。官方对称晶格变换可在加载阶段执行；本项目延续单晶体接口，在预处理阶段完成同一极分解表示。额外的 JSON 元数据是本项目的溯源格式，不等同于官方条件属性缓存。

```text
data/processed/mp_20/
├── metadata.json          # 格式版本、预处理参数、源文件 SHA256、版本和各划分数量
├── errors.jsonl           # 显式跳过的无效结构；没有错误时为空
├── train/
│   ├── atomic_numbers.npy # int64   [N_total]
│   ├── pos.npy            # float64 [N_total, 3]，分数坐标
│   ├── cell.npy           # float64 [S, 3, 3]，行向量晶格，单位 Å
│   ├── num_atoms.npy      # int64   [S]
│   ├── structure_id.npy   # Unicode [S]，不使用 object/pickle
│   └── records.jsonl     # 与结构顺序对齐的来源记录、其余 CSV 字段和解析警告
├── val/                  # 同样的六个文件
└── test/                 # 同样的六个文件
```

`S` 是该划分中的结构数，`N_total` 是该划分全部原子的总数。利用 `num_atoms` 的累计和定位每个结构的原子切片。构建时逐条解析，将数值数组流式写入临时文件，然后分块转为 NPY，避免同时保存整个数据集的 `Structure` 对象。

缓存采用独立 NPY，支持只读内存映射；一次取样只复制该晶体的张量，默认浮点类型为 `torch.float32`，也支持 `torch.float64`。对返回张量原地加噪不会修改磁盘缓存。每个 DataLoader 进程独立打开映射，支持 Windows spawn；Dataset 支持上下文管理器或显式 `close()`。

`records.jsonl` 保留 `cif` 之外的原始 CSV 列，包括带隙、形成能、化学式等，但不自动把这些字符串转换为训练标签。原始 CIF 可通过来源文件、记录序号与 SHA256 追溯。加载缓存不再需要读取原始 CSV。

这里只缓存干净结构，周期边在每个 batch 上调用现有几何函数生成。缓存因此不依赖 cutoff 或邻居上限；改变构图参数无需重做标准化。

## 加载 batch 并调用 build_periodic_graph

以下示例从项目根目录运行。若使用 `num_workers > 0`，在 Windows 上需要保留 `main` 保护。

```python
import torch
from torch.utils.data import DataLoader
from data import CrystalDataset, collate_crystals
from src.geometry import build_periodic_graph

def main():
    cache = r"E:\THU\AI2DFT\Diffusion-Material-Generation\diffusion_mattergen\data\processed\mp_20"
    device = torch.device("cpu")  # 也可以使用 cuda
    with CrystalDataset(cache, split="train") as dataset:
        loader = DataLoader(dataset, batch_size=64, shuffle=True,
                            num_workers=0, collate_fn=collate_crystals)
        for batch in loader:
            batch = {key: value.to(device) if torch.is_tensor(value) else value
                     for key, value in batch.items()}
            (edge_index, edge_image, edge_vector, edge_distance,
             num_edges, edge_ptr, edge_batch) = build_periodic_graph(
                batch["frac_coords"], batch["lattice"],
                batch=batch["batch"], cutoff=7.0, max_neighbor=50,
            )
            # 此处使用本 batch 的元素、晶格和周期图。
            # 如需扰动坐标或晶格，应先更新 batch，再调用构图函数。

if __name__ == "__main__":
    main()
```

`collate_crystals` 不补齐原子数，不会跨晶体连接节点：

| 字段 | 形状与含义 |
|---|---|
| `atomic_numbers` | `[N_batch]`，拼接后的元素 |
| `frac_coords` | `[N_batch,3]`，拼接后的分数坐标 |
| `lattice` | `[B,3,3]`，每个晶体的晶格 |
| `num_atoms` | `[B]`，每个晶体的原子数 |
| `batch` | `[N_batch]`，每个原子属于哪个晶体 |
| `ptr` | `[B+1]`，各晶体的原子切片边界 |
| `num_graphs` | Python 整数 `B` |
| `structure_id`、`split` | 长度为 `B` 的列表，用于定位来源 |

构图函数返回七项：`edge_index [2,E]`、`edge_image [E,3]`、`edge_vector [E,3]`、`edge_distance [E]`、`num_edges [B]`、`edge_ptr [B+1]`、`edge_batch [E]`。返回值命名沿用当前 `geometry.py`，没有把边预先存入数据集。

已提供可直接运行的全量构图检查脚本，在激活环境后执行：

```powershell
python -B ./data/examples/build_graphs.py ./data/processed/mp_20 --batch-size 64 --report ./data/processed/mp_20_graph_check.json
```

默认遍历所有已缓存划分，检查每个 batch 的边是否跨晶体、边向量和周期平移是否一致、距离是否有效，并核对全部结构和原子的覆盖数量。默认 CPU 使用一个线程以降低小矩阵操作的线程开销。可传 `--workers 2` 检查多进程加载，或传 `--max-batches 2` 仅做快速检查；快速检查的报告会标明是否覆盖完整划分。报告路径也不覆盖已有文件。

本模块与旧 `dataset_pipeline` 的 SQLite/预存图格式独立；新的 NPY 缓存使用本页的 `data.CrystalDataset` 读取。

## 测试

激活 diffusion 后，在项目根目录执行 `python -B -m unittest discover -s data/tests -v`。测试包括单晶体几何、CSV 划分与来源保存、原胞转换、失败回滚、跨划分重复 ID、只读缓存保护、多进程加载，以及与 pymatgen 独立邻居搜索对照的变长批次构图。

## 第一次运行

在 PowerShell 中执行下面这段命令。从 `diffusion_mattergen` 根目录运行示例，所有相对路径都以该目录为起点。

```powershell
Set-Location 'E:\THU\AI2DFT\Diffusion-Material-Generation\diffusion_mattergen'
. './scripts/activate_diffusion.ps1'
if ($env:CONDA_DEFAULT_ENV -ne 'diffusion' -or (Get-Command python).Source -ne 'D:\Anaconda\anaconda3\envs\diffusion\python.exe') {
    throw 'Expected the diffusion Python environment.'
}
python -B -m data data/Ba2PrCu3O7.cif --output results/Ba2PrCu3O7.npz
```

首次运行会创建 `results` 目录。成功后，终端显示输出路径、`num_atoms=13` 和 `volume=184.02153 A^3`。每次新开 PowerShell 都需要重新激活环境。标准化需要 pymatgen 和 NumPy，读取 NPZ 只需 NumPy，最后一个张量示例还需要 PyTorch。[示例 CIF](./Ba2PrCu3O7.cif)

### 命令行参数

| 参数 | 必填或默认值 | 用途 |
|---|---|---|
| `cif` | 必填，第一个位置参数 | UTF-8 编码的 CIF 文件路径，每个文件包含一个晶体结构。 |
| `--output PATH` | 必填 | 以 `.npz` 结尾的输出路径。缺少的父目录会自动创建。 |
| `--primitive-tolerance FLOAT` | 默认 `0.25` | 取原胞时识别平移等价位点的位置容差，单位 Å。取有限正数，较小的值会更严格地区分轻微偏离的位点。 |
| `--overwrite` | 默认关闭 | 允许在成功转换后替换同名输出。默认遇到已有文件会报错。 |
| `-h`、`--help` | 可选 | 显示用法及参数。 |

例如，用更小的容差重新转换并替换刚才的文件，可以在已激活的终端运行 `python -B -m data data/Ba2PrCu3O7.cif --output results/Ba2PrCu3O7.npz --primitive-tolerance 0.10 --overwrite`。包含空格的路径需加引号。

## 输入与返回值

输入应是沿三条晶格矢量周期重复的晶体。每个位点由一种元素占据，占位率为 1。CIF 省略整列占位率时按 1 读取。坐标和晶格参数必须是有限数值，三条基矢须线性无关。输入可以是原胞、常规胞或超胞，分数坐标也可以超出一个周期。

准备自己的 CIF 时，可参照示例文件提供三个晶胞长度（Å）、三个晶胞角（度）、位点标签 `_atom_site_label`、元素符号 `_atom_site_type_symbol` 和三列分数坐标 `_atom_site_fract_x/y/z`。如果文件只列出不对称单元中的位点，还应给出对应的空间群或对称操作，以便还原完整晶胞。

输出使用 Niggli 约化原胞并固定晶格的整体朝向。位点以元素和位置区分，氧化态、磁矩及自定义位点标签不参与区分。表中的 `N` 都指**输出原胞**的原子数。

| 字段 | NumPy 类型 | 形状 | 含义 |
|---|---|---|---|
| `atomic_numbers` | `int64` | `(N,)` | 各位点的原子序数 1–118。`atomic_numbers[i]` 对应 `frac_coords[i]`。 |
| `frac_coords` | `float64` | `(N, 3)` | 无量纲分数坐标，每个分量都在 `[0, 1)`。 |
| `lattice` | `float64` | `(3, 3)` | 对称正定的晶格矩阵，三**行**依次是基矢 a、b、c，单位 Å。 |
| `num_atoms` | `int64` | `()` | 零维数组。`int(cell["num_atoms"])` 取得 Python 整数。 |

从超胞取原胞后，原子数和总体积可能按同一倍数缩小。原子序号以本次输出为准。需要比较两个等价结构时，应结合元素和周期位置匹配，同一元素的位点顺序以及等价原点、等长晶轴的选择都可能不同。

## 在 Python 中直接转换

以下每个 Python 示例都可以独立运行。在已激活的环境中，从 `diffusion_mattergen` 根目录打开 Python，或把示例存成脚本并从该目录运行。

```python
from pymatgen.core import Lattice, Structure
from data import standardize_cell, standardize_cif

# CIF 路径可以是字符串，也可以是 pathlib.Path。
from_file = standardize_cif("data/Ba2PrCu3O7.cif")
print(int(from_file["num_atoms"]))  # 13

# 内存输入给出晶格、元素和分数坐标。这里建立晶体后沿 a 方向扩成超胞。
structure = Structure(Lattice.cubic(4.2), ["Cs", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
supercell = structure * [2, 1, 1]
from_memory = standardize_cell(supercell, primitive_tolerance=0.25)
print(len(supercell), int(from_memory["num_atoms"]))  # 4 2
```

两个函数均接受仅限关键字的参数 `primitive_tolerance`，单位和默认值与命令行一致。返回值是以字段名为键的字典。这里构造 `Structure` 时使用分数坐标，如已有 Å 为单位的笛卡尔坐标，可在构造 `Structure` 时传入 `coords_are_cartesian=True`。

## 读取文件并检查几何量

先完成“第一次运行”。`np.load` 读出的字段与 Python 接口相同。分数坐标右乘晶格矩阵得到笛卡尔坐标，晶格行列式给出原胞体积。

```python
import numpy as np

with np.load("results/Ba2PrCu3O7.npz", allow_pickle=False) as cell:
    numbers = cell["atomic_numbers"]
    frac = cell["frac_coords"]
    lattice = cell["lattice"]
    n = int(cell["num_atoms"])

cart_angstrom = frac @ lattice
volume_angstrom3 = np.linalg.det(lattice)
print(n, cart_angstrom.shape, f"{volume_angstrom3:.5f}")  # 13 (13, 3) 184.02153
print(f"{volume_angstrom3 / n:.5f}")  # 14.15550 Å³/原子
assert numbers.shape == (n,) and np.all((frac >= 0) & (frac < 1))
assert np.allclose(lattice, lattice.T) and np.all(np.linalg.eigvalsh(lattice) > 0)
```

## 转为 PyTorch 张量

下面的代码将坐标和晶格转成训练常用的 `float32`。原子保持 `[N]` 和 `[N,3]`，
晶格堆叠为 `[B,3,3]`，另用 `batch` 标记原子所属晶体。单个晶体的 `B=1`。

```python
import numpy as np
import torch
from src.geometry import frac_to_cart

with np.load("results/Ba2PrCu3O7.npz", allow_pickle=False) as cell:
    atomic_numbers = torch.from_numpy(cell["atomic_numbers"])
    frac_coords = torch.from_numpy(cell["frac_coords"]).float()
    lattice = torch.from_numpy(cell["lattice"]).float().unsqueeze(0)
    num_atoms = torch.from_numpy(cell["num_atoms"]).reshape(1)

batch = torch.repeat_interleave(torch.arange(num_atoms.numel()), num_atoms)
positions = frac_to_cart(frac_coords, lattice, batch)
print(atomic_numbers.shape, positions.shape, lattice.shape, num_atoms.tolist(), batch.shape)
# torch.Size([13]) torch.Size([13, 3]) torch.Size([1, 3, 3]) [13] torch.Size([13])
```

多个晶体按同一规则组合：沿原子维拼接 `atomic_numbers` 和 `frac_coords`，
按晶体维堆叠 `lattice`，保留 `num_atoms [B]` 和 `batch [N_total]`。
本模块的 `collate_crystals` 还提供原子边界 `ptr`，构图后得到边界 `edge_ptr`。
完整用法见本页“加载 batch 并调用 build_periodic_graph”。

## 常见问题

| 现象 | 处理方式 |
|---|---|
| Python 无法导入 `data` 或 `pymatgen` | 在 `diffusion_mattergen` 根目录重新执行环境激活和解释器检查。若示例另存为脚本，把脚本放在该根目录，或将根目录加入 Python 模块搜索路径。 |
| 提示 `Expected exactly one crystal structure` | 检查文件中是否存在带原子位点的结构块。若包含多个晶体，将每个晶体单独导出为一个 CIF 后再转换。 |
| 提示 `fully occupied` 或指出 `_atom_site_occupancy` | 检查混合元素、部分占位或未知占位率。为具体的有序构型提供每个位点的元素与满占位数据。 |
| 错误指出坐标、晶格字段或 `Degenerate lattice` | 在输入中补全缺失数值，检查长度为正、角度合理、晶胞有非零体积。 |
| 提示 `primitive_tolerance` 无效 | 使用以 Å 为单位的有限正数，例如 `0.25`。 |
| 提示 `Output already exists` | 改用新文件名，或用 `--overwrite` 替换输出。 |

命令行成功时退出码为 0，出错时返回非零退出码并在标准错误中说明原因。Python 接口对不合要求的晶体数据抛出 `ValueError`，路径不存在时抛出 `FileNotFoundError`。

晶胞表示与 MatterGen 的原胞、Niggli 约化和对称晶格约定相对应。[MatterGen 数据预处理](https://github.com/microsoft/mattergen/blob/main/mattergen/common/data/dataset.py) · [MatterGen 晶格表示](https://github.com/microsoft/mattergen/blob/main/mattergen/common/utils/data_utils.py)
