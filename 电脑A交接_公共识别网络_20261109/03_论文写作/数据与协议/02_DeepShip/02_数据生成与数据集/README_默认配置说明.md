# DeepShip Python数据生成器

代码目录：`D:\LSJ\Code\UART\deepship\02_数据生成与数据集`。原始DeepShip只读，生成数据单独保存。

## 执行

已有`zpfenv1`环境的依赖已核对，PowerShell中运行：

```powershell
cd 'D:\LSJ\Code\UART\deepship\02_数据生成与数据集'

# 可选：只保存选源与混合计划，不生成波形
.\run.ps1 -Mode plan

# 小规模贯通，输出到独立smoke目录
.\run.ps1 -Mode smoke

# 正式全量生成
.\run.ps1 -Mode full
```

启动器默认使用`C:\Users\Admin\anaconda3\envs\zpfenv1\python.exe`。需要其他解释器时传入`-PythonExe 'C:\path\python.exe'`。也可直接执行：

```powershell
& 'C:\Users\Admin\anaconda3\envs\zpfenv1\python.exe' .\run.py --mode full
```

新环境安装`requirements.txt`即可。计算使用CPU，波形按批写入磁盘；不启动识别模型训练。

## 默认数据安排

- 正式输出：`D:\LSJ\Data\DeepShip_Multitarget_0dB_v1`。
- 试生成输出：`D:\LSJ\Data\DeepShip_Multitarget_0dB_v1_smoke`。
- 四类：Cargo、Passengership、Tanker、Tug。11个集合标签：0为背景，1–4为四种Single，5–10为按上述顺序枚举的六种不同船型组合；数量为0／1／2。
- 沿用初筛的录音／事件归属；正式计划保留原Train每类700片，再补至900片；Val／Test每类150片及其源ID保持不变。每录音／事件累计最多20片，覆盖原609条录音。

| 集合 | Single | Double | Wenz合成背景 | 总数 |
| --- | ---: | ---: | ---: | ---: |
| Train | 3600 | 7200 | 1200 | 12000 |
| Val | 600 | 1200 | 200 | 2000 |
| Test | 600 | 1200 | 200 | 2000 |
| 合计 | 4800 | 9600 | 1600 | 16000 |

先从整录音重采样16kHz、去均值，再读取核心的真实前史／后视，使用随代码保存的39条Bellhop E99 FIR传播并截取5秒接收核心。每核心一份组件，由Single和Double复用；无需MATLAB或重新求解Bellhop。

Double在接收组件、包络之前设置SIR＝0dB；六种组合约均衡，片段对不重复。C0全程重叠、C1单侧占优、C2交替占优，比例约50／25／25。三集合统一随机占优区0.8–1.2秒、过渡0.2–0.3秒，弱路幅度0.1。最后对整条输出施加公共增益至RMS0.05。

背景是Wenz三成分合成负样本。Single基础数组不施加人工包络，Train加载器可显式启用。原船声仍含天然背景；后续在线AWGN的SNR定义为相对已有接收波形的追加噪声比，不保存六套含噪数据。

## 配置和输出接口

所有参数集中在`config.json`；相对路径均以该配置文件所在目录为基准。筛选CSV快照在`sources/prescreen`，FIR在`physical`，运行不依赖旧代码目录。

| 参数 | 用途 |
| --- | --- |
| `raw_root`、`output_directory`、`smoke_output_directory` | 原始、正式和试生成路径 |
| `selection.per_class`、`selection.max_per_recording` | 各类各集合配额及单录音上限 |
| `seed`、`bellhop` | 随机种子、已有FIR及传播开关 |
| `mixture` | Double／背景数量比例、SIR、C0／C1／C2及包络参数 |
| `background` | Wenz风速、航运因子与低频参数 |
| `single_augmentation` | 加载时可选的Train Single增强 |
| `export_wavs` | 默认false；需要时额外导出WAV |
| `runtime` | smoke规模、混合批大小和进度间隔 |

每个生成目录包含：

```text
CONFIG.json              本次配置
DATA_INDEX.csv           唯一最终样本排列索引，含来源、标签、组件和包络参数
SUMMARY.json / REPORT.md  实际数量、耗时及生成结果
components/audio.npy     可复用的单源接收组件
components/STATE.json    组件映射、信道与处理状态
data/Train/audio.npy     float32波形数组
data/Val/audio.npy
data/Test/audio.npy
```

默认正式波形约4.8GiB，加组件约6.2GiB。成功结束显示`COMPLETE`及报告路径；重复运行已完成版本直接返回。修改数据规则时建议使用新版本和新输出目录。

分阶段执行或明确重做时可使用：

```powershell
.\run.ps1 -Mode full -ComponentsOnly        # 只生成组件
.\run.ps1 -Mode full -MixOnly               # 复用已完成组件，生成最终波形
.\run.ps1 -Mode full -OverwritePlan         # 明确重建该目录的选源／混合计划
.\run.ps1 -Mode full -OverwriteComponents   # 明确重建组件
```

改变选源或信道后，旧组件不再适配，需新输出目录或显式重建组件。中断的组件重做使用`-OverwriteComponents`；原始录音不变。仅修改混合规则且组件仍相同时，可配合`-OverwritePlan -MixOnly`复用组件。

## 读取数据

```python
from dataset import MixedDataset, recognition_metrics, add_noise

train = MixedDataset(r'D:\LSJ\Data\DeepShip_Multitarget_0dB_v1', 'Train')
wave, target, label, count = train.batch([0, 1, 2])
# target为四维presence；label为11集合标签；count为0/1/2
wave_aug, target, label, count = train.batch([0, 1, 2], augment_single=True)
# 增强只作用于Train Single，Val/Test和Double不重复施加包络
noisy = add_noise(wave, snr_db=-5, seed=13, sample_ids=[0, 1, 2])
```

加载器采用内存映射，默认返回存储波形。`recognition_metrics`输出Single、Double、背景、各组合及数量指标。

已完成23项测试及159条真实smoke样本贯通：每集合53条、48个单源核心来自48条录音，三个集合均覆盖11标签，输出RMS0.05。96条Double重放逐点一致，包络前SIR误差小于4e−15dB。正式全量数据尚未生成，留由用户运行。
