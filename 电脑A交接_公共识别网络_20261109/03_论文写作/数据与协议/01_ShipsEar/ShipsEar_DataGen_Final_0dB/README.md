# ShipsEar 最终数据生成器

当前数据共 **4473 条**，Train/Val/Test 为 **2880/716/877**，波形及排列与已确认的 89.40% 实验一致。

唯一的数据排列索引是 **`datasets/shipsear_pf_mb_rv_0db/DATA_INDEX.csv`**。每行对应一个样本，保存集合、数组行号、标签、来源录音/片段、组件路径、C0/C1/C2、包络参数、训练增强类型、SIR和增益。它合并了原来分散的 manifest 和幅度记录，不再输出重复的排列清单、哈希表、冻结报告或 READY 文件。

## 固定数据与目标比例

- PF、MB、RV 三类全部有效来源参与 Single 和 Double。51/52/58/78 属于 Train，其余录音维持当前归属。
- 同一原录音的派生样本只属于一个集合，配对只能发生在集合内。仅声明录音级隔离；部分录音来自同一船舶或事件。
- 当前 Double/Single/E 约 60/30/10，双目标内部 C0/C1/C2 实际为 50/25/25。
- 比例是目标值，生成新版本时按整数取整，并根据来源覆盖与配对容量调整。没有精确比例相等或必须被4整除的断言。单目标保留全部核心，不强制把船型数量配平。
- 既有 `DATA_INDEX.csv` 优先用于复现当前排列，现有音频不会因修改目标比例而被重排或覆盖。修改比例后若要生成新数据，请在 `config.json` 中指定新的 `output_directory`。
- 原始来源信息保存在 `metadata/sources.json` 和 `recordings.json`，供 MATLAB 提取核心及建立新版本源池使用；它们不是额外的样本排列导出表。

## 流程

MATLAB 读取原音频、重采样和去均值，采用冻结的39个 Bellhop E99 FIR 传播，保存5秒、16 kHz接收组件。Python 在集合内配对，设置包络前 SIR，施加上下文包络、相加，最终公共增益归一化至 RMS 0.05。

本版 SIR=0 dB。C0 全程叠加；C1 单侧占优；C2 首尾分别占优、中间叠加。占优区随机0.8–1.2秒，过渡随机0.2–0.3秒，弱路幅度0.1，所有集合使用相同参数分布。标签采用官方主船型，生成来源数不等同于原始现场的绝对船舶数。

背景E独立选窗，不向目标音频额外注入背景或人工噪声。Single基础文件不加人工包络；训练加载器默认按索引启用约50%平坦、25%边缘变化、25%中间占优增强，Val/Test关闭。

种子42决定新计划的随机采样；源池、代码和输入顺序也会影响结果。随机子流仍沿用原种子派生方式，其中的摘要只作用于种子字符串，不读取或校验文件。精确复现这批数据请保留索引和数据配置。

## 必要检查

只检查来源归属不跨集合、标签与来源一致、数组行号和音频格式正确、音频有限且非静音、应参与的来源得到覆盖、片段对不重复。输出实际数量和比例供查看，不因合理的比例偏差终止。

加载器不再扫描文件哈希。特征缓存标识仅使用数据版本、索引/数组修改时间与增强开关；后续接入识别模型时重新建立缓存。

## 运行与读取

```powershell
.\RUN_GENERATE.ps1
.\RUN_GENERATE.ps1 -RebuildComponents
.\RUN_AUDIT.ps1
.\RUN_AUDIT.ps1 -Wavs
```

已有完整数据只检查并返回。组件缺失时从原音频重建；`-RebuildComponents` 显式重建组件。`-Wavs` 额外逐个比较 WAV 与数组采样值，不计算哈希。

从原始音频精确重建当前数据：复制代码、config、metadata、physical，以及数据目录内的 `CONFIG.json` 和 `DATA_INDEX.csv` 到空目录，再运行生成入口。未提供索引时，程序按当前目标比例和种子建立新计划，因此取整后的样本数可以与旧版本不同。

```python
import sys
from pathlib import Path
root = Path(r'D:\LSJ\Code\UART\shipsear\ShipsEar_DataGen_Final_0dB')
sys.path.insert(0, str(root / 'generator'))
from dataset import MixedDataset
train = MixedDataset(root / 'datasets/shipsear_pf_mb_rv_0db', 'Train')
audio, target, label, count = train.batch([0, 1, 2])
```

模型只获得波形及任务标签，不获得来源编号和包络参数。15%恢复源权重属于识别模型训练策略，不属于数据生成比例。

## 独立三目标扩展（2026-10-23）

已新增独立三目标分支，使用原集合内部的PF/MB/RV组件，单独生成576/144/176条Train/Val/Test样本。原4473条数据保持不变。运行 `./RUN_GENERATE.ps1 -TripleOnly`；预览加 `-PlanOnly`。默认不导出重复WAV。

完整配置、三路能量比和包络定义见[三目标分支说明](三目标分支说明.md)。`config.json` 的 `triple.enabled` 默认关闭，可通过命令行单次启用。三目标标签新增为7，数量为3；模型适配另行开展。
