# DeepShip严格自动初筛

从DeepShip原录音自动选出完整、连续、声学信息较充分的5秒基础源，供后续Bellhop传播和Single/Double混合生成使用。程序只读取原录音，输出来源索引与数量报告，不裁剪或删除原文件，不训练模型。

## 执行

在PowerShell中运行：

```powershell
Set-Location 'D:\LSJ\Code\UART\final\data_gen_v8_single_physical\DeepShip_Prescreen'

# 每类先扫描4个Train录音/事件组，确认程序与参数输出
.\run.ps1 -Mode pilot

# 全库自动初筛
.\run.ps1 -Mode full
```

启动器默认使用已有的`C:\Users\Admin\anaconda3\envs\zpfenv1\python.exe`，默认单进程逐录音扫描，无需GPU。每完成一条显示文件名、耗时、合格数和预计剩余时间；单条尚未完成时每5秒显示当前录音。需要其他Python时传入`-PythonExe 'C:\path\python.exe'`。

也可以直接执行：

```powershell
python -m pip install -r requirements.txt
python run.py --config config.json --mode pilot
python run.py --config config.json --mode full
```

## 输出

默认目录为`outputs/strict_v1/pilot`和`outputs/strict_v1/full`：

| 文件 | 内容 |
| --- | --- |
| `CONFIG.json` | 本次参数快照 |
| `RECORDINGS.csv` | 原录音归属、来源匹配、排除原因及扫描数量 |
| `SOURCE_INDEX.csv` | 唯一5秒源索引，包括保留与淘汰片段 |
| `SUMMARY.json` | 各类实际容量、淘汰原因、耗时及空间预算 |
| `REPORT.md` | 中文结果汇总 |
| `preview/` | 可选自动示例图，默认关闭 |

`SOURCE_INDEX.csv`中，`candidate_pass=1`表示通过全部自动声学门槛，`selected=1`表示同时处于录音和类别配额内，可交给后续生成器。超过配额的窗口会注明原因。全部决策由配置和程序确定，无需逐条处理。

## 配置

参数集中在`config.json`。相对路径以该配置所在目录为基准；`metadata.mapping_file`相对原始DeepShip根目录。

| 参数 | 默认 | 作用 |
| --- | --- | --- |
| `raw_root` | `D:/LSJ/Data/DeepShip` | 原录音目录，只读 |
| `output_root` | `outputs/strict_v1` | 输出版本；修改规则时建议改为新目录 |
| `metadata.require_match` | `false` | 元数据匹配不完整只标记，类别明确且音频正常的录音仍参与扫描 |
| `metadata.group_matched_event` | `true` | 已确认同一事件的多个文件不跨集合 |
| `metadata.duration_tolerance_seconds` | `2` | 元数据与音频时长的允许差值 |
| `split.seed` | `42` | 录音组划分种子 |
| `split.fractions` | `0.7/0.15/0.15` | Train/Val/Test目标比例 |
| `window_seconds` | `5` | 无重叠完整核心时长，不补短尾 |
| `screening.energy_keep_fraction` | `0.4` | 每录音优先考虑带内能量前40% |
| `screening.zero_run_ms` | `50` | 连续数字零值排除门槛 |
| `screening.activity_margin_db` | `6` | 录音内P90参考电平减6dB作为持续性门槛 |
| `screening.persistence_required` | `4` | 5个1秒子段至少4个达到门槛 |
| `screening.max_transient_energy_fraction` | `0.2` | 最强100ms能量占整个核心的上限 |
| `screening.spectrum.enabled` | `true` | 持续线谱或稳定宽带满足其一 |
| `selection.max_per_recording` | `20` | 每录音/可靠事件最多保留20个核心 |
| `selection.max_per_class` | `1000` | 每类总上限，按集合比例分配 |
| `runtime.workers` | `1` | 默认逐录音扫描；可用`-Workers 2`开启限量并行 |
| `runtime.heartbeat_seconds` | `5` | 单条任务等待时的进度输出间隔 |
| `runtime.recording_timeout_seconds` | `90` | 并行模式单任务上限；超时退出并显示录音路径 |
| `runtime.pilot_recordings_per_class` | `4` | pilot每类扫描的Train组数 |
| `preview.enabled` | `false` | 是否生成少量自动示例图 |
| `preview.max_per_class` | `2` | 每类示例图上限 |
| `preview.include_rejected` | `true` | 示例中是否包含淘汰窗口 |

更严格可降低`energy_keep_fraction`、降低`activity_margin_db`、提高`persistence_required`或降低瞬态占比上限。程序不自动放宽规则凑数。技术异常和孤立爆音先排除，再计算能量参考，避免异常值挤占候选。FLOAT幅值超过±1不当作整数削波。

## 来源与划分

工具联合原始`*-metafile`、文件名、日期和实际时长匹配来源，船名映射仅作提示。先固定录音/可靠事件归属，再筛选片段；不同时间的同一船舶仍可能跨集合。已有规则不依赖识别模型的输出。

当前先检查全部录音的合格容量。`REPORT.md`列出各类别、各集合的零合格录音数，以及每录音合格片段的最小值、非零最小值、中位数和最大值。逐录音数量看`RECORDINGS.csv`的`windows_candidate_pass`；该值在20片及每类1000片配额限制前计算。`windows_selected`只是当前临时配额内数量，正式生成配额尚未确定。

只查看来源和划分而不扫描波形：`.\run.ps1 -Mode plan`。如需复用该划分，将`split.partition_file`指向`outputs/strict_v1/plan/RECORDINGS.csv`。plan中的空窗口索引不用于生成。

## 重跑与后续接口

已有索引时默认不覆盖。使用新`output_root`保存新版本；明确重做或中断后重跑时运行`.\run.ps1 -Mode full -Overwrite`。中断/并行超时会保留已完成片段的原始指标，SUMMARY状态不标完成；重新运行从录音扫描开始，原文件不变。

并行模式只同时提交与worker数量相同的录音，不一次排队整个库。每5秒列出当前等待的文件；90秒仍未完成则停止本工具的子进程并报告路径。Ctrl+C会停止本工具自己的并行任务，不继续等待整库队列。默认单进程绕过进程池；其心跳用于显示较长录音的当前状态。

```python
from dataset import iter_selected, load_core
for row in iter_selected('outputs/strict_v1/full', split='Train'):
    waveform, fs = load_core(row, r'D:\LSJ\Data\DeepShip')
```

索引以原录音起点为零点：`start_sample`是目标采样率下的零基起点，`end_sample`为右开端点，默认`num_samples=80000`。`load_core`用于读取检查；Bellhop生成器须读取核心的真实前史/后视后再卷积，不能给每个独立核心边界补零传播。

自动声学门槛用于构建筛选子集，相对电平不是原始真实SNR；人工加噪后不再次删掉困难样本。全库扫描由用户执行，本工具尚未生成混合数据或启动训练。

自动流程已通过28项测试，包含慢任务心跳、并行任务超时和退出测试。此前4条实际Train录音的399窗口示例仍在`outputs/example_check`，正式pilot/full输出与其分开。

2026-10-31进度停顿处理：默认逐录音模式复测实际前40条，21.68秒完成，单条最长1.41秒，无扫描错误。此次只调整任务调度、进度和中断处理，筛选指标与阈值保持原值。未代替用户运行全库扫描。
