# 论文数据生成交接

本包对应已用于正式实验的两套数据。先读各自的生成记录，再按源码链接核对论文中的筛选、传播、包络和归一化公式。

| 数据 | 正式写作用途 | Train／Val／Test | 核对入口 |
| --- | --- | ---: | --- |
| ShipsEar 基础旧划分 | 已完成外部对比复用 | 2880／716／877 | [基础生成器](01_ShipsEar/ShipsEar_DataGen_Final_0dB/README.md) · [旧样本索引](01_ShipsEar/ShipsEar_DataGen_Final_0dB/datasets/shipsear_pf_mb_rv_0db/DATA_INDEX.csv) · [旧录音归属](01_ShipsEar/基础划分来源/RECORDING_SPLIT_旧划分.csv) |
| ShipsEar 最终 MB 划分 | 冻结 Q0 主模型与计划中的新版消融 | 2877／719／877 | [生成记录及 MB 交换](01_ShipsEar/shipsear_mb_train_val_20261104/REPORT.md) · [最终 DATA_INDEX](01_ShipsEar/shipsear_mb_train_val_20261104/data/DATA_INDEX.csv) · [最终录音归属](01_ShipsEar/shipsear_mb_train_val_20261104/RECORDING_SPLIT.csv) |
| DeepShip Energy80 SeenShip PTT | 三船型、七集合，当前论文对比数据 | 7500／1250／1250 | [PTT 生成链与数量](02_DeepShip/PTT最终索引/REPORT.md) · [PTT DATA_INDEX](02_DeepShip/PTT最终索引/DATA_INDEX.csv) · [PTT 录音归属](02_DeepShip/PTT最终索引/RECORDING_SPLIT.csv) |

## 论文核对顺序

1. **筛选与划分。** ShipsEar 从官方主船型及冻结源池出发，最终交换 MB 的 Train／Val 录音。DeepShip 先做录音组初筛，生成 SeenShip 父数据，再按 PTT 集合读取。ShipsEar 的旧表和新版主模型使用各自的 Train／Val，Test 索引一致。
2. **传播。** 对整录音选定通道、重采样并去均值。目标源核心用 RMS 定标后应用 E99 FIR，按历史、前瞻和数值延迟对齐。 [Bellhop 参数、缓存及源码](03_Bellhop_E99/README.md)
3. **混合与背景。** Double 在接收组件上、包络前设置源间 SIR 为 0 dB，施加 C0／C1／C2 包络，叠加后对整条波形施加公共增益至 RMS 0.05。ShipsEar 的 E 为真实背景录音窗，DeepShip 的 E 为 Wenz 三成分合成背景。 [ShipsEar 混合与包络](01_ShipsEar/ShipsEar_DataGen_Final_0dB/generator/generate.py) · [DeepShip 混合与 Wenz](02_DeepShip/02_数据生成与数据集/render.py) · [DeepShip 包络](02_DeepShip/02_数据生成与数据集/dataset.py)
4. **评估噪声。** 原始档是存储波形；10／5／0／−5／−10 dB 通过追加 AWGN 得到。源间 SIR 与额外 AWGN 的 SNR 分开表述。含目标片段以存储混合波形均方功率为参考，背景档以 0.05² 为参考，加噪后将整体 RMS 定到 0.05。正式训练的噪声抽样为原始档 50%，五个含噪档各 10%，人工 Single 包络关闭。 [ShipsEar 加噪实现](01_ShipsEar/正式数据口径/train_data.py) · [DeepShip 加噪实现](02_DeepShip/读取协议/Q0_DeepShip/noise_views.py)

## 使用本包

整目录复制即可做静态核查。包内保留实际生成代码、正式配置快照、原始索引、划分和生成报告，以及 E99 小型 FIR 和缓存索引。PTT 的索引和录音表为论文核查导出，父数据索引也一并保留。其余录音和波形数组、识别权重及 Bellhop 原始求解输出由源机保存。

需要运行时，再按目标机器设置源码和配置中的原录音、输出目录和缓存路径。配置快照保留原始绝对路径以便追溯；当前交接已核对索引和代码关系，适用于静态写作核查。
