# ShipsEar 最终 MB 划分与生成记录

依据封存生成配置、索引和样本摘要整理。版本 shipsear_mb21_33_train_27_50_val_20261104，每条为 5 秒、16 kHz 单声道。PF 为 Passenger ferry，MB 为 Motorboat，RV 为 Ro-Ro。七种输出覆盖背景、三种单船型及三种异类双船型集合。源数量取自构造时使用的源录音。

## 基础版与最终版

基础版是已经完成的旧划分外部对比所用数据，Train／Val／Test 为 2880／716／877。最终主模型采用下表。MB 21、33 号从 Val 调入 Train，27、50 号从 Train 调入 Val，依据开发时的验证难度及来源覆盖调整。Train 和 Val 分别重混 155 条和 115 条涉及换源的 Double。沿用另一源、组合标签、C0／C1／C2、包络和包络前 SIR。此交接逐行核对了基础版与最终版 Test 索引一致。原冻结记录也保存了 Test 波形一致的结论。

| 集合 | 背景 | Single | Double | 总数 | Double 的 C0／C1／C2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 288 | 861 | 1728 | 2877 | 864／432／432 |
| Val | 71 | 216 | 432 | 719 | 216／108／108 |
| Test | 87 | 262 | 528 | 877 | 264／132／132 |

[最终 DATA_INDEX](data/DATA_INDEX.csv) · [录音归属表](RECORDING_SPLIT.csv) · [最终生成配置](data/CONFIG.json) · [原生成摘要](data/SUMMARY.json) · [基础版 DATA_INDEX](../ShipsEar_DataGen_Final_0dB/datasets/shipsear_pf_mb_rv_0db/DATA_INDEX.csv)

## 源码入口

[MB 交换入口](prepare_data.py)调用[共用交换实现](../shipsear_val_background_20261104/prepare_pf_swap.py)，对涉及交换的双目标复用已传播的接收组件并调用[混合和包络源码](../ShipsEar_DataGen_Final_0dB/generator/generate.py)。背景 E 从真实背景录音切窗。源池与接收组件路径查 [sources.json](../ShipsEar_DataGen_Final_0dB/metadata/sources.json)；最终的归属以本目录索引的 split 和源录音 ID 为准。

生成配置保留了可选 Single 训练包络的历史字段。冻结 Q0 读取存储波形，正式训练关闭人工 Single 包络。 [正式数据配置](../正式数据口径/config.json) · [正式训练加噪源码](../正式数据口径/train_data.py)
