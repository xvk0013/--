# DeepShip PTT 论文数据视图

版本 deepship_multitarget_0db_energy80_seen_ship_ptt_v1。从已经生成的四船型 SeenShip 父数据读取存储波形，保留 Passengership、Tanker、Tug 的单船型与异类双船型集合，并保留背景。三种船型、七种集合，每段最多两个同时目标。波形来自父数组的原行。

## 生成链

1. DeepShip 对原录音按可靠事件或录音分组，划定集合，再做 5 秒核心初筛。SeenShip 正式源池采用 Train 能量前 80%、Val／Test 前 40%，共用至少 4 个 1 秒子段通过持续性门槛、P90 参考减 6 dB 的规则。完整技术门槛、频带能量与线谱／宽带条件见[初筛实现](../DeepShip_Prescreen/screening.py)及[80% 配置](../DeepShip_Prescreen/outputs/energy80_v1/full/CONFIG.json)、[40% 配置](../DeepShip_Prescreen/outputs/strict_v1/full/CONFIG.json)。
2. [SeenShip 划分脚本](../SeenShip_录音划分/prepare.py)生成[录音划分表](../SeenShip_录音划分/PARTITION.csv)。数据按录音组隔离，协议允许已见船舶的不同录音进入不同集合。生成器以源索引在集合内选源和配对，建立 16 kHz E99 接收组件，生成原始混合与 Wenz 背景。 [正式父配置](../Energy80_SeenShip父数据/CONFIG.json) · [父数据生成报告](../Energy80_SeenShip父数据/REPORT.md)
3. 正式 [PTT 数据读取器](../读取协议/Q0_DeepShip/data.py)按合法集合筛选、重编号，保留父行号作为数据和评价噪声依据。包内 [导出脚本](export_ptt_index.py)从父索引直接导出下表对应的 CSV，已经逐条与真实读取器对齐。该 CSV 的 array_index、label_index 和 target 为父数据字段，追加的 ptt_array_index、ptt_label_index 和 ptt_target 为 PTT 字段。ptt_array_index 表示过滤后的顺序；读取父波形和复现噪声使用 array_index。

## 最终数量

| 集合 | Wenz 背景 | Single | Double | 总数 | Double 的 C0／C1／C2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 1200 | 2700 | 3600 | 7500 | 1800／900／900 |
| Val | 200 | 450 | 600 | 1250 | 300／150／150 |
| Test | 200 | 450 | 600 | 1250 | 300／150／150 |

PTT 实际纳入 500 个源录音组，Train／Val／Test 分别为 350／76／74，三个集合的源组无交集。各类别数量与逐组归属见下列索引。

[PTT DATA_INDEX](DATA_INDEX.csv) · [PTT 录音与组归属](RECORDING_SPLIT.csv) · [PTT 汇总](SUMMARY.json) · [父数据 DATA_INDEX（本机出处）](../../../../90_本机路径.md#r019) · [父接收组件缓存索引](../Energy80_SeenShip父数据/components/STATE.json) · [PTT 正式配置](../读取协议/Q0_DeepShip/config_ptt_seen_ship.json)
