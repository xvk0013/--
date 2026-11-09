# 复现入口、验收和来源

## 1. 记录从哪里读

| 记录 | 用途 | 入口 |
| --- | --- | --- |
| 正式协议与参数模板 | 核对训练起点、六档噪声、Val 排序、损失和分组预算。具体一次训练优先读它的解析配置。模板 `正式实验配置.json` 在原模型 ZIP 的 GN 快照内 | [正式实验协议](../原始资料/research/shipsear_finalization_20261103/正式实验协议.md) · [原模型 ZIP](D:/LSJ/Code/UART/final/data_gen_v8_single_physical/论文模型交接_20261108.zip) |
| 正式阶段全登记 | 按实验 ID 查看数据版本、初始化、所选轮、配置与结果路径。包含旧划分、续训、同划分完整训练等不同阶段 | [实验登记表](../原始资料/research/shipsear_finalization_20261103/实验登记表.csv) |
| 原始文件与前包索引 | 列出每份实际复制文件的执行机来源、本文件夹内的位置，以及已经在旧包中的条目。派生表的源预测和数据索引也在这里 | [来源与已有交接清单](来源与已有交接清单.json) |
| 执行前的轻量验收 | 核对实际张量尺寸、短窗参数是否更新、优化器恢复后下一步是否一致，以及 GN 与 BN 初值一致性 | [GN 单批验收](../原始资料/research/shipsear_finalization_20261103/runs/short_local_GN_smoke.json) · [GN 结构核对](../原始资料/research/shipsear_finalization_20261103/runs/short_local_GN_structure_check.json) · [同段双线性验收](../原始资料/research/shipsear_finalization_20261103/runs/short_local_GN_Bilinear_smoke.json) |
| 数据与跨数据集的记录 | 保留录音划分的历史和当前版本，PTT 单列登记 | [历史重划分评估](../原始资料/research/shipsear_repartition_assessment_20261029/重划分评估.md) · [DeepShip PTT 登记](../原始资料/DeepShip执行目录/07_论文实验/实验登记表.csv) |

## 2. 按原入口辨认已做实验

ShipsEar 的研究训练脚本在已有模型 ZIP 的 GN 快照内。已登记的短窗路径、归一化和融合版本通过下列原参数区分。参数所用配置、数据位置和运行目录以实际解析配置为准。

| 原选项 | 原有实验用途 |
| --- | --- |
| `--short-local` | ShortLocal-BN 完整短窗路径 |
| `--short-local-long L1`、`L2`、`L3` | 三个长窗对照 |
| `--long-record-normalization` | 在 L3 上检查长窗归一化 |
| `--short-record-normalization` | ShortLocal-GN 完整版 |
| `--short-gn-cat`、`--short-gn-add`、`--short-gn-mask` | GN 正式消融 |
| `--short-gn-bilinear` | GN 同段双线性候选 |

短测和正式训练各有各的运行记录。重读历史代码时，续训与从构造初值启动分开登记，模型、数据及合法集合的映射使用同一实验 ID 的文件。统计脚本原件可用于核查字段或改造，运行旧脚本前按相应报告核对它引用的数据和依赖目录。 [已有模型 ZIP](D:/LSJ/Code/UART/final/data_gen_v8_single_physical/论文模型交接_20261108.zip) · [正式数据与训练协议](../原始资料/research/shipsear_finalization_20261103/正式实验协议.md)

## 3. 本次派生统计怎样核对

逐档匹配封存预测中的样本 ID、真实标签及交汇条件，再用各模型的 `count_map` 将集合映射为数量。全部 Test 综合准确率与各自 `results.json` 一致。错判分解之和等于对应样本数，混淆矩阵总数和对角线与预测相符。外部模型的来源是正式交接 ZIP 中的相应预测。 [逐模型计算来源](来源与已有交接清单.json) · [图表 CSV](../01_论文证据/README.md)
