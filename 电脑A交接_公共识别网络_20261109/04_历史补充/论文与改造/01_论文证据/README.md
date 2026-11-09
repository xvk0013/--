# 论文证据、图表与引用口径

## 1. 从哪张表开始

| 数据集 | 已核对运行数 | 六档错判行 | 双船交汇行 | 配对行 | Test 混淆矩阵行 |
| --- | ---: | ---: | ---: | ---: | ---: |
| ShipsEar_新MB | 19 | 228 | 684 | 612 | 2058 |
| DeepShip_PTT | 3 | 36 | 108 | 72 | 882 |

| 表 | 列含义和用途 | ShipsEar 新 MB | DeepShip PTT |
| --- | --- | --- | --- |
| 六档错判分解 | 每个模型、集合、信噪比一行。给出集合与数量正确数、真背景／Single／Double 分母和各方向错判 | [CSV](ShipsEar_新MB/六档错判分解.csv) | [CSV](DeepShip_PTT/六档错判分解.csv) |
| 双船交汇分组 | 仅在真 Double 中按 C0／C1／C2 分组，记录完全命中、数量正确但组合错误、漏成 Single 或背景 | [CSV](ShipsEar_新MB/双船交汇分组.csv) | [CSV](DeepShip_PTT/双船交汇分组.csv) |
| 同样本配对与保持率 | 按真实数量分组，记录双方独对、双方都对及双方都错。另存原始档共同判对样本在加噪后的保持数 | [CSV](ShipsEar_新MB/同样本配对与保持率.csv) | [CSV](DeepShip_PTT/同样本配对与保持率.csv) |
| 七集合混淆矩阵长表 | 真集合为行，预测集合为列。可按实验 ID 和信噪比透视成计数矩阵 | [CSV](ShipsEar_新MB/Test七集合混淆矩阵长表.csv) | [CSV](DeepShip_PTT/Test七集合混淆矩阵长表.csv) |

原始表示无额外 AWGN，仍保留存储波形的环境声。ShipsEar 的合法标签是背景、PF／MB／RV 和三种异类双船组合。DeepShip PTT 使用 Passengership／Tanker／Tug 的对应七集合。组合准确率要求整个集合一致。数量正确可以同时出现船型或双船组合错误。C0／C1／C2 分别表示全重叠、局部单方占优和两端由不同源占优。 [实验协议](../原始资料/research/shipsear_finalization_20261103/正式实验协议.md) · [源索引和逐模型来源](../03_复现核查/来源与已有交接清单.json)

配对表的“原始共同正确”只由该对模型各自的原始档预测确定。各噪声档使用同一队列，其分母写在 `common_original_correct_n`，保持率为该档保持正确数除以此分母。不同模型对的分母可能不同。两源混合会复用源录音，固定划分的切片数不能直接作为独立录音样本数。 [ShipsEar 录音敏感性及方法归属](../原始资料/research/shipsear_finalization_20261103/消融实验/README.md)

## 2. 供初稿和新机制讨论的错误读数

### ShipsEar 新 MB

| −10 dB Test | 完整集合判对 | 真 Double 判对 | Double→Single | Double→背景 | 数量对、双船组合错 | Single→Double | 背景→船 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ShortLocal-BN | 519／877 | 260／528 | 127 | 14 | 127 | 60 | 0 |
| ShortLocal-GN | 503／877 | 244／528 | 155 | 10 | 119 | 55 | 0 |
| GN-ADD | 490／877 | 230／528 | 181 | 2 | 115 | 51 | 0 |
| TResNet-LA 本地适配 | 437／877 | 345／528 | 21 | 0 | 162 | 199 | 56 |

相对等参数加法，GN 在这一档减少了 Double→Single，同时增加了 Double→背景。完整 Double 正确数的变化比单看数量方向更接近任务目标。TResNet-LA 的双船命中较高，同时出现更多真 Single 判双船和背景误报。讨论弱第二船型时，可以直接用这些并列指标说明取舍。 [逐档原始计数](ShipsEar_新MB/六档错判分解.csv)

### DeepShip PTT

| −10 dB Test | 完整集合判对 | 真 Double 判对 | Double→Single | Double→背景 | 数量对、双船组合错 | Single→Double | 背景→船 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PTT Q0 | 671／1250 | 203／600 | 191 | 2 | 204 | 71 | 2 |
| TResNet-LA 本地适配 | 677／1250 | 279／600 | 90 | 0 | 231 | 170 | 0 |

PTT 表中的 Q0 属于该数据集已训练的版本。与 ShipsEar 分表时，可比较相同的错误定义。两套数据的背景来源、船型与分母按各自协议描述。 [逐档原始计数](DeepShip_PTT/六档错判分解.csv)

## 3. 引用和方法归属

- **数据与评测。** 数据生成包负责 E99、源间 SIR、附加 AWGN、录音划分及背景生成的来源。固定 ShipsEar Test 在开发中多次使用，论文按录音隔离的开发基准表述。DeepShip PTT 按独立训练的另一数据集实现例表述。 [官方目录历史核查](../原始资料/research/shipsear_all_recording_metadata_audit_20261004/REPORT.md) · [91 号录音核查](../原始资料/research/shipsear_finalization_20261103/ShipsEar_91号录音核查.md)
- **模型组件。** 散射实现与 TabM 主干保留原出处，写清本项目首层交互、短窗路径和集合读出的实际改动。乘积与残差融合有已存档的先例。原始论文的不同划分、SIR／TIR／SNR 和不同准确率分母需各自注明。 [Q0 交互文献与对应改动](../原始资料/research/shipsear_finalization_20261103/Q0改造文献与对照.md) · [第三方归属说明](../../../03_论文写作/引用资料/1493_README.md)
- **相关工作与候选来源。** 阅读记录区分作者全文、出版社预览和作者摘要。该文档整理了本项目局部组合证据候选与 APA、LocalEvidence、H1、PairHead 的关系。当前外部基线仍按本地任务适配引用。 [11 月 8 日文献核对](../原始资料/research/shipsear_core_module_literature_20261108.md) · [历史近邻差异核查](../原始资料/research/ctcl_academic_assessment_20261025/文献重合与差异核查.md) · [与双分支框架较近的 CMM-DACTNet 原论文](https://www.sciencedirect.com/science/article/pii/S0029801826017749)
