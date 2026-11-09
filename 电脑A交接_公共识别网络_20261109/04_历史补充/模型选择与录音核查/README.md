# 电脑 A 实验补充核对

> 本文为 11 月 8 日归档，标题沿用当时的电脑称谓。 [当前分工](../../00_开始这里.md)

2026-11-08。数字来自封存 JSON、逐样本预测 NPZ 和数据索引。逐录音与配对表均为本次离线重算。**新增训练为 0。** “无记录”指本次核对的项目资料里缺少对应决定、运行或排程。

## P0-1 GN 与 BN 的模型身份和选择依据

**当前执行主线使用 GN 的封存权重作为消融和研究参照，论文最终主模型无记录。**

归一化报告在该轮对照结束时写的是“ShortLocal 继续作为主线”，并把 GN 留作候选。后来的 GN 冻结说明和统一收尾路线明确把 GN 用作研究参照。**两份文件都保留各自的阶段含义。对现行实验参照，以后出的冻结说明及收尾路线为准。论文定稿的 GN／BN 选择和选择理由均无记录。** 11 月 4 日的 Q0 冻结是更早的版本记录。 [当轮 BN／GN 结论（第 58 行）](../../03_论文写作/自研参照/报告/短窗GroupNorm完整训练报告.md) · [11 月 8 日执行路线（第 3 行）](../论文与改造/原始资料/research/shipsear_finalization_20261103/统一收尾路线.md) · [GN 冻结说明](../../03_论文写作/自研参照/ShortLocal-GN冻结说明.md)

**预先登记的 GN／BN 跨模型裁决标准，没有。** 记录里有各自的 Val 选轮规则。

| 项目 | 预先登记与实际用途 |
| --- | --- |
| Val 规则 | 预先用于各模型的轮次封存。依次为六档单双均衡分、六档综合均值、六档 Single／Double 均值中较低者、Val 原始综合。完全同分取较早轮次 |
| 逐录音稳定性 | 逐录音结果属于归档指标。归一化报告用 Val 背景录音作事后分析。预设稳定性门槛无记录 |
| 配对胜率 | 两版同样本配对。胜率是本次用封存预测重算的描述量，预先采用阈值无记录 |
| 背景鲁棒性 | 背景 ACC 是预定的报告指标。Test 背景全对与 Val 背景取舍是报告中的结果观察，用于 GN／BN 取舍的预定门槛无记录 |
| 原始档 | Val 原始综合是预定的第四排序项。短窗报告记有 Test 原始综合目标，GN 超过该目标是事后观察。用它作 GN／BN 跨模型采用条件无记录 |

[正式 Val 规则与封存协议（第 49 行）](../../03_论文写作/自研参照/报告/正式实验协议.md) · [原 ShortLocal 目标与报告](../论文与改造/原始资料/research/shipsear_finalization_20261103/runs/SE-Q0-ShortLocal-13/实验报告.md)

准确率单位为 %。背景六档顺序为原始、10、5、0、−5、−10 dB。GN 的 5 dB 与 0 dB 综合也高于 BN，完整档位列在 CSV。

| 模型及实验 ID | Val 六档单双均衡 | Val 综合均值 | Test 六档均值 | Test 原始 | Test −5 dB | Test −10 dB | Test 背景六档 | 参数量 | 封存轮 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BN · SE-Q0-ShortLocal-13 | 66.51 | 64.14 | 76.09 | 84.61 | 71.04 | 59.18 | 100.00／100.00／98.85／96.55／100.00／100.00 | 2,721,537 | 32 |
| GN · SE-Q0-ShortLocal-GN-13 | 64.51 | 61.73 | 75.56 | 85.18 | 68.07 | 57.35 | 100.00／100.00／100.00／100.00／100.00／100.00 | 2,721,537 | 29 |

[两版全部封存指标 CSV](P0-1_GN_BN封存指标.csv) · BN [结果](../../03_论文写作/自研参照/ShipsEar/SE-Q0-ShortLocal-13/results.json) · [配置](../../03_论文写作/自研参照/ShipsEar/SE-Q0-ShortLocal-13/config_resolved.json) · [封存](../../03_论文写作/自研参照/ShipsEar/SE-Q0-ShortLocal-13/selection_sealed.json) · GN [结果](../../03_论文写作/自研参照/ShipsEar/SE-Q0-ShortLocal-GN-13/results.json) · [配置](../../03_论文写作/自研参照/ShipsEar/SE-Q0-ShortLocal-GN-13/config_resolved.json) · [封存](../../03_论文写作/自研参照/ShipsEar/SE-Q0-ShortLocal-GN-13/selection_sealed.json)

按样本 ID、标签和交汇条件配对，整条目标集合正确为赢。GN 独对和 BN 独对为有胜负的样本，双方都对或都错为平局。每条录音计入它参与的样本，双源混合可分别计入两条录音。下表汇总全部样本的六档视图，逐录音表保留每档的样本数、两版正确数、各自独对、平局及净胜差。六档重复使用同一批样本。

| 集合 | 源录音数 | 各档样本数 | 六档预测视图数 | GN 独对 | BN 独对 | GN 在胜负样本中占比 |
| --- | --- | --- | --- | --- | --- | --- |
| Val | 13 | 719 | 4,314 | 261 | 365 | 41.69% |
| Test | 10 | 877 | 5,262 | 292 | 320 | 47.71% |

[Val 与 Test 每条录音六档配对 CSV](P0-1_GN_BN逐录音配对.csv) · [全样本逐档配对 CSV](P0-1_GN_BN全样本配对.csv) · [两版原始逐录音 JSON 的逐档路径](P1-3_结果文件清单.json)

## P0-2 首层替换候选的失败记录

**同段双线性池化**保留标准直通路径，将完整 GN 首层的门控乘积残差换成两路在四个粗时间格内投影、同格外积、取均值及线性投影后的残差。每路投影秩为 16，外积为 16×16。 [实现（本机出处）](../../90_本机路径.md#r462)

| 项目 | 封存记录 |
| --- | --- |
| 实验 ID | SE-Q0-ShortLocal-GN-Bilinear-13 |
| 数据版本及 Train／Val／Test | shipsear_mb21_33_train_27_50_val_20261104 · 2,877／719／877 |
| 训练起点与种子 | 模型构造初值，训练种子 13，评价噪声种子 13 |
| 训练视图 | stored，原始及五档加噪概率 50%／10%／10%／10%／10%／10% |
| 优化 | AdamW，学习率 0.0003，权重衰减 0.0001，梯度裁剪 5，训练／评价批量 32／64 |
| 损失 | 集合加权交叉熵，数量边际系数 0.1 |
| 预算与实际封存 | 40–80 轮，patience 10，有效刷新 0.2 pp。实际 40 轮，Val 封存第 28 轮 |
| 可训练参数 | 2,348,736，完整 GN 为 2,721,537 |

Test 单元格为候选 ACC（相对完整 GN 的差值）。前者单位 %，括号内单位百分点。

| Test 条件 | 综合 | Single | Double | 背景 | 数量 |
| --- | --- | --- | --- | --- | --- |
| 原始 | 86.66（+1.48） | 91.60（+2.29） | 82.01（+1.33） | 100.00（0.00） | 93.61（+0.11） |
| 10 dB | 82.90（−0.91） | 85.11（−0.76） | 78.98（−1.14） | 100.00（0.00） | 91.79（−0.23） |
| 5 dB | 80.16（−2.05） | 83.59（−0.76） | 75.19（−3.03） | 100.00（0.00） | 90.08（−1.37） |
| 0 dB | 75.48（−1.25） | 77.48（−3.05） | 70.45（−0.57） | 100.00（0.00） | 87.12（−1.37） |
| −5 dB | 67.39（−0.68） | 70.61（−3.44） | 60.42（+0.57） | 100.00（0.00） | 82.44（−0.34） |
| −10 dB | 55.87（−1.48） | 55.73（−9.92） | 48.67（+2.46） | 100.00（0.00） | 74.12（−0.11） |

| 预先登记的检查项 | 完整 GN | 同段双线性 | 通过 |
| --- | --- | --- | --- |
| Val 0 dB 与 −5 dB 的 Double→Single 应减少 | 232 | 241 | 否 |
| 同两档完整 Double 判对数应增加 | 419 | 427 | 是 |
| 同两档 Double 净增益应覆盖 Single 净损失 | — | Double +8，Single -15 | 否 |
| Val 六档单双均衡分应至少持平 | 64.51% | 63.70% | 否 |
| Val 原始和 −10 dB 各自最多下降 1 pp | 参照 | +1.81／−0.14 pp | 是 |

**归类为未达到预定 Val 采用门槛。** 封存文件记录完整训练并按 patience 结束。因训练失稳或参数／成本超限被淘汰的依据无记录。同参数的全时间配对对照无记录，因此时间对应机制的独立结论也无记录。 [预定采用条件（本机出处）](../../90_本机路径.md#r463) · [采用判断（第 29 行）](../../03_论文写作/自研参照/报告/ShortLocal-GN同段双线性完整训练报告.md) · [封存配对统计](../../03_论文写作/自研参照/ShipsEar/SE-Q0-ShortLocal-GN-Bilinear-13/paired_analysis.json) · [六档 CSV](P0-2_同段双线性六档.csv) · [结果](../../03_论文写作/自研参照/ShipsEar/SE-Q0-ShortLocal-GN-Bilinear-13/results.json) · [配置](../../03_论文写作/自研参照/ShipsEar/SE-Q0-ShortLocal-GN-Bilinear-13/config_resolved.json) · [封存](../../03_论文写作/自研参照/ShipsEar/SE-Q0-ShortLocal-GN-Bilinear-13/selection_sealed.json)

## P0-3 最终模型与实验时间表

| 问题 | 已有记录 |
| --- | --- |
| 论文最终模型的预计冻结日 | 无记录 |
| 最终候选清单 | 无记录。现有身份是 GN 研究参照、BN 历史归一化对照、Q0-MB 历史同划分参照。同段双线性已完成且未采纳 |
| 每周可完整训练次数 | 无记录 |
| 多种子 | 已有路线写先跑种子 13，必要时在完整模型与关键对照上补 42／2026。GN、BN、Q0-MB 与 PTT Q0 各只有一份种子 13 的封存运行。最终模型与具体对照、每组次数和排期无记录 |
| 最终结构的 DeepShip PTT 训练 | 有条件安排。若论文采用同一结构跨数据集验证，在结构选定后在 PTT 从初值完整训练一次。日历日期和正式排期无记录。现有正式自研运行属于 PTT Q0，ShortLocal 或 ShortLocal-GN 在 PTT 上的完整运行无记录 |
| 当前可用写法 | 两个已登记实现例，ShipsEar 用 ShortLocal-GN，DeepShip PTT 用 Q0。若后续同一最终结构在两套数据各自完整训练，再描述为同一结构的跨数据集验证 |

[种子约定与当前路线](../论文与改造/原始资料/research/shipsear_finalization_20261103/统一收尾路线.md) · [DeepShip 条件计划](../../03_论文写作/自研参照/ShortLocal-GN冻结说明.md#3-模型变更后各结果怎样处理) · [PTT 正式实验登记](../论文与改造/原始资料/DeepShip执行目录/07_论文实验/实验登记表.csv)

以下是实际运行的逐轮累计计时，包含每轮训练和 Val，不以这些记录推算未来算力或周容量。

| 已完成运行 | 实际训练轮 | 封存轮 | 已记录逐轮累计（分钟） | 来源 |
| --- | --- | --- | --- | --- |
| SE-Q0-MBTrainValSwap-13 | 50 | 40 | 13.84 | [封存 elapsed_seconds](../../03_论文写作/自研参照/ShipsEar/SE-Q0-MBTrainValSwap-13/selection_sealed.json) |
| SE-Q0-ShortLocal-13 | 42 | 32 | 11.51 | [封存 elapsed_seconds](../../03_论文写作/自研参照/ShipsEar/SE-Q0-ShortLocal-13/selection_sealed.json) |
| SE-Q0-ShortLocal-GN-13 | 40 | 29 | 11.10 | [封存 elapsed_seconds](../../03_论文写作/自研参照/ShipsEar/SE-Q0-ShortLocal-GN-13/selection_sealed.json) |
| SE-Q0-ShortLocal-GN-Bilinear-13 | 40 | 28 | 10.69 | [封存 elapsed_seconds](../../03_论文写作/自研参照/ShipsEar/SE-Q0-ShortLocal-GN-Bilinear-13/selection_sealed.json) |
| DS-Q0-PTT-SeenShip-13 | 40 | 29 | 28.16 | [完整 history.json 的 seconds 求和](../../03_论文写作/自研参照/DeepShip_PTT/DS-Q0-PTT-SeenShip-13/history.json) |

## P0-4 确认集与重复划分库存

录音范围按本地归档的官方目录和生成器正式主船型映射核对。PF 对应 Passenger ferry，MB 对应 Motorboat，RV 对应 Ro-Ro。当前使用以新 MB 索引两路源录音字段为准。既往使用以早期四分类封存逐片预测汇总中的录音记录为证据。

| 任务主船型 | 全量录音数 | 新 MB 已用编号 | 新 MB 未用编号（数量） | 早期已评价录音数 | 无既往使用的编号（数量） |
| --- | --- | --- | --- | --- | --- |
| PF | 30 | 6、7、8、9、10、11、12、13、14、17、32、34、35、36、38、40、41、42、43、53、54、55、59、60、61、62、63、64、65、67 | 无（0） | 30 | 无（0） |
| MB | 13 | 21、26、27、33、39、45、50、51、52、70、72、77、79 | 无（0） | 13 | 无（0） |
| RV | 5 | 18、19、20、58、78 | 无（0） | 5 | 无（0） |

[官方 90 条目录快照](../论文与改造/原始资料/research/shipsear_all_recording_metadata_audit_20261004/official_catalogue.json) · [项目主船型映射](../../03_论文写作/数据与协议/01_ShipsEar/ShipsEar_DataGen_Final_0dB/config.json) · [早期 78 条目标录音的预测汇总](../论文与改造/原始资料/research/shipsear_all_recording_metadata_audit_20261004/baseline_all_recordings_ranked.csv) · [新 MB 实际数据索引（本机出处）](../../90_本机路径.md#r408) · [本次逐录音库存 CSV](P0-4_ShipsEar录音库存.csv)

**现有库存无法组成三类未参与研发的整录音确认集。** PF、MB、RV 的任务录音均已用过。每类另留多少条及新的确认集来源无记录。

已有录音级重划分先例。下表是从现成索引核对出的布局库存，逐一核验源录音在各布局内只属于一个集合。7 份布局对应 4 种不同的 Test 源录音名单，名单之间可以重叠。A21 共享 O0 的 Train，B78 独立训练；它们属于历史开发方案。新模型正式重复划分的套数、每套录音配额及预计训练成本均无记录。

| 已归档布局 | 对应运行或方案编号 | Train 录音数 | Val 录音数 | Test 录音数 |
| --- | --- | --- | --- | --- |
| 新 MB | SE-Q0-MBTrainValSwap-13 | 37 | 13 | 10 |
| 历史 O0 | O0 | 37 | 13 | 10 |
| 历史 A21 | A21 | 37 | 12 | 11 |
| 历史 B78 | B78 | 37 | 12 | 11 |
| 背景交换 | SE-Q0-BackgroundSwap-13 | 37 | 13 | 10 |
| 背景与 PF65/53 交换 | SE-Q0-RecordingSwap-13 | 37 | 13 | 10 |
| 背景与 PF65/63 交换 | SE-Q0-RecordingSwap63-13 | 37 | 13 | 10 |

| 历史重划分运行 | 封存记录中的耗时 |
| --- | ---: |
| O0／A21／B78 旧开发对照整轮 | 72.97 分钟 |

[各布局的逐类录音编号与样本数 CSV](P0-4_已归档录音级布局.csv) · [历史录音级方案与共用训练规则](../论文与改造/原始资料/research/shipsear_repartition_trial_20261029/PROTOCOL.md) · [历史运行计时字段](../论文与改造/原始资料/research/shipsear_repartition_trial_20261029/runs/seed13/results.json)

**现 Test 已反复用于研发。** 基础旧划分和新 MB 的 Test 索引逐行一致，换的是 Train／Val。Q0-MB、ShortLocal、GN、GN 的消融和同段双线性都在这一固定 Test 上评价过。归一化对照报告明确把 Test 的背景和强噪声 Double 结果写入阶段取舍。历史重划分的探索协议还曾登记使用 Test 门槛决定开发取舍。正式各运行按 Val 封存轮次，当前 Test 应按已参与研发的录音级基准报告。 [新 MB 重建记录](../../03_论文写作/数据与协议/01_ShipsEar/shipsear_mb_train_val_20261104/REPORT.md) · [正式协议的 Test 角色（第 67 行）](../../03_论文写作/自研参照/报告/正式实验协议.md) · [BN／GN 阶段判断（第 58 行）](../../03_论文写作/自研参照/报告/短窗GroupNorm完整训练报告.md) · [历史探索性采用条件（第 25 行）](../论文与改造/原始资料/research/shipsear_repartition_trial_20261029/PROTOCOL.md)

## P1-1 短窗分支的单因素对照

**严格的 CNN 单因素对照无记录，最接近的完整对照已经有。** Q0-MB 与 ShortLocal-BN 共用长窗和公共 TabM 结构、划分、损失、加噪及 Val 排序，各自从构造初值完整训练。短窗替换还改变了窗长、谱表示和短窗输出归一化，论文可称为**整段短窗路径替换**。团队同意启动更严格对照的记录及排期均无记录。 [短窗试验方案与执行报告](../论文与改造/原始资料/research/shipsear_finalization_20261103/runs/SE-Q0-ShortLocal-13/实验报告.md)

| 模型 | 实验 ID | 短窗路径 | 参数量 | Val 封存轮 | 原始文件 |
| --- | --- | --- | --- | --- | --- |
| Q0-MB | SE-Q0-MBTrainValSwap-13 | 固定 STFT 对数功率图→二维散射→网格统计→Train 固定特征标准化 | 2,713,089 | 40 | [结果](../../03_论文写作/自研参照/ShipsEar/SE-Q0-MBTrainValSwap-13/results.json) · [配置](../../03_论文写作/自研参照/ShipsEar/SE-Q0-MBTrainValSwap-13/config_resolved.json) · [封存](../../03_论文写作/自研参照/ShipsEar/SE-Q0-MBTrainValSwap-13/selection_sealed.json) |
| ShortLocal-BN | SE-Q0-ShortLocal-13 | 短窗 log-Mel→可学习局部 CNN→网格统计→短窗输出 BN1d | 2,721,537 | 32 | [结果](../../03_论文写作/自研参照/ShipsEar/SE-Q0-ShortLocal-13/results.json) · [配置](../../03_论文写作/自研参照/ShipsEar/SE-Q0-ShortLocal-13/config_resolved.json) · [封存](../../03_论文写作/自研参照/ShipsEar/SE-Q0-ShortLocal-13/selection_sealed.json) |

单元格依次为综合／Single／Double／背景／数量 ACC，单位为 %。

| Test 条件 | Q0-MB | ShortLocal-BN |
| --- | --- | --- |
| 原始 | 87.91／92.37／83.71／100.00／94.18 | 84.61／87.79／80.49／100.00／92.93 |
| 10 dB | 84.15／84.35／81.44／100.00／90.76 | 84.26／85.50／81.06／100.00／90.88 |
| 5 dB | 80.96／82.06／77.27／100.00／89.40 | 81.30／85.11／76.52／98.85／89.85 |
| 0 dB | 74.23／77.10／68.56／100.00／85.52 | 76.17／82.82／69.51／96.55／86.43 |
| −5 dB | 63.28／68.70／54.55／100.00／77.54 | 71.04／77.86／62.88／100.00／82.33 |
| −10 dB | 46.52／65.27／28.41／100.00／62.94 | 59.18／65.65／49.24／100.00／76.62 |

[短窗路径全部指标 CSV](P1-1_短窗路径六档.csv)

## P1-2 导师／团队意见

| 问题 | 一句话结论 |
| --- | --- |
| 接受已有短窗局部 CNN 作为核心，或要求全新模块 | **无记录。** 本机现有交接、协议和实验记录中没有对应导师或团队决定 |
| 带三种船型证据、三种组合偏置及三种基数偏置的读出 | **初步意见无记录。** 旧划分 `SE-Q0-Shared-13` 实现的是背景分数、船型分数与组合偏置，指定的基数偏置方案和团队意见无记录 |

[历史共享船型读出报告](../论文与改造/原始资料/research/shipsear_finalization_20261103/Q0共享船型读出完整训练报告.md) · [已实现的读出代码（第 198 行）](../../03_论文写作/引用资料/1494_train_formal.py)

## P1-3 结果文件清单

| 模型及实验 ID | 版本状态 | 结果／封存／解析配置 | 六档逐样本 | 逐录音 |
| --- | --- | --- | --- | --- |
| SE-Q0-ShortLocal-GN-13 | 当前研究参照冻结版，论文最终主模型无记录 | [结果](../../03_论文写作/自研参照/ShipsEar/SE-Q0-ShortLocal-GN-13/results.json) · [配置](../../03_论文写作/自研参照/ShipsEar/SE-Q0-ShortLocal-GN-13/config_resolved.json) · [封存](../../03_论文写作/自研参照/ShipsEar/SE-Q0-ShortLocal-GN-13/selection_sealed.json) | [运行目录](../../03_论文写作/自研参照/ShipsEar/SE-Q0-ShortLocal-GN-13) 中 Val／Test 各六个 NPZ | [运行目录](../../03_论文写作/自研参照/ShipsEar/SE-Q0-ShortLocal-GN-13) 中 Val／Test 各六个 `*_recordings.json` |
| SE-Q0-ShortLocal-13 | 历史版，短窗归一化对照 | [结果](../../03_论文写作/自研参照/ShipsEar/SE-Q0-ShortLocal-13/results.json) · [配置](../../03_论文写作/自研参照/ShipsEar/SE-Q0-ShortLocal-13/config_resolved.json) · [封存](../../03_论文写作/自研参照/ShipsEar/SE-Q0-ShortLocal-13/selection_sealed.json) | [运行目录](../../03_论文写作/自研参照/ShipsEar/SE-Q0-ShortLocal-13) 中 Val／Test 各六个 NPZ | [运行目录](../../03_论文写作/自研参照/ShipsEar/SE-Q0-ShortLocal-13) 中 Val／Test 各六个 `*_recordings.json` |
| SE-Q0-MBTrainValSwap-13 | 历史版，同划分 Q0 独立发布 | [结果](../../03_论文写作/自研参照/ShipsEar/SE-Q0-MBTrainValSwap-13/results.json) · [配置](../../03_论文写作/自研参照/ShipsEar/SE-Q0-MBTrainValSwap-13/config_resolved.json) · [封存](../../03_论文写作/自研参照/ShipsEar/SE-Q0-MBTrainValSwap-13/selection_sealed.json) | [运行目录](../../03_论文写作/自研参照/ShipsEar/SE-Q0-MBTrainValSwap-13) 中 Val／Test 各六个 NPZ | [运行目录](../../03_论文写作/自研参照/ShipsEar/SE-Q0-MBTrainValSwap-13) 中 Val／Test 各六个 `*_recordings.json` |
| DS-Q0-PTT-SeenShip-13 | PTT Q0 封存版，当前 PTT 已完成自研结果 | [结果](../../03_论文写作/自研参照/DeepShip_PTT/DS-Q0-PTT-SeenShip-13/results.json) · [配置](../../03_论文写作/自研参照/DeepShip_PTT/DS-Q0-PTT-SeenShip-13/config_resolved.json) · [封存](../../03_论文写作/自研参照/DeepShip_PTT/DS-Q0-PTT-SeenShip-13/selection_sealed.json) | [运行目录](../../03_论文写作/自研参照/DeepShip_PTT/DS-Q0-PTT-SeenShip-13) 中 Val／Test 各六个 NPZ | 原始文件无记录 · [本次按索引重算](P1-3_PTT_Q0逐录音.csv) |

[四套模型每一个 JSON 与 NPZ 的绝对路径和外部 ZIP 条目](P1-3_结果文件清单.json)。GN 的快照文件已逐个与正式运行核对。PTT Q0 补算表按 PTT 的样本 ID、PTT 标签及交汇条件和封存预测对齐。合成背景无源录音 ID，背景总体正确率仍使用封存结果。

| 外部基线实验 ID | 已回传包内的正式逐档 NPZ |
| --- | --- |
| SE-EXT-SFAC32-MB-13 | 12／12 |
| SE-EXT-TRLA-MB-13 | 12／12 |
| SE-EXT-SW-MB-13 | 12／12 |
| SE-EXT-RCMOE-MB-13 | 12／12 |
| DS-EXT-SFAC32-PTT-13 | 12／12 |
| DS-EXT-TRLA-PTT-13 | 12／12 |

六个外部模型的正式 Val／Test 六档 NPZ 均已在回传包内，已核对样本 ID、标签和交汇条件。 [原外部对比包](../../03_论文写作/外部对比/00_开始这里.md)

## P0 完成度

| 项目 | 已提供 | 缺失与原因 |
| --- | --- | --- |
| P0-1 | 两份阶段决定及当前研究参照身份，选轮标准，封存指标，Val／Test 的逐录音与全样本配对 | 最终论文主模型、GN 对 BN 的跨模型预定标准及最终采用理由无记录 |
| P0-2 | 同段双线性结构、配置、封存轮、六档五项指标及相对 GN 的差值，逐项采用判断 | 同参数全时间配对对照无记录，时间对应机制的独立结论未形成 |
| P0-3 | 各已完成运行的计时、条件种子约定、PTT Q0 状态与条件计划 | 最终候选清单、冻结日、周容量、最终多种子及 PTT 运行排期无记录 |
| P0-4 | 全量官方录音盘点、任务三类已用编号、已有录音级布局库存和 Test 开发使用范围 | 三类剩余未用录音均为零。新确认集来源、正式重复划分的计划与预计成本无记录 |
