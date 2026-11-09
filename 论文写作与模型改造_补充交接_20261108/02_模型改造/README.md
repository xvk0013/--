# 模型改造资料导航

## 1. 当前实现从哪改

ShipsEar GN 的短测保存了三个输入张量。批量为 32 时，波形散射为 32×3,730，长窗谱散射为 32×1,600，短窗 log-Mel 为 32×1×128×663。短窗 CNN 汇聚成 1,600 维，和另两部分拼成 6,930 维后进入公共识别网络。卷积内三处使用四组 GN，短窗汇聚后还有 BN1d。公共首层并行计算直通与门控乘积残差，TabM 成员各输出集合概率，推理对成员概率取均值。代码入口和封存权重随已有模型 ZIP 提供。 [GN 实际张量及更新核对](../原始资料/research/shipsear_finalization_20261103/runs/short_local_GN_smoke.json) · [BN 到 GN 的结构核对](../原始资料/research/shipsear_finalization_20261103/runs/short_local_GN_structure_check.json) · [已有模型 ZIP](D:/LSJ/Code/UART/final/data_gen_v8_single_physical/论文模型交接_20261108.zip)

| 要动的位置 | 现成对照或实现 | 讨论时可回答的问题 |
| --- | --- | --- |
| 固定时频或短窗 CNN | Q0-MB 对 ShortLocal-BN 是整段短窗替换，同时改窗长、谱表示、编码和归一化。BN 对 GN 隔离了卷积内归一化 | 先区分整个表征路径与归一化效果 |
| 长窗 | L1 去长窗，L2 改固定散射汇聚网格，L3 改可学习长窗频率编码，L4 调整 L3 的长窗归一化。它们的短窗参照都是 BN | 删减和替换效果在对应结构内解释 |
| 交互整条残差与算子 | GN-CAT 去整条残差，GN-ADD 以同参数加法替代逐元素乘积并同步替换门控输入。屏蔽一路同时去掉该路参与的交互 | 分开整条残差、融合算子及分支在完整模型中的贡献 |
| 动态门控 | 历史选择性交互已有分来源门控续训。GN-GLOBAL 仍是待验证条件项 | 具体到当前 GN 时，另核查按样本／成员变化的效应 |
| 共享船型与组合读出 | 旧划分 SE-Q0-Shared-13 已完成背景分数、三种船型分数和静态组合偏置的对照。它保留原数量边际监督 | 带基数偏置或输入依赖组合证据的设计需单独界定 |
| 局部成分证据 | 新 MB 的 LocalEvidence 在汇聚后的双窗网格取证。旧 H1 做配对移源／交换。旧 PairHead 读汇聚后的类别信息并使用原数量路由 | 最近文献笔记提出汇聚前针对具体集合竞争取证。该项归为待验证设计 |
| 时序外积 | 同段双线性替换整条首层交互残差，未达预设 Val 采用线 | 时间对应本身仍缺同参数的全时间对照 |
| 噪声方向与表征解耦 | 已有冻结探针、噪声×真实加源因素分解、局部方向恢复、配对间隔、CUMI 和环境解耦 | 区分可读性诊断、短续训与结构对照，关注真实弱目标和噪声误报之间的取舍 |

## 2. 重要原件

- **近期前端与归一化。** [L1/L2 完整报告](../原始资料/research/shipsear_finalization_20261103/Q0长短窗互补L1_L2完整训练报告.md) · [L3 完整报告](../原始资料/research/shipsear_finalization_20261103/长窗频率局部编码L3完整训练报告.md) · [L4 与诊断](../原始资料/research/shipsear_finalization_20261103/长窗归一化诊断与L4完整训练报告.md)
- **首层、读出与局部证据。** [当前消融原索引](../原始资料/research/shipsear_finalization_20261103/消融实验/README.md) · [同段双线性报告](../原始资料/research/shipsear_finalization_20261103/ShortLocal-GN同段双线性完整训练报告.md) · [共享船型读出报告](../原始资料/research/shipsear_finalization_20261103/Q0共享船型读出完整训练报告.md) · [局部证据报告](../原始资料/research/shipsear_finalization_20261103/Q0局部船型证据迁移报告.md) · [LocalEvidence 实现](../原始资料/research/shipsear_finalization_20261103/local_evidence.py) · [PairHead 实现](../原始资料/research/shipsear_pair_evidence_20261022/pair_evidence_model.py)
- **研发失败与方法差异。** [历史失败证据矩阵](../原始资料/research/innovation_failure_audit_20261026/历史失败证据矩阵.md) · [T1 以来的训练链审计](../原始资料/research/shipsear_finalization_20261103/T1至今训练审计与复验优先级.md) · [噪声／真实加源诊断](../原始资料/research/shipsear_noise_target_factorial_20261102/阶段结论.md) · [方向重叠与联合恢复](../原始资料/research/shipsear_local_noise_path_20261102/结果解读_方向重叠与联合恢复.md) · [移源与交换报告](../原始资料/research/shipsear_intervention_set_20261027/实验报告.md)
- **当前待讨论的问题和文献。** [11 月 8 日候选核对](../原始资料/research/shipsear_core_module_literature_20261108.md) · [机制候选审稿评估](../原始资料/research/academic_method_exploration_20261026/核心方法候选与审稿评估.md)

## 3. 将现有证据接到下一轮

对照最好对应一个具体干预，例如改融合算子、改用局部证据或改组合读出。登记相同的输入、参数规格、训练起点和评价规则，明确哪一项误判应减少、哪些误报要同时报告。源级干预若重提，先和已有移源、等能量加源、加噪及探针记录核对。论文核心的确定和实际训练仍由新窗口讨论。 [可复用训练登记和验收](../03_复现核查/README.md)
