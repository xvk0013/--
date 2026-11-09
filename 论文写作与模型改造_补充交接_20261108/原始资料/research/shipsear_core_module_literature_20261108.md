# 水声多目标与低信噪比识别文献核对

核对日期为 2026-11-08。阅读范围为四篇原论文或作者全文、三篇出版社预览及一篇作者摘要。分区采用 2024 JIF 的 JCR 学科四分位。Ocean Engineering 在 Engineering, Ocean 为 Q1，Expert Systems with Applications 在 Computer Science, Artificial Intelligence 为 Q1，JASA 在 Acoustics 为 Q2。中科院分区另有口径。 [JCR 2024 学科分区表，2025 年 7 月更新](https://library.utem.edu.my/phocadownload/Research-Resources/JCR2024%20Updated%20July2025.pdf)

## 多目标和船声干扰

| 论文与阅读范围 | 方法和评估 | 对当前项目的含义 |
| --- | --- | --- |
| Sun、Wang，JASA，2022。原论文 §II–III 及作者代码 | 从 ShipsEar 取背景和三种船型，叠加 200 ms 片段。比较波形、复数或幅值 STFT、log-Mel、MFCC 以及实数或复数 CNN。八种合法标签配置的最高完整集合准确率 95.21%，包含同型重复的二十种配置为 51.41%。作者代码按组合随机切分样本。 | 单通道识别未知数量和完整组合已有直接先例。本文以切片划分、短片段和不同标签配置评估。 [论文 DOI](https://doi.org/10.1121/10.0009852) · [作者划分代码](https://github.com/QinggangSUN/keras_multiple_target_recognition/blob/master/recognition_mix_shipsear_s0tos3_preprocess.py) |
| Chen 等，Ocean Engineering，2025。出版社预览 | 将基于 Hu 等工作的通道注意力加入 ResNet，使用多标签二分类识别类型和数量。5 秒片段的目标间能量比为 −5 至 5 dB、0 dB 时，论文分别报告 97.16%、98.09% 的识别准确率。预览的指标和源录音划分细节有限。 | 通道重标定和目标间强弱控制都有直接水声先例。这里的 dB 描述参与叠加的船声强弱。 [期刊论文](https://www.sciencedirect.com/science/article/abs/pii/S0029801824031792) |
| Chen 等，Expert Systems with Applications，2026。原论文图 7、§5.1、表 8 | TResNet-LA 在 TResNet-L 的末段加入一处 SE 通道注意力，针对非目标船干扰。三套 TIR 数据上，同骨干的完整集合准确率分别由 47.41% 升至 50.94%、64.98% 升至 68.98%、89.09% 升至 91.21%。等能对应最后一套。 | TIR 以非目标船声为参照。论文在构造混合数据后按类将样本划分为 70%／15%／15%，该节未交代源录音隔离。本地适配另按新 MB 协议比较，−10 dB 双船命中较高，同时真单船被判双船更多。 [期刊论文](https://www.sciencedirect.com/science/article/abs/pii/S0957417425027393) · [存档正文表 8](D:/LSJ/Code/UART/final/data_gen_v8_single_physical/research/low_snr_literature_20261025/chen_multitarget_extracted.txt:1615) · [新 MB 适配报告](D:/LSJ/Code/UART/shipsear/07_论文实验/01_外部模型对比/TResNet-LA_新MB实验报告.md) · [错判对照](D:/LSJ/Code/UART/final/data_gen_v8_single_physical/research/shipsear_finalization_20261103/消融实验/README.md:53) |
| Li、Yang，Ocean Engineering，2024。出版社预览和作者摘要 | 在多处深层特征上施加听觉被动注意损失，按类别组织表示。任务是在干扰船存在时辨别感兴趣船型。按两船相对距离构造的 −18 dB 条件下，平均 F1 为 67.43%。 | 可借鉴的是让中间特征承担辨认**指定类别**的职责。原指标是感兴趣目标的分类 F1，整段船型集合另需验证。 [期刊论文](https://www.sciencedirect.com/science/article/abs/pii/S0029801824010114) · [作者摘要](https://ssrn.com/abstract=4691104) |
| Xiao、Guo、Han，JASA，2025。作者摘要 | 针对多目标叠加与小样本的类内分布偏移，在波形、人工声学特征和隐层语义特征空间构造 mixup 样本。 | 适合作为数据增强与跨录音泛化的先例。配方、划分和逐条件数字仍需正文核实。 [作者摘要及 DOI](https://pubmed.ncbi.nlm.nih.gov/40985707/) |

## 低信噪比和模块消融

| 论文与阅读范围 | 方法和评估 | 对当前项目的含义 |
| --- | --- | --- |
| Deng、Hong，Ocean Engineering，2025。出版社预览 | 将两张注意力图相减的差分注意力与 PolyLoss 用于 MFCC 船型分类，在 ShipsEar 和 DeepShip 报告 −10 dB 下相对标准 Transformer 的收益。公开预览的划分及低信噪比消融细节有限。 | 差分注意力已有水声抗噪先例。可用作借鉴模块或对照。双船场景中，抑制某个噪声相关方向对弱船型的影响需单独测量。 [期刊论文](https://www.sciencedirect.com/science/article/abs/pii/S0029801825023510) · [原差分注意力论文](https://arxiv.org/abs/2410.05258) |
| Xu、Xie、Wang，Ocean Engineering，2023。作者全文 §2.3、§3.2、表 4 | 用原始样本做分类，以原始和加噪视图预测间的双向 KL 约束平滑性。30 秒窗，训练和测试按录音隔离，验证从训练数据随机抽取 15%。在 −15 至 10 dB 区间，ShipsEar 的普通噪声增强与平滑正则分别为 71.69%、79.69%，DeepShip 为 68.67%、73.03%。表 4 单独比较平滑正则。 | 可借鉴固定主干比较训练目标和宽噪声区间压力测试。相同标签的扰动可以加一致性约束，实际新增船型则应让相应集合得分变化。 [期刊论文](https://doi.org/10.1016/j.oceaneng.2023.114926) · [作者全文](https://arxiv.org/html/2306.06945v3) |
| Xie、Ren、Li、Xu，JASA，2024。作者全文 §3、§4.2、表 4 | 以船舶大小作辅助任务，用一维频域特征对专家进行任务相关路由。在两个种子下，九类识别的 ResNet-18 单任务、普通多任务、M3 分别为 83.62%、85.34%、86.21%。M3-TSE 为 87.07%，同时调整了辅助任务权重。 | 一个有具体物理含义的辅助任务和逐层消融能把贡献拆开。普通多任务已经解释了部分总增益，门控比较的量级小于系统总增益。 [期刊论文](https://doi.org/10.1121/10.0026481) · [作者全文表 4](https://arxiv.org/html/2411.02787v1#S5.SS2) |

这些论文的准确率各自对应片段长度、录音划分、目标数量、噪声源或指标定义。当前项目应以同划分的本地适配和完整集合指标比较系统。 [现有实验与数据协议](D:/LSJ/Code/UART/final/data_gen_v8_single_physical/论文技术部分_阶段模型与实验交接_20261108.md:21)

## 模块候选与已有证据

**优先考虑让局部证据直接参与具体的单双船竞争。** 借鉴 APA 的类别选择思路。以 ShortLocal 的时频特征为例，在汇聚成全局向量前，针对候选组合 A+B 分别读取支持 B 相对 A、支持 A 相对 B 的局部信息，再据此修正该组合的分数。推理对所有合法候选打分。问题是正确的 B 能否把 A+B 从 A 及 A+C 中区分出来。项目已有 Q0 LocalEvidence，对汇聚后的双窗 8×4 网格做独立船型池化和存在监督，−10 dB 综合为 46.52% → 46.41%。H1 还完成了移源配对和类别槽交换训练。更早的 PairHead 已用汇聚后的类别信息对组合做支持与反证，固定数量路由。因此候选的实质应落在**保留的局部输入和它作用于哪对集合的判别**，以同输入、等容量的普通类别池化作机制对照。 [APA](https://doi.org/10.1016/j.oceaneng.2024.117674) · [Q0 局部证据实现](D:/LSJ/Code/UART/final/data_gen_v8_single_physical/research/shipsear_finalization_20261103/local_evidence.py) · [Q0 局部证据报告](D:/LSJ/Code/UART/final/data_gen_v8_single_physical/research/shipsear_finalization_20261103/Q0局部船型证据迁移报告.md) · [H1 证据交换报告](D:/LSJ/Code/UART/final/data_gen_v8_single_physical/research/shipsear_intervention_set_20261027/实验报告.md) · [PairHead 代码](D:/LSJ/Code/UART/final/data_gen_v8_single_physical/research/shipsear_pair_evidence_20261022/pair_evidence_model.py:98)

**可用同一验证集上的数量偏置作为便宜的筛查。** 对现有七集合分数扫统一的双船偏置，绘出真实双船完整命中率与真单船判双船的关系，并记录背景误报。候选需要在相近误报预算下改善完整组合命中，同时在配对样本上固定底船和加噪规则，比较加入 B、加入等能量的另一类 C 和仅加噪时，A+B 的分数是否有选择地响应。V1 的局部干预显示，真实加源的变化方向与噪声变化空间重合，部分抑制提高 Single 却严重降低 Double。这个结果让泛化的去噪或差分注意力更适合作为对照。 [V1 局部干预](D:/LSJ/Code/UART/final/data_gen_v8_single_physical/research/shipsear_local_noise_path_20261102/结果解读_方向重叠与联合恢复.md) · [GN 封存结果与消融](D:/LSJ/Code/UART/final/data_gen_v8_single_physical/research/shipsear_finalization_20261103/ShortLocal-GN正式消融报告.md)

上述候选属于待验证设计。沿用文献中成熟算子时，应保留归属。若能在录音隔离的同协议比较中看到双船组合改善、误报受控，并用匹配消融定位到该模块的具体职责，才有依据将它作为论文方法贡献。
