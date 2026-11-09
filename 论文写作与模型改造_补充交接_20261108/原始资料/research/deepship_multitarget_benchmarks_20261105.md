# DeepShip 多船混合识别的公开成绩与协议

核查日期为 2026 年 11 月 5 日。范围为从同一段混合船声识别同时出现的多个目标，并区分独立数据集实验、普通单目标分类和混合环境中的指定目标检测。

## 直接相关研究

| 论文 | DeepShip 在实验中的角色与公布成绩 | 数据与评估口径 |
| --- | --- | --- |
| Yu 等，CMM-DACTNet，*Ocean Engineering* 359（2026），125940 | 与 ShipsEar 并列构造多船辐射噪声数据。摘要依次给出 ShipsEar **88.2444%–99.9384%** 和 DeepShip **91.8097%–99.7771%**。 | 出版社预览说明采用论文所称的 harmony superposition 和循环音频叠加，列出 DSRN、TSRN、FSRN 子集，并报告五折交叉验证。公开预览未提供从原录音划分折的证据，也未核实各区间端点对应的具体组合、评分公式、SIR 或附加噪声 SNR。 [出版社摘要与预览](https://www.sciencedirect.com/science/article/abs/pii/S0029801826017749) |
| Shan 等，SKANN，*CAAI Transactions on Intelligence Technology* 10（2025），1548–1558 | 从 DeepShip 构造 M-DeepShip，论文报告 **93.6% 峰值 Val**。 | 四船型按 10 秒线性叠加，覆盖 15 个非空集合，正文报告 28,125 条、Train／Val 为 80%／20%。§3.1 对同一原录音的奇偶切片分配 Train／Val。§5.1 式（3）按真实船型被找回的比例给部分分，额外预测船型也不会直接扣分。该实验的分组和评分单位与录音隔离下的双目标完整组合命中不同。§4 说明线性叠加，未给出与我们接收端 0 dB SIR 或分档加噪 SNR 一致的设置。 [原文 §3.1、§4、§5.1–5.2](https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/cit2.70037) |
| Duan 等，*IEEE Journal of Oceanic Engineering* 50（2025），1760–1771 | IEEE 摘要确认用公开 DeepShip 合成多标签船声，预览只定性报告性能。Yu 等在自己的论文引言中转述 **94.59%**。 | Duan 原文公开摘要未列具体标签范围、源录音划分、评分公式、混合能量比和附加噪声；94.59% 的原始结果表仍待核查。 [IEEE 原始摘要](https://ieeexplore.ieee.org/abstract/document/10993418/) · [Yu 等论文引言](https://www.sciencedirect.com/science/article/abs/pii/S0029801826017749) |

SKANN 式（3）可写为每样本的 `|预测集合 ∩ 真实集合| / |真实集合|` 再平均。真实为 Cargo＋Tug 而预测 Cargo 时得 0.5，预测 Cargo＋Tug＋Tanker 时得 1。完整组合准确率对这两种预测都计 0。 [SKANN 原文式（3）及表 4](https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/cit2.70037)

## 容易混入的数字

- Chen 等的通道注意力多目标识别论文从 ShipsEar 取 PF、MB、RV，摘要报告 5 秒下在源间能量比 −5 至 +5 dB 时为 **97.16%**、等能量时为 **98.09%**。同团队干扰研究以 ShipsEar 派生的 SE_en0 在等能条件下报告 **91.21%**。这三项成绩均来自 ShipsEar 派生数据。 [Ocean Engineering 原文](https://www.sciencedirect.com/science/article/abs/pii/S0029801824031792) · [ESWA 原文](https://www.sciencedirect.com/science/article/pii/S0957417425027393)
- DCMT 的 DeepShip **97.53%** 属于四船型单目标分类。它在 QiandaoEar22 上的混合背景实验以指定目标与其他信号的区分为任务。QiandaoEar22 原始论文的 **99.56%** 也是混合音频中的指定单船检测结果。 [DCMT 原文](https://www.nature.com/articles/s41598-025-32401-3) · [QiandaoEar22 原文](https://doi.org/10.1186/s13634-024-01181-9)

## 本地同一预测的两种评分

对已封存 Q0 原始条件 Test 的同一组预测重算两种指标。每版均取 1,200 条双目标。计算部分找回率时排除真实为空的背景，避免分母为零。

| 版本 | 双目标完整组合 ACC | 双目标平均真实标签找回率 |
| --- | --- | --- |
| Energy80 录音级基线 | 47.75% | 71.79% |
| SeenShip 录音级同船名覆盖 | 48.75% | 72.50% |

我们的双目标完整组合 ACC 对船型集合做严格相等判断。相同预测换成部分找回率后可增加约 24 个百分点，这个差异来自指标。Energy80 和 SeenShip 的全部 1,800 条含目标样本按部分找回率分别为 71.31% 和 74.17%。 [Energy80 封存预测](D:/LSJ/Code/UART/deepship/Q0_DeepShip/runs/DS-Q0-Energy80-13/Test_original.npz) · [SeenShip 封存预测](D:/LSJ/Code/UART/deepship/Q0_DeepShip/runs/DS-Q0-Energy80-SeenShip-13/Test_original.npz)

## 可以比较到什么程度

CMM-DACTNet 是本次检索中最接近“ShipsEar 加 DeepShip 双数据源的多船识别”的公开结果。SKANN 提供了最清楚的 DeepShip 派生混合实验协议。已核实材料中，仍缺与我们同口径的公开结果，即源录音先分集合、各集独立混合，并单报 5 秒双目标完整集合 Test ACC。当前应分别引用论文在其协议下的报告值和本地的同划分实验分数。 [CMM-DACTNet 公开预览](https://www.sciencedirect.com/science/article/abs/pii/S0029801826017749) · [SKANN 原文](https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/cit2.70037) · [本地生成协议](D:/LSJ/Data/DeepShip_Multitarget_0dB_Energy80_SeenShip_v1/REPORT.md)
