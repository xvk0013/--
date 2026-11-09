# ShipsEar 外部模型对比

整理日期 2026-11-06。四组已有外部结果统一按旧录音级划分归档。每组包含独立模型包，报告放在本目录，公共训练与评估入口单列。

| 模型 | 独立模型包 | 实验报告 | 旧划分原始 Test |
| --- | --- | --- | ---: |
| ShuffleFAC-32 | [01_ShuffleFAC-32（本机出处）](../../90_本机路径.md#r484) | [ShuffleFAC-32_实验报告.md（本机出处）](../../90_本机路径.md#r485) | 已完成，81.64% |
| TResNet-LA | [02_TResNet-LA（本机出处）](../../90_本机路径.md#r486) | [TResNet-LA_实验报告.md（本机出处）](../../90_本机路径.md#r487) | 已完成，65.79% |
| Sun–Wang CNN | [03_Sun-Wang-CNN（本机出处）](../../90_本机路径.md#r488) | [Sun-Wang-CNN_实验报告.md（本机出处）](../../90_本机路径.md#r489) | 已完成，61.35% |
| RCMoE | [04_RCMoE（本机出处）](../../90_本机路径.md#r490) | [RCMoE_实验报告.md（本机出处）](../../90_本机路径.md#r491) | 已完成，60.89% |

[公共接口（本机出处）](../../90_本机路径.md#r492) · [旧划分 Q0 对照](../../04_历史补充/论文与改造/原始资料/research/shipsear_finalization_20261103/runs/SE-Q0-13/results.json) · [旧六档总表（本机出处）](../../90_本机路径.md#r024) · [DeepShip 对比目录](1505_README.md)

## 新 MB 划分正式外部对比

四组已在新 MB 划分完整训练和评价，可与当前 Q0 同表报告。 [新MB六档对比汇总.md（本机出处）](../../90_本机路径.md#r440)

| 模型 | 新 MB 实验报告 | 原始 Test |
| --- | --- | ---: |
| ShuffleFAC-32 | [六档结果（本机出处）](../../90_本机路径.md#r493) | 83.24% |
| TResNet-LA | [六档结果（本机出处）](../../90_本机路径.md#r388) | 72.63% |
| Sun–Wang CNN | [六档结果（本机出处）](../../90_本机路径.md#r494) | 66.48% |
| RCMoE | [六档结果（本机出处）](../../90_本机路径.md#r495) | 70.13% |

上方旧划分结果可与旧 Q0 86.09% 同表作协议对照。新划分使用当前 Q0 87.91%。两版 Test 样本一致，Train／Val 的录音归属分别登记。

## 新 MB 划分执行规则

1. **公平比较固定任务和评测。** 各组使用本数据集冻结的源录音划分、5 秒／16 kHz 波形、七个合法目标集合和相同的六档评价视图。Train 使用与 Q0 一致的存储波形与 50／10／10／10／10／10 噪声日程。种子 13，从各模型构造初值完整训练。保留各方法适合的前端、损失、优化器和训练轮数。
2. **按 Val 选轮和选配置。** 各组使用冻结协议的六档排序。原始 Val 综合 ACC 约 65–75% 是检查基线是否发挥正常的参考。结合 Single、Double、背景及六档表现判断适配质量。已有的历史 Test 只供解释历史结果，正式配置、阈值和轮次均由预先登记的规则及当前 Val 决定。
3. **限制微调预算。** 首选论文或作者代码的同数据集配置。若 Val 显示未收敛或单项适配明显偏弱，每个方法在每套数据上至多再完整训练一次表中指定的单因素备选。两次都按同一六档排序选用。TResNet-LA 的统一阈值属于封存权重上的 Val 校准，无需加训。所有候选配置、Train／Val 曲线与选择理由留档。
4. **配置封存后评估 Test。** 同一权重跑完六档，报告实际的完整集合综合 ACC、Single／Double／背景和样本数。65–75% 无成绩上限或入表门槛。论文按「同任务适配」标注前端、输出、损失和训练预算的实际差异。

## 首选配置和有限备选

| 编号与模型 | 首选参数 | 单因素备选及适配 |
| --- | --- | --- |
| SE-EXT-SFAC32-MB-13 · ShuffleFAC-32 | 作者 DeepShip 配方。128 频带幅度 log-Mel，Hamming 4096 点、步长 2048，0–8 kHz。作者 32 宽结构与池化。Adam `1e-3`、batch 48、普通 CE，训练上限 200 轮 | 5 秒／40 帧和七集合输出为本任务适配。若 Val 显示数量分组失衡，备选为同一 CE 加入 10／30／60 的背景／Single／Double 组预算。 [论文训练设置](https://arxiv.org/html/2601.13679v1#S3.SS3) · [作者配置](https://github.com/KNU-LMAP/ShuffleFAC/blob/master/default.yaml) |
| SE-EXT-TRLA-MB-13 · TResNet-LA | 作者 ShipsEar 配方。保留网络、128 频带 Mel 与三船型存在输出。AdamW `1e-4`、batch 32、80 轮、ASL 正／负指数 0／6 | 16 kHz 下采用 512 点 Hann 窗、160 点步长、FFT 1024、0–8 kHz，保留约 500 帧。ASL clip 0.05、weight decay `1e-4` 为本地实现值。Val 校准见下节。 [原论文 §5.1 与附录 C](https://doi.org/10.1016/j.eswa.2025.129122) · [本地实现说明（本机出处）](../../90_本机路径.md#r496) |
| SE-EXT-SW-MB-13 · Sun–Wang 实值 CNN | 复用本地 PyTorch 实值 ResNet 七集合适配。恢复作者 ShipsEar 前端约 30 ms 窗、7.5 ms 步长，对应 480／120 点，128 频带 log-Mel，FFT 1024 零填充，保留声谱的时间维。Adam `1e-3`、batch 32、上限 80 轮 | CE 七集合读出、batch 与上限为本地适配。Val 备选将 log-Mel 换为作者也比较过的幅度 STFT，保持窗口和优化器。 [论文](https://doi.org/10.1121/10.0009852) · [作者预处理](https://github.com/QinggangSUN/keras_multiple_target_recognition/blob/master/prepare_data_shipsear_recognition_mix_s0tos3.py) · [作者训练脚本](https://github.com/QinggangSUN/keras_multiple_target_recognition/blob/master/train_recognition_mix.py) |
| SE-EXT-RCMOE-MB-13 · RCMoE | 恢复作者 ShipsEar 参数。幅度 STFT 50 ms 窗、25 ms 步长，16 kHz 下为 800／400 点，频带 100 Hz 至 Nyquist。沿用本地作者特征复现的 Hamming 窗、预加重与片段标准化。采用作者公开的残差专家结构与普通平均池化，四专家 sigmoid top-1，CE 加 `0.01` 负载均衡。AdamW 峰值 `5e-4`、weight decay `1e-5`、batch 32、上限 200 轮 | 七集合头、输入长度和 batch 为本地适配。公开实现使用普通池化，论文描述了注意力池化，论文表中注明所用实现。前端属于本地重建。Val 备选只把 STFT 换为论文中的 300 频带 Mel。 [原论文 §§4–5](https://arxiv.org/html/2402.11919v1#S4) · [作者代码](https://github.com/YuanX9/UATR-CMoE/blob/main/cmoe_model.py) · [本地特征复现（本机出处）](../../90_本机路径.md#r497) |

ShuffleFAC-32、Sun–Wang、RCMoE 的早停可沿用最少 40 轮、patience 10 和 0.2 个百分点时钟刷新量。TResNet-LA 按作者预算跑 80 轮，再用六档 Val 排序取最佳轮次。每组登记真实批量、实际轮数、峰值学习率及调度。

## TResNet-LA 的统一 Val 校准

先用无偏置合法集合读出按六档 Val 选轮。固定该权重，对各候选集合按 `sum(集合内船型 logits) - b × 集合大小` 计算分数。Val 试 `b = 0, 0.25, …, 2.0`，按相同六档排序选定一个覆盖所有条件的值，同分选较小值。最终封存轮次、偏置和 Val 得分，再评估 Test。主表使用该预定流程，原始解码可在附表并列。
