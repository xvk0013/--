# DeepShip PTT 机制分析

1. 按六档拆解 Single→Double、Double→Single、含目标→背景、以及数量正确但船型集合错误。真 Double 的错误以全部真 Double 为分母。原始条件下 Q0 有 177／600 条 Double 识别错误，其中 144 条数量正确、组合错误。这个分解可直接从封存预测复核。
2. 逐档列出 Passengership＋Tanker、Passengership＋Tug、Tanker＋Tug 的命中率和样本数，配合七集合混淆矩阵定位误判。正文聚焦 0／−10 dB，附表保留六档。
3. 外部对比和双路拼接完成后，按相同样本 ID、标签和噪声档位对齐预测，报告相对 Q0 的纠正、损坏和净变化。

第一、二项的六档统计已导出。第三项待对应组封存。 [Q0 封存预测目录](../../01_模型/Q0_DeepShip/runs/DS-Q0-PTT-SeenShip-13)

## 已整理的 Q0 表

[Val／Test 六档指标与错误计数](Q0六档指标.csv) · [Val／Test 逐集合准确率](Q0逐集合准确率.csv)。均从冻结运行的结果 JSON 导出，ACC 数值单位为百分比。C0／C1／C2 对应全重叠、单方占优、交替占优三种双船混合情形。