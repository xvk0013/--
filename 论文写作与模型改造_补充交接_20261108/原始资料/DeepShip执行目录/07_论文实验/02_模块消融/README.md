# DeepShip PTT 模块消融

| 编号 | 变体 | 职责 |
| --- | --- | --- |
| DS-ABL-CAT-PTT-13 | 双散射拼接 ＋ 标准 TabM | **正式组**。保留同样两路、三块 256 宽、32 成员、七集合输出与数量边际监督，去掉定制首块交互和门控。与 PTT 完整 Q0 比较 |
| DS-ABL-ADD-PTT-13 | 逐元素乘积换为等参数加法 | **补充组**。保留直通和门控，用于需要单独验证乘积算子的跨数据集作用时 |

两组在 PTT 的 Train 上拟合统计，从构造初值完整训练，按本目录的共同 Val 规则封存后评估。两分支各自的消融与全局门控对照优先由 ShipsEar 完成。

DeepShip 已有的 LocalEvidence 属于 Energy80 SeenShip 四船型，作为已完成的探索单独归档。 [PTT 模型入口](../../01_模型/Q0_DeepShip/train.py) · [LocalEvidence 历史运行](../../对比实验/Q0_LocalEvidence) · [ShipsEar 对应消融](../../../shipsear/07_论文实验/02_模块消融/README.md)