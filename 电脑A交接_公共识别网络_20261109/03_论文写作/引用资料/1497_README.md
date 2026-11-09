# 异构重建第一步：独立神经与散射基线

本轮只建立N0、S0两条独立识别基线，暂不引入新声学模块或公共融合。不改动Bellhop信道、混合波形、录音归属及C0/C1/C2比例，不生成含噪数据文件。

- N0：双频带声谱和DEMON → 本任务自训练M2编码器 → 普通七类MLP。旧数量头被移除，全部编码层、DEMON编码器和新分类头实际训练，BatchNorm统计仅在Train更新。
- S0：固定小波散射全局/四段统计 → 本任务自训练TabM → 32成员概率等权平均。TabM全部训练，不读取N0状态、数量或预测。
- 两路最终均直接选择七集合概率的最大项。数量由集合确定；集合概率汇总数量后接受0.1权重的辅助监督，没有硬数量路由。

N0编码器和S0的初始化资产来自保留的clean模型，分别是本任务训练的quantity.pt（只加载声学部分）和tabm.pt。N0七类头随机初始化。这里是本任务权重初始化后的独立识别重建，不是随机初始化训练，也不使用BEATs或其他外部预训练权重。旧神经条件成员加权、MLP数量修正和源辅助头均不加载。

固定Train2880、Val716、Test877，种子13。原始/10/5/0/−5/−10dB训练比例为30/10/10/10/20/20。沿用既有在线AWGN、Single包络及低SNR Single取消额外包络规则；两路每轮读取相同样本顺序和完全相同的波形。固定变换可以缓存，训练中的神经特征每次重新计算。

两路独立优化、独立保存选中权重，每路最多40轮、至少15轮、连续10轮未改善停止。N0声学部分lr1e-5、新头1e-3，S0 lr1e-4；AdamW、weight decay0.001、batch32。主损失均为七集合交叉熵，TabM对32成员取平均；辅助项为由各自集合概率汇总的数量NLL。保持背景10%、Single30%、Double60%的训练损失预算。

按六档Val的Single/Double/背景宏平均选择每路唯一权重，同分看原始Val综合ACC。Test只在两路均完成选模后读取。所有信噪比使用同一权重。报告包含最终集合ACC、Single/Double/C0/C1/C2/背景、数量错误、每类船型/组合，以及给定真实数量的条件类型诊断。条件类型成绩和两路择优上限只用于诊断。

```powershell
& 'C:\Users\Admin\anaconda3\envs\zpfenv1\python.exe' -B -X utf8 -W ignore .\run.py --validate
& 'C:\Users\Admin\anaconda3\envs\zpfenv1\python.exe' -B -X utf8 -W ignore .\run.py
```

结果：`runs/seed13/results.json`与`实验报告.md`；每路`N0.pt`、`S0.pt`可由`baseline_model.IndependentRecognizer`直接读取波形推理。原模型与已有结果保留。
