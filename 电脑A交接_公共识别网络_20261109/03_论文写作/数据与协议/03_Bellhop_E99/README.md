# Bellhop E99 信道与缓存

两套正式生成器使用同一份冻结的 39 通道 E99 FIR，包内的两份信道缓存已核对逐字节一致。距离为 1–4 km、步长 0.25 km，源深 5／10／15 m，接收深度 1000 m。Bellhop 网格频率为 50–7950 Hz、步长 100 Hz。接收波形采样率 16 kHz。

- [信道参数配置](../01_ShipsEar/ShipsEar_DataGen_Final_0dB/physical/CHANNEL_PROTOCOL.json) · [DeepShip 缓存说明](../02_DeepShip/02_数据生成与数据集/physical/BANK_INFO.json) · [冻结 E99 FIR 文件（本机出处）](../../../90_本机路径.md#r020)
- [逐通道支持区间、数值延迟和能量占比](../01_ShipsEar/ShipsEar_DataGen_Final_0dB/physical/FIR_99_SUPPORT.tsv) · [逐频率表](FIR_99_FREQUENCY.tsv) · [E99 裁剪源码](v9_trim_fir_99.m) · [80 频率收敛验收](ACCEPTANCE.md)
- [ShipsEar MATLAB 传播](../01_ShipsEar/ShipsEar_DataGen_Final_0dB/physical/matlab/v9_propagate_99.m) · [DeepShip Python 传播](../02_DeepShip/02_数据生成与数据集/physics.py)

核查公式时，从含零滞后的最短连续 99% 能量支持区间开始。对裁剪后的 FIR，令 M 为长度减一，D 为数值延迟，抽取核心前 M−D 个真实历史采样和核心后 D 个真实前瞻采样。以源核心 RMS 的倒数缩放整段上下文，再线性卷积并按 M 截取对应的 5 秒接收核心。只有超出原录音边界的上下文才填零。生成时读取已封存 FIR，Bellhop 的冻结物理模板在 ShipsEar 的 physical/environment 下。

此处保存数据生成实际使用的 E99 缓存、索引与物理验收。完整未裁剪信道及原始 ARR 求解输出保存在源机历史工程。
