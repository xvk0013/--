# 源干预证据与异构协同：首轮实验

本轮承接固定数据上综合ACC 83.01%的自训练M2＋散射专家，验证类别条件移除关系能否改善Single和数量边界。数据、前端统计、原编码器与任务头、散射变换及SVM均保留。

## 当前实验结构

```text
混合音频 → 冻结原编码器 → 共享时频图 → 三类条件移除 → 原／剩余／变化证据 ─┐
        └→ 原模型与固定散射专家 → 等价七集合基础评分 ─────────────────────┤
                                                                          ↓
                                                              零初始化七评分修正
                                                                          ↓
                                                                最终集合及其数量
```

此轮是新机制的过渡验证，原神经Double和数量头仍参与基础评分。它不代表已经完成独立集合判别器的替换或旧头删除。

- `ClassRemoval`：19,536参数；192→48的共享压缩、局部卷积、类别FiLM条件、192通道恢复，输出预测剩余图。
- `RemovedEvidenceReadout`：129,639参数；原始、剩余与差分的全局／四段均值，经共享小投影后与原混合DEMON拼接，输出七维修正。
- 原混合DEMON始终只是混合上下文，不能视为预测的剩余来源DEMON。
- 源辅助监督取Train实际贡献项，完整保留传播、源增益、包络和共同增益。移除source1匹配source2，反之亦然。
- 保留类监督复用冻结查询及Single读出，只在Train配合真实保留源自身DEMON；正式推理不使用真实源。
- 缺席主标签的查询约束保持；Single移除唯一来源后没有真实背景目标，不用零波形伪造背景。
- 共现录音继续参与正式训练，不强制实际移除后某物理类别一定消失。

## 对照和选择

两臂同结构、同初始化、同批次顺序，各36轮和相同实际更新数：

| 阶段 | 普通适配 | 源干预适配 |
|---|---|---|
| 前12轮 | 正式七集合分类及边缘数量监督 | 剩余表征匹配、缺席保持、保留类别辅助 |
| 后24轮 | 正式七集合分类及边缘数量监督 | 同样正式任务＋移除辅助约束 |

普通臂获得更多正式任务更新；这是完整源引导方案对普通残差适配的比较，尚非单个损失的严格消融。

固定种子13；第13轮两臂共同重置优化器和dropout随机流。编码器冻结，只缓存不变的特征。Train增强波形的散射分数由冻结专家重新计算一次，避免神经与散射输入不一致。Val/Test正常识别只使用原始混合音频。

按Val综合ACC选最佳实际训练轮次；另记录满足Single提高、Double不降、综合不降、背景≥95%、MB最终准确率不降的候选。epoch0只表示原基线，不能替代实际训练结果。两臂选模固化后统一评价Test。

## 运行

推荐沿用已配置CUDA扫描的Python：

```powershell
& 'C:\Users\Admin\anaconda3\envs\zpfenv1\python.exe' -B -u -W ignore run_experiment.py --output runs\removal_seed13
```

音频推理与独立评价：

```powershell
& 'C:\Users\Admin\anaconda3\envs\zpfenv1\python.exe' -B -W ignore inference.py --manifest runs\removal_seed13\source_removal_best_trained_model.json --split Test --output runs\removal_seed13\audio_replay
```

本轮复用的固定底座入口：
`D:\LSJ\Code\UART\shipsear\ShipsEar_M2_ScatteringExpert_20261021\runs\scattering_double_seed13_20261021\artifacts\model.json`。

## 必要文件

- `cache_data.py`：固定混合音频、Train成分与冻结表征。
- `removal_model.py`：移除变换、证据读出及等价集合评分。
- `run_experiment.py`：两臂训练、Val选择、最终评价。
- `inference.py`：只输入混合音频的部署入口。
- `runs/removal_seed13/实验结果.md`：指标、诊断和保留决定。
- `protocol.json`、`selection.json`、选中权重与逐样本结果：用于复现。

大体积特征缓存是训练加速文件，不属于模型部署所需内容。

## 有限上下文联合适配

冻结阶段后补做一次8轮对照，仅解冻编码器最后阶段、投影与类别查询，并训练新增模块；旧任务头、其余编码器及BN统计不更新。每臂从自身保护排序版Val最佳新模块和同一83.01%编码器开始。受影响表征实时重算，只缓存固定前端。

```powershell
& 'C:\Users\Admin\anaconda3\envs\zpfenv1\python.exe' -B -u -W ignore joint_adaptation.py
```

`joint_adaptation.py`保存更新后的自训练M2与新模块两份权重；相应manifest的`m2_checkpoint`覆盖冻结底座路径，`inference.py`可直接读取。`diagnose_removal.py`仅在候选确定后离线分析Train／Val源，不参与优化或选模，不接受Test源。

汇总决定见`本轮结论.md`：本轮继续保留83.01%主模型，尚未替换或删除旧判别头。
