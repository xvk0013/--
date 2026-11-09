# 固定数量与Single的Double接口实验

数据与原部署入口保持不变。本目录保存G1普通MLP、G2单向类别—组合校验、G3双向校验及按Val触发的G4存在支持版本。各候选的模型入口为`runs/pair_study_seed13/<组名>/model.json`。按Val选中的组见`recommendation.json`，完整说明见`Double类别组合校验实验报告.md`。

## 使用

从本目录导入：

```python
from pair_evidence_model import PairRecognizer
model = PairRecognizer('runs/pair_study_seed13/G4_support_only/model.json')
result = model.predict(waves)  # float32 [batch,80000], 16 kHz, 5 seconds
predictions = result['predictions']
```

输出标签顺序为E、PF、MB、RV、PF+MB、PF+RV、MB+RV。推理不输入真实数量、船型或源音频。模型读取原神经底座、散射TabM和固定数量权重；三个模块均为本项目已训练权重，未新引入外部BEATs预训练。旧七集合校验头不再加载。

## 运行

```powershell
& 'C:\Users\Admin\anaconda3\envs\zpfenv1\python.exe' -B -W ignore run_pair_study.py --stage all
```

完成目录会拒绝被覆盖；需要重复实验时应使用一个新的实验目录，保留原记录。`--stage report`只重新生成报告。冻结特征缓存从原正式目录只读复用，未复制或改写波形数据。最终每个候选都已用全部716条Val和877条Test混合音频验证，与缓存评价预测完全一致。

本次所有新训练seed13；共同冻结专家包含历史训练种子权重。这是固定底座上的接口对照，不称为全模型从随机初始化训练。
