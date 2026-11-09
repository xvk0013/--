# CUMI共享—互补表征：Double接口对照

模型及波形入口：`cumi_model.py`中的`CumiRecognizer`。实验入口：`run_experiment.py`。详细条件见`PROTOCOL.md`，结果见`CUMI共享互补实验报告.md`。

所有四组真实训练，种子13；新增接口使用相同神经/散射信息，原编码器、TabM、预测数量与Single路径固定。R2/R3的推理架构和初始化一致。正式模型入口保持原状，完整CUMI组尚未建立识别收益。

候选入口：`runs/cumi_seed13/<R0_mlp|R1_common|R2_common_private_ce|R3_cumi>/model.json`。

```python
from cumi_model import CumiRecognizer
model = CumiRecognizer('runs/cumi_seed13/R3_cumi/model.json')
result = model.predict(waves)  # float32 [batch,80000], 16 kHz
```

```powershell
# 在本目录，用现有实验结果更新报告
& 'C:\Users\Admin\anaconda3\envs\zpfenv1\python.exe' run_experiment.py --stage report
# 完整重新运行时选择新输出目录，保留已有结果
& 'C:\Users\Admin\anaconda3\envs\zpfenv1\python.exe' run_experiment.py --stage all --output runs/reproduce_seed13
```

训练复用固定模型的特征缓存；波形推理会实际提取声学与散射特征并使用已选数量权重，不依赖标签、真实数量或源音频。原始数据、继承权重与特征目录记录在模块常量及每个候选manifest中。

本实现借鉴CUMI论文并作任务适配；alpha2核熵、共同来源边际化等细节均已记录。`method_review.md`保存原文与作者示例代码核对结果，不将此实验冒充原论文数值复现。
