# 当前开发模型：短窗时频块T1第27轮

用户已认可该版作为下一阶段起点，实验暂停。原始Test Single88.17%、Double83.33%、背景91.95%、综合85.63%；−10dB Single63.74%、Double40.34%、背景98.85%、综合53.14%。

## 使用入口

- `model.py`：独立结构与`Recognizer`，无需导入历史实验目录或加载其他历史权重。
- `weights/current.pt`：T1第27轮当前主线。
- `weights/reference_S0.pt`：保留的S0对照。
- `predict.py`：读取16kHz、5秒的numpy波形数组，输出目标集合。
- `vendor`：原工程所用TabM、Kymatio及许可证。

```python
from model import Recognizer
result = Recognizer().predict(waves)  # shape(N,80000)
print(result['sets'])
```

源checkpoint中的旧路径只作训练来源记录，独立推理不访问它们。训练脚本、初始化来源、配置、原始记录保存在相邻“训练来源与复现”目录。独立包严格加载原权重；当前模型及S0的完整Val原始/−10dB回放均为零预测差异，见独立模型回放核对.json。未进行重训。

## 恢复工作时的顺序

先保住Single、减少真实Double被判Single，再处理原始背景边界，最后检查六档曲线。−10dB175条Double→Single中126条的正确Double组合仍排名第一；不要统一提高Double分数。0到−5dB Single反升仅净1条，不为修整该曲线反复改模型。

该版未通过旧实验预设的全部保护条件，历史报告如实保留；采用它是后续开发选择，不声称已满足最终验收。三目标已取消。
