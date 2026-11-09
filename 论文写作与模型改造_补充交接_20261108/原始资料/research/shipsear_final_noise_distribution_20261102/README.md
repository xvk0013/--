# V1 固定模型与噪声训练分布收尾实验

本目录保留 V1 的完整计算结构，仅比较两种训练噪声采样比例，不加入早层 skip，也不改变特征、归一化、交互机制或分类头。

```text
16 kHz / 5秒波形
    ├─ 固定一维波形散射 → 全局/四段均值、标准差 → 3730维
    └─ 原生长短窗对数功率谱 → 固定二维谱散射 → 3200维
                         ↓ 拼接、原 Train 归一化
       公共 TabM：32成员，3层×256，首层 CalibratedInteraction 门控
                         ↓
                  7集合读出 + 数量读出
                         ↓ 成员概率平均
          E / PF / MB / RV / PF+MB / PF+RV / MB+RV
```

两路固定特征均从同一观测波形计算；推理不输入真实数量、源波形、纯净参考或信噪比。特征提取批量固定为32，分类批量固定为64；每成员先在 float32 中计算 softmax，再在 CPU 用 NumPy 求成员均值。模型共 **2,737,761** 个可训练参数。

| 实验臂 | 原始波形 | 10 / 5 / 0 / −5 / −10 dB，各档 |
|---|---:|---:|
| P25_noise75 | 25% | 15% |
| P50_noise50 | 50% | 10% |

两个实验臂均继承同一份 V1 权重及原 Train 归一化。P50 在满足本目录复用检查后可接续与其训练协议相同的 A_continue 前20轮；复用与选模细节由 `config.json`、`reuse_audit.md` 和运行报告记录。比例作用于训练视图，模型推理结构完全相同。

文件说明：

- `model.py`：独立模型、固定特征、权重初始化与 `Recognizer` 推理接口。`make_models` 中相对的初始权重路径以本目录为基准。
- `predict.py`：`.npy` 波形批量预测入口；输入为 `(N,80000)` 的16 kHz、5秒波形，不增加额外预处理。
- `weights/V1_reference.pt`：原 `shipsear_single_stored_view_20261031/runs/seed13/V1_stored_single_best.pt` 的保留参考，来自 V1 全局第51轮；可直接用于推理。
- `vendor/`：实际运行所需原版 TabM 0.0.3、rtdl_num_embeddings 0.0.12、Kymatio 0.3.0；来源、版本和许可证见 `vendor/README.md`。运行时不动态导入历史实验模块。
- `config.json`、`data.py`、训练入口及报告文件：本轮数据、训练与结果管理；不参与额外模型计算。

在本目录执行：

```powershell
python predict.py --checkpoint weights/V1_reference.pt --input waves.npy --output predictions.npz --device cuda
```

也可以指定 P25/P50 的保存权重，或使用 `--device cpu`。命令行中的文件路径相对当前工作目录解析。输出 `.npz` 包含 `probabilities`（N×7）、`predictions`（集合索引）、`counts`（0/1/2）和 `sets`（集合名称），均可用 `numpy.load(..., allow_pickle=False)` 读取。

Python 接口：

```python
from model import Recognizer
result = Recognizer('weights/V1_reference.pt', device='cuda').predict(waves)
```

运行环境需要 PyTorch、NumPy、SciPy、packaging 和 typing_extensions。TabM/rtdl 源码遵循 Apache 2.0，Kymatio 遵循 BSD 3-Clause，许可证已随本地依赖保留。
