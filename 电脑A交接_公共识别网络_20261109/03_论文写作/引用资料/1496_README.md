# M2与独立散射Double专家

保留固定数据与原M2初判，用固定小波散射＋SVM提供双目标组合排序，不使用外部预训练权重。

## 模型流程

同一条5秒音频进入两条路径：

- 原STFT＋DEMON＋M2输出七集合初判，取最高分集合所对应的目标数量。
- 固定一维小波散射提取整段与四段时间统计，标准化后由七分类RBF SVM输出分数。

M2初判为0或1时，保留其原集合；初判为2时，在SVM的PF+MB、PF+RV、MB+RV三个分数中选最高者。数量来自M2最终集合，不改成数量头直接取最大值，也不输入真实数量。SVM分数只用于组合排序，不视为校准概率。

原Double头保留并参与M2初判。本轮没有修改其接法，也没有训练M2。该分工不会修正原数量错误或提高Single、背景的准确率。

## 训练与选模

M2全部冻结，小波散射为固定变换。标准化只在Train拟合；SVM保留七类Train样本、原有Single包络增强和相对样本权重。数据排列、混合波形与Train/Val/Test标签不变，不额外加入AWGN。

首先复现已有C=1诊断；然后比较RBF核C=0.1、1、10。C=1可复用已拟合SVM，其余候选重新拟合。只按**完整路由系统的Val综合ACC**选择C，选定后评价Test。前一轮已评价过C=1的Test结果；本轮不使用Test选择候选。

## 运行

在本目录执行：

```powershell
python -B run_experiment.py --output runs/scattering_double_seed13_20261021
```

重新读取导出的完整模型进行Test评价：

```powershell
python -B evaluate_model.py runs/scattering_double_seed13_20261021/artifacts/model.json --split Test
```

重新生成结果报告：

```powershell
python -B report_results.py runs/scattering_double_seed13_20261021
```

## 输出

运行目录下保留：

- `artifacts/model.json`：完整模型入口，记录M2与散射专家的加载设置及路由。
- `artifacts/m2.pt`：固定的自训练M2权重。
- `artifacts/scattering_svm.joblib`：选定的标准化与SVM。
- `protocol.json`：固定数据路径、M2来源、散射设置、种子和样本权重。
- `selection.json`：各C的Val指标、是否复用及拟合耗时。
- `metrics.json`：Train/Val/Test的原M2、C=1参考与选定专家结果。
- `timings.json`、逐样本预测及`实验结果.md`：耗时、错误分析与汇总。

只检查运行必需的输入、特征与标签对齐、路由及导出模型输出一致性，不添加哈希流程。
