# CUMI 原文与官方代码核对

核对时间：2026-10-23。范围：原文公式、作者公开代码与本轮 Double 融合接口的对应关系；未执行训练。

## 1. 本轮应采用的数学对象

两视图记为神经特征 `x_N` 与散射特征 `x_S`。分别学习共同编码 `c_N,c_S` 和独有编码 `u_N,u_S`。随机选取一个共同编码 `C`，分类输入为 `[C,u_N,u_S]`。独有表示是视图特有信息，不能直接解释为噪声或录音风格；共同表示也不能预先解释为类别语义。

原文式 (12) 的最小化目标为：

\[
L=L_{CE}+\sum_i\operatorname{MSE}(x_i,\hat x_i)-\beta H_\alpha(C)+\gamma\operatorname{TC}_\alpha(C,u_N,u_S).
\]

对批内 Gram 矩阵 `K`，先作 `A=K/tr(K)`，再计算：

\[
H_\alpha(A)=\frac{\log_2\sum_j\lambda_j(A)^\alpha}{1-\alpha},\qquad
\operatorname{TC}_\alpha=H_\alpha(A_C)+H_\alpha(A_N)+H_\alpha(A_S)
-H_\alpha\left(\frac{A_C\circ A_N\circ A_S}{\operatorname{tr}(A_C\circ A_N\circ A_S)}\right).
\]

原文式 (8) 约束不同视图导出的共同变量一致；该目标的独立性条件与可逆生成假设，不能直接作为我们冻结声学特征实验的理论保证。原文实现节使用 MLP，`beta,gamma` 在 `{0.001,0.01,0.1}` 中调节。来源：[原文 3.2、3.3、4.2.3 节](https://arxiv.org/html/2406.15043v1)。

## 2. 作者代码实际执行什么

检查的是 `archy666/CUMI` 的 `main` 分支现有文件，下面是静态代码事实。

| 文件 | 观察 | 对本轮的意义 |
|---|---|---|
| `model/model.py:80` | `random.choice(com_features)` 选择整批次的一个视图；没有逐样本抽取 | 两个视图各以 1/2 概率提供该批共同编码 |
| `model/model.py:54-60` | 每个视图先以自己的 `c_i,u_i` 重建本视图，再选共同编码分类 | 代码没有显式 `MSE(c_N,c_S)`；随机共同编码没有被送去跨视图重建 |
| `model/model.py:80-84` | `eval()` 也执行随机共同选择，未改为均值 | 本轮应预先确定评价规则；若对两种选择的 logits 求平均，属于确定性评价适配 |
| `wrapper.py:38` | 目标中为 **正** `beta[0]*H_zc` | 与论文式 (12) 的 **负** 熵项不同；按原文实现应使用负号 |
| `wrapper.py:37` | TC 仅接收 `uni_features` | 缺少共同变量，不能实现论文 `TC(C,U_N,U_S)` |
| `utils.py:70` | 用 `torch.tensor([entropy_tensor,...])` 收集边际熵 | 边际熵梯度被切断；应以 `torch.stack` 保留计算图 |
| `utils.py:74-75` | TC 默认除以最大边际熵 | 这是额外归一化，原文式 (14) 没有该分母 |

来源：[model/model.py](https://raw.githubusercontent.com/archy666/CUMI/main/model/model.py)、[wrapper.py](https://raw.githubusercontent.com/archy666/CUMI/main/wrapper.py)、[utils.py](https://raw.githubusercontent.com/archy666/CUMI/main/utils.py)。

官方每视图共同编码器宽度依次为 `ceil(1.2d),ceil(.5d),10K`，独有编码器末维为 `5K`，二者都有 `dropout=.5`；分类器为拼接后的线性层。解码器最后也使用 ReLU。若本轮输入按 Train 均值方差标准化，会出现负值，解码器应允许输出负值；轻量投影宽度与线性解码输出均应作为任务适配记录。上述结构来源同 `model/model.py`。

## 3. 核熵的实际参数与数值注意点

官方 `utils.py` 固定 `alpha=1.01`。其核为 `exp(-squared_distance/sigma)`；`get_kernelsize` 返回的是 `sigma=.15*median(pairwise_Euclidean_distance)`，中位数排除对角线，带宽从 detached CPU 数组估计，未设置有效的正下界。注意这个 `sigma` 不是标准 Gaussian 核中直接可解释的标准差。来源：[utils.py:10-27、99-114](https://raw.githubusercontent.com/archy666/CUMI/main/utils.py)、[wrapper.py:32-33](https://raw.githubusercontent.com/archy666/CUMI/main/wrapper.py)。

以下是对本轮的数值分析与建议，不是作者原文结论：

- 高维潜变量或较大特征范数会使核近单位阵；此时 `H` 接近 `log2(B)`，三变量 TC 接近 `2log2(B)`，损失可能很大但几乎没有有效梯度。相反，全部核接近全 1 矩阵时，熵接近 0。应检查非对角核均值和梯度，不能只确认 loss 非零。
- 官方带宽分母随特征缩放一次、距离平方随缩放两次，最大化熵可能受到尺度变化驱动。可采用 `exp(-d²/(2h²))` 并用批内中位平方距离确定 detached `h²`，或者先规范潜变量尺度再固定带宽。两种均需明确登记为核估计适配。
- `alpha=1.01` 使 `1/(1-alpha)` 放大浮点误差。小批量核熵宜采用 float64、对称化 Gram、`eigvalsh` 和仅用于数值误差的非负特征值处理。批大小、带宽方案在各组一致；最后过小批次可合并或训练时丢弃。
- 不要把 TC 缩小单独解释为成功：全部私有变量塌缩也会降低依赖。共同变量方差、有效秩、重建误差和条件 Double 分类要一起检查。

## 4. 四组实验与可解释边界

本轮固定既有数量预测和 Single 排序，只替换 Double 融合；所有组使用相同神经与散射原始特征、Train 标准化与训练预算。

1. **普通 MLP：**相同输入与近似容量，回答增加散射原始特征本身能带来多少收益。
2. **共同表示：**仅用共同编码分类，回答仅依赖一致信息的效果。
3. **共同＋独有 CE：**与完整组保留相同分类结构，关闭重建、共同熵和 TC，回答通路拆分的结构收益。
4. **完整 CUMI 适配：**采用原文式 (12) 的方向和完整三变量 TC，回答约束是否在相同结构上有额外价值。

完整组相对第 3 组比较的是一组联合约束，不能分别声称 TC、熵最大化、重建各自有效。若共同编码加入一致性 MSE、随机 C 跨视图重建、评价均值化或更改带宽，都应记录；普通正交损失、余弦一致性或停止梯度的 TC 不能称作原文完整 CUMI。

这轮首先验证“三创新”中的**核心模块**，并检验异构框架是否能更好地组织共同与互补证据；它不单独证明任务设置的新颖性，也不直接证明跨录音泛化已经解决。若相同输入 MLP 更好，保留的是新的问题定位和基线结果，不能把分解通路直接认定为已成立的创新。
