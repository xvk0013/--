# Train内部录音留出的专家误差训练

固定正式数据与原工作模型。Train内部选取一组完整录音，排除任何含这些录音的Single和Double后，从随机初始化训练辅助M2＋DEMON数量／船型专家及散射TabM。另训练同结构、相同初始化和更新预算的全Train辅助专家作见过录音的对照。各TabM标准化仅使用本组专家的训练数据。

两组507参数的路由校正器在相同内部校正样本上训练，分别读取两套辅助专家的13维语义概率。评价时两者都接回同一原工作模型；不混用不同编码器的隐藏特征，不输入真实数量。背景判定、Single船型与Double组合排序固定，修改单／双选择。

辅助专家固定60个全Train轮次等效更新预算；同组实际样本数不同，但两套专家的更新次数相同。校正器最多60轮、早停15轮，按Val封存后评价Test。不搜索随机种子，不移动正式集合，不改变原增强，不增加噪声，不使用外部预训练权重。

```powershell
& 'C:\Users\Admin\anaconda3\envs\zpfenv1\python.exe' -B -u run_holdout.py
```

正式输出位于`runs/holdout_seed13`。辅助专家不进入最终推理，路由器通过`HoldoutRouterRecognizer(model.json).predict(waves)`读取混合音频。
