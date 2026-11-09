# DeepShip生成结果

状态：complete；版本：deepship_multitarget_0db_energy80_seen_ship_v1；模式：full。

| 集合 | Single | Double | 合成背景 | 总数 | 单源录音数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 3600 | 7200 | 1200 | 12000 | 426 |
| Val | 600 | 1200 | 200 | 2000 | 93 |
| Test | 600 | 1200 | 200 | 2000 | 90 |

原录音归属沿用筛选索引；源片段每录音/事件上限为20，每类各集合配额为{'Train': 900, 'Val': 150, 'Test': 150}。preserve_existing_selected=True，开启时保留初筛已经选入的来源，再补到目标数。smoke仅用于小规模贯通。

信道：已有Bellhop E99 FIR。使用真实核心前史/后视，不给每个5秒边界人为补零。

Double在接收组件、包络之前设置SIR=0.0dB；C0全重叠、C1单侧占优、C2交替占优，三个集合使用同一包络参数分布。最后施加公共增益至RMS=0.05。

背景为Wenz三成分合成负样本。Single基础数组不加人工包络，Train加载器可显式启用包络增强。原船声保留天然背景；后续SNR指相对已有接收波形的追加噪声比。

DATA_INDEX.csv为最终唯一样本排列索引；data/Train、Val、Test/audio.npy为float32波形。components/audio.npy用于复用单源传播，STATE.json保存组件映射与物理处理信息。

## 类别与组合

| 集合 | Single各类 | Double各组合 | C0/C1/C2 |
| --- | --- | --- | --- |
| Train | {'Cargo': 900, 'Passengership': 900, 'Tanker': 900, 'Tug': 900} | {'Cargo+Passengership': 1200, 'Cargo+Tanker': 1200, 'Cargo+Tug': 1200, 'Passengership+Tanker': 1200, 'Passengership+Tug': 1200, 'Tanker+Tug': 1200} | {'C2': 1800, 'C0': 3600, 'C1': 1800} |
| Val | {'Cargo': 150, 'Passengership': 150, 'Tanker': 150, 'Tug': 150} | {'Cargo+Passengership': 200, 'Cargo+Tanker': 200, 'Cargo+Tug': 200, 'Passengership+Tanker': 200, 'Passengership+Tug': 200, 'Tanker+Tug': 200} | {'C1': 300, 'C0': 600, 'C2': 300} |
| Test | {'Cargo': 150, 'Passengership': 150, 'Tanker': 150, 'Tug': 150} | {'Cargo+Passengership': 200, 'Cargo+Tanker': 200, 'Cargo+Tug': 200, 'Passengership+Tanker': 200, 'Passengership+Tug': 200, 'Tanker+Tug': 200} | {'C0': 600, 'C1': 300, 'C2': 300} |

## 波形输出

| 集合 | RMS最小 | RMS最大 | 绝对峰值最大 |
| --- | ---: | ---: | ---: |
| Train | 0.05000000 | 0.05000000 | 2.53381467 |
| Val | 0.05000000 | 0.05000000 | 1.24286258 |
| Test | 0.05000000 | 0.05000000 | 1.73530495 |

耗时：2.42分钟。
