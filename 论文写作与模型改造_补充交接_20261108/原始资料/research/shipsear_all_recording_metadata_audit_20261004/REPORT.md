# ShipsEar 全部目标录音的元数据适用性审计

日期：2026-10-04。核查官方数据库五页共90条，完整匹配原始78条目标录音、1999个5秒切片；其余12条为自然背景。此前排除52/58/66的实验仍保留75条1897片；本次新增列出的71及弱信号8条等都仍在该实验目标集合中。CSV单列标明该实验的保留状态。审计只使用官方目录事实及当前划分清单，未读取模型预测或准确率，未修改实验清单、音频或模型。

## 明确结论

除56的无发动机通过、23的靠泊结束背景声外，没有发现官方说明中同等明确的新增无动力或事件后背景目标录音。这并不证明其余每片都有类别信息。即使23/56，也不能说已经证明波形中绝对不存在船舶声学特征；它们被排除的充分依据是与“有动力船舶目标事件”的任务范围不符。

71是前一轮只检查52/58/66时漏掉的关键干扰记录，应补入：目标信号弱且其他类别船只干扰强。因此，先前3条排除清单并不是完整的元数据质量审计结果。

## 1. 任务范围明确不符，以及新发现的强干扰

|编号|划分|类别|5秒片数|官方说明的相关事实|
|---|---|---|---:|---|
|23|Test|D · Ocean liner|1|明确记为靠泊操作结束后的背景声；也有Rio Caxil及港口背景。不等于证明所有机器已关闭。|
|56|Test|B · Sailboat|9|帆船近距离无发动机通过，背景噪声较大；不满足有动力目标任务，但不等于绝对无声。|
|71|Train|D · Ocean liner|27|邮轮目标信号小且干扰高；快艇及Playa Beluso渔船干扰；D与B/A跨类，无视频。|

来源：[官方第2页](https://underwaternoise.atlanttic.uvigo.es/indexDB.php?pageNum_leerMedidas=1&totalRows_leerMedidas=90)、[第3页](https://underwaternoise.atlanttic.uvigo.es/indexDB.php?pageNum_leerMedidas=2&totalRows_leerMedidas=90)、[第4页](https://underwaternoise.atlanttic.uvigo.es/indexDB.php?pageNum_leerMedidas=3&totalRows_leerMedidas=90)。23未明确说邮轮所有发动机/辅机关闭；71未明确说目标没有特征。

## 2. 除71外，另8条明确弱目标

|编号|划分|类别|5秒片数|官方说明的相关事实|
|---|---|---|---:|---|
|26|Train|B · Motorboat|4|快艇通过，目标信号小；另有Rio Caxil及港口背景。|
|29|Train|B · Pilot boat|18|引航船在邮轮到来前进港，目标信号小；另有Rio Caxil及港口背景。|
|34|Train|C · Passenger ferry|24|目标客船进港，信号低，另一客船在远处；同类多船及港口声。|
|35|Train|C · Passenger ferry|19|目标客船离港，信号低，另一客船远处进港；同类多船及港口声。|
|36|Val|C · Passenger ferry|20|目标客船离港，信号低，另一客船远处进港；同类多船及港口声。|
|39|Train|B · Motorboat|14|快艇进港，信号低，港口噪声较大。|
|41|Train|C · Passenger ferry|10|目标客船进港，信号低，另一客船远处离港；同类多船及港口声。|
|45|Train|B · Motorboat|18|未识别身份的游艇离港，信号低，港口噪声较大。|

这8条共127片，Train 7条107片，Val 1条20片，没有Test；加71后弱信号共9条154片。官方只给相对强弱描述，没有SNR阈值或逐片可识别性标注。“信号低”可以形成预先固定的质量子集条件，但不能写成“没有类别特征”。来源：[官方第2页](https://underwaternoise.atlanttic.uvigo.es/indexDB.php?pageNum_leerMedidas=1&totalRows_leerMedidas=90)。

## 3. 其他混合来源或背景问题

|编号|划分|类别|5秒片数|官方说明的相关事实|
|---|---|---|---:|---|
|25|Train|D · Ocean liner|28|靠泊进港；渔船和协助靠泊的引航船可能干扰，另有Rio Caxil背景。|
|48|Test|A · Mussel boat|16|贻贝作业船录音中有远处客船；标题标记干扰，强度及片段界限不明。|
|51|Train|B · Motorboat|20|快艇高速通过，其他船只造成干扰且并非始终可见；干扰船类别未给出。|
|72|Val|B · Motorboat|24|快艇通过，录音末尾可能有干扰；未明确弱信号或无动力。|
|75|Val|A · Fishing boat|10|约8米渔船通过，有来源未知的背景声；不是目标缺失证据。|
|78|Train|D · Ro-Ro|37|滚装船通过，后有引航船跟随；跨类船出现，但未说明声学干扰强度。|

51明确有其他船干扰，不能认证为纯单源素材。25/72是可能干扰；48有远处其他类别客船，未量化强度；75来源不明；78只是说明引航船跟随，不能把视觉/场景共现直接等同于强声学干扰。

为完整起见：22—31整组共有Rio Caxil小幅背景船声及港口声；32/34/35/36/38/40/41/43/60另有同类别客船出现；62则注明另一船交会似乎不构成干扰。同类多船未必破坏四分类标签，但不能由此称为物理纯单源。52/58/66是此前已检查的跨类干扰，其中只有58在官方原文中明确使用强干扰程度描述；52/66说明有干扰但未量化程度。

来源：[官方第2页](https://underwaternoise.atlanttic.uvigo.es/indexDB.php?pageNum_leerMedidas=1&totalRows_leerMedidas=90)、[第3页](https://underwaternoise.atlanttic.uvigo.es/indexDB.php?pageNum_leerMedidas=2&totalRows_leerMedidas=90)、[第4页](https://underwaternoise.atlanttic.uvigo.es/indexDB.php?pageNum_leerMedidas=3&totalRows_leerMedidas=90)。

## 4. 容易被误排的录音

|编号|划分|类别|5秒片数|官方说明的相关事实|
|---|---|---|---:|---|
|7|Train|C · Passenger ferry|12|等待旅客下船；未说明发动机关闭。|
|9|Test|C · Passenger ferry|12|等待旅客登船；未说明发动机关闭。|
|14|Val|C · Passenger ferry|12|等待旅客登船；未说明发动机关闭。|
|16|Train|D · Ocean liner|32|水听器与船之间无直接视线；不是没有船声的证据。|
|31|Test|A · Tugboat|12|拖船离泊时发动机启停，官方评价信号中等至好；不能称为没有发动机特征。|
|37|Val|B · Sailboat|8|帆船启动发动机离港，官方评价信号好。|
|57|Val|B · Sailboat|3|远处帆船使用发动机；不能归为无动力。|
|68|Train|B · Sailboat|59|测量用帆船返回回收水听器；未明确交代是否开动发动机，不能推定关闭。|
|93|Test|A · Dredger|6|疏浚船项目测量、海底主要为泥；未说明发动机关闭或目标信号缺失。|

等待不等于无动力；启停不等于全段关机；远距离不等于目标消失；低准确率不等于没有信号。18—21的H2饱和备注也不够证明当前发布音频削波。69/70没有弱信号或无动力说明。

来源：[官方第1页](https://underwaternoise.atlanttic.uvigo.es/indexDB.php)、[第2页](https://underwaternoise.atlanttic.uvigo.es/indexDB.php?pageNum_leerMedidas=1&totalRows_leerMedidas=90)、[第3页](https://underwaternoise.atlanttic.uvigo.es/indexDB.php?pageNum_leerMedidas=2&totalRows_leerMedidas=90)、[第4页](https://underwaternoise.atlanttic.uvigo.es/indexDB.php?pageNum_leerMedidas=3&totalRows_leerMedidas=90)、[第5页](https://underwaternoise.atlanttic.uvigo.es/indexDB.php?pageNum_leerMedidas=4&totalRows_leerMedidas=90)。

## 使用边界与下一步

建议任务范围排除23/56；明确跨类干扰隔离52/58/66/71；51另列确定混合来源；弱目标8条独立成组。每个排除理由独立报告，不用识别结果来决定保留谁。本次没有执行新增训练，也没有把候选列表写进现有正式数据生成或排除清单。

若后续据此改变测试集，必须同时保留原完整测试结果，并将新结果称为筛选子集结果。历史Test已查看，不能把此后固定的规则说成首次盲测前预注册。元数据审计可以界定任务适用范围，不能单独证明所有保留录音逐片可识别，更不能保证超过某个准确率。

## 交付物

- `all_78_recording_audit.csv`：全部78条逐录音事实、证据标记、解释、官方来源及原划分。
- `target_metadata.json` / `official_catalogue.json`：本地目标与官方事实元数据的对应表。
- `sources/official_page_0.html`—`official_page_4.html`及`source_manifest.json`：网页快照与SHA256。
- `collect_metadata.py`、`write_audit.py`：采集及生成脚本。

自然背景类81—92未在78条目标列表中，不混入本次待排除船舶清单。此次没有听音判定或逐片目标归属标注。
