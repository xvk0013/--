# ShipsEar 全量官方元数据独立复核

日期：2026-10-04。范围：独立重新阅读官方目录第 0–4 页全部 90 条记录，并与本地 78 条目标录音清单比对。复核覆盖目标 ID 共 78 个，与本地集合差异为 0。本轮没有听检音频、看视频、运行模型或根据预测筛选文件。

## 结论

**没有找到 56、23 之外，官方 notes 同等明确标注“无发动机通过”或“靠泊结束后的背景事件”的目标录音。** 这也不能把 56、23 提升为数学意义的“完全没有可识别特征”：56 缺发动机声不代表缺全部船体/水动力声；23 的事件描述不能排除残余船声。能够确认的是它们不适合未经核验就当作典型动力船辐射噪声。

新增最值得优先核验的是 **71（Discovery UK）**：官方同时注明目标声弱、干扰强，并点名快艇和渔船。这属于“目标信息可能被遮蔽”，不能据此断言“没有目标信息”。

## 1. 完整的明确弱信号清单

官网使用低/小信号描述的目标录音共 **9 条：26、29、34、35、36、39、41、45、71**。其中 26、29、71 对应 small signal；其余 6 条对应 low signal。没有把远距、small yacht、低海浪、低背景声或中等增益误算为目标声弱。

|ID|本地类别 / split / 5 s core 数|最小证据摘要|
|---|---|---|
|26|B / Train / 4|快艇通过，信号小。|
|29|B / Train / 18|引航船到来，信号小。|
|34|C / Train / 24|Mar de Onza 到港，信号低，远处还有另一客船。|
|35|C / Train / 19|Mar de Onza 离港，信号低，远处还有另一客船。|
|36|C / Val / 20|Minho uno 离港，信号低，远处还有另一客船。|
|39|B / Train / 14|快艇 Duda 到港，信号低。|
|41|C / Train / 10|Minho uno 到港，信号低，远处还有另一客船。|
|45|B / Train / 18|游艇离港，信号低。|
|71|D / Train / 27|Discovery UK 离开海湾，目标声小且干扰强。|

26–45 的证据来自 [官方第 1 页](https://underwaternoise.atlanttic.uvigo.es/indexDB.php?pageNum_leerMedidas=1&totalRows_leerMedidas=90)；71 来自 [官方第 3 页](https://underwaternoise.atlanttic.uvigo.es/indexDB.php?pageNum_leerMedidas=3&totalRows_leerMedidas=90)。split/core 数来自 [本地清单](../shipsear_metadata_inventory_20261004.csv)，这些是清单计数，并非新做切片。总计 154 个 core，不能将该数量解释成无效样本数。

## 2. 明确强干扰：只有 58、71 使用这个强度级别的描述

|ID|官方所述目标及问题|本地信息|
|---|---|---|
|58|RORO 通过时受到贻贝船强干扰；>100 m；H2。|D / Train / 47 core|
|71|邮轮声弱，且快艇、Playa Beluso 渔船干扰；>100 m；H3；无视频。|D / Train / 27 core|

来源：[58，第 2 页](https://underwaternoise.atlanttic.uvigo.es/indexDB.php?pageNum_leerMedidas=2&totalRows_leerMedidas=90)、[71，第 3 页](https://underwaternoise.atlanttic.uvigo.es/indexDB.php?pageNum_leerMedidas=3&totalRows_leerMedidas=90)。这里特指船只干扰强度，不把普遍的港噪、风浪或岸边石块噪声都称为强跨类干扰。

## 3. 其他混合声源/干扰证据：不能一律归成“无特征”

|证据性质|ID|简要说明|
|---|---|---|
|可能跨类别干扰|25|邮轮靠泊时可能有渔船和引航船干扰。|
|同类多船场景|38、40、43|多艘客船同时进出；不能仅因多船就判四大类标签错。|
|远处同类船|32、34、35、36、41、60|备注提到另一客船；34/35/36/41 也在弱信号清单。|
|远处异类船|48|贻贝船录音，远处有客船。|
|来源未全部可视的船声干扰|51|快艇高速通过，其他船只干扰并非始终可见。|
|明确跨类别干扰|52、66|52 快艇受贻贝船干扰；66 贻贝船中段受开机帆船干扰。|
|局部可疑干扰|72|快艇录音末尾可能有干扰。|
|未知背景|75|渔船通过，背景声来源不明。|
|跨类别伴行|78|RORO 后面跟着引航船；notes 未说明引航船声是否主导。|

来源：25、32、34–36、41 为 [第 1 页](https://underwaternoise.atlanttic.uvigo.es/indexDB.php?pageNum_leerMedidas=1&totalRows_leerMedidas=90)；38、40、43 为 [第 0 页](https://underwaternoise.atlanttic.uvigo.es/indexDB.php?pageNum_leerMedidas=0&totalRows_leerMedidas=90)；48、51、52、60、66 为 [第 2 页](https://underwaternoise.atlanttic.uvigo.es/indexDB.php?pageNum_leerMedidas=2&totalRows_leerMedidas=90)；72、75、78 为 [第 3 页](https://underwaternoise.atlanttic.uvigo.es/indexDB.php?pageNum_leerMedidas=3&totalRows_leerMedidas=90)。

此外，22–31 的现场记录共同提到 Rio Caxil 的较小背景声及港噪，不能把这种背景存在直接升级为强干扰或标签错误。[第 1 页](https://underwaternoise.atlanttic.uvigo.es/indexDB.php?pageNum_leerMedidas=1&totalRows_leerMedidas=90)

## 4. 对容易误判的录音逐项排除过度解释

- **69**：官网是 Costa Voyager 离开海湾、约 15 节、H2；没有无动力、弱信号或高干扰记录。**70**：6 m 游艇通过、H2；small 指船尺寸，不是信号弱。**72**：仅末尾可能干扰，不能认定整条无特征。[第 3 页](https://underwaternoise.atlanttic.uvigo.es/indexDB.php?pageNum_leerMedidas=3&totalRows_leerMedidas=90)
- **7、9、14**：等候上下客不等于发动机关闭。**16**：目标与水听器位置没有直接视线，不等于无声传播。**18–21**：H2 饱和是设备通道情况，不能未经检查认定本地发布音频已削波，且 notes 说明无明显船只干扰。[第 0 页](https://underwaternoise.atlanttic.uvigo.es/indexDB.php?pageNum_leerMedidas=0&totalRows_leerMedidas=90)
- **31**：拖船发动机启停，但备注评价信号中等至好；**37**：帆船启动发动机后离港，信号好。两者不应类比 56。[第 1 页](https://underwaternoise.atlanttic.uvigo.es/indexDB.php?pageNum_leerMedidas=1&totalRows_leerMedidas=90)
- **46、57**：远处采集不等于无信息；57 还明确开机。**62**：末尾有船交会，但备注认为未形成明显干扰，不能只匹配到“船交会”就删除。[第 2 页](https://underwaternoise.atlanttic.uvigo.es/indexDB.php?pageNum_leerMedidas=2&totalRows_leerMedidas=90)
- **93–96**：均描述挖泥船项目测量和泥质海床，没有目标停机或声源缺失说明。[第 4 页](https://underwaternoise.atlanttic.uvigo.es/indexDB.php?pageNum_leerMedidas=4&totalRows_leerMedidas=90)

## 5. 覆盖核验

每页目标 ID 如下；这是对照索引，不是异常清单。90 条里其余 12 条是官方环境噪声类，不属于本地 78 条目标录音。

|官网 pageNum|已审目标 ID|目标数|
|---|---|---|
|0|80,10,14,15,6,7,13,11,12,17,8,9,16,21,18,19,20,38,40,43|20|
|1|42,41,39,45,26,27,28,31,29,30,24,25,22,23,37,36,34,33,35,32|20|
|2|46,54,56,57,47,50,48,53,51,52,49,55,62,63,67,58,66,60,59,65|20|
|3|61,64,68,71,77,75,79,73,76,72,74,70,69,78|14|
|4|93,94,95,96|4|

使用这些 ID 与本地 inventory 的 recording_id 做集合比较，78 对 78、差异 0。不存在“未看第 3 页而漏掉 71”的覆盖问题。

## 6. 可以给用户的准确回答

若要求的是“像 56、23 一样，有官方事件描述直接质疑是否在录目标动力声”，目前没有第三条同等明确记录。若问“哪些录音的目标可辨识性值得优先怀疑”，应首先指向 71 和 58，其次是 26、29、34、35、36、39、41、45 的弱信号，再单独核验 25、48、51、52、66、72、75、78 等归属/干扰问题。

这一列表给出人工复核顺序，绝不是整文件剔除名单。要判断具体 5 s 是否没有足够类别证据，还需不看模型预测的片段听检、声谱检查和可得事件证据，并保留“无法确认”。现有 metadata 也不能把没有异常注释的录音证明为全部干净。训练/测试正确与否不参与本次证据判定。
