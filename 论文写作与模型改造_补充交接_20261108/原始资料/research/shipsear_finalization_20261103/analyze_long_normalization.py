"""Summarize the frozen-weight BN diagnostic and the complete L4 training."""

import json
import sys
from pathlib import Path

import numpy as np

from analyze_long_complement import benchmark_models, paired_analysis
from train_formal import CONDITIONS, ranking, write_json


HERE = Path(__file__).resolve().parent
RUNS = {
    'ShortLocal': HERE / 'runs/SE-Q0-ShortLocal-13',
    'L3': HERE / 'runs/SE-Q0-ShortLocal-L3-13',
    'L4': HERE / 'runs/SE-Q0-ShortLocal-L4-13',
}
FIELDS = ('acc', 'single_acc', 'double_acc', 'noise_acc')


def conclusion_lines(results, scores):
    l3, l4, reference = (results[key] for key in ('L3', 'L4', 'ShortLocal'))
    mean_acc = lambda result: float(np.mean([result['Test'][c]['acc'] for c in CONDITIONS]))
    replay = HERE / 'runs/SE-Q0-ShortLocal-L4-13/cleanup_replay_verified.json'
    return [
        '', '## 结果判断与保留方案', '',
        '1. Train 精确统计回算未改善 Val 背景，说明本次失败不能仅由旧的 BN 均值、方差估计解释。'
        '本轮随后完成的 L4 是归一化算子替换的从头训练对照。',
        '2. L4 在 Test 上明显缓解 L3 的背景问题：'
        f'5 dB 背景由 {l3["Test"]["5dB"]["noise_acc"]:.2f}% 升至 '
        f'{l4["Test"]["5dB"]["noise_acc"]:.2f}%，0 dB 由 '
        f'{l3["Test"]["0dB"]["noise_acc"]:.2f}% 升至 '
        f'{l4["Test"]["0dB"]["noise_acc"]:.2f}%。'
        '但是各档 Val 背景反而降低，尚不能把 Test 的恢复解释为稳定的跨录音背景增益。',
        '3. L4 的 Test 在六档条件下均为 Single 高于 Double，Single、Double 和综合 ACC '
        '均随信噪比恶化而逐档下降。这符合本轮希望得到的曲线方向。'
        f'但 −10 dB Single 仅 {l4["Test"]["-10dB"]["single_acc"]:.2f}%，低于 '
        f'ShortLocal 的 {reference["Test"]["-10dB"]["single_acc"]:.2f}%；'
        '差距收拢同时伴随低信噪比 Single 识别能力损失。',
        f'4. L4 六档 Test 综合均值为 {mean_acc(l4):.2f}%，ShortLocal 为 '
        f'{mean_acc(reference):.2f}%；Val 单双均衡分分别为 '
        f'{scores["L4"][0]:.2f}% 和 {scores["ShortLocal"][0]:.2f}%。'
        'L4 原始综合 ACC 为 84.61%，尚未达到原始综合高于 85% 的目标。'
        '因此本轮将 L4 保留为曲线更规整的归一化备选，暂不替换 ShortLocal 主线。',
        '', '## 结构整理与回放验收', '',
        'ShortLocal 前端已移除随后被丢弃的旧 128 ms 短窗 STFT 计算，直接生成实际使用的 '
        '30 ms log-Mel。保留一维散射、长窗路径、短窗编码器和公共 TabM。'
        '本轮没有新增分类头、分支或可学习参数。', '',
        'ShortLocal、L1、L2、L3 在六档波形夹具、32／7 两种批量下的全部输出特征逐值一致。'
        + ('L4 封存权重已用整理后的源码回放六档 Val，指标、逐条预测和概率全部精确一致。'
           if replay.exists() else 'L4 整理后回放验收记录尚未生成。'), '',
        '[前端整理核对](runs/unused_short_stft_cleanup.json) · '
        '[L4 回放核对](runs/SE-Q0-ShortLocal-L4-13/cleanup_replay_verified.json)',
    ]


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    diagnostic = json.loads(
        (HERE / 'runs/SE-Q0-LongBN-Diagnostic-13/results.json').read_text(encoding='utf-8'))
    results = {name: json.loads((folder / 'results.json').read_text(encoding='utf-8'))
               for name, folder in RUNS.items()}
    histories = {name: json.loads((folder / 'history.json').read_text(encoding='utf-8'))
                 for name, folder in RUNS.items()}
    for source in ('ShortLocal', 'L3'):
        for left, right in zip(histories[source], histories['L4']):
            if left['condition_counts'] != right['condition_counts']:
                raise ValueError(f'L4 sampling differs from {source}.')
    paired = paired_analysis(RUNS)
    latency = benchmark_models(results, RUNS)
    scores = {name: ranking(result['Val']) for name, result in results.items()}
    foreground = {}
    for name, result in results.items():
        foreground[name] = {}
        for condition, row in result['Test'].items():
            confusion = np.asarray(row['confusion'])
            count = int(confusion[1:].sum())
            correct = int(np.trace(confusion[1:, 1:]))
            foreground[name][condition] = {
                'n': count, 'correct': correct, 'acc': 100 * correct / count,
                'background_confusion': confusion[0].tolist()}
    write_json(HERE / 'runs/long_normalization_L4_analysis.json', {
        'diagnostic_gate': diagnostic['gate'], 'validation_rank': scores,
        'highest_validation_rank': max(scores, key=scores.get), 'paired_common_original': paired,
        'end_to_end_latency': latency, 'foreground_diagnostic': foreground,
        'training_noise_counts_matched': True,
    })
    lines = [
        '# 长窗归一化诊断与 L4 完整训练报告', '',
        '统计诊断使用封存 L3 权重，仅估计 Train 上的长窗 BN 统计量。'
        'L4 按预先规定的诊断分支，将长窗 BatchNorm2d 替换为 GroupNorm，将输出 BatchNorm1d 替换为 LayerNorm，'
        '从种子 13 的模型构造初值完整训练。', '',
        '## 冻结权重统计诊断', '',
        '校准使用全部 Train 音频，按原加噪概率生成固定视图，先回算上游层，再估计下游层。'
        '只修改长窗 BN 的均值与方差，全部可学习参数及其余缓冲区经逐值核对一致，优化器更新为零。'
        '效果判断采用六档 Val。每格为综合／Single／Double／背景，单位为 %。', '',
        '| 条件 | 原 L3 | Train 统计回算后 |', '| --- | ---: | ---: |',
    ]
    for condition in CONDITIONS:
        cells = ['／'.join(f'{diagnostic[split][condition][key]:.2f}' for key in FIELDS)
                 for split in ('Val_before', 'Val_after')]
        lines.append('| ' + ' | '.join((condition, *cells)) + ' |')
    gate = diagnostic['gate']
    lines.extend([
        '', f'10／5／0 dB 的 Val 背景均值变化为 {gate["background_gain_pp"]:+.2f} 个百分点，'
        f'Single、Double 六档均值分别变化 {gate["single_change_pp"]:+.2f}、'
        f'{gate["double_change_pp"]:+.2f} 个百分点。预定门槛未通过，采用 record 分支完整重训。', '',
        '## 完整训练与选模', '',
        '共有可学习参数初值已核对一致。三版使用同一录音划分、损失和噪声协议，共同轮次的采样计数一致。'
        '逐条 Val／Test 的编号、标签、上下文经配对核对。', '',
        '| 模型 | 参数量 | 训练轮数 | 封存轮次 | Val 单双均衡分 | Val 综合均值 | Test 综合均值 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |',
    ])
    for name, result in results.items():
        selection = result['selection']
        test_mean = np.mean([result['Test'][c]['acc'] for c in CONDITIONS])
        lines.append(
            f'| {name} | {result["parameters"]:,} | {selection["epochs_trained"]} | '
            f'{selection["epoch"]} | {scores[name][0]:.2f}% | {scores[name][1]:.2f}% | {test_mean:.2f}% |'
        )
    for split in ('Val', 'Test'):
        lines.extend(['', f'## {split}', '', '每格为综合／Single／Double／背景，单位为 %。', '',
                      '| 条件 | ShortLocal | L3 | L4 |', '| --- | ---: | ---: | ---: |'])
        for condition in CONDITIONS:
            cells = ['／'.join(f'{results[name][split][condition][key]:.2f}' for key in FIELDS)
                     for name in RUNS]
            lines.append('| ' + ' | '.join((condition, *cells)) + ' |')
    lines.extend(['', '## 原始档共同判对样本保持率', '',
                  '每格依次为 ShortLocal／L3／L4。无共同判对样本的分组以 — 表示。', '',
                  '| 集合与分组 | 样本数 | 0 dB | −5 dB | −10 dB |',
                  '| --- | ---: | ---: | ---: | ---: |'])
    for split in ('Val', 'Test'):
        for group in ('ship', 'single', 'double', 'background'):
            row = paired[split][group]
            cells = []
            for condition in ('0dB', '-5dB', '-10dB'):
                rates = [row['conditions'][condition][name]['retention'] for name in RUNS]
                cells.append('／'.join('—' if rate is None else f'{rate:.2f}' for rate in rates))
            lines.append('| ' + ' | '.join((f'{split} {group}', str(row['n']), *cells)) + ' |')
    lines.extend(['', '## Test 数量误判', '',
                  '每格依次为 ShortLocal／L3／L4，单位为条。', '',
                  '| 条件 | Single→Double | Single→背景 | Double→Single | Double→背景 | Double组合错 |',
                  '| --- | ---: | ---: | ---: | ---: | ---: |'])
    for condition in CONDITIONS:
        cells = ['／'.join(str(results[name]['Test'][condition][key]) for name in RUNS)
                 for key in ('single_to_double', 'single_to_background', 'double_to_single',
                             'double_to_background', 'double_pair_errors')]
        lines.append('| ' + ' | '.join((condition, *cells)) + ' |')
    lines.extend(['', '## 计算成本', '',
                  '推理采用同一批 32 条 Train 原始波形，包含全部特征提取，三个预热批后取十次测量中位数。', '',
                  '| 模型 | 每轮中位耗时（秒） | 波形推理批耗时（毫秒） |',
                  '| --- | ---: | ---: |'])
    for name, result in results.items():
        lines.append(
            f'| {name} | {result["benchmark"]["median_epoch_seconds"]:.2f} | '
            f'{latency[name]["median_batch_ms"]:.2f} |')
    lines.extend(conclusion_lines(results, scores))
    lines.extend(['', '[实验协议](长窗归一化L4实验协议.md) · '
                  '[统计诊断](runs/SE-Q0-LongBN-Diagnostic-13/results.json) · '
                  '[成对分析](runs/long_normalization_L4_analysis.json)', ''])
    for name, folder in RUNS.items():
        relative = folder.relative_to(HERE).as_posix()
        lines.extend([
            f'{name} 的 [配置]({relative}/config_resolved.json)、'
            f'[Val 封存]({relative}/selection_sealed.json)、[完整结果]({relative}/results.json)。', ''])
    path = HERE / '长窗归一化诊断与L4完整训练报告.md'
    path.write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({'report': str(path), 'validation_rank': scores}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
