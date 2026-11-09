"""Report the learned long-window L3 candidate against ShortLocal."""

import json
import sys
from pathlib import Path

import numpy as np

from analyze_long_complement import FIELDS, benchmark_models, paired_analysis
from train_formal import CONDITIONS, ranking, write_json


HERE = Path(__file__).resolve().parent
RUNS = {
    'ShortLocal': HERE / 'runs/SE-Q0-ShortLocal-13',
    'L3': HERE / 'runs/SE-Q0-ShortLocal-L3-13',
}


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    results = {
        name: json.loads((folder / 'results.json').read_text(encoding='utf-8'))
        for name, folder in RUNS.items()
    }
    histories = {
        name: json.loads((folder / 'history.json').read_text(encoding='utf-8'))
        for name, folder in RUNS.items()
    }
    for old, new in zip(histories['ShortLocal'], histories['L3']):
        if old['condition_counts'] != new['condition_counts']:
            raise ValueError('Training noise sampling differs.')
    paired = paired_analysis(RUNS)
    latency = benchmark_models(results, RUNS)
    scores = {name: ranking(result['Val']) for name, result in results.items()}
    winner = max(scores, key=scores.get)
    write_json(HERE / 'runs/short_local_L3_analysis.json', {
        'validation_rank': scores, 'highest_validation_rank': winner,
        'training_noise_counts_matched': True, 'paired_common_original': paired,
        'end_to_end_latency': latency,
    })
    lines = [
        '# 长窗频率局部编码 L3 完整训练报告', '',
        'L3 在同一 512 ms 长窗对数功率谱上，以频率重排和轻量残差编码替换固定二维散射。'
        '两版均从种子 13 的构造初值训练，共有参数初值已核对一致，使用共同的新 MB 划分、损失和加噪协议。',
        '全部 Val、Test 的样本编号、标签与上下文逐条对齐，共同轮次的六档训练采样计数一致。', '',
        '| 模型 | 参数量 | 训练轮数 | 封存轮次 | Val 单双均衡分 | Val 综合均值 | Test 综合均值 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    for name, result in results.items():
        selection = result['selection']
        mean = np.mean([result['Test'][c]['acc'] for c in CONDITIONS])
        lines.append(
            f'| {name} | {result["parameters"]:,} | {selection["epochs_trained"]} | '
            f'{selection["epoch"]} | {scores[name][0]:.2f}% | {scores[name][1]:.2f}% | {mean:.2f}% |'
        )
    lines.extend(['', f'按既定六档 Val 排序，最高分为 {winner}。', '',
                  '每格依次为综合／Single／Double／背景准确率，单位为 %。'])
    for split in ('Val', 'Test'):
        lines.extend(['', f'## {split}', '', '| 条件 | ShortLocal | L3 | 综合变化（百分点） |',
                      '| --- | ---: | ---: | ---: |'])
        for condition in CONDITIONS:
            old, new = (results[name][split][condition] for name in RUNS)
            cells = ['／'.join(f'{row[key]:.2f}' for key in FIELDS) for row in (old, new)]
            lines.append('| ' + ' | '.join((condition, *cells, f'{new["acc"]-old["acc"]:+.2f}')) + ' |')
    lines.extend(['', '## 原始档共同判对样本的加噪保持率', '',
                  '每格依次为 ShortLocal／L3，分母在各噪声档保持一致。', '',
                  '| 集合与分组 | 样本数 | 0 dB | −5 dB | −10 dB |',
                  '| --- | ---: | ---: | ---: | ---: |'])
    for split in ('Val', 'Test'):
        for group in ('ship', 'single', 'double', 'background'):
            row = paired[split][group]
            cells = ['／'.join(f'{row["conditions"][c][name]["retention"]:.2f}' for name in RUNS)
                     for c in ('0dB', '-5dB', '-10dB')]
            lines.append('| ' + ' | '.join((f'{split} {group}', str(row['n']), *cells)) + ' |')
    lines.extend(['', '## Test 数量误判', '',
                  '每格依次为 ShortLocal／L3，单位为条。', '',
                  '| 条件 | Single→Double | Single→背景 | Double→Single | Double→背景 | Double组合错 |',
                  '| --- | ---: | ---: | ---: | ---: | ---: |'])
    error_keys = ('single_to_double', 'single_to_background', 'double_to_single',
                  'double_to_background', 'double_pair_errors')
    for condition in CONDITIONS:
        cells = ['／'.join(str(results[name]['Test'][condition][key]) for name in RUNS)
                 for key in error_keys]
        lines.append('| ' + ' | '.join((condition, *cells)) + ' |')
    lines.extend(['', '## 计算成本', '',
                  '波形到集合 logits 的推理计时采用同一批 32 条 Train 原始波形，包含传输、固定特征提取和网络。'
                  '三个预热批后，对十次测量取中位数。', '',
                  '| 模型 | Train 原始特征准备（秒） | Val 六档特征准备（秒） | 每轮中位耗时（秒） | 波形推理批耗时（毫秒） |',
                  '| --- | ---: | ---: | ---: | ---: |'])
    for name, result in results.items():
        bench = result['benchmark']
        lines.append(
            f'| {name} | {bench["train_cache_seconds"]:.2f} | {bench["val_cache_seconds"]:.2f} | '
            f'{bench["median_epoch_seconds"]:.2f} | {latency[name]["median_batch_ms"]:.2f} |'
        )
    lines.extend(['', '[实验协议](长窗频率局部编码L3实验协议.md) · '
                  '[结构实现](short_local.py) · [成对统计](runs/short_local_L3_analysis.json)', ''])
    for name, folder in RUNS.items():
        rel = folder.relative_to(HERE).as_posix()
        lines.extend([
            f'{name} 的 [解析配置]({rel}/config_resolved.json)、'
            f'[Val 封存]({rel}/selection_sealed.json)、[完整指标]({rel}/results.json)。', '',
        ])
    target = HERE / '长窗频率局部编码L3完整训练报告.md'
    target.write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({'report': str(target), 'validation_rank': scores,
                      'highest_validation_rank': winner}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
