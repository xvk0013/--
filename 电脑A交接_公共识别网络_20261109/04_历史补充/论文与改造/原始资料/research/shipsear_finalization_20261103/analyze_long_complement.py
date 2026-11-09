"""Summarize the complete L1/L2 runs and their paired noise predictions."""

import json
import time
from pathlib import Path

import numpy as np
import torch

from train_formal import CONDITIONS, FixedDataset, create_model, ranking, write_json
from short_local import ShortLocalViews


HERE = Path(__file__).resolve().parent
RUNS = {
    'ShortLocal': HERE / 'runs/SE-Q0-ShortLocal-13',
    'L1': HERE / 'runs/SE-Q0-ShortLocal-L1-13',
    'L2': HERE / 'runs/SE-Q0-ShortLocal-L2-13',
}
FIELDS = ('acc', 'single_acc', 'double_acc', 'noise_acc')
ERRORS = ('single_to_double', 'single_to_background', 'double_to_single',
          'double_to_background', 'double_pair_errors')


def paired_analysis(runs=RUNS):
    analysis = {}
    for split in ('Val', 'Test'):
        aligned = {}
        reference = None
        for condition in CONDITIONS:
            aligned[condition] = {}
            for name, folder in runs.items():
                with np.load(folder / f'{split}_{condition}.npz') as saved:
                    if reference is None:
                        reference = {key: saved[key] for key in ('ids', 'labels', 'contexts')}
                    for key, expected in reference.items():
                        np.testing.assert_array_equal(saved[key], expected)
                    aligned[condition][name] = saved['predictions'] == reference['labels']
        counts = np.asarray((0, 1, 1, 1, 2, 2, 2))[reference['labels']]
        groups = {'ship': counts > 0, 'single': counts == 1,
                  'double': counts == 2, 'background': counts == 0}
        common = np.logical_and.reduce(list(aligned['original'].values()))
        analysis[split] = {}
        for group, mask in groups.items():
            mask = mask & common
            rows = {}
            for condition, correct in aligned.items():
                rows[condition] = {
                    name: {'correct': int((values & mask).sum()),
                           'retention': float(values[mask].mean() * 100) if mask.any() else None}
                    for name, values in correct.items()
                }
            analysis[split][group] = {'n': int(mask.sum()), 'conditions': rows}
    return analysis


@torch.inference_mode()
def benchmark_models(results, runs=RUNS):
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    data = FixedDataset(results['ShortLocal']['config']['data_root'], 'Train')
    waves = data.batch(np.arange(32))
    timing = {}
    for name, folder in runs.items():
        saved = torch.load(folder / 'best.pt', map_location='cpu', weights_only=False)
        model, optimizer = create_model(saved['config'], saved['normalization'])
        del optimizer
        model.load_state_dict(saved['model'], strict=True)
        model.eval()
        fixed = ShortLocalViews(saved['config']).cuda().eval()
        measurements = []
        for index in range(13):
            torch.cuda.synchronize()
            begun = time.perf_counter()
            features = fixed(waves)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                model(*features)
            torch.cuda.synchronize()
            if index >= 3:
                measurements.append((time.perf_counter() - begun) * 1000)
        timing[name] = {'batch_size': 32, 'warmup_batches': 3, 'measured_batches': 10,
                        'median_batch_ms': float(np.median(measurements)),
                        'median_sample_ms': float(np.median(measurements) / 32)}
        del fixed, model, saved, features
    return timing


def main():
    results = {
        name: json.loads((folder / 'results.json').read_text(encoding='utf-8'))
        for name, folder in RUNS.items()
    }
    histories = {
        name: json.loads((folder / 'history.json').read_text(encoding='utf-8'))
        for name, folder in RUNS.items()
    }
    for variant in ('L1', 'L2'):
        for left, right in zip(histories['ShortLocal'], histories[variant]):
            if left['condition_counts'] != right['condition_counts']:
                raise ValueError(f'Training noise sampling differs in {variant}.')
    paired = paired_analysis()
    latency = benchmark_models(results)
    scores = {name: ranking(result['Val']) for name, result in results.items()}
    winner = max(scores, key=scores.get)
    write_json(HERE / 'runs/short_local_long_analysis.json', {
        'validation_rank': scores, 'highest_validation_rank': winner,
        'training_noise_counts_matched': True, 'paired_common_original': paired,
        'end_to_end_latency': latency,
    })

    lines = [
        '# Q0 长短窗互补 L1/L2 完整训练报告', '',
        '基线为 ShortLocal。L1 删除长窗并缩小公共输入，L2 将长窗统计网格改为 32×1。'
        '三者均使用新 MB 划分、种子 13 的模型构造初值和共同的训练及六档 Val 规则。',
        '共同轮次的六档训练采样计数一致，全部 Val、Test 的样本编号、标签及上下文逐条一致。', '',
        '| 模型 | 参数量 | 训练轮数 | 封存轮次 | Val 单双均衡分 | Val 综合均值 | Test 综合均值 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    for name, result in results.items():
        selection = result['selection']
        test_mean = np.mean([result['Test'][c]['acc'] for c in CONDITIONS])
        lines.append(
            f'| {name} | {result["parameters"]:,} | {selection["epochs_trained"]} | '
            f'{selection["epoch"]} | {scores[name][0]:.2f}% | {scores[name][1]:.2f}% | '
            f'{test_mean:.2f}% |'
        )
    lines.extend(['', f'按预定六档 Val 排序，三者中的最高分为 {winner}。', '',
                  '每格依次为综合／Single／Double／背景准确率，单位为 %。'])
    for split in ('Val', 'Test'):
        lines.extend(['', f'## {split}', '',
                      '| 条件 | ShortLocal | L1 | L2 |',
                      '| --- | ---: | ---: | ---: |'])
        for condition in CONDITIONS:
            cells = ['／'.join(f'{results[name][split][condition][key]:.2f}' for key in FIELDS)
                     for name in RUNS]
            lines.append('| ' + ' | '.join((condition, *cells)) + ' |')
    lines.extend(['', '## Test 相对 ShortLocal 的变化', '',
                  '单位为百分点，每格依次为综合／Single／Double／背景。', '',
                  '| 条件 | L1 | L2 |', '| --- | ---: | ---: |'])
    for condition in CONDITIONS:
        reference = results['ShortLocal']['Test'][condition]
        cells = ['／'.join(f'{results[name]["Test"][condition][key] - reference[key]:+.2f}'
                          for key in FIELDS) for name in ('L1', 'L2')]
        lines.append('| ' + ' | '.join((condition, *cells)) + ' |')
    lines.extend(['', '## 原始档共同判对样本的保持率', '',
                  '各行分母为三版在原始档共同判对的样本数。每格依次为 ShortLocal／L1／L2。', '',
                  '| 集合与分组 | 样本数 | 0 dB | −5 dB | −10 dB |',
                  '| --- | ---: | ---: | ---: | ---: |'])
    for split in ('Val', 'Test'):
        for group in ('ship', 'single', 'double', 'background'):
            row = paired[split][group]
            cells = ['／'.join(f'{row["conditions"][c][name]["retention"]:.2f}' for name in RUNS)
                     for c in ('0dB', '-5dB', '-10dB')]
            lines.append('| ' + ' | '.join((f'{split} {group}', str(row['n']), *cells)) + ' |')
    lines.extend(['', '## Test 数量误判', '',
                  '每格依次为 ShortLocal／L1／L2，单位为条。', '',
                  '| 条件 | Single→Double | Single→背景 | Double→Single | Double→背景 | Double组合错 |',
                  '| --- | ---: | ---: | ---: | ---: | ---: |'])
    for condition in CONDITIONS:
        cells = ['／'.join(str(results[name]['Test'][condition][key]) for name in RUNS)
                 for key in ERRORS]
        lines.append('| ' + ' | '.join((condition, *cells)) + ' |')
    lines.extend(['', '## 计算成本', '',
                  '特征准备和训练耗时来自各次完整训练的实际记录。波形到集合 logits 的推理计时使用同一批 32 条 Train 原始波形，'
                  '包含 CPU 到 GPU 的波形传输、全部固定特征变换和公共网络，三个预热批后取十次测量的中位数。', '',
                  '| 模型 | Train 原始特征准备（秒） | Val 六档特征准备（秒） | 每轮中位耗时（秒） | 波形推理批耗时（毫秒） |',
                  '| --- | ---: | ---: | ---: | ---: |'])
    for name, result in results.items():
        bench = result['benchmark']
        lines.append(
            f'| {name} | {bench["train_cache_seconds"]:.2f} | {bench["val_cache_seconds"]:.2f} | '
            f'{bench["median_epoch_seconds"]:.2f} | {latency[name]["median_batch_ms"]:.2f} |'
        )
    lines.extend(['', '[实验协议](长短窗互补L1_L2实验协议.md) · '
                  '[成对统计](runs/short_local_long_analysis.json)', ''])
    for name, folder in RUNS.items():
        relative = folder.relative_to(HERE).as_posix()
        lines.append(
            f'{name} 的 [解析配置]({relative}/config_resolved.json)、'
            f'[Val 封存]({relative}/selection_sealed.json)、'
            f'[完整结果与预测目录]({relative}/results.json)。'
        )
        lines.append('')
    target = HERE / 'Q0长短窗互补L1_L2完整训练报告.md'
    target.write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({'report': str(target), 'validation_rank': scores,
                      'highest_validation_rank': winner}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
