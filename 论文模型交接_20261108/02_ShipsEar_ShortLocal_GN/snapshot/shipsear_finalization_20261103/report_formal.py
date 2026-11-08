"""Report sealed full-model runs and their historical P50 reference."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean, stdev


CONDITIONS = ('original', '10dB', '5dB', '0dB', '-5dB', '-10dB')
LABELS = ('原始', '10 dB', '5 dB', '0 dB', '−5 dB', '−10 dB')
SPLITS = ('Val', 'Test')
METRICS = ('single_acc', 'double_acc', 'noise_acc', 'acc')
HEADERS = ('Single', 'Double', '背景', '综合 ACC')
MEAN_CONDITION = 'six_condition_mean'
DISPLAY = dict(zip(CONDITIONS, LABELS), six_condition_mean='六档均值')
BENCHMARK_NAMES = {
    'train_cache_seconds': 'Train 原始特征缓存构建（秒）',
    'val_cache_seconds': 'Val 六档特征缓存构建（秒）',
    'feature_batch_seconds': '固定特征每批秒数',
    'train_batch_seconds': '训练每批秒数',
    'val_batch_seconds': '验证每批秒数',
    'projected_minutes': '预计总时长（分钟）',
}
STOP_NAMES = {'early_stop': '达到早停条件', 'patience': '达到早停条件',
              'max_epochs': '达到轮数上限'}


def _table(headers, rows):
    def row(values):
        cells = [str(value).replace('|', r'\|').replace('\n', ' ') for value in values]
        return '| ' + ' | '.join(cells) + ' |'

    return '\n'.join([row(headers), row(['---'] * len(headers)), *(row(r) for r in rows)])


def _fmt(value, digits=2, *, signed=False):
    if value is None:
        return '—'
    return f'{value:+.{digits}f}' if signed else f'{value:.{digits}f}'


def _setting(value):
    if value is None:
        return '—'
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _metric(values, condition, metric):
    if condition == MEAN_CONDITION:
        entries = [_metric(values, c, metric) for c in CONDITIONS]
        return mean(entries) if all(value is not None for value in entries) else None
    return values.get(condition, {}).get(metric)


def _subtract(value, reference):
    return value - reference if value is not None and reference is not None else None


def _collect_rows(results):
    reference = results[0]['source_reference']
    rows = []
    for split in SPLITS:
        for condition in (*CONDITIONS, MEAN_CONDITION):
            baseline = {key: _metric(reference[split], condition, key) for key in METRICS}
            seed_values = []
            for result in results:
                other_reference = result['source_reference'][split]
                if any(_metric(other_reference, condition, k) != baseline[k] for k in METRICS):
                    raise ValueError(f'P50 reference differs across runs for {split} / {condition}.')
                values = {key: _metric(result[split], condition, key) for key in METRICS}
                seed_values.append(values)
                rows.append(_record('formal_seed', split, condition, values, baseline,
                                    result['experiment_id'], result['seed'], 1))
            if len(results) > 1:
                columns = {key: [values[key] for values in seed_values] for key in METRICS}
                complete = {key: all(value is not None for value in values)
                            for key, values in columns.items()}
                means = {key: mean(values) if complete[key] else None
                         for key, values in columns.items()}
                deviations = {key: stdev(values) if complete[key] else None
                              for key, values in columns.items()}
                rows.append(_record('formal_mean', split, condition, means, baseline,
                                    'F', '', len(results), deviations))
            rows.append(_record('development_reference', split, condition, baseline, None,
                                'P50', '', ''))
    return rows


def _record(group, split, condition, values, baseline, experiment_id, seed, n_seeds, deviations=None):
    record = dict(group=group, split=split, condition=condition,
                  experiment_id=experiment_id, seed=seed, n_seeds=n_seeds)
    for key in METRICS:
        record[key] = values[key]
        record[key + '_std_pp'] = deviations[key] if deviations is not None else None
        record[key + '_delta_vs_p50_pp'] = (
            _subtract(values[key], baseline[key]) if baseline is not None else None)
    return record


def _configuration(results):
    fields = (
        ('协议', 'protocol_id'), ('数据版本', 'dataset_version'),
        ('Train／Val／Test 样本数', 'split_sizes'), ('训练波形', 'waveform_view'),
        ('初始化', 'initialization'), ('归一化拟合', 'normalization'),
        ('评价噪声种子', 'evaluation_noise_seed'), ('优化器', 'optimizer'),
        ('学习率', 'classifier_lr'), ('权重衰减', 'weight_decay'),
        ('梯度裁剪', 'gradient_clip'), ('训练批量', 'batch_size'),
        ('评价分类批量', 'evaluation_batch_size'),
        ('数量边际损失权重', 'count_marginal_weight'),
        ('最少轮数', 'min_epochs'), ('最多轮数', 'max_epochs'),
        ('早停等待轮数', 'patience_epochs'), ('早停主分刷新量（百分点）', 'early_stop_min_delta_pp'),
    )
    settings = []
    for label, key in fields:
        values = [_setting(r['config'].get(key)) for r in results]
        settings.append([label, *values])
    for i, label in enumerate(LABELS):
        values = []
        for result in results:
            probabilities = result['config'].get('noise_probabilities')
            values.append(f'{100 * probabilities[i]:g}%' if probabilities is not None else '—')
        settings.append([f'Train {label}抽样概率', *values])
    return _table(['项目', *(f"种子 {r['seed']}" for r in results)], settings)


def _score_cells(score):
    if isinstance(score, (list, tuple)) and len(score) == 4:
        return [_fmt(value, 4) for value in score]
    raise ValueError('Expected the four recorded Val ranking values.')


def _performance_table(rows, split, group):
    displayed = []
    for row in rows:
        if row['split'] != split or row['group'] != group:
            continue
        values = [_fmt(row[key]) for key in METRICS]
        if group == 'formal_mean':
            values = [f"{value} ± {_fmt(row[key + '_std_pp'])}" if row[key] is not None else value
                      for key, value in zip(METRICS, values)]
        prefix = [DISPLAY[row['condition']]]
        if group == 'formal_seed':
            prefix.insert(0, row['seed'])
        displayed.append([*prefix, *values])
    prefix = ['种子', '条件'] if group == 'formal_seed' else ['条件']
    return _table([*prefix, *HEADERS], displayed)


def _comparison_table(rows, split, group):
    displayed = []
    for row in rows:
        if row['split'] != split or row['group'] != group:
            continue
        values = [_fmt(row[key + '_delta_vs_p50_pp'], signed=True) for key in METRICS]
        displayed.append([DISPLAY[row['condition']], *values])
    return _table(['条件', *HEADERS], displayed)


def _benchmark_table(results):
    rows = []
    for result in results:
        benchmark = result.get('benchmark') or result['config'].get('benchmark') or {}
        for key, value in benchmark.items():
            shown = _fmt(value, 3) if isinstance(value, (int, float)) else _setting(value)
            rows.append([result['seed'], BENCHMARK_NAMES.get(key, key), shown])
    return _table(['种子', '计时项目', '实测或估计值'], rows) if rows else '逐轮训练和 Val 的累计耗时见训练记录。'


def render(results: list[dict], out: Path):
    """Write the report and numeric CSV for one or more sealed full-model runs."""
    results = sorted(results, key=lambda r: r['seed'])
    if not results or len({r['seed'] for r in results}) != len(results):
        raise ValueError('Expected at least one result and unique training seeds.')
    rows = _collect_rows(results)
    summary = []
    for result in results:
        selection = result['selection']
        reason = result['stop_reason']
        summary.append([result['experiment_id'], result['seed'], result['parameters'],
                        result['epochs_trained'], selection['epoch'], STOP_NAMES.get(reason, reason),
                        _fmt(result['elapsed_minutes']), *_score_cells(selection['score'])])
    score_headers = ['实验 ID', '种子', '参数量', '训练轮数', 'Val 最佳轮', '结束原因', '训练与逐轮 Val 耗时（分钟）',
                     'Val 六档平衡分', 'Val 六档综合均值', 'Val 较弱组均值', 'Val 原始综合']
    seeds = '、'.join(str(r['seed']) for r in results)
    lines = [
        '# ShipsEar 正式完整模型训练报告',
        f'本报告汇总训练种子 {seeds} 的完整模型。每个种子以各自的 Val 最佳权重评价六档 Test。'
        'P50 开发参考来自续训第 20 轮，累计第 71 轮。',
        '## 训练记录与 Val 选轮',
        _table(score_headers, summary),
        'Val 依次按六档 Single／Double 平衡分、六档综合 ACC 均值、两组六档均值中的较小值、原始综合 ACC 排序。'
        '平衡分为每档 Single 和 Double 的算术平均，再对六档取均值。完全同分时取较早轮。'
        '训练达到最少轮数后，按主分的累计有效刷新量和等待轮数判断早停。',
        '## 训练设置',
        '公共识别网络从模型构造初值开始训练，固定散射特征用原始 Train 拟合逐维归一化。'
        '各档噪声概率作用于存储波形，评价噪声种子与训练种子分别登记。',
        _configuration(results),
        '## 基准计时',
        _benchmark_table(results),
        '## 六档评价',
        '表中准确率单位为 %。六档均值对每个档位等权。'
        + ('均值 ± 样本标准差按训练种子计算，标准差单位为百分点，分母为种子数减一。'
           '六档均值行先在每个种子内跨档取均值，再跨种子统计。' if len(results) > 1 else ''),
    ]
    for split in SPLITS:
        lines.extend([f'### {split} 逐种子', _performance_table(rows, split, 'formal_seed')])
        if len(results) > 1:
            lines.extend([f'### {split} 跨种子均值 ± 样本标准差',
                          _performance_table(rows, split, 'formal_mean')])
        lines.extend([f'### {split} P50 开发参考',
                      _performance_table(rows, split, 'development_reference'),
                      f'### {split} 正式结果相对 P50 的差值（百分点）',
                      _comparison_table(rows, split, 'formal_mean' if len(results) > 1 else 'formal_seed')])

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / '完整训练报告.md').write_text('\n\n'.join(lines) + '\n', encoding='utf-8')
    with (out / '性能表.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
