"""Evaluate sealed formal models on the existing Val second-source pairs."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import torch


CONDITIONS = ('original', '5dB', '0dB')
ARMS = ('F', 'B', 'C')
METRICS = (
    ('absent_exact', '无第二源时单目标正确'),
    ('false_double', '无第二源时双目标误报'),
    ('false_candidate', '无第二源时指定船型误报'),
    ('weak_candidate', '弱第二源船型召回'),
    ('weak_exact', '弱双目标组合正确'),
    ('both_exact', '配对两端同时正确'),
)


def _formal_model(config, normalization):
    # Importing train_formal defines its classes without invoking its guarded main.
    from train_formal import EvidenceModel, JointModel, STRUCTURE_ARM

    if 'evidence_arm' in config:
        return EvidenceModel(config, normalization)
    return JointModel(config, STRUCTURE_ARM, normalization)


def _probabilities(panel_code, panel, fixed, models, condition, strength, mode, device):
    parts = {arm: [] for arm in models}
    batch_size = panel_code.CFG['evaluation_batch_size']
    for start in range(0, len(panel.plans), batch_size):
        ids = np.arange(start, min(start + batch_size, len(panel.plans)))
        waves, _ = panel.waves(ids, mode, condition, strength)
        features = panel_code.extract(fixed, waves)
        for arm, model in models.items():
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                logits = model(*features)
            parts[arm].append(logits.float().softmax(-1).cpu().numpy().mean(1))
    return {arm: np.concatenate(values) for arm, values in parts.items()}


def _summarize(panel_code, panel, absent, weak):
    single = absent.argmax(-1)
    double = weak.argmax(-1)
    single_correct = single == panel.anchor_labels
    double_correct = double == panel.pair_labels
    events = {
        'absent_exact': single_correct,
        'false_double': panel_code.SET_COUNTS[single] == 2,
        'false_candidate': panel_code.MEMBERS[single, panel.candidate_columns],
        'weak_candidate': panel_code.MEMBERS[double, panel.candidate_columns],
        'weak_exact': double_correct,
        'both_exact': single_correct & double_correct,
    }
    return {
        name: {
            'correct_or_event_count': int(values.sum()),
            'percent': float(100 * values.mean()),
            'recording_macro_percent': panel_code.macro_correct(panel, values)['recording_macro_acc'],
        }
        for name, values in events.items()
    }


def _report(result):
    lines = [
        '# SE-F／B／C 成对Val证据诊断', '',
        f"冻结同种子Val封存权重，复用{result['pairs']}个C0有向来源对，共涉及"
        f"{result['anchor_recordings']}份anchor录音。每对固定原来源与噪声实现。"
        '弱第二源比anchor低6dB。原始条件表示不额外添加AWGN。', '',
        '主口径固定anchor相对噪声的功率。配对两端同时正确要求无第二源时正确识别单目标，'
        '并在加入弱第二源后正确识别两类组合。指定船型误报包含把单目标错认成该船型的情况。', '',
        '| 条件 | 功率口径 | 组别 | ' + ' | '.join(label for _, label in METRICS) + ' |',
        '|---|---|---|' + '---:|' * len(METRICS),
    ]
    for condition, modes in result['conditions'].items():
        for mode, arms in modes.items():
            for arm, values in arms.items():
                columns = ' | '.join(f"{values[name]['percent']:.2f}%" for name, _ in METRICS)
                lines.append(f'| {condition} | {mode} | {arm} | {columns} |')
    lines.extend(['', '表中为有向配对百分比。JSON同时记录按来源及anchor录音聚合后的录音宏平均。'
                  '这些来源重建视图用于Val诊断，正式存储混合的六档结果另见各组封存报告。', ''])
    return '\n'.join(lines)


@torch.no_grad()
def evaluate_pairs(fixed, here, model_factory=None, *, include_total_snr=False):
    """Evaluate F/B/C seed 13 checkpoints without changing their Val selections."""
    here = Path(here)
    runs = here / 'runs'
    device = next(fixed.buffers()).device
    factory = model_factory if model_factory is not None else _formal_model
    models, checkpoints = {}, {}
    for arm in ARMS:
        experiment_id = f'SE-{arm}-13'
        out = runs / experiment_id
        seal = json.loads((out / 'selection_sealed.json').read_text(encoding='utf-8'))
        path = out / 'best.pt'
        saved = torch.load(path, map_location='cpu', weights_only=False)
        if (saved['experiment_id'], saved['epoch'], saved['score']) != (
                experiment_id, seal['epoch'], seal['score']):
            raise RuntimeError(f'{experiment_id} checkpoint differs from the Val seal.')
        config = saved['config']
        model = factory(config, saved['normalization'])
        model.load_state_dict(saved['model'], strict=True)
        models[arm] = model.to(device).eval().requires_grad_(False)
        checkpoints[arm] = {'experiment_id': experiment_id, 'epoch': saved['epoch'],
                            'path': str(path.resolve())}

    panel_dir = here.parent / 'shipsear_noise_target_factorial_20261102'
    spec = importlib.util.spec_from_file_location('shipsear_paired_val_panel', panel_dir / 'run.py')
    panel_code = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(panel_code)
    panel = panel_code.Panel('Val')
    for model in models.values():
        config = model.config
        if (Path(config['data_root']).resolve() != panel.data.root
                or config['batch_size'] != panel_code.CFG['batch_size']
                or config['evaluation_batch_size'] != panel_code.CFG['evaluation_batch_size']):
            raise RuntimeError('Formal model and paired panel use different data or batch settings.')
    with np.load(panel_dir / 'runs/seed13/Val_responses.npz') as archived:
        if not np.array_equal(archived['pair_ids'], [row['pair_id'] for row in panel.plans]):
            raise RuntimeError('Paired Val source order differs from the archived panel.')

    result = {
        'split': 'Val',
        'pairs': len(panel.plans),
        'anchor_recordings': int(len(np.unique(panel.recordings))),
        'weak_second_db_relative_to_anchor': -6,
        'checkpoints': checkpoints,
        'conditions': {},
    }
    for condition in CONDITIONS:
        absent = _probabilities(panel_code, panel, fixed, models, condition,
                                'absent', 'anchor_fixed', device)
        modes = ['anchor_fixed']
        if include_total_snr and condition != 'original':
            modes.append('total_snr')
        result['conditions'][condition] = {}
        for mode in modes:
            weak = _probabilities(panel_code, panel, fixed, models, condition,
                                  'weak6dB', mode, device)
            result['conditions'][condition][mode] = {
                arm: _summarize(panel_code, panel, absent[arm], weak[arm]) for arm in ARMS
            }

    (runs / 'paired_evidence_val.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    (here / '成对证据诊断_Val.md').write_text(_report(result), encoding='utf-8')
    return result
