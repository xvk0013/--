"""Train complete ShipsEar models from their constructor initialization."""

import argparse
import csv
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent / 'shipsear_final_noise_distribution_20261102'
sys.path.insert(0, str(PACKAGE))
from data import CONDITIONS, FixedDataset, evaluation_views, recognition_metrics, recording_results, training_views
from model import (COUNT_MAP, SCATTER_DIM, SPECTRAL_DIM, FixedViews, JointModel,
                   CalibratedInteraction, LinearEnsemble, TabM, count_probabilities)

# JointModel and the packaged Recognizer use this label for the shared structure.
STRUCTURE_ARM = 'P50_noise50'
ROOT = HERE / 'runs'
EVIDENCE_ARMS = {'B': 'late_residual', 'C': 'early_skip'}
BAND_ARMS = {'B': 'shared_temporal', 'C': 'frequency_temporal'}
DECISION_ARMS = ('I0', 'Q0')


class EvidenceModel(JointModel):
    def __init__(self, config, normalization):
        super().__init__(config, STRUCTURE_ARM, normalization)
        self.evidence_arm = config['evidence_arm']
        width = config['model']['d_block']
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(config['seed'] + 8731)
            self.tail_skip = nn.Sequential(nn.Linear(width, 64, bias=False), nn.GELU(),
                                           nn.Linear(64, width, bias=False))
            nn.init.zeros_(self.tail_skip[2].weight)

    def forward(self, scatter, spectral):
        values = torch.cat((scatter.float(), spectral.float()), -1)
        values = (values - self.feature_mean) / self.feature_scale
        early = self.tabm.backbone.blocks[0](self.tabm.ensemble_view(values))
        hidden = early
        for block in self.tabm.backbone.blocks[1:]:
            hidden = block(hidden)
        evidence = early if self.evidence_arm == 'C' else hidden
        hidden = hidden + self.tail_skip(evidence).to(dtype=hidden.dtype)
        return self.tabm.output(hidden) + self.tabm.count_readout(hidden)[..., COUNT_MAP]


class DirectSetModel(JointModel):
    def __init__(self, config, normalization):
        super().__init__(config, STRUCTURE_ARM, normalization)
        del self.tabm.count_readout

    def forward(self, scatter, spectral, *, masked_view=None):
        values = torch.cat((scatter.float(), spectral.float()), -1)
        values = (values - self.feature_mean) / self.feature_scale
        if masked_view == '1d':
            values[..., :SCATTER_DIM] = 0.
        elif masked_view == '2d':
            values[..., SCATTER_DIM:] = 0.
        return self.tabm(values)


class ShortLocalSetModel(DirectSetModel):
    def __init__(self, config, normalization):
        from short_local import ShortLocalEncoder

        fixed = normalization['F1']
        short_dim = SPECTRAL_DIM // 2
        stats = {
            'mean': nn.functional.pad(fixed['mean'], (0, short_dim), value=0.),
            'scale': nn.functional.pad(fixed['scale'], (0, short_dim), value=1.),
        }
        super().__init__(config, {'F1': stats})
        self.short_local = ShortLocalEncoder(config)
        self.masked_view = config.get('short_local_mask')
        fusion = config.get('short_local_fusion')
        first_block = self.tabm.backbone.blocks[0]
        if fusion == 'concat':
            first_block[0] = first_block[0].base
        elif fusion == 'additive':
            first_block[0].combine_views = scaled_addition
        elif fusion == 'local_bilinear':
            from short_local import LocalBilinearInteraction

            first_block[0] = LocalBilinearInteraction(
                first_block[0].base, config['spectral_scattering']['summary_grid'][1], config['bilinear_rank'])

    def forward(self, scatter, long_summary, short_mel):
        spectral = torch.cat((long_summary, self.short_local(short_mel)), dim=-1)
        return super().forward(scatter, spectral, masked_view=self.masked_view)


class ShortLocalSingleViewModel(nn.Module):
    """Use a standard TabM on one branch at its native input width."""

    def __init__(self, config, normalization):
        super().__init__()
        stats = normalization['F1']
        mean, scale = stats['mean'], stats['scale']
        two_dimensional = config['short_local_view'] == '2d'
        width = SPECTRAL_DIM if two_dimensional else SCATTER_DIM
        self.tabm = TabM.make(n_num_features=width, d_out=7, arch_type='tabm', **config['model'])
        if two_dimensional:
            from short_local import ShortLocalEncoder

            self.short_local = ShortLocalEncoder(config)
            mean = nn.functional.pad(mean, (0, SPECTRAL_DIM // 2), value=0.)
            scale = nn.functional.pad(scale, (0, SPECTRAL_DIM // 2), value=1.)
        if mean.numel() != width or scale.numel() != width:
            raise ValueError('Normalization width differs from the retained input view.')
        self.register_buffer('feature_mean', mean.clone())
        self.register_buffer('feature_scale', scale.clone())

    def forward(self, fixed_features, short_mel=None):
        values = fixed_features.float()
        if short_mel is not None:
            values = torch.cat((values, self.short_local(short_mel)), dim=-1)
        return self.tabm((values - self.feature_mean) / self.feature_scale)


class ShortLongLocalSetModel(ShortLocalSetModel):
    """Replace the long-view scattering by a learned frequency encoder."""

    def __init__(self, config, normalization):
        from short_local import LongLocalEncoder

        stats = normalization['F1']
        placeholder = {
            'mean': nn.functional.pad(stats['mean'], (0, SPECTRAL_DIM // 2)),
            'scale': nn.functional.pad(stats['scale'], (0, SPECTRAL_DIM // 2), value=1.),
        }
        super().__init__(config, {'F1': placeholder})
        self.long_local = LongLocalEncoder(config)

    def forward(self, scatter, long_image, short_mel):
        return super().forward(scatter, self.long_local(long_image), short_mel)


class ShortLocalNoLongSetModel(nn.Module):
    """Remove the long view and resize the same Q0 interaction and TabM."""

    def __init__(self, config, normalization):
        from short_local import ShortLocalEncoder

        super().__init__()
        short_dim = SPECTRAL_DIM // 2
        stats = normalization['F1']
        self.register_buffer('feature_mean', nn.functional.pad(stats['mean'], (0, short_dim)))
        self.register_buffer('feature_scale', nn.functional.pad(stats['scale'], (0, short_dim), value=1.))
        features = SCATTER_DIM + short_dim
        if self.feature_mean.numel() != features or self.feature_scale.numel() != features:
            raise ValueError('L1 normalization must contain the waveform-scattering statistics.')
        self.tabm = TabM.make(n_num_features=features, d_out=7, arch_type='tabm', **config['model'])
        LinearEnsemble(config['model']['d_block'], 3, k=config['model']['k'])
        interaction = SimpleNamespace(
            base=self.tabm.backbone.blocks[0][0], split=SCATTER_DIM,
            left=nn.Linear(SCATTER_DIM, 64, bias=False),
            right=nn.Linear(short_dim, 64, bias=False),
            project=nn.Linear(64, config['model']['d_block'], bias=False),
        )
        nn.init.zeros_(interaction.project.weight)
        self.tabm.backbone.blocks[0][0] = CalibratedInteraction(interaction, 'adaptive')
        self.short_local = ShortLocalEncoder(config)

    def forward(self, scatter, short_mel):
        features = torch.cat((scatter.float(), self.short_local(short_mel)), dim=-1)
        return self.tabm((features - self.feature_mean) / self.feature_scale)


class LocalEvidenceModel(DirectSetModel):
    def __init__(self, config, normalization):
        from local_evidence import LocalShipEvidence

        super().__init__(config, normalization)
        names = config['set_names']
        ship_types = [name for name, count in zip(names, config['count_map']) if count == 1]
        self.register_buffer('presence_targets', torch.tensor([
            [ship in name.split('+') for ship in ship_types] for name in names
        ], dtype=torch.float32), persistent=False)
        self.local_evidence = LocalShipEvidence(
            SPECTRAL_DIM, config['spectral_scattering']['summary_grid'], len(ship_types),
            config['local_evidence']['hidden_dim'])

    def forward(self, scatter, spectral, *, return_presence=False):
        values = torch.cat((scatter.float(), spectral.float()), dim=-1)
        values = (values - self.feature_mean) / self.feature_scale
        spectral, presence = self.local_evidence(values[:, SCATTER_DIM:])
        logits = self.tabm(torch.cat((values[:, :SCATTER_DIM], spectral), dim=-1))
        return (logits, presence) if return_presence else logits


class SharedTypeReadout(nn.Module):
    def __init__(self, original, config):
        super().__init__()
        names, counts = config['set_names'], config['count_map']
        single_names = [name for name, count in zip(names, counts) if count == 1]
        components = 1 + len(single_names)
        if counts != [0] + [1] * len(single_names) + [2] * (len(names) - components):
            raise ValueError('Readout requires background, then singles, then pairs.')
        pairs = [[single_names.index(member) for member in name.split('+')] for name in names[components:]]
        self.register_buffer('pair_index', torch.tensor(pairs, dtype=torch.long), persistent=False)
        self.components = LinearEnsemble(original.in_features, components, k=original.k)
        with torch.no_grad():
            self.components.weight.copy_(original.weight[..., :components])
            self.components.bias.copy_(original.bias[..., :components])
            pair_bias = original.bias[..., components:] - original.bias[..., 1:components][..., self.pair_index].sum(-1)
        self.pair_bias = nn.Parameter(pair_bias)

    def forward(self, hidden):
        components = self.components(hidden)
        pair_logits = components[..., 1:][..., self.pair_index].sum(-1) + self.pair_bias
        return torch.cat((components, pair_logits), dim=-1)


def rms_signed_sqrt(product):
    values = product.float()
    rooted = values * (values.abs() + 1e-6).rsqrt()
    norm = torch.linalg.vector_norm(values, dim=-1, keepdim=True)
    rooted_norm = torch.linalg.vector_norm(rooted, dim=-1, keepdim=True)
    scale = norm / rooted_norm.clamp_min(1e-6 * values.shape[-1] ** .5)
    return (rooted * scale).to(product.dtype)


def scaled_addition(left, right):
    return ((left.float() + right.float()) * (2 ** -.5)).to(left.dtype)


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def save_state(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def log(event, **fields):
    row = {'time': now(), 'event': event, **fields}
    write_json(ROOT / 'status.json', row)
    print(json.dumps(row, ensure_ascii=False), flush=True)


def model_state(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def create_model(config, normalization):
    torch.manual_seed(config['seed'])
    if 'alignment_arm' in config:
        from aligned_dual import AlignedModel
        model = AlignedModel(config, normalization).cuda()
    elif 'band_arm' in config:
        from band_temporal import BandTemporalModel
        model = BandTemporalModel(config, config['band_arm'], normalization).cuda()
    elif 'evidence_arm' in config:
        model = EvidenceModel(config, normalization).cuda()
    elif 'local_evidence' in config:
        model = LocalEvidenceModel(config, normalization).cuda()
    elif 'short_local_view' in config:
        model = ShortLocalSingleViewModel(config, normalization).cuda()
    elif 'short_local' in config:
        if config.get('short_local_long', {}).get('arm') == 'L1':
            model = ShortLocalNoLongSetModel(config, normalization).cuda()
        elif config.get('short_local_long', {}).get('arm') == 'L3':
            model = ShortLongLocalSetModel(config, normalization).cuda()
        else:
            model = ShortLocalSetModel(config, normalization).cuda()
    elif config.get('decision_arm') == 'Q0':
        model = DirectSetModel(config, normalization)
        if config.get('shared_readout'):
            model.tabm.output = SharedTypeReadout(model.tabm.output, config)
        elif config.get('power_product'):
            model.tabm.backbone.blocks[0][0].combine_views = lambda left, right: rms_signed_sqrt(left * right)
        elif config.get('additive_interaction'):
            model.tabm.backbone.blocks[0][0].combine_views = scaled_addition
        model = model.cuda()
    else:
        model = JointModel(config, STRUCTURE_ARM, normalization).cuda()
        if config.get('decision_arm') == 'I0':
            first_block = model.tabm.backbone.blocks[0]
            first_block[0] = first_block[0].base
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['classifier_lr'], weight_decay=config['weight_decay'])
    return model, optimizer


@torch.no_grad()
def feature_cache(fixed, dataset, config, condition='original'):
    parts = None
    for start in range(0, len(dataset), config['batch_size']):
        ids = np.arange(start, min(start + config['batch_size'], len(dataset)))
        waves = dataset.batch(ids) if condition == 'original' else evaluation_views(
            dataset, ids, condition, config['evaluation_noise_seed'])
        values = fixed(waves)
        if parts is None:
            parts = [[] for _ in values]
        for part, feature in zip(parts, values):
            part.append(feature.cpu())
    return tuple(torch.cat(part, dim=0) for part in parts)


def fit_normalization(train, features, config):
    if train.split != 'Train':
        raise ValueError('Feature normalization requires Train.')
    values = torch.cat(features, dim=1).double()
    mean = values.mean(0)
    scale = (values.square().mean(0) - mean.square()).clamp_min(0).sqrt()
    scale.clamp_min_(config['normalization']['std_floor'])
    if not torch.isfinite(mean).all() or not torch.isfinite(scale).all():
        raise ValueError('Nonfinite Train feature normalization.')
    return {'F1': {'mean': mean.float(), 'scale': scale.float()}}


def training_features(fixed, train, originals, config, ids, epoch):
    waves, choices = training_views(train, ids, epoch, config['noise_probabilities'], config['seed'])
    features = tuple(values[ids].cuda() for values in originals)
    noisy = np.flatnonzero(choices)
    if len(noisy):
        for base, replacement in zip(features, fixed(waves[noisy])):
            base[noisy] = replacement
    return features, choices


def train_step(model, optimizer, features, data, ids, config, epoch, index):
    torch.manual_seed(config['seed'] + epoch * 100000 + index)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    y = torch.as_tensor(data.labels[ids], device='cuda')
    n = torch.as_tensor(data.counts[ids], device='cuda')
    weights = torch.as_tensor(data.weights[ids] * len(data), device='cuda')
    local_presence = None
    with torch.autocast('cuda', dtype=torch.bfloat16):
        if 'local_evidence' in config:
            logits, local_presence = model(*features, return_presence=True)
        else:
            logits = model(*features)
    log_probs = logits.float().log_softmax(-1)
    members = logits.shape[1]
    set_loss = -log_probs.gather(-1, y[:, None, None].expand(-1, members, 1)).squeeze(-1)
    counts = count_probabilities(log_probs.exp()).clamp_min(1e-12)
    count_loss = -counts.log().gather(-1, n[:, None, None].expand(-1, members, 1)).squeeze(-1)
    weighted_set = (set_loss.mean(1) * weights).mean()
    weighted_count = (count_loss.mean(1) * weights).mean()
    loss = weighted_set + config['count_marginal_weight'] * weighted_count
    if local_presence is not None:
        presence_bce = nn.functional.binary_cross_entropy_with_logits(
            local_presence, model.presence_targets[y], reduction='none')
        presence_loss = (presence_bce.mean(1) * weights).mean()
        loss = loss + config['local_evidence']['presence_weight'] * presence_loss
    if not torch.isfinite(loss):
        raise RuntimeError('Nonfinite training loss.')
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'], error_if_nonfinite=True)
    optimizer.step()
    metrics = {'loss': float(loss.detach()), 'set_loss': float(weighted_set.detach()),
               'count_loss': float(weighted_count.detach())}
    if local_presence is not None:
        metrics['local_presence_loss'] = float(presence_loss.detach())
    return metrics


@torch.no_grad()
def evaluate(model, data, cache, config, out=None):
    model.eval()
    metrics = {}
    for condition, features in cache.items():
        probabilities = []
        for start in range(0, len(data), config['evaluation_batch_size']):
            batch = tuple(feature[start:start + config['evaluation_batch_size']].cuda() for feature in features)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits = model(*batch)
            probabilities.append(logits.float().softmax(-1).cpu().numpy().mean(1))
        p = np.concatenate(probabilities)
        predictions = p.argmax(-1)
        metrics[condition] = recognition_metrics(data.labels, predictions, data.counts, data.contexts)
        if out is not None:
            stem = f'{data.split}_{condition}'
            np.savez_compressed(out / (stem + '.npz'), ids=data.ids, labels=data.labels,
                                contexts=data.contexts, predictions=predictions, probabilities=p)
            write_json(out / (stem + '_recordings.json'), recording_results(data, predictions))
    return metrics


def ranking(metrics):
    single = float(np.mean([metrics[c]['single_acc'] for c in CONDITIONS]))
    double = float(np.mean([metrics[c]['double_acc'] for c in CONDITIONS]))
    return [(single + double) / 2, float(np.mean([metrics[c]['acc'] for c in CONDITIONS])),
            min(single, double), metrics['original']['acc']]


def checkpoint(config, epoch, score, normalization, model):
    arm = EVIDENCE_ARMS[config['evidence_arm']] if 'evidence_arm' in config else STRUCTURE_ARM
    if 'band_arm' in config:
        arm = BAND_ARMS[config['band_arm']]
    if 'fixed_blend' in config:
        arm = 'fixed_long_blend'
    if 'alignment_arm' in config:
        arm = 'spectral_' + config['alignment_arm']
    if 'decision_arm' in config:
        arm = config['decision_arm']
    if config.get('power_product'):
        arm = 'Q0-Power'
    if config.get('additive_interaction'):
        arm = 'Q0-A0'
    if config.get('shared_readout'):
        arm = 'Q0-Shared'
    if config.get('recording_swap') == 'mb_train_val':
        arm = 'Q0-MBTrainValSwap'
    elif config.get('recording_swap') == 'background_only':
        arm = 'Q0-BackgroundSwap'
    elif config.get('recording_swap'):
        arm = 'Q0-RecordingSwap'
    if 'local_evidence' in config:
        arm = 'Q0-LocalEvidence'
    if 'short_local' in config:
        arm = 'Q0-ShortLocal'
    if 'short_local_long' in config:
        arm += '-' + config['short_local_long']['arm']
    if config.get('long_normalization') == 'record':
        arm = 'Q0-LongNorm-Record'
    if config.get('short_normalization') == 'record':
        arm = 'Q0-ShortLocal-GN'
    if config.get('short_local_fusion') == 'concat':
        arm = 'ShortLocal-GN-CAT'
    elif config.get('short_local_fusion') == 'additive':
        arm = 'ShortLocal-GN-ADD'
    elif config.get('short_local_fusion') == 'local_bilinear':
        arm = 'ShortLocal-GN-Bilinear'
    if 'short_local_view' in config:
        arm = 'ShortLocal-GN-' + config['short_local_view'].upper()
    if 'short_local_mask' in config:
        arm = 'ShortLocal-GN-Mask-' + config['short_local_mask'].upper()
    return {'arm': arm, 'experiment_id': config['experiment_id'], 'config': config,
            'seed': config['seed'], 'epoch': epoch, 'global_epoch': epoch, 'score': score,
            'normalization': normalization, 'model': model_state(model)}


def resume_state(path, config, model, optimizer):
    saved = torch.load(path, map_location='cpu', weights_only=False)
    if saved['config'] != config:
        raise ValueError(f'Stored configuration differs for {config["experiment_id"]}.')
    model.load_state_dict(saved['model'], strict=True)
    optimizer.load_state_dict(saved['optimizer'])
    return saved


def update_registry(config, out, status, epoch=''):
    path = HERE / '实验登记表.csv'
    with path.open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        fields, rows = reader.fieldnames, list(reader)
    identifier = config['experiment_id']
    row = next((row for row in rows if row['experiment_id'] == identifier), None)
    if row is None:
        row = {'experiment_id': identifier}
        rows.append(row)
    rel = out.relative_to(HERE).as_posix()
    if 'band_arm' in config:
        arm, role = 'BT-' + config['band_arm'], '长窗频带时序结构对照'
    elif 'fixed_blend' in config:
        arm, role = 'BT-Fixed', '长窗探针固定处理完整复验'
    elif 'alignment_arm' in config:
        arm, role = 'TF-' + config['alignment_arm'], '双分支时间段配合对照'
    elif config.get('power_product'):
        arm, role = 'Q0-Power', '乘积坐标幂归一化'
    elif config.get('additive_interaction'):
        arm, role = 'Q0-A0', '等参数加性交互对照'
    elif config.get('shared_readout'):
        arm, role = 'Q0-Shared', '共享船型证据读出'
    elif 'local_evidence' in config:
        arm, role = 'Q0-LocalEvidence', '共享谱散射局部船型证据迁移'
    elif 'short_local' in config:
        arm, role = 'Q0-ShortLocal', '短窗局部卷积后汇聚'
        if 'short_local_long' in config:
            variant = config['short_local_long']['arm']
            arm += '-' + variant
            role = {'L1': '删除长窗并缩小公共网络', 'L2': '长窗频率优先汇聚',
                    'L3': '长窗频率重排与局部编码'}[variant]
        if config.get('long_normalization') == 'record':
            arm, role = 'Q0-LongNorm-Record', '长窗样本内归一化替换'
        if config.get('short_normalization') == 'record':
            arm, role = 'Q0-ShortLocal-GN', '短窗卷积样本内归一化替换'
        if config.get('short_local_fusion') == 'concat':
            arm, role = 'ShortLocal-GN-CAT', 'GN 双路拼接与标准 TabM 消融'
        elif config.get('short_local_fusion') == 'additive':
            arm, role = 'ShortLocal-GN-ADD', '完整 GN 等参数加法替换乘积'
        elif config.get('short_local_fusion') == 'local_bilinear':
            arm, role = 'ShortLocal-GN-Bilinear', '同段双线性池化替换乘积门控残差'
        if 'short_local_view' in config:
            view = config['short_local_view'].upper()
            arm, role = f'ShortLocal-GN-{view}', f'{view} 单路与标准 TabM 消融'
        if 'short_local_mask' in config:
            view = config['short_local_mask'].upper()
            arm, role = f'ShortLocal-GN-Mask-{view}', f'完整 GN 屏蔽 {view} 路'
    elif config.get('recording_swap') == 'mb_train_val':
        arm, role = 'Q0-MBTrainValSwap', 'MB源录音Train与Val交换'
    elif config.get('recording_swap') == 'background_only':
        arm, role = 'Q0-BackgroundSwap', '仅Train与Val背景录音交换'
    elif config.get('recording_swap'):
        arm, role = 'Q0-RecordingSwap', '背景与PF录音交换对照'
    elif 'decision_arm' in config:
        arm, role = config['decision_arm'], '公共判别接口对照'
    else:
        arm = config.get('evidence_arm', 'F')
        role = '正式完整模型' if arm == 'F' else '首块直通等预算结构对照'
    row.update(state=status, role=role, dataset=config['dataset_version'],
               training_seed=config['seed'], arm=arm, initialization='模型构造初值',
               training_views='stored 50/10/10/10/10/10', val_rule=config['protocol_id'],
               epoch=epoch, cumulative_epoch=epoch, config=rel + '/config_resolved.json',
               history=rel + '/history.json', checkpoint=rel + '/best.pt' if epoch else '',
               results=rel + '/results.json' if status == '已完成' else '',
               notes=f'评价噪声种子 {config["evaluation_noise_seed"]}')
    temporary = path.with_suffix('.csv.tmp')
    with temporary.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def train(config, out, fixed, train_data, originals, val_data, val_cache, normalization, allow_resume):
    out.mkdir(parents=True, exist_ok=True)
    model, optimizer = create_model(config, normalization)
    resume_path = out / 'resume.pt'
    history, best, clock, elapsed, stop_reason = [], None, None, 0.0, None
    if resume_path.exists():
        if not allow_resume:
            raise FileExistsError(f'{resume_path} already exists. Use --resume.')
        saved = resume_state(resume_path, config, model, optimizer)
        history, best, clock = saved['history'], saved['best'], saved['clock']
        elapsed, stop_reason = saved['elapsed_seconds'], saved['stop_reason']
        for key in ('mean', 'scale'):
            if not torch.equal(saved['normalization']['F1'][key], normalization['F1'][key]):
                raise ValueError('Train normalization differs on resume.')
        del saved
    write_json(out / 'config_resolved.json', config)
    update_registry(config, out, '训练中')
    log('training_started', experiment_id=config['experiment_id'], next_epoch=len(history) + 1,
        evaluation_noise_seed=config['evaluation_noise_seed'])
    while stop_reason is None:
        begun = time.perf_counter()
        epoch = len(history) + 1
        order = np.random.default_rng([config['seed'], 700, epoch]).permutation(len(train_data))
        counts = np.zeros(len(CONDITIONS), dtype=np.int64)
        losses = {'loss': 0.0, 'set_loss': 0.0, 'count_loss': 0.0}
        for index, start in enumerate(range(0, len(order), config['batch_size'])):
            ids = order[start:start + config['batch_size']]
            features, choices = training_features(fixed, train_data, originals, config, ids, epoch)
            counts += np.bincount(choices, minlength=len(CONDITIONS))
            for key, loss in train_step(model, optimizer, features, train_data, ids, config, epoch, index).items():
                losses[key] = losses.get(key, 0.0) + loss * len(ids) / len(train_data)
        metrics = evaluate(model, val_data, val_cache, config)
        score = ranking(metrics)
        improved = best is None or score > best['score']
        if improved:
            best = checkpoint(config, epoch, score, normalization, model)
        if clock is None or score[0] >= clock['anchor'] + config['early_stop_min_delta_pp'] - 1e-10:
            clock = {'anchor': score[0], 'last_epoch': epoch}
        if epoch >= config['max_epochs']:
            stop_reason = 'max_epochs'
        elif epoch >= config['min_epochs'] and epoch - clock['last_epoch'] >= config['patience_epochs']:
            stop_reason = 'patience'
        seconds = time.perf_counter() - begun
        elapsed += seconds
        row = {'epoch': epoch, 'optimizer_updates': epoch * (index + 1), **losses,
               'condition_counts': dict(zip(CONDITIONS, counts.tolist())), 'Val': metrics, 'score': score,
               'best_epoch': best['epoch'], 'clock': clock, 'seconds': seconds, 'stop_reason': stop_reason}
        history.append(row)
        save_state(resume_path, {'config': config, 'model': model_state(model),
                   'optimizer': optimizer.state_dict(), 'normalization': normalization, 'history': history,
                   'best': best, 'clock': clock, 'elapsed_seconds': elapsed, 'stop_reason': stop_reason})
        if improved:
            save_state(out / 'best.pt', best)
        write_json(out / 'history.json', history)
        log('epoch_finished', experiment_id=config['experiment_id'], epoch=epoch, score=score,
            best_epoch=best['epoch'], seconds=round(seconds, 2), stop_reason=stop_reason)
    save_state(out / 'best.pt', best)
    write_json(out / 'history.json', history)
    model.load_state_dict(best['model'], strict=True)
    final_val = evaluate(model, val_data, val_cache, config, out)
    if ranking(final_val) != best['score']:
        raise RuntimeError('Restored best checkpoint differs from its recorded Val rank.')
    selection = {'experiment_id': config['experiment_id'], 'seed': config['seed'], 'epoch': best['epoch'],
                 'score': best['score'], 'checkpoint': str(out / 'best.pt'), 'sealed_at': now(),
                 'epochs_trained': len(history), 'elapsed_seconds': elapsed, 'stop_reason': stop_reason,
                 'parameters': sum(p.numel() for p in model.parameters()), 'Val': final_val}
    write_json(out / 'selection_sealed.json', selection)
    update_registry(config, out, 'Val已封存', best['epoch'])
    log('selection_sealed', experiment_id=config['experiment_id'], epoch=best['epoch'])
    return selection


def smoke(configs, fixed, train_data, originals, normalization, evidence=False, band=False):
    report, initial_states = {}, []
    alignment = 'alignment_arm' in configs[0]
    short_local = 'short_local' in configs[0]
    local_bilinear = configs[0].get('short_local_fusion') == 'local_bilinear'
    short_add = configs[0].get('short_local_fusion') == 'additive'
    single_view = configs[0].get('short_local_view')
    branch_masks = 'short_local_mask' in configs[0]
    has_short_encoder = short_local and single_view != '1d'
    long_variant = configs[0].get('short_local_long', {}).get('arm')
    smoke_tag = 'L4' if configs[0].get('long_normalization') == 'record' else long_variant
    if configs[0].get('short_normalization') == 'record':
        smoke_tag = 'GN'
    if configs[0].get('short_local_fusion') == 'concat':
        smoke_tag = 'GN_CAT'
    elif short_add:
        smoke_tag = 'GN_ADD'
    elif local_bilinear:
        smoke_tag = 'GN_Bilinear'
    if single_view:
        smoke_tag = f'GN_{single_view.upper()}'
    elif branch_masks:
        smoke_tag = 'GN_Mask'
    decision = 'decision_arm' in configs[0] and not short_local
    power_product = configs[0].get('power_product', False)
    additive_interaction = short_add or configs[0].get('additive_interaction', False)
    shared_readout = configs[0].get('shared_readout', False)
    local_evidence = 'local_evidence' in configs[0]
    recording_swap = configs[0].get('recording_swap', False)
    pf_replacement = configs[0].get('pf_replacement', 53)
    baseline_config = {key: value for key, value in configs[0].items()
                       if key not in ('evidence_arm', 'band_arm', 'decision_arm', 'power_product',
                                      'additive_interaction', 'shared_readout', 'recording_swap', 'pf_replacement',
                                      'local_evidence')}
    if power_product or additive_interaction or shared_readout or recording_swap:
        baseline_config['decision_arm'] = 'Q0'
    if short_add:
        baseline_config.pop('short_local_fusion')
    baseline = (create_model(baseline_config, normalization)[0].eval()
                if evidence or band or decision or short_add else None)
    if branch_masks:
        full, full_optimizer = create_model(dict(configs[0], short_local_mask=None), normalization)
        full_initial = model_state(full)
        del full, full_optimizer
    for config in configs:
        model, optimizer = create_model(config, normalization)
        initial = model_state(model)
        first_parameter = next(iter(dict(model.named_parameters())))
        initial_states.append(initial)
        ids = np.random.default_rng([config['seed'], 700, 1]).permutation(len(train_data))[:config['batch_size']]
        begun = time.perf_counter()
        features, choices = training_features(fixed, train_data, originals, config, ids, 1)
        if baseline is not None:
            expected = model_state(baseline)
            if config.get('decision_arm') == 'I0':
                prefix = 'tabm.backbone.blocks.0.0.'
                expected = {name.replace(prefix + 'base.', prefix, 1): value
                            for name, value in expected.items()
                            if not name.startswith(prefix) or name.startswith(prefix + 'base.')}
            elif config.get('decision_arm') == 'Q0':
                expected = {name: value for name, value in expected.items()
                            if not name.startswith('tabm.count_readout.')}
            if shared_readout:
                output = model.tabm.output
                width = output.components.out_features
                weight = expected.pop('tabm.output.weight')
                bias = expected.pop('tabm.output.bias')
                expected['tabm.output.components.weight'] = weight[..., :width]
                expected['tabm.output.components.bias'] = bias[..., :width]
                indices = output.pair_index.cpu()
                expected['tabm.output.pair_bias'] = bias[..., width:] - bias[..., 1:width][..., indices].sum(-1)
            if ((decision and not local_evidence) or short_add) and initial.keys() != expected.keys():
                raise RuntimeError('The decision variant changed unexpected state keys.')
            if any(not torch.equal(value, initial[name]) for name, value in expected.items()):
                raise RuntimeError('The retained parameters do not match the formal baseline initialization.')
            original_features = features
            if band:
                with torch.no_grad():
                    from band_temporal import baseline_spectral
                    original_features = (features[0], baseline_spectral(config['band_position'], *features[1:], fixed.grid))
            if shared_readout:
                check_shared_readout(baseline, model, features, config)
            else:
                with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                    if not torch.equal(baseline(*original_features), model.eval()(*features)):
                        raise RuntimeError('The variant changes the initial predictions.')
        if power_product:
            product_check = check_power_product(baseline, model, features)
        if additive_interaction:
            check_additive_interaction(model, features)
        if local_bilinear:
            cat, _ = create_model(dict(config, short_local_fusion='concat'), normalization)
            prefix = 'tabm.backbone.blocks.0.0.'
            for name, value in cat.state_dict().items():
                candidate_name = name.replace(prefix, prefix + 'base.', 1) if name.startswith(prefix) else name
                if not torch.equal(initial[candidate_name], value.cpu()):
                    raise RuntimeError(f'Bilinear candidate changed CAT initialization for {name}.')
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                if not torch.equal(cat.eval()(*features), model.eval()(*features)):
                    raise RuntimeError('The zero bilinear projection differs from CAT.')
            del cat
        masked_view = config.get('short_local_mask')
        if masked_view:
            if initial.keys() != full_initial.keys() or any(
                    not torch.equal(value, full_initial[name]) for name, value in initial.items()):
                raise RuntimeError('The mask changed the full GN state or its initial values.')
            inputs, products = [], []
            input_hook = model.tabm.register_forward_pre_hook(lambda module, args: inputs.append(args[0]))
            projection = model.tabm.backbone.blocks[0][0].project
            product_hook = projection.register_forward_pre_hook(lambda module, args: products.append(args[0]))
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                model.eval()(*features)
            input_hook.remove()
            product_hook.remove()
            waveform, spectral = inputs[0].split((SCATTER_DIM, SPECTRAL_DIM), dim=-1)
            masked, retained = (waveform, spectral) if masked_view == '1d' else (spectral, waveform)
            if torch.count_nonzero(masked) or not torch.count_nonzero(retained) or torch.count_nonzero(products[0]):
                raise RuntimeError('The standardized mask or its interaction product is incorrect.')
        loss = train_step(model, optimizer, features, train_data, ids, config, 1, 0)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - begun
        state = model_state(model)
        if torch.equal(initial[first_parameter], state[first_parameter]):
            raise RuntimeError('Smoke batch did not update public model weights.')
        if has_short_encoder and masked_view != '2d' and torch.equal(
                initial['short_local.stem.0.weight'], state['short_local.stem.0.weight']):
            raise RuntimeError('Smoke batch did not update the short-window convolution.')
        if long_variant == 'L3' and torch.equal(initial['long_local.stem.0.weight'],
                                                state['long_local.stem.0.weight']):
            raise RuntimeError('Smoke batch did not update the long-window convolution.')
        if alignment:
            for prefix in ('spectral_encoder.', 'scatter_projection.', 'tabm.'):
                if not any(not torch.equal(initial[name], state[name])
                           for name, _ in model.named_parameters() if name.startswith(prefix)):
                    raise RuntimeError(f'Smoke batch did not update {prefix}')
        temporary = ROOT / f'smoke_resume_{config["experiment_id"]}.pt'
        save_state(temporary, {'config': config, 'model': state, 'optimizer': optimizer.state_dict()})
        restored, restored_optimizer = create_model(config, normalization)
        resume_state(temporary, config, restored, restored_optimizer)
        left = train_step(model, optimizer, features, train_data, ids, config, 1, 1)
        right = train_step(restored, restored_optimizer, features, train_data, ids, config, 1, 1)
        if left != right or any(not torch.equal(x, y) for x, y in zip(model.state_dict().values(), restored.state_dict().values())):
            raise RuntimeError('Restored model/optimizer does not reproduce the next update.')
        if evidence and not torch.count_nonzero(model.tail_skip[2].weight):
            raise RuntimeError('The added evidence projection did not learn in the smoke check.')
        if band and not torch.count_nonzero(model.temporal_taps):
            raise RuntimeError('The local temporal operator did not learn in the smoke check.')
        if additive_interaction and not torch.count_nonzero(model.tabm.backbone.blocks[0][0].project.weight):
            raise RuntimeError('The additive projection did not learn in the smoke check.')
        if local_bilinear and not torch.count_nonzero(model.tabm.backbone.blocks[0][0].project.weight):
            raise RuntimeError('The bilinear projection did not learn in the smoke check.')
        if local_evidence and not torch.count_nonzero(model.local_evidence.project.weight):
            raise RuntimeError('The local evidence projection did not learn in the smoke check.')
        if shared_readout:
            for name in ('components.weight', 'pair_bias'):
                key = f'tabm.output.{name}'
                if torch.equal(initial[key], state[key]):
                    raise RuntimeError(f'The shared readout did not update {name}.')
        temporary.unlink()
        report[config['experiment_id']] = {'loss': loss, 'first_batch_seconds': seconds,
            'feature_shapes': [list(feature.shape) for feature in features],
            'condition_counts': dict(zip(CONDITIONS, np.bincount(choices, minlength=6).tolist())),
            'restored_update_exact': True}
        if power_product:
            report[config['experiment_id']]['product_check'] = product_check
        if additive_interaction:
            report[config['experiment_id']]['same_addition_in_projection_and_gate'] = True
        if short_add:
            report[config['experiment_id']]['full_gn_initial_state_and_logits_exact'] = True
        if local_bilinear:
            report[config['experiment_id']]['cat_initial_state_and_logits_exact'] = True
            report[config['experiment_id']]['bilinear_projection_updated'] = True
        if masked_view:
            report[config['experiment_id']].update(masked_view=masked_view, full_initial_state_exact=True,
                                                  standardized_slice_and_product_zero=True)
        if shared_readout:
            report[config['experiment_id']]['pair_labels_checked'] = config['set_names'][width:]
    if not decision and len(initial_states) == 2:
        if baseline is not None or alignment:
            if any(not torch.equal(initial_states[0][name], value)
                   for name, value in initial_states[1].items()):
                raise RuntimeError('The matched arms do not share their complete initialization.')
        elif not branch_masks and torch.equal(
                *(state[first_parameter] for state in initial_states)):
            raise RuntimeError('Both seeds produced the same initial weights.')
    fixed_blend = 'fixed_blend' in configs[0]
    swap_smoke = 'recording_swap_smoke.json' if pf_replacement == 53 else f'recording_swap_{pf_replacement}_smoke.json'
    name = (f'short_local_{smoke_tag}_smoke.json' if smoke_tag else
            'short_local_smoke.json' if short_local else
            'local_evidence_smoke.json' if local_evidence else
            'mb_train_val_swap_smoke.json' if recording_swap == 'mb_train_val' else
            'background_swap_smoke.json' if recording_swap == 'background_only' else
            swap_smoke if recording_swap else
            'shared_readout_smoke.json' if shared_readout else
            'additive_interaction_smoke.json' if additive_interaction else
            'power_product_smoke.json' if power_product else
            'decision_smoke.json' if decision else
            'aligned_dual_smoke.json' if alignment else
            'fixed_blend_smoke.json' if fixed_blend else 'band_temporal_smoke.json' if band
            else 'evidence_smoke.json' if evidence else 'smoke.json')
    check = ('full_gn_branch_mask_verified' if branch_masks else
             'short_local_weights_updated' if has_short_encoder else
             'waveform_only_classifier_updated' if single_view == '1d' else
             'same_initial_q0_parameters_and_logits' if local_evidence else
             'same_initial_backbone_background_and_single_parameters' if shared_readout else
             'same_initial_parameters' if alignment else
             'exact_probe_features' if fixed_blend else 'same_initial_weights_and_logits' if baseline is not None
             else 'distinct_initial_weights')
    write_json(ROOT / name, {'verified_at': now(), 'train_rows': len(train_data),
                             check: True, 'runs': report})
    log('smoke_passed', experiments=list(report), batches=report)


@torch.no_grad()
def check_power_product(baseline, model, features):
    products = []

    def capture(_module, inputs):
        products.append(inputs[0])

    for variant in (baseline, model):
        projection = variant.tabm.backbone.blocks[0][0].project
        handle = projection.register_forward_pre_hook(capture)
        try:
            with torch.autocast('cuda', dtype=torch.bfloat16):
                variant.eval()(*features)
        finally:
            handle.remove()
    if not torch.equal(products[1], rms_signed_sqrt(products[0])):
        raise RuntimeError('The projection did not receive the prescribed product transform.')
    raw, transformed = (values.float() for values in products)
    raw_norm = torch.linalg.vector_norm(raw, dim=-1)
    transformed_norm = torch.linalg.vector_norm(transformed, dim=-1)
    max_relative_error = ((transformed_norm - raw_norm).abs() / raw_norm.clamp_min(1e-6)).max()
    shares = []
    for values in (raw, transformed):
        energy = values.square()
        shares.append(float((energy.topk(4, dim=-1).values.sum(-1) / energy.sum(-1)).median()))
    if max_relative_error > .005 or not shares[1] < shares[0]:
        raise RuntimeError('The product transform failed the scale or concentration check.')
    return {'max_relative_rms_error': float(max_relative_error),
            'top_four_energy_share_median': dict(raw=shares[0], transformed=shares[1])}


@torch.no_grad()
def check_additive_interaction(model, features):
    layer = model.tabm.backbone.blocks[0][0]
    seen = {}

    def capture(name, use_input=False):
        def hook(_module, inputs, output):
            seen[name] = inputs[0] if use_input else output
        return hook

    hooks = [module.register_forward_hook(capture(name, use_input))
             for name, module, use_input in (('left', layer.left, False), ('right', layer.right, False),
                                             ('projection', layer.project, True), ('gate', layer.gater, True))]
    try:
        with torch.autocast('cuda', dtype=torch.bfloat16):
            model.eval()(*features)
            fused = scaled_addition(seen['left'], seen['right'])
            width = fused.shape[-1]
            gate_coordinates = nn.functional.layer_norm(fused, (width,))
            gate_strength = torch.log1p(fused.float().square().mean(-1).clamp_min(1e-12).sqrt())
    finally:
        for hook in hooks:
            hook.remove()
    if (not torch.equal(seen['projection'], fused)
            or not torch.equal(seen['gate'][..., 2 * width:3 * width], gate_coordinates)
            or not torch.equal(seen['gate'][..., -2], gate_strength)):
        raise RuntimeError('The projection and gate do not use the same additive interaction.')


@torch.no_grad()
def check_shared_readout(baseline, model, features, config):
    model.eval()
    with torch.autocast('cuda', dtype=torch.bfloat16):
        actual = model(*features)
        reference = baseline(*features)
    output = model.tabm.output
    width = output.components.out_features
    torch.testing.assert_close(actual[..., :width], reference[..., :width], rtol=.01, atol=.002)
    for index, name in enumerate(config['set_names'][width:]):
        first, second = (config['set_names'].index(label) for label in name.split('+'))
        expected = actual[..., first] + actual[..., second] + output.pair_bias[..., index]
        torch.testing.assert_close(actual[..., width + index], expected, rtol=0, atol=0)


def render_band_temporal(results, here, title='频带时序完整训练报告'):
    lines = [f'# {title}', '', '表格单元依次为综合／Single／Double／背景 ACC，单位为 %。']
    for split in ('Val', 'Test'):
        lines.extend(['', f'## {split}', '',
                      '| 实验 | 原始 | 10 dB | 5 dB | 0 dB | −5 dB | −10 dB | 低四档综合均值 |',
                      '| --- | --- | --- | --- | --- | --- | --- | --- |'])
        for result in results:
            metrics = result[split]
            cells = ['／'.join(f'{metrics[c][k]:.2f}' for k in ('acc', 'single_acc', 'double_acc', 'noise_acc'))
                     for c in CONDITIONS]
            mean = np.mean([metrics[c]['acc'] for c in CONDITIONS[2:]])
            lines.append('| ' + ' | '.join([result['experiment_id'], *cells, f'{mean:.2f}']) + ' |')
    (here / f'{title}.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume', action='store_true')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--evidence', action='store_true', help='Train the matched B/C arms at seed 13.')
    modes.add_argument('--band-temporal', action='store_true', help='Train the screened band-temporal arms at seed 13.')
    modes.add_argument('--fixed-probe-blend', action='store_true', help='Train F on the fixed long-window probe representation.')
    modes.add_argument('--aligned-dual', action='store_true', help='Train global and time-aligned spectral fusion at seed 13.')
    modes.add_argument('--decision', action='store_true', help='Train I0 and Q0 decision-interface controls at seed 13.')
    modes.add_argument('--power-product', action='store_true', help='Train the Q0 power-normalized interaction at seed 13.')
    modes.add_argument('--additive-interaction', action='store_true', help='Train the matched additive Q0 control at seed 13.')
    modes.add_argument('--shared-readout', action='store_true', help='Train Q0 with shared vessel-type evidence at seed 13.')
    modes.add_argument('--recording-swap', action='store_true', help='Train Q0 on the documented background and PF recording swap.')
    modes.add_argument('--background-swap', action='store_true', help='Train Q0 on the background-only Train/Val swap.')
    modes.add_argument('--mb-train-val-swap', action='store_true', help='Train Q0 on the MB-only Train/Val swap.')
    modes.add_argument('--local-evidence', action='store_true', help='Train the transferred local evidence module on the frozen Q0 split.')
    modes.add_argument('--short-local', action='store_true', help='Replace the Q0 short-window scattering with a local CNN on dense log-Mel.')
    modes.add_argument('--short-local-long', choices=('L1', 'L2', 'L3'),
                       help='Remove the long view (L1), use its 32x1 summary (L2), or encode it locally (L3).')
    modes.add_argument('--long-record-normalization', action='store_true',
                       help='Train L4 with GroupNorm/LayerNorm in the L3 long-window encoder.')
    modes.add_argument('--short-record-normalization', action='store_true',
                       help='Train ShortLocal with four-group normalization inside the short-window CNN.')
    modes.add_argument('--short-gn-cat', action='store_true',
                       help='Train the ShortLocal-GN concatenation ablation with standard TabM.')
    modes.add_argument('--short-gn-add', action='store_true',
                       help='Train ShortLocal-GN with scaled addition in place of the interaction product.')
    modes.add_argument('--short-gn-bilinear', action='store_true',
                       help='Replace the ShortLocal-GN residual with same-bin bilinear pooling.')
    modes.add_argument('--short-gn-single-view', choices=('1d', '2d'),
                       help='Train one ShortLocal-GN input view with standard TabM.')
    modes.add_argument('--short-gn-mask', action='store_true',
                       help='Train both full-width GN ablations with one standardized branch masked.')
    parser.add_argument('--pf-replacement', type=int, choices=(53, 63), default=53)
    args = parser.parse_args()
    short_local_run = (args.short_local or args.short_local_long is not None
                       or args.long_record_normalization or args.short_record_normalization
                       or args.short_gn_cat or args.short_gn_add or args.short_gn_bilinear or args.short_gn_mask
                       or args.short_gn_single_view is not None)
    if args.pf_replacement != 53 and not args.recording_swap:
        parser.error('--pf-replacement requires --recording-swap')
    swap_suffix = '' if args.pf_replacement == 53 else f'_{args.pf_replacement}'
    data_swap = args.recording_swap or args.background_swap or args.mb_train_val_swap
    ROOT.mkdir(exist_ok=True)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    config = json.loads((HERE / '正式实验配置.json').read_text(encoding='utf-8'))
    runtime = {'device': torch.cuda.get_device_name(), 'torch': str(torch.__version__),
               'cuda': torch.version.cuda, 'numpy': np.__version__, 'python': sys.executable}
    baseline = None
    position = None
    if args.long_record_normalization:
        baseline = json.loads((ROOT / 'SE-Q0-ShortLocal-L3-13/results.json').read_text(encoding='utf-8'))
        config = dict(baseline['config'], experiment_id='SE-Q0-ShortLocal-L4-13', runtime=runtime,
                      long_normalization='record')
        configs = [config]
    elif args.short_record_normalization:
        baseline = json.loads((ROOT / 'SE-Q0-ShortLocal-13/results.json').read_text(encoding='utf-8'))
        config = dict(baseline['config'], experiment_id='SE-Q0-ShortLocal-GN-13', runtime=runtime,
                      short_normalization='record')
        configs = [config]
    elif args.short_gn_cat:
        baseline = json.loads((ROOT / 'SE-Q0-ShortLocal-GN-13/results.json').read_text(encoding='utf-8'))
        config = dict(baseline['config'], experiment_id='SE-ABL-ShortLocal-GN-CAT-13', runtime=runtime,
                      short_local_fusion='concat')
        configs = [config]
    elif args.short_gn_add:
        baseline = json.loads((ROOT / 'SE-Q0-ShortLocal-GN-13/results.json').read_text(encoding='utf-8'))
        config = dict(baseline['config'], experiment_id='SE-ABL-ShortLocal-GN-ADD-13', runtime=runtime,
                      short_local_fusion='additive')
        configs = [config]
    elif args.short_gn_bilinear:
        baseline = json.loads((ROOT / 'SE-Q0-ShortLocal-GN-13/results.json').read_text(encoding='utf-8'))
        config = dict(baseline['config'], experiment_id='SE-Q0-ShortLocal-GN-Bilinear-13', runtime=runtime,
                      short_local_fusion='local_bilinear', bilinear_rank=16)
        configs = [config]
    elif args.short_gn_single_view:
        baseline = json.loads((ROOT / 'SE-ABL-ShortLocal-GN-CAT-13/results.json').read_text(encoding='utf-8'))
        view = args.short_gn_single_view
        config = dict(baseline['config'], experiment_id=f'SE-ABL-ShortLocal-GN-{view.upper()}-13',
                      runtime=runtime, short_local_view=view)
        configs = [config]
    elif args.short_gn_mask:
        baseline = json.loads((ROOT / 'SE-Q0-ShortLocal-GN-13/results.json').read_text(encoding='utf-8'))
        config = dict(baseline['config'], runtime=runtime)
        configs = [dict(config, experiment_id=f'SE-ABL-ShortLocal-GN-Mask-{view.upper()}-13',
                        short_local_mask=view) for view in ('1d', '2d')]
    elif args.short_local_long:
        baseline = json.loads((ROOT / 'SE-Q0-ShortLocal-13/results.json').read_text(encoding='utf-8'))
        long_config = {'arm': args.short_local_long}
        if args.short_local_long == 'L2':
            long_config['summary_grid'] = [32, 1]
        elif args.short_local_long == 'L3':
            long_config.update(frequency_block=4, width=16)
        config = dict(baseline['config'],
                      experiment_id=f'SE-Q0-ShortLocal-{args.short_local_long}-13',
                      runtime=runtime, short_local_long=long_config)
        configs = [config]
    elif args.short_local:
        baseline = json.loads((ROOT / 'SE-Q0-MBTrainValSwap-13/results.json').read_text(encoding='utf-8'))
        config = dict(baseline['config'], experiment_id='SE-Q0-ShortLocal-13', runtime=runtime,
                      short_local={'window_length': 480, 'hop_length': 120, 'n_fft': 1024,
                                   'mel_bins': 128, 'width': 16})
        configs = [config]
    elif args.local_evidence:
        baseline = json.loads((ROOT / 'SE-Q0-MBTrainValSwap-13/results.json').read_text(encoding='utf-8'))
        config = dict(baseline['config'], experiment_id='SE-Q0-LocalEvidence-13', runtime=runtime,
                      local_evidence={'hidden_dim': 32, 'presence_weight': 0.1})
        configs = [config]
    elif (args.evidence or args.band_temporal or args.fixed_probe_blend
            or args.aligned_dual or args.decision or args.power_product
            or args.additive_interaction or args.shared_readout or data_swap):
        baseline = json.loads((ROOT / 'SE-F-13/results.json').read_text(encoding='utf-8'))
        expected = dict(config, seed=13, experiment_id='SE-F-13', runtime=runtime)
        if expected != baseline['config']:
            raise ValueError('The current protocol or runtime differs from the stored SE-F-13 baseline.')
        if args.power_product or args.additive_interaction or args.shared_readout or data_swap:
            baseline = json.loads((ROOT / 'SE-Q0-13/results.json').read_text(encoding='utf-8'))
            q0 = dict(expected, experiment_id='SE-Q0-13', decision_arm='Q0')
            if q0 != baseline['config']:
                raise ValueError('The current protocol or runtime differs from the stored Q0 baseline.')
            if data_swap:
                if args.mb_train_val_swap:
                    dataset = HERE.parent / 'shipsear_mb_train_val_20261104' / 'data'
                else:
                    folder = 'data' if args.background_swap else f'data_pf65_{args.pf_replacement}'
                    dataset = HERE.parent / 'shipsear_val_background_20261104' / folder
                dataset_config = json.loads((dataset / 'CONFIG.json').read_text(encoding='utf-8'))
                summary = json.loads((dataset / 'SUMMARY.json').read_text(encoding='utf-8'))
                config = dict(q0, data_root=str(dataset), dataset_version=dataset_config['version'],
                              split_sizes={split: row['samples'] for split, row in summary.items()})
                if args.mb_train_val_swap:
                    configs = [dict(config, experiment_id='SE-Q0-MBTrainValSwap-13', recording_swap='mb_train_val')]
                elif args.background_swap:
                    configs = [dict(config, experiment_id='SE-Q0-BackgroundSwap-13', recording_swap='background_only')]
                else:
                    experiment_tag = '' if args.pf_replacement == 53 else str(args.pf_replacement)
                    if experiment_tag:
                        config['pf_replacement'] = args.pf_replacement
                    configs = [dict(config, experiment_id=f'SE-Q0-RecordingSwap{experiment_tag}-13', recording_swap=True)]
            elif args.shared_readout:
                configs = [dict(q0, experiment_id='SE-Q0-Shared-13', shared_readout=True)]
            elif args.additive_interaction:
                configs = [dict(q0, experiment_id='SE-Q0-A0-13', additive_interaction=True)]
            else:
                configs = [dict(q0, experiment_id='SE-Q0-Power-13', power_product=True)]
        elif args.decision:
            configs = [dict(expected, experiment_id=f'SE-{arm}-13', decision_arm=arm) for arm in DECISION_ARMS]
        elif args.aligned_dual:
            configs = [dict(expected, experiment_id=f'SE-TF-{tag}-13', alignment_arm=arm,
                            normalization=dict(expected['normalization'], scope='waveform_scattering_only'),
                            spectral_frontend='author Mel, 4 retained frequency regions and 10 time cells',
                            window_embedding=128)
                       for tag, arm in (('G', 'global'), ('A', 'aligned'))]
        elif args.band_temporal:
            screen = json.loads((ROOT / 'position_probe.json').read_text(encoding='utf-8'))
            position = screen['winner']
            if position not in ('short', 'long') or not screen['gates'][position]['passed']:
                raise ValueError('No band-temporal insertion position passed Train/Val screening.')
            configs = [dict(expected, experiment_id=f'SE-BT-{arm}-13', band_arm=arm, band_position=position)
                       for arm in BAND_ARMS]
        elif args.fixed_probe_blend:
            screen = json.loads((ROOT / 'position_probe.json').read_text(encoding='utf-8'))
            if screen['winner'] != 'long' or screen['filter_weights'] != [.25, .5, .25]:
                raise ValueError('The stored probe does not match the fixed long-window replication.')
            configs = [dict(expected, experiment_id='SE-BT-Fixed-13',
                            fixed_blend={'position': 'long', 'weights': screen['filter_weights'], 'padding': 'replicate'})]
        else:
            configs = [dict(expected, experiment_id=f'SE-{arm}-13', evidence_arm=arm) for arm in EVIDENCE_ARMS]
    else:
        configs = [dict(config, seed=seed, experiment_id=f'SE-F-{seed}', runtime=runtime)
                   for seed in config['training_seeds']]
    dirs = [ROOT / cfg['experiment_id'] for cfg in configs]
    if not args.smoke and not args.resume and any((out / 'resume.pt').exists() for out in dirs):
        raise FileExistsError('A formal run already exists. Use --resume.')
    if short_local_run:
        from short_local import ShortLocalViews
        fixed = ShortLocalViews(config).cuda().eval()
    elif args.aligned_dual:
        from aligned_dual import AlignedViews
        fixed = AlignedViews(config).cuda().eval()
    elif args.band_temporal:
        from band_temporal import BandViews, baseline_spectral
        fixed = BandViews(config, position).cuda().eval()
    elif args.fixed_probe_blend:
        from position_probe import FixedLongBlendViews, PositionViews
        fixed = FixedLongBlendViews(config).cuda().eval()
    else:
        fixed = FixedViews(config).cuda().eval()
    train_data = FixedDataset(config['data_root'], 'Train')
    if args.fixed_probe_blend and args.smoke:
        ids = np.arange(config['batch_size'])
        waves, _ = training_views(train_data, ids, 1, config['noise_probabilities'], 13)
        probe = PositionViews(config).cuda().eval()
        one_d, long, short = probe(waves)[2]
        exact = (one_d, torch.cat((long, short), dim=1))
        if any(not torch.equal(left, right) for left, right in zip(fixed(waves), exact)):
            raise RuntimeError('The full-training front end differs from the screened probe.')
        del probe, exact, one_d, long, short
    begun = time.perf_counter()
    originals = feature_cache(fixed, train_data, config)
    if args.short_local_long == 'L3' or args.long_record_normalization:
        normalization = fit_normalization(train_data, originals[:1], config)
    elif short_local_run:
        fixed_features = originals if args.short_gn_single_view == '1d' else originals[:-1]
        normalization = fit_normalization(train_data, fixed_features, config)
    elif args.aligned_dual:
        normalization = fit_normalization(train_data, originals[:1], config)
    elif args.band_temporal:
        untouched, coefficients = originals[1:]
        spectra = [baseline_spectral(position, a.cuda(), b.cuda(), fixed.grid).cpu()
                   for a, b in zip(untouched.split(config['batch_size']), coefficients.split(config['batch_size']))]
        normalization = fit_normalization(train_data, (originals[0], torch.cat(spectra)), config)
        del spectra, untouched, coefficients
    else:
        normalization = fit_normalization(train_data, originals, config)
    train_cache_seconds = time.perf_counter() - begun
    if args.smoke:
        smoke(configs, fixed, train_data, originals, normalization, args.evidence, args.band_temporal)
        return
    val_data = FixedDataset(config['data_root'], 'Val')
    begun = time.perf_counter()
    val_cache = {c: feature_cache(fixed, val_data, config, c) for c in CONDITIONS}
    val_cache_seconds = time.perf_counter() - begun
    prep_name = (f'short_local_GN_{args.short_gn_single_view.upper()}_preparation.json'
                 if args.short_gn_single_view else
                 'short_local_GN_Mask_preparation.json' if args.short_gn_mask else
                 'short_local_GN_ADD_preparation.json' if args.short_gn_add else
                 'short_local_GN_Bilinear_preparation.json' if args.short_gn_bilinear else
                 'short_local_GN_CAT_preparation.json' if args.short_gn_cat else
                 'short_local_GN_preparation.json' if args.short_record_normalization else
                 'short_local_L4_preparation.json' if args.long_record_normalization else
                 f'short_local_{args.short_local_long}_preparation.json' if args.short_local_long else
                 'short_local_preparation.json' if args.short_local else
                 'local_evidence_preparation.json' if args.local_evidence else
                 'mb_train_val_swap_preparation.json' if args.mb_train_val_swap else
                 'background_swap_preparation.json' if args.background_swap else
                 f'recording_swap{swap_suffix}_preparation.json' if args.recording_swap else
                 'shared_readout_preparation.json' if args.shared_readout else
                 'additive_interaction_preparation.json' if args.additive_interaction else
                 'power_product_preparation.json' if args.power_product else
                 'decision_preparation.json' if args.decision else
                 'aligned_dual_preparation.json' if args.aligned_dual else
                 'fixed_blend_preparation.json' if args.fixed_probe_blend else
                 'band_temporal_preparation.json' if args.band_temporal else
                 'evidence_preparation.json' if args.evidence else 'preparation.json')
    write_json(ROOT / prep_name, {'runtime': runtime, 'prepared_at': now(),
               'train_rows': len(train_data), 'val_rows': len(val_data), 'normalization_split': 'Train',
               'train_cache_seconds': train_cache_seconds, 'val_cache_seconds': val_cache_seconds})
    selections = []
    for cfg, out in zip(configs, dirs):
        if (out / 'selection_sealed.json').is_file():
            saved = json.loads((out / 'config_resolved.json').read_text(encoding='utf-8'))
            if saved != cfg:
                raise ValueError('Sealed run configuration differs.')
            selections.append(json.loads((out / 'selection_sealed.json').read_text(encoding='utf-8')))
        else:
            selections.append(train(cfg, out, fixed, train_data, originals, val_data, val_cache, normalization, args.resume))
    del originals
    if not args.evidence:
        del val_cache
    log('test_started', sealed=[selection['experiment_id'] for selection in selections])
    test_data = FixedDataset(config['data_root'], 'Test')
    test_cache = {c: feature_cache(fixed, test_data, config, c) for c in CONDITIONS}
    reference = json.loads((PACKAGE / 'runs/seed13/results.json').read_text(encoding='utf-8'))
    if args.evidence:
        from report_evidence import render_evidence as render
        results = [baseline]
    elif args.band_temporal:
        render, results = render_band_temporal, [baseline]
    elif args.fixed_probe_blend:
        def render(results, here):
            render_band_temporal(results, here, '长窗固定时序完整复验报告')

        results = [baseline]
    elif args.aligned_dual:
        def render(results, here):
            render_band_temporal(results, here, '双分支时序配合完整训练报告')

        shuffle_result = HERE.parent / 'external_baselines_20261026/formal_runs/ShuffleFACAuthor/results.json'
        results = [baseline, json.loads(shuffle_result.read_text(encoding='utf-8'))]
    elif args.power_product:
        def render(results, here):
            render_band_temporal(results, here, 'Q0乘积幂归一化完整训练报告')

        results = [baseline]
    elif args.additive_interaction:
        def render(results, here):
            render_band_temporal(results, here, 'Q0等参数加性交互完整训练报告')

        results = [baseline]
    elif args.shared_readout:
        def render(results, here):
            render_band_temporal(results, here, 'Q0共享船型读出完整训练报告')

        results = [baseline]
    elif args.local_evidence:
        def render(results, here):
            render_band_temporal(results, here, 'Q0局部船型证据迁移报告')

        results = [baseline]
    elif args.long_record_normalization:
        def render(results, here):
            render_band_temporal(results, here, '长窗样本内归一化L4完整训练报告')

        results = [baseline]
    elif args.short_record_normalization:
        def render(results, here):
            render_band_temporal(results, here, '短窗GroupNorm完整训练报告')

        results = [baseline]
    elif args.short_gn_cat:
        def render(results, here):
            render_band_temporal(results, here, 'ShortLocal-GN直接拼接消融报告')

        results = [baseline]
    elif args.short_gn_add:
        def render(results, here):
            render_band_temporal(results, here, 'ShortLocal-GN正式消融报告')

        results = [baseline]
        for arm in ('Mask-1D', 'Mask-2D', 'CAT'):
            path = ROOT / f'SE-ABL-ShortLocal-GN-{arm}-13' / 'results.json'
            results.append(json.loads(path.read_text(encoding='utf-8')))
    elif args.short_gn_bilinear:
        def render(results, here):
            render_band_temporal(results, here, 'ShortLocal-GN同段双线性完整训练报告')

        cat = json.loads((ROOT / 'SE-ABL-ShortLocal-GN-CAT-13/results.json').read_text(encoding='utf-8'))
        results = [baseline, cat]
    elif args.short_gn_single_view:
        def render(results, here):
            render_band_temporal(results, here, f'ShortLocal-GN-{args.short_gn_single_view.upper()}单路消融报告')

        results = [baseline]
    elif args.short_gn_mask:
        def render(results, here):
            render_band_temporal(results, here, 'ShortLocal-GN完整模型分支消融报告')

        results = [baseline]
    elif args.short_local_long:
        def render(results, here):
            render_band_temporal(results, here, f'Q0长短窗互补{args.short_local_long}完整训练报告')

        results = [baseline]
    elif args.short_local:
        def render(results, here):
            render_band_temporal(results, here, 'Q0短窗局部编码完整训练报告')

        results = [baseline]
    elif args.mb_train_val_swap:
        def render(results, here):
            render_band_temporal(results, here, 'Q0仅MB录音交换完整训练报告')

        results = [baseline]
    elif args.background_swap:
        def render(results, here):
            render_band_temporal(results, here, 'Q0仅背景录音交换完整训练报告')

        results = [baseline]
    elif args.recording_swap:
        def render(results, here):
            title = ('Q0录音交换完整训练报告' if args.pf_replacement == 53
                     else f'Q0录音交换65与{args.pf_replacement}完整训练报告')
            render_band_temporal(results, here, title)

        results = [baseline]
    elif args.decision:
        def render(results, here):
            render_band_temporal(results, here, '公共判别接口完整训练报告')

        results = [baseline]
    else:
        from report_formal import render
        results = []
    for cfg, out, selection in zip(configs, dirs, selections):
        path = out / 'results.json'
        if path.is_file():
            result = json.loads(path.read_text(encoding='utf-8'))
        else:
            saved = torch.load(out / 'best.pt', map_location='cpu', weights_only=False)
            if (saved['epoch'], saved['score']) != (selection['epoch'], selection['score']):
                raise RuntimeError('Checkpoint differs from the sealed selection.')
            model, _ = create_model(cfg, saved['normalization'])
            model.load_state_dict(saved['model'], strict=True)
            metrics = evaluate(model, test_data, test_cache, cfg, out)
            history = json.loads((out / 'history.json').read_text(encoding='utf-8'))
            result = dict(seed=cfg['seed'], experiment_id=cfg['experiment_id'], config=cfg,
                          elapsed_minutes=selection['elapsed_seconds'] / 60,
                          epochs_trained=selection['epochs_trained'], stop_reason=selection['stop_reason'],
                          selection=selection, Val=selection['Val'], Test=metrics, parameters=selection['parameters'],
                          source_reference={split: reference[split]['P50_noise50'] for split in ('Val', 'Test')},
                          benchmark={'train_cache_seconds': train_cache_seconds, 'val_cache_seconds': val_cache_seconds,
                                     'median_epoch_seconds': float(np.median([row['seconds'] for row in history]))})
            if args.evidence:
                with torch.no_grad():
                    model.tail_skip[2].weight.zero_()
                result['path_disabled'] = {
                    'Val': evaluate(model, val_data, val_cache, cfg),
                    'Test': evaluate(model, test_data, test_cache, cfg),
                }
            write_json(path, result)
        update_registry(cfg, out, '已完成', selection['epoch'])
        results.append(result)
        render(results, HERE)
        log('test_finished', experiment_id=cfg['experiment_id'], epoch=selection['epoch'])
    if args.evidence:
        from paired_evidence import evaluate_pairs
        evaluate_pairs(fixed, HERE)
    log('completed', experiments=[result['experiment_id'] for result in results])


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        ROOT.mkdir(exist_ok=True)
        log('failed', error=repr(error))
        raise
