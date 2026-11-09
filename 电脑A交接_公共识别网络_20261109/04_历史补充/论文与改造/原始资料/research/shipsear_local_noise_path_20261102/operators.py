"""Train-only diagnostic noise subspaces and matched projection controls.

Fitting uses frozen hidden states, not gradients or new V1 task weights.
Projection is evaluated in Train-scaled coordinates independently for each
member. Reference-dependent correction is an offline diagnostic operation.
"""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
import torch

import network


_EPS = 1e-12


def _fit_weights(panel, n):
    """Equal recordings, then cores within recording, then repeated pairs."""
    if getattr(panel, 'split', 'Train') != 'Train':
        raise ValueError('Noise operators must be fitted on Train only.')
    if hasattr(panel, 'sources') and hasattr(panel, 'recordings'):
        sources = [str(value) for value in panel.sources]
        recordings = [str(value) for value in panel.recordings]
    else:
        sources = [str(row['anchor_source']) for row in panel.plans]
        recordings = [str(row['anchor_recording']) for row in panel.plans]
    if len(sources) != n or len(recordings) != n:
        raise ValueError('The panel row count does not match hidden states.')
    cores = defaultdict(set)
    repetitions = Counter(zip(recordings, sources))
    for recording, source in zip(recordings, sources):
        cores[recording].add(source)
    weights = torch.tensor([
        1 / (len(cores) * len(cores[recording]) * repetitions[(recording, source)])
        for recording, source in zip(recordings, sources)
    ], dtype=torch.float64)
    weights /= weights.sum()
    return weights, len(cores), len(set(sources))


def _state(cache, key, site):
    value = torch.as_tensor(cache[key]['states'][site])
    if value.device.type != 'cpu':
        raise ValueError('Operator fitting accepts CPU Train hidden states only.')
    if value.ndim != 3 or not value.is_floating_point():
        raise ValueError('Expected native Train states [sample,member,channel].')
    return value


def _canonical_sign(basis):
    indices = basis.abs().argmax(dim=1)
    largest = basis.gather(1, indices[:, None, :]).squeeze(1)
    sign = torch.where(largest < 0, -torch.ones_like(largest), torch.ones_like(largest))
    return basis * sign[:, None, :]


def _scale(cache, keys, site, weights, threshold, chunk_size):
    shape = _state(cache, keys[0], site).shape[1:]
    mean = torch.zeros(shape, dtype=torch.float64)
    second = torch.zeros_like(mean)
    for key in keys:
        states = _state(cache, key, site)
        if states.shape != (len(weights), *shape):
            raise ValueError('Train views must have identical sample/member coordinates.')
        for start in range(0, len(states), chunk_size):
            x = states[start:start + chunk_size].double()
            w = weights[start:start + len(x), None, None] / len(keys)
            mean += (w * x).sum(0)
            second += (w * x.square()).sum(0)
    std = (second - mean.square()).clamp_min(0).sqrt()
    constant = std < threshold
    result = torch.where(constant, torch.ones_like(std), std)
    if not torch.isfinite(result).all() or not (result > 0).all():
        raise ValueError('Invalid Train interface scale.')
    return result, int(constant.sum())


def _delta_chunk(cache, key_a, key_b, site, start, stop, scale):
    # Conversion precedes subtraction, preserving the native-state difference.
    a = _state(cache, key_a, site)[start:stop].double()
    b = _state(cache, key_b, site)[start:stop].double()
    return (a - b) / scale


def _energy_fraction(cache, key_a, key_b, site, scale, basis, weights, chunk_size):
    members = basis.shape[0]
    total = torch.zeros(members, dtype=torch.float64)
    retained = torch.zeros_like(total)
    for start in range(0, len(weights), chunk_size):
        stop = min(start + chunk_size, len(weights))
        delta = _delta_chunk(cache, key_a, key_b, site, start, stop, scale)
        coords = torch.einsum('nkd,kdr->nkr', delta, basis)
        w = weights[start:stop, None]
        total += (w * delta.square().sum(-1)).sum(0)
        retained += (w * coords.square().sum(-1)).sum(0)
    fraction = torch.where(total > _EPS, retained / total.clamp_min(_EPS), torch.zeros_like(total))
    return float(fraction.mean())


@torch.no_grad()
def fit(train_cache, panel, cfg):
    """Fit separate member-local scales and rank-r noise subspaces per site.

    The six views are original/0dB crossed with absent/equal/weak source.
    Scales use all six; the *uncentered* noise second moment uses only the
    0dB-minus-original absent view, keeping the common mean noise shift.
    """
    keys = list(train_cache)
    if len(keys) != 6 or {key[0] for key in keys} != {'original', '0dB'}:
        raise ValueError('Expected six Train views: original/0dB by three strengths.')
    if ('original', 'absent') not in train_cache or ('0dB', 'absent') not in train_cache:
        raise ValueError('Missing paired absent-source Train views.')
    sites = tuple(cfg.get('sites', network.SITES))
    chunk_size = int(cfg.get('operator_fit_batch_size', 128))
    threshold = float(cfg.get('scale_constant_threshold', 1e-6))
    if chunk_size < 1 or threshold <= 0:
        raise ValueError('Invalid operator fitting settings.')
    n = len(_state(train_cache, keys[0], sites[0]))
    weights, recording_count, source_count = _fit_weights(panel, n)
    operators, metadata = {}, {}
    for site_index, site in enumerate(sites):
        scale, constant_count = _scale(train_cache, keys, site, weights, threshold, chunk_size)
        members, dimensions = scale.shape
        rank = int(cfg.get('rank', 16))
        if not 1 <= rank <= dimensions:
            raise ValueError('Projection rank must lie within the interface dimension.')
        moment = torch.zeros((members, dimensions, dimensions), dtype=torch.float64)
        for start in range(0, n, chunk_size):
            stop = min(start + chunk_size, n)
            delta = _delta_chunk(train_cache, ('0dB', 'absent'), ('original', 'absent'),
                                 site, start, stop, scale)
            weighted = delta * weights[start:stop, None, None].sqrt()
            batch = weighted.permute(1, 0, 2).contiguous()
            moment += torch.bmm(batch.transpose(1, 2), batch)
        moment = (moment + moment.transpose(1, 2)) * .5
        eigenvalues, eigenvectors = torch.linalg.eigh(moment)
        eigenvalues = eigenvalues.clamp_min(0)
        basis = _canonical_sign(eigenvectors[:, :, -rank:].flip(-1))
        total = eigenvalues.sum(-1)
        retained = eigenvalues[:, -rank:].sum(-1)
        fraction = torch.where(total > _EPS, retained / total.clamp_min(_EPS), torch.zeros_like(total))
        generator = torch.Generator(device='cpu').manual_seed(
            int(cfg.get('seed', 13)) + int(cfg.get('random_seed_tag', 7701)) + site_index)
        gaussian = torch.randn(members, dimensions, rank, dtype=torch.float64, generator=generator)
        random = _canonical_sign(torch.linalg.qr(gaussian, mode='reduced').Q)
        operators[site] = dict(scale=scale.float(), noise=basis.float(), random=random.float())
        identity = torch.eye(rank)
        errors = [float((u.transpose(1, 2) @ u - identity).abs().max())
                  for u in (operators[site]['noise'], operators[site]['random'])]
        info = dict(rank=rank, noise_energy_fraction=float(fraction.mean()),
                    noise_energy_fraction_min=float(fraction.min()), noise_energy_fraction_max=float(fraction.max()),
                    scale_constant_coordinates=constant_count, fit_n=n,
                    fit_recordings=recording_count, fit_sources=source_count,
                    basis_orthogonality_maxerror=max(errors),
                    fitting='Train only; equal recordings/cores/repeated pairs; uncentered noise second moment.')
        for strength in {key[1] for key in keys} - {'absent'}:
            name = 'weak' if strength.lower().startswith('weak') else strength
            for condition, label in (('original', 'raw'), ('0dB', '0dB')):
                info[f'train_source_energy_fraction_{name}_{label}'] = _energy_fraction(
                    train_cache, (condition, strength), (condition, 'absent'), site,
                    scale, basis, weights, chunk_size)
        metadata[site] = info
    return operators, metadata


@torch.no_grad()
def correction(delta, op, basis_kind):
    """Return an offline component correction plus scale/energy diagnostics.

    ``random_energy`` matches the noise-subspace projected norm independently
    for each sample/member, in standardized coordinates. Degenerate random
    projections are set to zero and counted rather than divided by tiny zero
    norms. This function does not cast the correction back to a native hidden
    dtype; callers do that at the suffix boundary.
    """
    if basis_kind not in ('noise', 'random_rank', 'random_energy'):
        raise ValueError('Unknown projection control: ' + str(basis_kind))
    values = delta.float()
    scale = torch.as_tensor(op['scale'], dtype=torch.float32, device=values.device)
    basis = op['noise'] if basis_kind == 'noise' else op['random']
    corr = network.project(values, scale, basis)
    normalized = values / scale
    delta_norm = torch.linalg.vector_norm(normalized, dim=-1)
    ratio_max = 1.
    skipped = 0
    target_norm = None
    matched = None
    if basis_kind == 'random_energy':
        target = network.project(values, scale, op['noise'])
        target_norm = torch.linalg.vector_norm(target / scale, dim=-1)
        random_norm = torch.linalg.vector_norm(corr / scale, dim=-1)
        nonzero_target = target_norm > _EPS
        valid = random_norm > _EPS
        ratio = torch.where(valid, target_norm / random_norm.clamp_min(_EPS), torch.zeros_like(random_norm))
        valid = valid & torch.isfinite(ratio)
        ratio = torch.where(valid, ratio, torch.zeros_like(ratio))
        skipped = int((nonzero_target & ~valid).sum())
        ratio_max = float(ratio.max()) if ratio.numel() else 0.
        corr = corr * ratio[..., None]
        matched = nonzero_target & valid
    projected_norm = torch.linalg.vector_norm(corr / scale, dim=-1)
    active = delta_norm > _EPS
    fraction = (projected_norm / delta_norm.clamp_min(_EPS)).square()
    chosen = fraction[active]
    stats = dict(
        projection_fraction_mean=float(chosen.mean()) if chosen.numel() else 0.,
        projection_fraction_min=float(chosen.min()) if chosen.numel() else 0.,
        projection_fraction_max=float(chosen.max()) if chosen.numel() else 0.,
        random_energy_ratio_max=ratio_max,
        energy_match_skipped=skipped,
        native_displacement_rms_mean=float(corr.square().mean(-1).sqrt().mean()),
        n_member_vectors=int(delta_norm.numel()), valid_delta_vectors=int(active.sum()),
    )
    if target_norm is not None:
        errors = (projected_norm - target_norm).abs() / target_norm.clamp_min(_EPS)
        stats['energy_matched_vectors'] = int(matched.sum())
        stats['energy_match_relative_error_max'] = float(errors[matched].max()) if matched.any() else 0.
    return corr, stats
