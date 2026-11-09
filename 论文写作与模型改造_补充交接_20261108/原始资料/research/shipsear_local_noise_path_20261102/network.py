"""Native-state capture and suffix replay for frozen V1 path interventions.

This module neither changes V1 nor fits an intervention. Hidden states retain
their complete member axis and native dtype; callers cast a perturbed state
back to that dtype before suffix replay. Final probabilities follow V1's
CUDA member softmax followed by the CPU NumPy member mean.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
_SOURCE_FILE = HERE.parent / 'shipsear_noise_target_factorial_20261102' / 'model.py'
_spec = importlib.util.spec_from_file_location('local_noise_path_v1_loader', _SOURCE_FILE)
_frozen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_frozen)

SITES = ('first', 'second', 'last')
SOURCE = _frozen.SOURCE
COUNT_MAP = _frozen.COUNT_MAP
SET_NAMES = _frozen.SET_NAMES


def _check_model(model):
    if len(model.tabm.backbone.blocks) != len(SITES):
        raise ValueError('The local path diagnostic requires V1\'s three blocks.')
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError('The V1 classifier must remain frozen.')
    model.eval()


def _check_hidden(model, hidden):
    expected = (model.config['model']['k'], model.config['model']['d_block'])
    if hidden.ndim != 3 or tuple(hidden.shape[1:]) != expected:
        raise ValueError('Hidden state must retain all [batch,member,channel] coordinates.')
    if not hidden.is_floating_point():
        raise ValueError('Hidden states must use a floating-point dtype.')


def load(device='cuda'):
    """Load the retained V1, preserving its task weights and normalization."""
    fixed, model, saved = _frozen.load(device)
    _check_model(model)
    if any(parameter.requires_grad for parameter in fixed.parameters()):
        raise RuntimeError('The acoustic extractor must remain frozen.')
    return fixed, model, saved


@torch.no_grad()
def capture(model, features):
    """Capture all three actual block outputs without casting hidden states.

    ``features`` is V1's ordinary two-vector acoustic input. The caller keeps
    acoustic extraction at batch 32 and classification at batch 64. Only the
    probability softmax is promoted to float32; hidden tensors are detached,
    cloned and copied to CPU with their native dtype intact.
    """
    if len(features) != 2:
        raise ValueError('V1 expects its two ordinary acoustic feature vectors.')
    _check_model(model)
    captured = {}

    def save(name):
        def hook(_module, _inputs, output):
            captured[name] = output.detach().clone()
        return hook

    handles = [block.register_forward_hook(save(site))
               for site, block in zip(SITES, model.tabm.backbone.blocks)]
    handles.append(model.tabm.output.register_forward_hook(save('main_logits')))
    device_type = next(model.parameters()).device.type
    try:
        with torch.autocast(device_type, dtype=torch.bfloat16,
                            enabled=device_type == 'cuda'):
            logits = model(*features)
        for site in SITES:
            _check_hidden(model, captured[site])
        return dict(
            states={site: captured[site].cpu() for site in SITES},
            probability=logits.float().softmax(-1).cpu().numpy().mean(1),
            main_probability=captured['main_logits'].float().softmax(-1).cpu().numpy().mean(1),
        )
    finally:
        for handle in handles:
            handle.remove()


@torch.no_grad()
def suffix(model, site, hidden, return_states=False):
    """Run the unchanged remaining blocks and both original V1 readouts.

    ``site`` denotes the block output supplied in ``hidden``. No hidden dtype
    conversion is performed here, so unmodified captured states can replay
    the original computation exactly at the same classifier batch size.
    If ``return_states`` is true, return the probability plus native GPU (or
    source-device) states at this site and all subsequent blocks, without CPU
    copies, so the caller can examine the same intervention's propagation.
    """
    if site not in SITES:
        raise ValueError('Unknown intervention site: ' + str(site))
    _check_model(model)
    _check_hidden(model, hidden)
    device = next(model.parameters()).device
    value = hidden.to(device=device)
    states = {site: value.detach().clone()} if return_states else None
    with torch.autocast(device.type, dtype=torch.bfloat16,
                        enabled=device.type == 'cuda'):
        for name, block in zip(SITES[SITES.index(site) + 1:],
                               model.tabm.backbone.blocks[SITES.index(site) + 1:]):
            value = block(value)
            if return_states:
                states[name] = value.detach().clone()
        logits = model.tabm.output(value) + model.tabm.count_readout(value)[..., COUNT_MAP]
    probability = logits.float().softmax(-1).cpu().numpy().mean(1)
    return dict(probability=probability, states=states) if return_states else probability


@torch.no_grad()
def project(delta, scale, basis):
    """Return a per-member normalized-subspace component in original units.

    For ``delta[B,K,D]``, ``scale[K,D]`` and orthonormal ``basis[K,D,R]``, this
    computes ``scale * U U^T (delta / scale)`` in float32 on delta's device.
    Coordinates and members are neither pooled nor fitted here. Compute a
    native-state difference *after* converting both states to float32 to
    avoid quantizing the difference before this function receives it.
    """
    if delta.ndim != 3 or not delta.is_floating_point():
        raise ValueError('delta must have shape [batch,members,channels].')
    values = delta.float()
    scale = torch.as_tensor(scale, device=values.device, dtype=torch.float32)
    basis = torch.as_tensor(basis, device=values.device, dtype=torch.float32)
    if tuple(scale.shape) != tuple(values.shape[1:]):
        raise ValueError('scale must match the member and channel dimensions.')
    if basis.ndim != 3 or tuple(basis.shape[:2]) != tuple(scale.shape):
        raise ValueError('basis must have shape [members,channels,rank].')
    if not torch.isfinite(scale).all() or not (scale > 0).all():
        raise ValueError('Projection scales must be finite and positive.')
    normalized = values / scale
    coordinates = torch.einsum('bkd,kdr->bkr', normalized, basis)
    return torch.einsum('bkr,kdr->bkd', coordinates, basis) * scale
