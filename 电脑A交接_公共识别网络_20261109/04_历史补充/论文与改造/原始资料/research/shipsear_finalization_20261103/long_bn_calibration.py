"""Train-only calibration of the learned long-window BatchNorm statistics."""

import numpy as np
import torch

from data import CONDITIONS, training_views


@torch.no_grad()
def calibration_cache(fixed, train, config):
    if train.split != 'Train':
        raise ValueError('Normalization calibration requires Train.')
    parts = []
    counts = torch.zeros(len(CONDITIONS), dtype=torch.int64)
    representation = config['representation']
    bins = int(representation['frequency_range_hz'][1]
               * representation['long_fft'] / config['sample_rate']) + 1
    for start in range(0, len(train), config['batch_size']):
        indices = np.arange(start, min(start + config['batch_size'], len(train)))
        waves, choices = training_views(
            train, indices, 1001, config['noise_probabilities'], config['seed'])
        counts += torch.from_numpy(np.bincount(choices, minlength=len(CONDITIONS)))
        audio = torch.as_tensor(waves, device='cuda', dtype=torch.float32)
        spectrum = torch.stft(
            audio, representation['long_fft'], representation['long_hop'],
            representation['long_fft'], fixed.fixed.long_window,
            center=True, return_complex=True)
        image = torch.log(spectrum[:, :bins].abs().square() + 1e-6).unsqueeze(1)
        parts.append(image.half().cpu())
    return torch.cat(parts), dict(zip(CONDITIONS, counts.tolist()))


@torch.no_grad()
def recalibrate_long_bn(model, images, batch_size):
    """Estimate each layer under already calibrated upstream evaluation layers."""
    model.eval()
    layers = [(name, layer) for name, layer in model.long_local.named_modules()
              if isinstance(layer, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d))]
    if not layers:
        raise ValueError('The long-window encoder has no BatchNorm layers.')
    report = {}
    for name, layer in layers:
        count, mean, square_deviation = 0, None, None

        def collect(_module, inputs):
            nonlocal count, mean, square_deviation
            values = inputs[0].float()
            dimensions = (0, *range(2, values.ndim))
            batch_var, batch_mean = torch.var_mean(values, dim=dimensions, correction=0)
            batch_var, batch_mean = batch_var.double(), batch_mean.double()
            batch_count = values.numel() // values.shape[1]
            if count == 0:
                count, mean = batch_count, batch_mean
                square_deviation = batch_var * batch_count
            else:
                total = count + batch_count
                delta = batch_mean - mean
                mean = mean + delta * (batch_count / total)
                square_deviation = (square_deviation + batch_var * batch_count
                                    + delta.square() * (count * batch_count / total))
                count = total

        hook = layer.register_forward_pre_hook(collect)
        try:
            for batch in images.split(batch_size):
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    model.long_local(batch.cuda())
        finally:
            hook.remove()
        variance = (square_deviation / count).clamp_min(0)
        if not torch.isfinite(mean).all() or not torch.isfinite(variance).all():
            raise RuntimeError(f'Nonfinite calibration statistics for {name}.')
        old_mean, old_var = layer.running_mean.clone(), layer.running_var.clone()
        layer.running_mean.copy_(mean)
        layer.running_var.copy_(variance)
        report[name] = {
            'observations_per_channel': count,
            'mean_shift_rms_in_old_std': float(
                ((mean - old_mean) / (old_var + layer.eps).sqrt()).square().mean().sqrt()),
            'variance_ratio_median': float(
                ((variance + layer.eps) / (old_var + layer.eps)).median()),
        }
    return report
