"""Learn local responses on dense short-window log-Mel before Q0 pooling."""

from functools import partial
import math

import torch
import torchaudio
from torch import nn
from torch.nn import functional as F

from band_temporal import spectral_summary
from model import FixedViews, SCATTER_DIM, SPECTRAL_DIM


class ShortLocalViews(FixedViews):
    """Extract the configured waveform and time-frequency views."""

    def __init__(self, config, device='cuda'):
        super().__init__(config, device=device)
        self.input_view = config.get('short_local_view', 'both')
        del self.spectral[1]
        long = config.get('short_local_long', {})
        self.keep_long = long.get('arm') != 'L1' and self.input_view != '1d'
        self.learn_long = long.get('arm') == 'L3'
        if self.keep_long and not self.learn_long:
            self.grid = tuple(long.get('summary_grid', self.grid))
        else:
            del self.spectral
        if self.input_view == '1d':
            return
        if self.input_view == '2d':
            del self.fixed.scattering
        short = config['short_local']
        self.window_length = short['window_length']
        self.hop_length = short['hop_length']
        self.n_fft = short['n_fft']
        low, high = config['representation']['frequency_range_hz']
        self.register_buffer('mel_window', torch.hann_window(self.window_length, device=device),
                             persistent=False)
        banks = torchaudio.functional.melscale_fbanks(
            self.n_fft // 2 + 1, low, high, short['mel_bins'], config['sample_rate'],
            norm='slaney', mel_scale='slaney')
        self.register_buffer('mel_banks', banks.to(device), persistent=False)
        self.decibels = torchaudio.transforms.AmplitudeToDB(stype='power', top_db=80)

    @torch.no_grad()
    def forward(self, waves):
        audio = torch.as_tensor(waves, device=self.fixed.long_window.device, dtype=torch.float32)
        with torch.autocast(audio.device.type, enabled=False):
            if self.keep_long:
                representation = self.config['representation']
                spectrum = torch.stft(
                    audio, representation['long_fft'], representation['long_hop'],
                    representation['long_fft'], self.fixed.long_window,
                    center=True, return_complex=True)
                bins = self.shapes[0][0]
                long = torch.log(spectrum[:, :bins].abs().square() + 1e-6).unsqueeze(1).half()
            if self.input_view != '2d':
                coefficients = self.fixed.scattering(audio.contiguous())
                values = coefficients[:, self.fixed.positive_order].clamp_min(1e-12).log()
                sections = (values, *torch.tensor_split(values, 4, dim=-1))
                scatter = torch.cat([
                    statistic for section in sections
                    for statistic in (section.mean(-1), section.std(-1, unbiased=False))
                ], dim=-1)
                if self.input_view == '1d':
                    return (scatter.float(),)
            if self.keep_long:
                long_view = long if self.learn_long else self._spectral_summary(long, self.spectral[0])
            frames = audio.unfold(1, self.window_length, self.hop_length) * self.mel_window
            power = torch.fft.rfft(frames, n=self.n_fft).abs().square()
            mel = (power @ self.mel_banks).transpose(1, 2).unsqueeze(1)
            short = self.decibels(mel).half()
        spectral = (long_view, short) if self.keep_long else (short,)
        if self.input_view == '2d':
            return spectral
        return (scatter.float(), *spectral)


class LocalBilinearInteraction(nn.Module):
    """Pool outer products of the four corresponding acoustic time bins."""

    def __init__(self, base, time_bins, rank):
        super().__init__()
        if time_bins != 4:
            raise ValueError('The spectral grid must match the four waveform sections.')
        self.base = base
        self.time_bins = time_bins
        self.left = nn.Linear(SCATTER_DIM // (time_bins + 1), rank, bias=False)
        self.right = nn.Linear(SPECTRAL_DIM // time_bins, rank, bias=False)
        self.project = nn.Linear(rank * rank, base.weight.shape[0], bias=False)
        nn.init.zeros_(self.project.weight)

    def forward(self, x):
        waveform, spectral = (x * self.base.r).split((SCATTER_DIM, SPECTRAL_DIM), dim=-1)
        # Keep the full-clip waveform statistics in the standard TabM path.
        waveform = waveform.unflatten(-1, (self.time_bins + 1, self.left.in_features))[..., 1:, :]
        # Long and short summaries both store time as the innermost axis.
        spectral = spectral.unflatten(-1, (self.right.in_features, self.time_bins)).transpose(-1, -2)
        left = self.left(waveform)
        right = self.right(spectral)
        pooled = (left.transpose(-1, -2) @ right) / self.time_bins
        return self.base(x) + self.project(pooled.flatten(-2)) * self.base.s


class ShortLocalEncoder(nn.Module):
    """Learn local time–frequency responses before the existing 8 × 4 summary."""

    def __init__(self, config):
        super().__init__()
        self.grid = tuple(config['spectral_scattering']['summary_grid'])
        width = config['short_local']['width']
        output_dim = SPECTRAL_DIM // 2
        channels, remainder = divmod(output_dim, 2 * math.prod(self.grid))
        if remainder or channels == 0:
            raise ValueError('The summary grid must divide the Q0 short-view width.')
        normalizer = partial(nn.BatchNorm2d, width)
        if config.get('short_normalization') == 'record':
            normalizer = partial(nn.GroupNorm, num_groups=4, num_channels=width)
        self.stem = nn.Sequential(
            nn.Conv2d(1, width, 3, stride=(2, 1), padding=1, bias=False),
            normalizer(), nn.ReLU(),
        )
        self.local = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1, bias=False),
            normalizer(), nn.ReLU(),
            nn.Conv2d(width, width, 3, padding=1, bias=False),
            normalizer(),
        )
        self.project = nn.Conv2d(width, channels, 3, stride=2, padding=1, bias=False)
        self.normalize = nn.BatchNorm1d(output_dim, affine=False)

    def forward(self, mel):
        stem = self.stem(mel.float())
        local = F.relu(stem + self.local(stem))
        responses = self.project(local)
        return self.normalize(spectral_summary(responses.float(), self.grid))


class LongLocalEncoder(nn.Module):
    """Encode nearby frequency positions before the shared long-view summary."""

    def __init__(self, config):
        super().__init__()
        settings = config['short_local_long']
        self.frequency_block = settings['frequency_block']
        self.grid = tuple(config['spectral_scattering']['summary_grid'])
        width = settings['width']
        record_normalization = config.get('long_normalization') == 'record'
        normalizer = nn.BatchNorm2d
        normalization_options = {'num_features': width}
        if record_normalization:
            normalizer = nn.GroupNorm
            normalization_options = {'num_groups': 4, 'num_channels': width}
        channels, remainder = divmod(SPECTRAL_DIM // 2, 2 * math.prod(self.grid))
        if remainder or self.frequency_block < 1:
            raise ValueError('Invalid long-window summary size or frequency block.')
        self.stem = nn.Sequential(
            nn.Conv2d(self.frequency_block, width, (5, 1), padding=(2, 0), bias=False),
            normalizer(**normalization_options), nn.ReLU(),
        )
        self.local = nn.Sequential(
            nn.Conv2d(width, width, (5, 1), padding=(2, 0), bias=False),
            normalizer(**normalization_options), nn.ReLU(),
            nn.Conv2d(width, width, 3, padding=1, bias=False),
            normalizer(**normalization_options),
        )
        self.project = nn.Conv2d(width, channels, 1, bias=False)
        if record_normalization:
            self.normalize = nn.LayerNorm(SPECTRAL_DIM // 2, elementwise_affine=False)
        else:
            self.normalize = nn.BatchNorm1d(SPECTRAL_DIM // 2, affine=False)

    def rearrange_frequency(self, image):
        batch, _, frequencies, frames = image.shape
        padding = (-frequencies) % self.frequency_block
        image = F.pad(image, (0, 0, 0, padding), mode='replicate')
        groups = (frequencies + padding) // self.frequency_block
        image = image.reshape(batch, 1, groups, self.frequency_block, frames)
        return image.permute(0, 1, 3, 2, 4).reshape(batch, self.frequency_block, groups, frames)

    def forward(self, image):
        grouped = self.rearrange_frequency(image.float())
        stem = self.stem(grouped)
        responses = self.project(F.relu(stem + self.local(stem)))
        return self.normalize(spectral_summary(responses.float(), self.grid))
