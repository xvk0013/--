"""Q0 AWGN and noise schedules on stored waveforms."""

import numpy as np

CONDITIONS = ('original', '10dB', '5dB', '0dB', '-5dB', '-10dB')
SNRS = (10, 5, 0, -5, -10)


def add_noise(wave, count, snr, random):
    x = np.asarray(wave, dtype=np.float64)
    noise = random.standard_normal(len(x))
    noise -= noise.mean()
    noise /= np.sqrt(np.mean(noise * noise))
    reference = float(np.mean(x * x)) if count > 0 else .05 ** 2
    mixed = x + noise * np.sqrt(reference / 10 ** (snr / 10))
    mixed *= .05 / np.sqrt(np.mean(mixed * mixed))
    return mixed.astype(np.float32)


def training_views(data, indices, epoch, probabilities, seed):
    indices = np.asarray(indices)
    choices = np.array([np.random.default_rng(np.random.SeedSequence(
        [seed, 100, epoch, int(data.indices[i])])).choice(6, p=probabilities) for i in indices])
    waves = data.batch(indices)
    for j, (i, condition) in enumerate(zip(indices, choices)):
        if condition:
            rng = np.random.default_rng(np.random.SeedSequence([seed, 101, epoch, int(data.indices[i])]))
            waves[j] = add_noise(waves[j], data.counts[i], SNRS[condition - 1], rng)
    return waves, choices


def evaluation_views(data, indices, condition, seed):
    indices = np.asarray(indices)
    waves = data.batch(indices)
    if condition == 'original':
        return waves
    snr = SNRS[CONDITIONS.index(condition) - 1]
    code = 1 if data.split == 'Val' else 2
    for j, i in enumerate(indices):
        rng = np.random.default_rng(np.random.SeedSequence([seed, code, int(data.indices[i])]))
        waves[j] = add_noise(waves[j], data.counts[i], snr, rng)
    return waves


