"""Run waveform inference with the frozen ShipsEar ShortLocal-GN model."""
import argparse
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
FORMAL = HERE / 'snapshot' / 'shipsear_finalization_20261103'
RUN_ID = 'SE-Q0-ShortLocal-GN-13'
sys.path.insert(0, str(FORMAL))

from train_formal import ShortLocalSetModel
from short_local import ShortLocalViews


@torch.inference_mode()
def predict(waves, checkpoint, device):
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if saved.get('experiment_id') != RUN_ID:
        raise ValueError(f'Expected the {RUN_ID} checkpoint.')

    config = saved['config']
    waves = np.atleast_2d(np.asarray(waves, dtype=np.float32))
    expected_samples = config['sample_rate'] * config['duration_seconds']
    if waves.ndim != 2 or waves.shape[1] != expected_samples:
        raise ValueError(f'Expected waveforms with shape (N, {expected_samples}).')

    device = torch.device(device)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    fixed = ShortLocalViews(config, device).eval()
    model = ShortLocalSetModel(config, saved['normalization']).to(device).eval()
    model.load_state_dict(saved['model'], strict=True)

    feature_batch = config['batch_size']
    evaluation_batch = config['evaluation_batch_size']
    set_names = np.asarray(config['set_names'], dtype=str)
    probabilities = np.empty((len(waves), len(set_names)), dtype=np.float32)
    for start in range(0, len(waves), evaluation_batch):
        batch = waves[start:start + evaluation_batch]
        parts = [fixed(batch[index:index + feature_batch])
                 for index in range(0, len(batch), feature_batch)]
        features = [torch.cat(view_parts, dim=0) for view_parts in zip(*parts)]
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            logits = model(*features)
        probabilities[start:start + len(batch)] = logits.float().softmax(-1).cpu().numpy().mean(1)

    predictions = probabilities.argmax(-1)
    return dict(probabilities=probabilities, predictions=predictions,
                counts=np.asarray(config['count_map'], dtype=np.int64)[predictions],
                sets=set_names[predictions])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=FORMAL / 'runs' / RUN_ID / 'best.pt')
    parser.add_argument('--input', required=True, type=Path, help='5 s, 16 kHz waveforms in a .npy array.')
    parser.add_argument('--output', required=True, type=Path, help='Destination .npz.')
    parser.add_argument('--device', default='cuda', help='PyTorch device, for example cuda or cpu.')
    args = parser.parse_args()

    waves = np.load(args.input, allow_pickle=False)
    result = predict(waves, args.checkpoint, args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **result)
    print(f'Saved {len(result["predictions"])} predictions to {args.output}')


if __name__ == '__main__':
    main()
