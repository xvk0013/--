"""Standalone waveform inference for the DeepShip Q0 model."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from model import Recognizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path, help='DeepShip best.pt checkpoint.')
    parser.add_argument('--input', required=True, type=Path,
                        help='Waveforms stored in a .npy array, shape (N, 80000), 16 kHz.')
    parser.add_argument('--output', required=True, type=Path,
                        help='Destination .npz with probabilities, predictions, counts and sets.')
    parser.add_argument('--device', default='cuda',
                        help='PyTorch device, for example cuda, cuda:0 or cpu (default: cuda).')
    args = parser.parse_args()

    waves = np.load(args.input, allow_pickle=False)
    recognizer = Recognizer(args.checkpoint, args.device)
    result = recognizer.predict(waves)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output,
                        probabilities=result['probabilities'],
                        predictions=result['predictions'],
                        counts=result['counts'],
                        sets=np.asarray(result['sets'], dtype=str))
    print(f'Saved {len(result["predictions"])} predictions to {args.output}')


if __name__ == '__main__':
    main()
