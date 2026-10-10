"""Fit QM9S per-element reference coefficients without using test labels.

Fit y ~= atom_counts @ coefficients, with no intercept (not molecular means).
Use train+val, explicitly excluding test, as requested for the public table.
"""
from pathlib import Path
import argparse
import json

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
ELEMENTS = (1, 6, 7, 8, 9)
TARGETS = (1, 6, 7, 8, 9, 10, 11)


def fit_atomrefs(counts, labels, splits):
    counts, labels = np.asarray(counts, dtype=np.float64), np.asarray(labels, dtype=np.float64)
    if counts.shape != (len(labels), len(ELEMENTS)):
        raise ValueError("Expected aligned molecular labels and H/C/N/O/F counts")
    parts = []
    for name in ("train", "val", "test"):
        values = np.asarray(splits['idx_' + name])
        if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
            raise ValueError("Split indices must be one-dimensional integer arrays")
        if (values < 0).any() or (values >= len(labels)).any():
            raise ValueError("Split index out of range")
        parts.append(values)
    combined = np.concatenate(parts)
    if len(np.unique(combined)) != len(combined):
        raise ValueError("Duplicate or overlapping split indices")
    selected = np.sort(np.concatenate(parts[:2]))
    x = counts[selected]
    y = labels[selected][:, TARGETS]
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Selected fitting rows must have finite counts and labels")
    coefficients, _, rank, _ = np.linalg.lstsq(x, y, rcond=None)
    if rank != len(ELEMENTS):
        raise ValueError("Atom-count matrix is rank-deficient")
    return dict(elements=list(ELEMENTS), fit_split="train+val", intercept=False,
                counts=dict(zip(("train", "val", "test"), map(len, parts))),
                fit_rows=len(selected),
                atomref={str(k): coefficients[:, i].tolist() for i, k in enumerate(TARGETS)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT/'datasets/qm9sp/processed/qm9s_with_uv_ir_raman_smiles.pt')
    parser.add_argument('--splits', type=Path, default=ROOT/'datasets/qm9sp/splits.npz')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists; choose a new file')
    data, slices = torch.load(args.data, map_location='cpu', weights_only=False)[:2]
    z, offsets = data.z.numpy().reshape(-1), slices['z'].numpy()
    counts = np.stack([np.add.reduceat((z == el).astype(np.float64), offsets[:-1]) for el in ELEMENTS], axis=1)
    if not np.array_equal(counts.sum(1), np.diff(offsets)):
        raise ValueError('Unexpected elements outside H/C/N/O/F')
    with np.load(args.splits) as splits:
        report = fit_atomrefs(counts, data.y.numpy().reshape(len(counts), -1), splits)
    report.update(data=str(args.data.resolve()), splits=str(args.splits.resolve()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
