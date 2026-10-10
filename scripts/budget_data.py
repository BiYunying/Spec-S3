"""Build Task5 supervision from raw FG26 spectra and fixed molecular splits."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scripts.budget import check_partitions, key_and_size, negative_ids
from torchmdnet.data import ExactProcessedDataset
from torchmdnet.tokenizer import CanonicalNonIsomericFG26Tokenizer, SMILES_DICTIONARY


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', choices=['data', 'features'], default='data')
    p.add_argument('--data', default='datasets/FG26/processed/fg26_uv_ir_raman_smiles_aligned.pt')
    p.add_argument('--splits', default='datasets/FG26/splits.npz')
    p.add_argument('--output', default='datasets/FG26/task5_generated')
    p.add_argument('--checkpoint', help='Matching cross-attention warmup; required for --phase features')
    p.add_argument('--device', default='cpu')
    for name, value in [('train-size', 4000), ('val-size', 200), ('test-size', 100),
                        ('bank-size', 40000), ('replay-size', 512), ('small-min', 1),
                        ('small-max', 13), ('large-min', 18), ('large-max', 0),
                        ('topk', 5), ('negatives', 4), ('hard-pool', 256),
                        ('batch-size', 32), ('seed', 2026)]:
        p.add_argument('--'+name, type=int, default=value)
    p.add_argument('--budgets', type=int, nargs='+', default=[9, 11, 13])
    p.add_argument('--temperature', type=float, default=.025)
    p.add_argument('--gap-min', type=float, default=.05)
    p.add_argument('--gap-max', type=float, default=.25)
    return p


def resolve(path):
    path = Path(path).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def portable(path):
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def validate_args(args):
    for name in ('train_size', 'val_size', 'test_size', 'bank_size', 'replay_size',
                 'topk', 'negatives', 'hard_pool', 'batch_size'):
        if getattr(args, name) < 1:
            raise ValueError(f'{name} must be positive')
    if not 1 <= args.small_min <= args.small_max < args.large_min:
        raise ValueError('Require 1 <= small-min <= small-max < large-min')
    if args.small_max > 30:
        raise ValueError('The retained budget embedding supports heavy-atom counts up to 30')
    if args.large_max and args.large_max < args.large_min:
        raise ValueError('large-max must be 0 (unlimited) or >= large-min')
    if not args.budgets or len(set(args.budgets)) != len(args.budgets):
        raise ValueError('budgets must be a nonempty list of distinct integers')
    if any(b < args.small_min or b > args.small_max for b in args.budgets):
        raise ValueError('Each budget must lie within small-min/small-max')
    if args.replay_size > args.bank_size or args.topk > args.bank_size:
        raise ValueError('replay-size and topk cannot exceed bank-size')
    if not np.isfinite(args.temperature) or args.temperature <= 0:
        raise ValueError('temperature must be finite and positive')
    if not 0 < args.gap_min <= args.gap_max <= 2:
        raise ValueError('Require 0 < gap-min <= gap-max <= 2')


def checked_splits(path, size):
    with np.load(path) as archive:
        splits = {tag: archive['idx_'+tag].copy() for tag in ('train', 'val', 'test')}
    seen = set()
    for tag, ids in splits.items():
        if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer):
            raise ValueError(f'{tag} split must be a one-dimensional integer array')
        current = set(ids.tolist())
        if len(current) != len(ids) or seen & current or (ids < 0).any() or (ids >= size).any():
            raise ValueError('Split IDs are duplicated, overlapping or outside the dataset')
        seen.update(current)
    return splits


def select_panels(data, splits, args):
    """Deduplicate by non-isomeric graph, with train > val > test precedence.

    Check against every preceding split molecule, including unselected ones,
    so test references cannot share a graph with the warmup training split.
    """
    tokenizer = CanonicalNonIsomericFG26Tokenizer(SMILES_DICTIONARY)
    seen, pools = set(), {}
    for tag, ids in splits.items():
        small, large = [], []
        for i, row_id in enumerate(sorted(ids.tolist())):
            sample = data[row_id]
            smiles = str(sample.smiles)
            key, ha, eligible = key_and_size(smiles)
            if not key or key in seen:
                continue
            seen.add(key)
            is_small = tag == 'train' and args.small_min <= ha <= args.small_max
            is_large = ha >= args.large_min and (not args.large_max or ha <= args.large_max)
            if not eligible or not (is_small or is_large):
                continue
            if any(not torch.isfinite(getattr(sample, m)).all() or
                   getattr(sample, m).float().norm() == 0 for m in ('uv', 'ir', 'raman')):
                continue
            if is_small:
                try:
                    tokenizer([smiles], max_length=80)  # BOS + up to79 predicted tokens incl EOS
                except (ValueError, RuntimeError):
                    continue
            row = dict(source_row_id=row_id, smiles=smiles, key=key, ha=ha)
            (small if is_small else large).append(row)
            if (i + 1) % 25000 == 0:
                print(f'Inspecting {tag}: {i+1}/{len(ids)}', flush=True)
        pools[tag] = (small, large)
        print(f'{tag}: eligible small={len(small)}, large={len(large)}', flush=True)
    rng = np.random.default_rng(args.seed)

    def choose(rows, count, name):
        if len(rows) < count:
            raise ValueError(f'{name}: requested {count}, only {len(rows)} eligible after graph deduplication')
        return pd.DataFrame([rows[i] for i in rng.choice(len(rows), count, replace=False)])

    bank = choose(pools['train'][0], args.bank_size, 'bank')
    sources = {tag: choose(pools[tag][1], getattr(args, tag+'_size'), tag) for tag in ('train', 'val')}
    test = choose(pools['test'][1], args.test_size, 'test')
    test['query_id'] = test.source_row_id  # Stable IDs, not a historical query-number mapping.
    return dict(bank=bank, sources=sources, test=test,
                replay_ids=rng.choice(len(bank), args.replay_size, replace=False).tolist(),
                target_tokens=tokenizer(bank.smiles.tolist(), max_length=80)['input_ids'],
                budgets=args.budgets,
                size_limits={k: getattr(args, k) for k in ('small_min', 'small_max', 'large_min', 'large_max')})


def raw_descriptors(data, ids, device):
    parts = []
    samples = [data[int(i)] for i in ids]
    for name in ('uv', 'ir', 'raman'):
        x = torch.stack([getattr(s, name).reshape(-1) for s in samples]).float().to(device)
        if not torch.isfinite(x).all() or (x.norm(dim=1) == 0).any():
            raise ValueError('Invalid raw spectrum')
        parts.append(F.normalize(x, dim=1))
    return torch.cat(parts, dim=1) / np.sqrt(3.)


@torch.no_grad()
def build_pairs(data, payload, args):
    ids = payload['bank'].source_row_id.to_numpy()
    first = raw_descriptors(data, ids[:args.batch_size], args.device)
    bank = torch.empty(len(ids), first.shape[1], device=args.device)
    bank[:len(first)] = first
    for start in range(len(first), len(ids), args.batch_size):
        bank[start:start+args.batch_size] = raw_descriptors(data, ids[start:start+args.batch_size], args.device)
    sizes = torch.tensor(payload['bank'].ha.to_numpy(), device=args.device)
    eligible = {b: torch.where(sizes <= b)[0] for b in args.budgets}
    for b, pool in eligible.items():
        if len(pool) < args.topk:
            raise ValueError(f'Budget {b}: bank has only {len(pool)} eligible targets for topk={args.topk}')
    examples, records = {}, []
    rng = np.random.default_rng(args.seed)
    for tag in ('train', 'val'):
        examples[tag] = []
        ids = payload['sources'][tag].source_row_id.to_numpy()
        for start in range(0, len(ids), args.batch_size):
            descriptor = raw_descriptors(data, ids[start:start+args.batch_size], args.device)
            scores = (descriptor @ bank.T).clamp(-1, 1)
            for j, similarities in enumerate(scores):
                for b, pool in eligible.items():
                    best = pool[torch.argsort(similarities[pool], descending=True, stable=True)[:args.topk]]
                    weights = ((similarities[best] - similarities[best].max()) / args.temperature).softmax(0)
                    examples[tag].append(dict(source_index=start+j, budget=b,
                                              target_bank_ids=best.tolist(), weights=weights.tolist()))
                    if tag == 'train':
                        positive = int(best[0])
                        for negative in negative_ids(similarities, sizes, positive, b, rng,
                                k=args.negatives, gap_min=args.gap_min, gap_max=args.gap_max, hard_pool=args.hard_pool):
                            records.append(dict(source_index=start+j, budget=b, positive=positive, negative=negative,
                                                gap=float(similarities[positive]-similarities[negative])))
            print(f'Paired {tag}: {min(start+args.batch_size, len(ids))}/{len(ids)}', flush=True)
    if not records:
        raise ValueError('No DPO pairs satisfy the same-size/gap constraints; enlarge the bank or adjust gap bounds')
    payload['examples'] = examples
    return {key: torch.tensor([r[key] for r in records], dtype=torch.float32 if key == 'gap' else torch.long)
            for key in records[0]}


def build_features(output, checkpoint, device):
    from torchmdnet.checkpoint import load_model
    from torchmdnet.utils import read_config
    from scripts.budget import prepare_features

    if not checkpoint:
        raise ValueError('--phase features requires --checkpoint (matching cross-attention warmup)')
    config = read_config(output/'config.yaml')
    model = load_model(resolve(checkpoint), args={'task': 'smiles'}, device=device).requires_grad_(False).eval()
    payload = torch.load(output/'prepared.pt', map_location='cpu', weights_only=False)
    check_partitions(payload)
    prepare_features(model, payload, dict(model.public_config, data_path=config['data_path'],
        features=str(output/'features.pt'), log_dir=str(output)), device, patches=False)
    raw = json.loads((output/'config.yaml').read_text())
    raw['checkpoint'] = portable(resolve(checkpoint))
    (output/'config.yaml').write_text(json.dumps(raw, indent=2)+'\n')


def main(argv=None):
    args = parser().parse_args(argv)
    torch.set_num_threads(4)
    output = resolve(args.output)
    if args.phase == 'features':
        build_features(output, args.checkpoint, args.device)
        return
    validate_args(args)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f'Output directory is not empty: {output}. Use --output with a new name.')
    data_path, split_path = resolve(args.data), resolve(args.splits)
    data = ExactProcessedDataset(data_path)
    splits = checked_splits(split_path, len(data))
    payload = select_panels(data, splits, args)
    check_partitions(payload)
    pairs = build_pairs(data, payload, args)
    config = json.loads((ROOT/'examples/budget.yaml').read_text())
    config.update(data_path=portable(data_path), splits=portable(split_path), budgets=args.budgets,
                  seed=args.seed, prepared=portable(output/'prepared.pt'), pairs=portable(output/'preference_pairs.pt'),
                  features=portable(output/'features.pt'), input=portable(output/'queries.csv'))
    config.pop('features_encoder_sha256', None)
    if args.checkpoint:
        config['checkpoint'] = portable(resolve(args.checkpoint))
    config['data_settings'] = {k: v for k, v in vars(args).items()
                               if k not in ('phase', 'checkpoint', 'device', 'data', 'splits', 'output')}
    output.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output/'prepared.pt')
    torch.save(pairs, output/'preference_pairs.pt')
    payload['test'].to_csv(output/'queries.csv', index=False)
    (output/'config.yaml').write_text(json.dumps(config, indent=2)+'\n')
    print(f'Wrote {len(pairs["gap"])} preference pairs and {len(payload["test"])} test queries to {output}', flush=True)
    print('Features are built automatically by training, or use --phase features --checkpoint PATH.', flush=True)


if __name__ == '__main__':
    main()
