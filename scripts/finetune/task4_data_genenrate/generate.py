"""Prepare FG26 Task 4 data, build OT/PCA conditions, and optionally fine-tune."""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy.optimize import linear_sum_assignment

from torchmdnet.checkpoint import load_model, sha256
from torchmdnet.data import ExactProcessedDataset
from torchmdnet.tokenizer import CanonicalNonIsomericFG26Tokenizer, SMILES_DICTIONARY
from torchmdnet.utils import read_config, write_json
from scripts.multiobjective import build_conditions, train as train_decoder


def local_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def aligned_properties(data_path, property_csv, splits, duplicate_policy="last"):
    """Align real DFT values by full InChIKey; never use row order or proxies."""
    props = pd.read_csv(property_csv)
    if not {'smiles', 'homo', 'lumo'}.issubset(props.columns):
        raise ValueError('Property CSV requires smiles,homo,lumo columns, energies in eV')
    mapping, conflicts = {}, []
    for row in props.itertuples(index=False):
        mol = Chem.MolFromSmiles(str(row.smiles))
        if mol is None or not np.isfinite([row.homo, row.lumo]).all():
            continue
        key = Chem.MolToInchiKey(mol)
        value = (float(row.homo), float(row.lumo))
        if key in mapping and not np.allclose(mapping[key], value, atol=1e-6, rtol=0):
            if duplicate_policy == 'error':
                raise ValueError(f'Conflicting DFT labels for InChIKey {key}')
            conflicts.append(dict(inchikey=key, previous=list(mapping[key]), retained=list(value)))
        mapping[key] = value
    data = ExactProcessedDataset(data_path)
    membership = np.full(len(data), '', dtype=object)
    with np.load(splits) as split:
        for part in ('train', 'val', 'test'):
            ids = np.asarray(split['idx_' + part])
            if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer):
                raise ValueError('Split indices must be one-dimensional integers')
            if len(set(ids.tolist())) != len(ids) or np.any(ids < 0) or np.any(ids >= len(data)):
                raise ValueError(f'Invalid {part} split indices')
            if np.any(membership[ids] != ''):
                raise ValueError('Train/val/test row indices overlap')
            membership[ids] = part
    raw = data._data
    smiles_list = raw.smiles
    keys = getattr(raw, 'inchikey', None)
    rows = []
    for index, smiles in enumerate(smiles_list):
        if not membership[index]:
            continue
        mol = Chem.MolFromSmiles(str(smiles)) if keys is None else None
        if keys is None and mol is None:
            continue
        key = str(keys[index]) if keys is not None else Chem.MolToInchiKey(mol)
        if key not in mapping:
            continue
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            continue
        identity = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
        homo, lumo = mapping[key]
        rows.append(dict(row_id=index, split=membership[index], smiles=identity,
                         true_homo=homo, true_lumo=lumo))
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError('No processed molecules align to the supplied DFT table')
    # Conservative identity-level split separation: held-out identities cannot
    # also be selected as training/validation queries or fit train thresholds.
    frame['priority'] = frame['split'].map({'test': 0, 'val': 1, 'train': 2})
    frame = frame.sort_values(['priority', 'row_id']).drop_duplicates('smiles').drop(columns='priority')
    return frame.sort_values('row_id').reset_index(drop=True), dict(
        property_rows=len(props), unique_property_keys=len(mapping),
        duplicate_property_policy=duplicate_policy, conflicting_property_rows=conflicts,
        matched_rows_before_deduplication=len(rows), unique_aligned_molecules=len(frame),
        identity_policy='non-isomeric canonical identity; test > val > train',
    )


def choose_protocol(frame, objective, quantile, anchor_count, seed, rank, model=None,
                    homo_threshold=None, lumo_threshold=None):
    train = frame[frame.split == 'train']
    if len(train) < 2:
        raise ValueError('At least two labelled, identity-disjoint train molecules are required')
    large = objective == 'large_gap'
    ht = float(homo_threshold if homo_threshold is not None else train.true_homo.quantile(1-quantile if large else quantile))
    lt = float(lumo_threshold if lumo_threshold is not None else train.true_lumo.quantile(quantile if large else 1-quantile))
    frame = frame.copy()
    hgood = frame.true_homo <= ht if large else frame.true_homo >= ht
    lgood = frame.true_lumo >= lt if large else frame.true_lumo <= lt
    frame['role'] = np.select([hgood & lgood, hgood & ~lgood, ~hgood & lgood], ['D', 'A', 'B'], default='neither')
    tokenizer = CanonicalNonIsomericFG26Tokenizer(SMILES_DICTIONARY)
    def representable(smiles):
        if model is not None:
            return model._normalize_gpt_smiles_target(smiles) is not None
        try:
            # Existing public FG26 GPT uses block size 79: at most 78 SMILES tokens.
            return len(tokenizer._tokenize(tokenizer._normalize(smiles))) <= 78
        except ValueError:
            return False
    frame = frame[frame.smiles.map(representable)].copy()
    dual = frame[frame.role == 'D'].copy()
    count = anchor_count or len(dual)
    if count < 2 or len(dual) < count or not 1 <= rank < count:
        raise ValueError(f'Found {len(dual)} representable dual molecules; requested {count}, PCA rank {rank}. Set --anchor-count/--pca-rank or explicit thresholds.')
    # Preserve train/non-train groups required by existing decoder selection.
    if not (dual.split == 'train').any() or not (dual.split != 'train').any():
        raise ValueError('Decoder selection needs both train and non-train disclosed anchors')
    rng = np.random.default_rng(seed)
    mandatory = [rng.choice(dual[dual.split == 'train'].index), rng.choice(dual[dual.split != 'train'].index)]
    others = rng.permutation(dual.index.difference(mandatory))[:count-2]
    anchors = dual.loc[sorted(mandatory + others.tolist())].copy().reset_index(drop=True)
    anchors['murcko_scaffold'] = anchors.smiles.map(lambda s: Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(Chem.MolFromSmiles(s))))
    fp = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    anchor_fp = [fp.GetFingerprint(Chem.MolFromSmiles(s)) for s in anchors.smiles]
    queries = {}
    for part in ('val', 'test'):
        chosen = {}
        for role in ('A', 'B'):
            pool = frame[(frame.split == part) & (frame.role == role)].reset_index(drop=True)
            if len(pool) < count:
                raise ValueError(f'{part}/{role} has {len(pool)} eligible parents, needs {count}; reduce anchor count or change thresholds')
            fps = [fp.GetFingerprint(Chem.MolFromSmiles(s)) for s in pool.smiles]
            similarity = np.array([DataStructs.BulkTanimotoSimilarity(f, fps) for f in anchor_fp])
            left, right = linear_sum_assignment(-similarity)
            assert np.array_equal(left, np.arange(count))
            chosen[role] = pool.iloc[right].reset_index(drop=True)
        records = []
        for i, d in anchors.iterrows():
            a, b = chosen['A'].iloc[i], chosen['B'].iloc[i]
            row = dict(query_id=i, a_row_id=int(a.row_id), b_row_id=int(b.row_id), d_row_id=int(d.row_id),
                       source_a_row_id=int(a.row_id), source_b_row_id=int(b.row_id), source_d_row_id=int(d.row_id),
                       a_smiles=a.smiles, b_smiles=b.smiles, d_smiles=d.smiles, a_split=part, b_split=part,
                       a_role='HOMO_only', b_role='LUMO_only', d_split=d.split)
            for label, parent in (('a', a), ('b', b), ('d', d)):
                row.update({f'{label}_true_homo':float(parent.true_homo), f'{label}_true_lumo':float(parent.true_lumo)})
            records.append(row)
        queries[part] = pd.DataFrame(records)
    return anchors, queries, dict(homo_threshold=ht, lumo_threshold=lt,
        homo_direction='minimize' if large else 'maximize', lumo_direction='maximize' if large else 'minimize',
        threshold_source='explicit_eV' if homo_threshold is not None else 'identity_disjoint_official_train_quantiles',
        quantile=quantile, dual_pool_count=len(dual), candidate_counts=frame.groupby(['split','role']).size().to_dict())


def expand_pairs(queries, draws, targets=None):
    rows = queries.loc[queries.index.repeat(draws)].reset_index(drop=True).copy()
    rows['flow_query_id'] = np.repeat(np.arange(len(queries)), draws)
    rows['flow_draw_id'] = np.tile(np.arange(draws), len(queries))
    rows['pair_id'] = np.arange(len(rows))
    if targets is not None:
        if targets.shape != (len(queries), draws):
            raise ValueError('Target shape does not match pair ordering')
        rows['transport_target_row_id'] = targets.reshape(-1)
    return rows


def write_summary(output, config):
    """Optional human-readable record; never an input to new builds."""
    large = config['objective'] == 'large_gap'
    thresholds = {
        'homo_max' if large else 'homo_min': config['homo_threshold'],
        'lumo_min' if large else 'lumo_max': config['lumo_threshold'],
    }
    anchors = pd.read_csv(config['known_reference'])
    pairs = pd.read_csv(output/'test_pairs.csv')
    write_json(output/'protocol_summary.json', dict(
        objective=config['objective'], threshold_from=config['threshold_from'], **thresholds,
        anchors=len(anchors),
        anchor_splits={part:int((anchors['split'] == part).sum()) for part in ('train','val','test')},
        test_queries=len(pairs.drop_duplicates(['a_row_id','b_row_id','d_row_id'])),
        generation_budget=len(pairs),
    ))


def prepare(args, objective, frame, mapping, hashes, model):
    output = args.output_dir / objective
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Use a new output directory; existing artifacts will not be overwritten: {output}')
    quantile = args.quantile if args.quantile is not None else (0.87625 if objective == 'large_gap' else 0.87825)
    anchors, queries, summary = choose_protocol(frame, objective, quantile, args.anchor_count, args.seed,
        args.pca_rank, model, args.homo_threshold, args.lumo_threshold)
    config = read_config(ROOT / 'examples/multiobjective.yaml', dict(dataset='fg26', objective=objective))
    config.update(task='multiobjective', public_task='multiobjective', phase='decoder', data_path=str(args.data_path),
        splits=str(args.splits), checkpoint=str(args.checkpoint) if args.checkpoint else None,
        init_checkpoint=str(args.init_decoder) if args.init_decoder else None,
        known_reference=str(output/'known_dual_anchors.csv'), input=str(output/'test_pairs.csv'),
        training_conditions=str(output/'val_conditions.npz'), conditions=str(output/'test_conditions.npz'),
        decoder_checkpoint=str(output/'training/decoder_final.pt'), log_dir=str(output/'training'),
        output=str(output/'generation'), seed=args.seed, pca_rank=args.pca_rank, projection_magnitude=args.beta,
        homo_threshold=summary['homo_threshold'], lumo_threshold=summary['lumo_threshold'],
        random_smiles_probability=0.0, val_draws=args.val_draws, test_draws=args.test_draws,
        input_sha256=hashes, threshold_from='explicit' if args.homo_threshold is not None else 'train_dft')
    for key in ('num_epochs','batch_size','decoder_lr','patience','weight_decay','unfreeze_last_decoder_blocks'):
        value = getattr(args, key, None)
        if value is not None:
            config[key] = value
    config['device'] = args.device
    if getattr(args, 'log_dir', None) is not None:
        log_dir = args.log_dir / objective if args.objective == 'both' else args.log_dir
        config.update(log_dir=str(log_dir), decoder_checkpoint=str(log_dir/'decoder_final.pt'))
    output.mkdir(parents=True, exist_ok=True)
    anchors[['row_id','split','smiles','murcko_scaffold','true_homo','true_lumo']].to_csv(output/'known_dual_anchors.csv',index=False)
    for part in ('val','test'):
        expand_pairs(queries[part], getattr(args,part+'_draws')).to_csv(output/f'{part}_pairs.csv',index=False)
    write_summary(output, config)
    write_json(output/'config.yaml',config)
    print(f'Prepared {output}: {len(anchors)} anchors, {len(anchors)*args.val_draws}/{len(anchors)*args.test_draws} val/test rows',flush=True)



def build(args, objective, model):
    output=args.output_dir/objective
    config=json.loads((output/'config.yaml').read_text())
    # Migrate already-prepared outputs from the previous layout once. New
    # configs contain all build inputs and work even if the summary is deleted.
    if 'input_sha256' not in config:
        previous=json.loads((output/'protocol_summary.json').read_text())
        config.update({key:previous[key] for key in ('input_sha256','val_draws','test_draws')})
    if 'threshold_from' not in config:
        old_path = output/'protocol_summary.json'
        old_summary = json.loads(old_path.read_text()) if old_path.exists() else {}
        source = old_summary.get('threshold_source')
        config['threshold_from'] = old_summary.get('threshold_from') or {
            'explicit_eV': 'explicit',
            'identity_disjoint_official_train_quantiles': 'train_dft',
            'official_train_real_dft_only': 'train_dft',
        }.get(source, 'unknown')
    for key in ('data_path','splits'):
        if sha256(config[key]) != config['input_sha256'][key]:
            raise ValueError(f'{key} changed since preparation')
    checkpoint=str(args.checkpoint)
    config['checkpoint']=checkpoint
    for part in ('val','test'):
        if (output/f'{part}_conditions.npz').exists():
            raise FileExistsError(f'{part} conditions already exist; choose a new output directory')
        table=pd.read_csv(output/f'{part}_pairs.csv')
        queries=table.drop_duplicates(['a_row_id','b_row_id','d_row_id']).reset_index(drop=True)
        for column in ('a_smiles','b_smiles','d_smiles'):
            if any(model._normalize_gpt_smiles_target(s) is None for s in queries[column]):
                raise ValueError('Prepared parent/anchor SMILES are incompatible with checkpoint vocabulary/length; prepare with this checkpoint')
        draws=int(config[part+'_draws'])
        with tempfile.TemporaryDirectory(dir=output) as temporary:
            build_config=dict(config, dataset='FG26', input=str(output/f'{part}_pairs.csv'), output=temporary,
                              draws=draws, weight_seed=config['seed']+(1000 if part=='val' else 2000))
            print(f'Encoding {objective}/{part}: {len(queries)} x {draws} conditions',flush=True)
            path=build_conditions(build_config,model=model,device=args.device)
            with np.load(path) as cache:
                np.testing.assert_array_equal(cache['source_row_ids'],queries[['a_row_id','b_row_id','d_row_id']].to_numpy())
                targets=cache['target_row_ids'].copy()
            metadata=json.loads((Path(temporary)/'conditions.json').read_text())
            metadata['config']['output']=str(output)
            Path(path).rename(output/f'{part}_conditions.npz')
            write_json(output/f'{part}_conditions.json',metadata)
        expand_pairs(queries,draws,targets).to_csv(output/f'{part}_pairs.csv',index=False)
    write_summary(output, config)
    write_json(output/'config.yaml',config)



def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=['prepare','build','all','finetune'],default='all')
    p.add_argument('--dataset',type=str.lower,choices=['fg26'],default='fg26',help='Automatic data preparation currently supports FG26')
    p.add_argument('--objective',choices=['large_gap','small_gap','both'],default='both')
    p.add_argument('--data-path',type=local_path,default=ROOT/'datasets/FG26/processed/fg26_uv_ir_raman_smiles_aligned.pt')
    p.add_argument('--property-csv',type=local_path,default=ROOT/'datasets/FG26/fg26_property.csv')
    p.add_argument('--splits',type=local_path,default=ROOT/'datasets/FG26/splits.npz')
    p.add_argument('--checkpoint',type=local_path,help='Trained FG26 GPT/warmup checkpoint, required for build/all; not a pretraining-only checkpoint')
    p.add_argument('--init-decoder',type=local_path,help='Optional anchor decoder for subsequent training')
    p.add_argument('--output-dir',type=local_path,default=ROOT/'datasets/FG26/task4_generated')
    p.add_argument('--anchor-count',type=int,default=40,help='0 uses all eligible dual anchors; number of queries equals anchor count')
    p.add_argument('--val-draws',type=int,default=16)
    p.add_argument('--test-draws',type=int,default=64)
    p.add_argument('--pca-rank',type=int,default=12)
    p.add_argument('--beta',type=float,default=0.15)
    p.add_argument('--quantile',type=float)
    p.add_argument('--homo-threshold',type=float,help='Explicit threshold in eV, supply both HOMO and LUMO')
    p.add_argument('--lumo-threshold',type=float)
    p.add_argument('--duplicate-property-policy', choices=['last','error'], default='last',
                   help='last matches the original property mapper and reports conflict count; error rejects them')
    p.add_argument('--seed',type=int,default=1)
    p.add_argument('--device',default=None,help='Defaults to cuda:0 when available, otherwise cpu')
    p.add_argument('--log-dir',type=local_path,help='Training output; both objectives get separate subdirectories')
    for name in ('num-epochs','batch-size','patience','unfreeze-last-decoder-blocks'):
        p.add_argument('--'+name,type=int)
    p.add_argument('--lr','--decoder-lr',dest='decoder_lr',type=float)
    p.add_argument('--weight-decay',type=float)
    args=p.parse_args()
    if args.anchor_count<0 or min(args.val_draws,args.test_draws,args.pca_rank)<1 or not np.isfinite(args.beta) or args.beta<0:
        p.error('Counts/rank must be positive (anchor-count may be 0), beta finite and nonnegative')
    if args.quantile is not None and not 0<args.quantile<1:p.error('quantile must lie in (0,1)')
    if (args.homo_threshold is None)!=(args.lumo_threshold is None):p.error('Supply both HOMO and LUMO thresholds')
    if args.homo_threshold is not None and (args.objective=='both' or not np.isfinite([args.homo_threshold,args.lumo_threshold]).all()):
        p.error('Explicit finite thresholds require one objective')
    if args.phase!='prepare' and args.checkpoint is None:
        p.error('--checkpoint is required: use a trained GPT, e.g. experiments/warmup_gpt/last.ckpt')
    for name in ('num_epochs','batch_size','patience','unfreeze_last_decoder_blocks'):
        value=getattr(args,name)
        if value is not None and value < 1:p.error(f'--{name.replace("_","-")} must be positive')
    if args.decoder_lr is not None and (not np.isfinite(args.decoder_lr) or args.decoder_lr<=0):p.error('--lr must be finite and positive')
    if args.weight_decay is not None and (not np.isfinite(args.weight_decay) or args.weight_decay<0):p.error('--weight-decay must be finite and nonnegative')
    args.device=args.device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
    objectives=['large_gap','small_gap'] if args.objective=='both' else [args.objective]
    if args.phase in ('prepare','all','finetune'):
        for objective in objectives:
            output=args.output_dir/objective
            if output.exists() and any(output.iterdir()):
                p.error(f'Output already exists: {output}; use a new --output-dir')
    if args.phase == 'finetune' and args.log_dir is not None:
        for objective in objectives:
            output=args.log_dir/objective if args.objective=='both' else args.log_dir
            if output.exists() and any(output.iterdir()):
                p.error(f'Training output already exists: {output}; use a new --log-dir')
    torch.manual_seed(args.seed)
    model=load_model(args.checkpoint,args={'task':'A+B'},device=args.device).eval() if args.checkpoint else None
    if args.phase in ('prepare','all','finetune'):
        frame,mapping=aligned_properties(args.data_path,args.property_csv,args.splits,args.duplicate_property_policy)
        if mapping['conflicting_property_rows']:
            print(f"DFT label conflicts: {len(mapping['conflicting_property_rows'])}; retained last CSV values (use --duplicate-property-policy error to reject).", flush=True)
        hashes={name:sha256(getattr(args,name)) for name in ('data_path','property_csv','splits')}
        for objective in objectives:prepare(args,objective,frame,mapping,hashes,model)
    if args.phase in ('build','all','finetune'):
        for objective in objectives:build(args,objective,model)
    if args.phase == 'finetune':
        # Decoder fitting reloads this checkpoint; release the encoding copy first.
        del model
        if torch.cuda.is_available():torch.cuda.empty_cache()
        for objective in objectives:
            config_path=args.output_dir/objective/'config.yaml'
            config=json.loads(config_path.read_text())
            Path(config['log_dir']).mkdir(parents=True,exist_ok=True)
            write_json(Path(config['log_dir'])/'config.json',config)
            print(f'Fine-tuning {objective}: {config_path}',flush=True)
            train_decoder(config)
            print(f'Saved decoder: {config["decoder_checkpoint"]}\nGeneration/evaluation config: {config_path}',flush=True)


if __name__=='__main__':
    main()
