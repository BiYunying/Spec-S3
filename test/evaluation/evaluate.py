"""Evaluate saved candidates with task-specific reported metrics and denominators."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--task",
        required=True,
        choices=(
            "property_prediction",
            "translation",
            "scaffold_editing",
            "multiobjective",
            "budget",
        ),
    )
    p.add_argument("--conf", required=True)
    for name in (
        "input",
        "output",
        "dataset",
        "objective",
        "data-path",
        "splits",
        "checkpoint",
        "device",
        "queries",
        "proxy-dir",
        "proxy-path",
        "known-reference",
        "dft-results",
        "selection",
    ):
        p.add_argument("--" + name)
    p.add_argument("--property-source", choices=("proxy", "dft"),
                   help="Task 4: direct proxy evaluation, or Top-K selection / DFT CSV evaluation")
    p.add_argument("--topk", type=int, help="Task 4 DFT selection size (default: 50)")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    from torchmdnet.utils import read_config, write_json

    config = read_config(args.conf, vars(args))
    if args.dry_run:
        import json

        print(json.dumps(config, indent=2))
        return
    import numpy as np
    import pandas as pd
    if args.task == "multiobjective":
        from test.evaluation.multiobjective import evaluate_multiobjective

        summary = evaluate_multiobjective(config)
        print(summary)
        return
    from test.evaluation.metrics import (
        identity,
        tanimoto,
    )

    # Property labels/inputs already live in the .pt file. A CSV is optional
    # for explicitly requested subsets or previously computed predictions.
    frame = None
    if args.task != "property_prediction" or config.get("input"):
        frame = pd.read_csv(config["input"]).fillna({"generated_smiles": ""})
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    if args.task == "property_prediction":
        if frame is None or not {"y_true", "y_pred"}.issubset(frame):
            frame = predict_property_rows(frame, config)
        a, b = frame.y_true.to_numpy(), frame.y_pred.to_numpy()
        if len(a) == 0 or not np.isfinite(a + b).all():
            raise ValueError("Empty or nonfinite property predictions")
        summary = dict(
            n=len(a),
            mae=float(np.abs(a - b).mean()),
        )
    else:
        if not {"query_id", "generated_smiles"}.issubset(frame):
            raise ValueError("Candidate CSV requires query_id and generated_smiles")
        identities = pd.DataFrame([identity(s) for s in frame.generated_smiles])
        for col in identities:
            frame[col] = identities[col]
        raw = frame[
            ~frame.get("empty_beam", pd.Series(False, index=frame.index))
            .fillna(False)
            .astype(bool)
        ]
        summary = {}
        queries_path = Path(
            config.get("queries") or Path(config["input"]).with_name("queries.csv")
        )
        if args.task != "multiobjective" and not queries_path.exists():
            raise ValueError(
                "Supply queries.csv so failed queries remain in the evaluation denominator"
            )
        if args.task == "budget":
            queries = pd.read_csv(queries_path)
            rows = []
            for q in queries.to_dict("records"):
                group = raw[
                    (raw.query_id == q["query_id"]) & (raw.budget == q["budget"])
                ]
                valid = group[group.valid]
                ranked = (
                    valid.sort_values("beam_rank")
                    .drop_duplicates("canonical_smiles")
                    .head(10)
                )
                eligible = ranked[ranked.eligible & (ranked.heavy_atoms <= q["budget"])]
                selected = eligible.iloc[0].canonical_smiles if len(eligible) else ""
                row = dict(
                    q,
                    top10_budget_coverage=float(bool(selected)),
                    selected_smiles=selected,
                    tanimoto=float("nan"),
                )
                if selected:
                    row["tanimoto"] = tanimoto(q["source_smiles"], selected)
                rows.append(row)
            query_frame = pd.DataFrame(rows)
            query_frame = enrich_spectra(query_frame, config, "selected_smiles")
            # Keep selected-pair results, not arbitrary input metadata or
            # redundant audit flags. Missing spectral scores remain empty.
            query_columns = [
                "query_id", "source_row_id", "source_smiles", "budget",
                "selected_smiles", "top10_budget_coverage", "tanimoto",
                "exact_uv_cosine", "exact_ir_cosine", "exact_raman_cosine",
                "exact_multimodal_cosine",
            ]
            query_frame.reindex(columns=query_columns).to_csv(output / "per_query.csv", index=False)
            summary["budget_rates"] = []
            for budget, g in query_frame.groupby("budget"):
                scores = [c for c in g if c.startswith("exact_") and c.endswith("_cosine")]
                summary["budget_rates"].append(
                    dict(
                        budget=int(budget),
                        query_count=len(g),
                        selected_count=int(g.top10_budget_coverage.sum()),
                        top10_budget_coverage=float(g.top10_budget_coverage.mean()),
                        spectral_count=int(g.spectra_available.sum()) if "spectra_available" in g else 0,
                        **{c:float(g[c].mean()) if g[c].notna().any() else None for c in scores},
                    )
                )
        elif args.task == "scaffold_editing":
            summary, scored, per_query = evaluate_scaffold(raw, pd.read_csv(queries_path), config)
            scored.to_csv(output / "candidates_audited.csv", index=False)
            per_query.to_csv(output / "per_query.csv", index=False)
        else:
            queries = pd.read_csv(queries_path)
            rows = []
            for q in queries.to_dict("records"):
                group = (
                    raw[(raw.query_id == q["query_id"]) & raw.valid]
                    .sort_values("beam_rank")
                    .drop_duplicates("canonical_smiles")
                )
                reference = identity(q["source_smiles"])["canonical_smiles"]
                match_values = group.canonical_smiles
                from rdkit import Chem

                normalize = lambda s: Chem.MolToSmiles(
                    Chem.MolFromSmiles(s), isomericSmiles=False
                )
                reference = normalize(reference)
                match_values = match_values.map(normalize)
                row = dict(q, selected_smiles="")
                for k in (1, 5, 10):
                    row[f"exact_top{k}"] = float(reference in set(match_values.head(k)))
                if len(group):
                    row["selected_smiles"] = group.iloc[0].canonical_smiles
                rows.append(row)
            per_query = pd.DataFrame(rows)
            per_query.to_csv(output / "per_query.csv", index=False)
            summary["queries"] = len(queries)
            summary["query_means"] = {
                k: float(per_query[k].mean()) if per_query[k].notna().any() else None
                for k in per_query
                if k.startswith("exact_top")
            }
        if args.task not in ("budget", "scaffold_editing"):
            frame.to_csv(output / "candidates_audited.csv", index=False)
    write_json(output / "summary.json", summary)
    print(summary)


def evaluate_scaffold(raw, queries, config):
    """Query-macro Task3 metrics on unique valid candidates; no spectra scoring."""
    import numpy as np
    import pandas as pd
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    from test.evaluation.metrics import tanimoto
    from test.evaluation.properties import predict_editing_properties

    if (queries.empty or not {'query_id', 'source_smiles'}.issubset(queries)
            or queries.query_id.duplicated().any()):
        raise ValueError("Unique query IDs and source_smiles are required")
    if not set(raw.query_id).issubset(set(queries.query_id)):
        raise ValueError("Candidate query IDs are absent from queries.csv")

    def structure(smiles):
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None or not mol.GetNumAtoms():
            raise ValueError(f"Invalid reference molecule: {smiles!r}")
        return (Chem.MolToSmiles(mol, isomericSmiles=False),
                Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol), isomericSmiles=False))

    references = {q.query_id: structure(q.source_smiles) for q in queries.itertuples()}
    columns = ['query_id', 'beam_rank', 'source_smiles', 'generated_smiles',
               'canonical_smiles', 'valid', 'recovered', 'scaffold_hop', 'tanimoto']
    columns += [prefix + name for name in ('homo', 'lumo', 'gap')
                for prefix in ('reference_', 'predicted_', 'abs_delta_')]
    rows = []
    for record in raw.to_dict('records'):
        reference, ref_scaffold = references[record['query_id']]
        row = dict(query_id=record['query_id'], beam_rank=record.get('beam_rank'),
                   source_smiles=reference, generated_smiles=record['generated_smiles'],
                   canonical_smiles='', valid=bool(record['valid']))
        if row['valid']:
            candidate, scaffold = structure(row['generated_smiles'])
            row.update(canonical_smiles=candidate, recovered=candidate == reference,
                       scaffold_hop=scaffold != ref_scaffold,
                       tanimoto=tanimoto(reference, candidate))
        rows.append(row)
    scored = pd.DataFrame(rows, columns=columns)
    valid = scored[scored.valid.astype(bool)]
    molecules = list(dict.fromkeys(valid.source_smiles.tolist() + valid.canonical_smiles.tolist()))
    # Predict each identity once; per-query statistics below retain one copy
    # per identity and query, even if that identity occurs under other queries.
    predictions = predict_editing_properties(molecules, config)
    if predictions.shape != (len(molecules), 3) or not np.isfinite(predictions).all():
        raise ValueError("Expected finite HOMO/LUMO/GAP predictions")
    lookup = dict(zip(molecules, predictions))
    for index, row in valid.iterrows():
        ref, pred = lookup[row.source_smiles], lookup[row.canonical_smiles]
        for j, name in enumerate(('homo', 'lumo', 'gap')):
            scored.loc[index, 'reference_' + name] = ref[j]
            scored.loc[index, 'predicted_' + name] = pred[j]
            scored.loc[index, 'abs_delta_' + name] = abs(pred[j] - ref[j])
    query_rows = []
    metric_names = ['validity_pct', 'uniqueness_pct', 'non_recovery_pct', 'hopping_pct',
                    'tanimoto', 'mae_homo', 'mae_lumo', 'mae_gap']
    for q in queries.to_dict('records'):
        group = scored[scored.query_id == q['query_id']]
        good = group[group.valid.astype(bool)]
        unique = good.drop_duplicates('canonical_smiles')
        n, nv, nu = len(group), len(good), len(unique)
        row = dict(query_id=q['query_id'],source_smiles=references[q['query_id']][0],
                   generated=n,valid=nv,unique=nu,
                   validity_pct=100.*nv/n if n else 0.,
                   uniqueness_pct=100.*nu/nv if nv else None,
                   non_recovery_pct=100.*(1.-float(unique.recovered.mean())) if nu else None,
                   hopping_pct=100.*float(unique.scaffold_hop.mean()) if nu else None,
                   tanimoto=float(unique.tanimoto.mean()) if nu else None,
                   **{'mae_'+name:float(unique['abs_delta_'+name].mean()) if nu else None
                      for name in ('homo','lumo','gap')})
        query_rows.append(row)
    per_query = pd.DataFrame(query_rows)
    summary = dict(
        queries=len(queries), valid_queries=int((per_query.valid > 0).sum()),
        generated=int(per_query.generated.sum()), valid=int(per_query.valid.sum()),
        unique=int(per_query.unique.sum()), aggregation='query_macro',
        **{k:float(per_query[k].mean()) if per_query[k].notna().any() else None for k in metric_names},
        property_unit='eV', property_source='independent HOMO/LUMO/GAP proxies',
        proxy_dir=str(config['proxy_dir']),
    )
    return summary, scored, per_query


def enrich_spectra(frame, config, smiles_column):
    if not config.get("data_path") or smiles_column not in frame:
        return frame
    from torchmdnet.data import ExactProcessedDataset
    from test.evaluation.metrics import identity, spectral_pair
    import pandas as pd

    data = ExactProcessedDataset(config["data_path"])
    lookup = {}
    for i in range(len(data)):
        key = identity(str(data[i].smiles))["canonical_smiles"]
        if key:
            lookup.setdefault(key, i)
    modalities = ("uv", "ir", "raman")
    values = []
    for row in frame.to_dict("records"):
        index = lookup.get(row.get(smiles_column))
        scores = {"spectra_available": False, **{"exact_" + m + "_cosine": float("nan") for m in modalities}}
        if len(modalities) > 1:
            scores["exact_multimodal_cosine"] = float("nan")
        if index is not None:
            a, b = data[int(row["source_row_id"])], data[index]
            scores.update(
                spectral_pair(
                    {
                        m: getattr(a, m).reshape(-1).numpy()
                        for m in modalities
                    },
                    {
                        m: getattr(b, m).reshape(-1).numpy()
                        for m in modalities
                    },
                    modalities=modalities,
                )
            )
            scores["spectra_available"] = True
        values.append(scores)
    return pd.concat([frame.reset_index(drop=True), pd.DataFrame(values)], axis=1)


def predict_property_rows(frame, config):
    import numpy as np
    import pandas as pd
    import torch
    from torch.utils.data import Subset
    from torch_geometric.loader import DataLoader
    from torchmdnet.data import ExactProcessedDataset, SpectraPreprocessedDataset
    from torchmdnet.checkpoint import load_model

    data = ExactProcessedDataset(config["data_path"])
    if frame is None:
        if not config.get("splits"):
            raise ValueError("Property evaluation requires the fixed splits NPZ")
        with np.load(config["splits"]) as splits:
            ids = np.asarray(splits["idx_test"])
            for key in ("idx_train", "idx_val"):
                if key in splits and np.intersect1d(ids, splits[key]).size:
                    raise ValueError("Property test split overlaps training/validation rows")
        frame = pd.DataFrame({"source_row_id": ids})
    else:
        ids = frame.source_row_id.to_numpy()
    if (ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer)
            or not len(ids) or len(np.unique(ids)) != len(ids)
            or (ids < 0).any() or (ids >= len(data)).any()):
        raise ValueError("Property row IDs must be nonempty, unique integers within the processed dataset")
    ids = ids.tolist()
    device = config.get("device") or "cpu"
    model = load_model(
        config["checkpoint"],
        args={
            "task": "regression",
        },
        device=device,
    ).eval()
    target_index = model.public_config.get("target_index")
    if target_index is None:
        raise ValueError("Property checkpoint must identify its trained target via target_index or a saved QM9S dataset_arg")
    if config.get("target_index", target_index) != target_index:
        raise ValueError("Evaluation target_index differs from the trained checkpoint")
    dataset = SpectraPreprocessedDataset(
        Subset(data, ids),
        model.public_config,
        is_train=False,
        peak_mask=False,
    )
    predictions = []
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=config.get("batch_size", 32)):
            batch = batch.to(device)
            out = model(
                batch.z,
                batch.pos,
                batch.spectra,
                batch.patch_masks,
                batch.smiles,
                task="regression",
                batch=batch.batch,
            )[0]
            predictions.extend(out.reshape(-1).cpu().tolist())
    frame = frame.copy()
    frame["y_pred"] = predictions
    frame["y_true"] = [
        float(data[int(i)].y.reshape(-1)[target_index]) for i in ids
    ]
    return frame


if __name__ == "__main__":
    main()
