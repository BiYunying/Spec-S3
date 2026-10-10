"""Generate candidates with the task's fixed decoder architecture."""

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
            "translation",
            "scaffold_editing",
            "ir_editing",
            "multiobjective",
            "budget",
        ),
    )
    p.add_argument("--conf", required=True)
    for name in (
        "checkpoint",
        "decoder-checkpoint",
        "adapter-checkpoint",
        "input",
        "output",
        "data-path",
        "conditions",
        "dataset",
        "objective",
        "device",
    ):
        p.add_argument("--" + name)
    for name in ("beam-width", "max-length", "count", "seed"):
        p.add_argument("--" + name, type=int)
    p.add_argument("--latent-noise", type=float)
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
    import torch
    from torchmdnet.checkpoint import load_model, load_decoder, sha256
    from torchmdnet.data import ExactProcessedDataset
    from test.generate.conditions import encode, ir_edit
    from test.generate.decoding import (
        decoder_function,
        completed_beams,
        sample_sequence,
    )
    from test.evaluation.metrics import identity

    device = config.get("device") or ("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    model = load_model(
        config["checkpoint"], args={"task": config["task"]}, device=device
    ).eval()
    # Preprocessing parameters must come from the model, not generic CLI defaults.
    preprocess = dict(model.public_config, spec_augment=False, peak_mask=False)
    if config.get("decoder_checkpoint"):
        load_decoder(model, config["decoder_checkpoint"])
    vocab = model.SMILES_DICTIONARY
    logits = decoder_function(model, config["decoder_type"])
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    adapter = None
    if args.task == "budget":
        from scripts.budget import load_adapter, PatchConditionDecoder

        model, adapter = load_adapter(model, config["adapter_checkpoint"], device)
        logits = adapter.decode
    rows = []
    queries = []
    if args.task == "multiobjective":
        if not config.get("conditions"):
            from scripts.multiobjective import build_conditions

            config["conditions"] = str(build_conditions(config, model, device))
        with np.load(config["conditions"]) as cache:
            z = cache["operator_z"]
            targets = cache["target_row_ids"]
            sources = cache["source_row_ids"]
        for i in range(len(z)):
            queries.append(
                dict(query_id=i, source_row_id=int(sources[i, 2]), draws=len(z[i]))
            )
            for j in range(len(z[i])):
                memory = torch.tensor(z[i, j], device=device).reshape(1, 1, -1)
                record = sample_sequence(
                    logits,
                    memory,
                    vocab,
                    max_new_tokens=min(
                        config.get("max_length", 79), model.hopping_decoder.block_size
                    ),
                    temperature=config.get("temperature", 1.0),
                    top_p=config.get("top_p", 0.9),
                )
                rows.append(
                    dict(
                        query_id=i,
                        draw_id=j,
                        source_row_id=int(sources[i, 2]),
                        target_row_id=int(targets[i, j]),
                        **record,
                    )
                )
    else:
        dataset = ExactProcessedDataset(config["data_path"])
        manifest = pd.read_csv(config["input"])
        if config.get("count"):
            manifest = manifest.iloc[: config["count"]]
        if "query_id" not in manifest:
            manifest["query_id"] = range(len(manifest))
        if manifest.query_id.duplicated().any():
            raise ValueError("Duplicate query IDs")
        for q in manifest.to_dict("records"):
            row_id = int(q.get("source_row_id", q.get("row_id", -1)))
            if row_id < 0:
                raise ValueError("Query manifest requires source_row_id")
            source = dataset[row_id]
            if args.task == "ir_editing":
                source = ir_edit(
                    source,
                    dataset[int(q["high_row_id"])],
                    config["crossfade_start"],
                    config["crossfade_end"],
                )
            memory = encode(
                model,
                [source],
                preprocess,
                device,
                patches=args.task == "budget" and isinstance(adapter, PatchConditionDecoder),
                project=config["decoder_type"] == "gpt",
            )
            if args.task == "scaffold_editing":
                memory = memory + float(
                    config.get("latent_noise", 0.01)
                ) * torch.randn_like(memory)
            budgets = config.get("budgets", [None]) if args.task == "budget" else [None]
            for budget in budgets:
                condition = (
                    adapter.memory(memory, torch.tensor([budget], device=device))
                    if adapter
                    else memory
                )
                max_length = config.get("max_length", 79)
                if config["decoder_type"] == "gpt":
                    max_length = min(max_length, model.hopping_decoder.block_size)
                else:
                    max_length = min(
                        max_length, model.smiles_decoder.positional_encoding.pe.shape[1]
                    )
                with torch.no_grad():
                    candidates = completed_beams(
                        logits,
                        condition,
                        width=config.get("beam_width", 20),
                        limit=config.get("candidate_limit", 20),
                        max_new_tokens=max_length,
                        penalty=config.get("length_penalty", 1.0),
                        vocab=vocab,
                    )
                query = dict(
                    query_id=q["query_id"],
                    source_row_id=row_id,
                    source_smiles=str(source.smiles),
                    budget=budget,
                )
                queries.append(query)
                seen = set()
                unique_rank = 0
                selected = False
                for record in candidates:
                    key = record["canonical_smiles"]
                    first = record["valid"] and key not in seen
                    if first:
                        seen.add(key)
                        unique_rank += 1
                    record.update(
                        is_first_unique=first,
                        candidate_rank=unique_rank if first else None,
                        selected=False,
                    )
                    if (
                        budget is not None
                        and first
                        and unique_rank <= 10
                        and record["eligible"]
                        and record["heavy_atoms"] <= budget
                        and not selected
                    ):
                        record["selected"] = True
                        selected = True
                    rows.append(dict(query, **record))
                if not candidates:
                    rows.append(
                        dict(
                            query,
                            generated_smiles="",
                            terminated_eos=False,
                            empty_beam=True,
                            beam_rank=0,
                            candidate_rank=None,
                            selected=False,
                            **identity(""),
                        )
                    )
    pd.DataFrame(rows).to_csv(output / "candidates.csv", index=False)
    pd.DataFrame(queries).to_csv(output / "queries.csv", index=False)
    write_json(
        output / "run.json",
        dict(
            config=config,
            checkpoint_sha256=sha256(config["checkpoint"]),
            decoder_sha256=(
                sha256(config["decoder_checkpoint"])
                if config.get("decoder_checkpoint")
                else None
            ),
            adapter_sha256=(
                sha256(config["adapter_checkpoint"])
                if config.get("adapter_checkpoint")
                else None
            ),
            queries=len(queries),
            records=len(rows),
            preprocessing=preprocess,
        ),
    )
    print(f"Wrote {len(rows)} records to {output}")


if __name__ == "__main__":
    main()
