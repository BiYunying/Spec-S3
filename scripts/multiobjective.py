"""FG26 tri-modal fusion, balanced hard OT and visible-anchor PCA."""

import math
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence, Tuple
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


def transport(fused, source_ids, prototypes, prototype_ids, rank=12, beta=0.15):
    """Balanced one-to-one cosine assignment excluding the source D, then PCA residual."""
    if fused.shape[0] != len(prototype_ids):
        raise ValueError("Balanced OT requires one query per visible anchor")
    if len(set(prototype_ids.tolist())) != len(prototype_ids):
        raise ValueError("Visible anchor IDs must be unique")
    if not 1 <= rank <= min(prototypes.shape[0] - 1, prototypes.shape[1]):
        raise ValueError("PCA rank exceeds the visible-anchor dimensions")
    if not np.isfinite(fused).all() or not np.isfinite(prototypes).all():
        raise ValueError("Nonfinite conditions")
    centered = prototypes - prototypes.mean(0, keepdims=True)
    basis = np.linalg.svd(centered, full_matrices=False)[2][:rank]
    result = np.empty_like(fused)
    targets = np.empty(fused.shape[:2], dtype=np.int64)
    allowed = np.asarray(source_ids)[:, None] != np.asarray(prototype_ids)[None, :]
    for draw in range(fused.shape[1]):
        score = normalized(fused[:, draw]) @ normalized(prototypes).T
        rows, cols = linear_sum_assignment(-score + (~allowed) * 1e6)
        if (
            not np.array_equal(rows, np.arange(len(source_ids)))
            or not allowed[rows, cols].all()
        ):
            raise RuntimeError("No valid balanced assignment excluding source D")
        atlas = prototypes[cols]
        residual = fused[:, draw] - atlas
        result[:, draw] = atlas + beta * ((residual @ basis.T) @ basis)
        targets[:, draw] = np.asarray(prototype_ids)[cols]
    return result, targets, basis


def build_conditions(config, model=None, device="cpu"):
    if str(config.get("dataset", "")).lower() != "fg26":
        raise ValueError("Task 4 supports FG26 only")
    import pandas as pd
    from torchmdnet.data import ExactProcessedDataset
    from torchmdnet.checkpoint import load_model, sha256
    from torchmdnet.utils import write_json
    from test.generate.conditions import encode, fuse_spectra

    model = model or load_model(
        config["checkpoint"], args={"task": "A+B"}, device=device
    )
    model.eval()
    data = ExactProcessedDataset(config["data_path"])
    source_table = pd.read_csv(config["input"])
    columns = ["a_row_id", "b_row_id", "d_row_id"]
    if not set(columns).issubset(source_table):
        columns = ["source_a_row_id", "source_b_row_id", "source_d_row_id"]
    # Historical transported manifests repeat a query once per draw.
    triples = source_table[columns].drop_duplicates().to_numpy(dtype=np.int64)
    anchors = pd.read_csv(config["known_reference"]).row_id.to_numpy(dtype=np.int64)
    preprocess = model.public_config
    prototypes = np.concatenate(
        [
            encode(model, [data[int(i)]], preprocess, device, project=True)
            .cpu()
            .numpy()[:, 0]
            for i in anchors
        ]
    )
    draws = int(config.get("draws", 64))
    if config.get("weights_path"):
        weights = np.load(config["weights_path"])
    else:
        args = SimpleNamespace(
            min_single_mass=0.10,
            max_single_mass=0.45,
            initial_single_mass=0.20,
            min_source_fraction=0.15,
            logistic_std=math.exp(-1.2),
            source_subset="ABD",
        )
        weights = sampled_weights(
            len(triples), draws, args, int(config.get("weight_seed", config["seed"]))
        )
    if (
        weights.shape != (len(triples), draws, 3, 3)
        or (weights < 0).any()
        or not np.allclose(weights.sum(-1), 1)
    ):
        raise ValueError(
            "Expected normalized weights [queries,draws,modalities,sources]"
        )
    fused = np.stack(
        [
            np.concatenate(
                [
                    encode(
                        model,
                        [fuse_spectra([data[int(k)] for k in triple], weights[i, j])],
                        preprocess,
                        device,
                        project=True,
                    )
                    .cpu()
                    .numpy()[:, 0]
                    for j in range(draws)
                ]
            )
            for i, triple in enumerate(triples)
        ]
    )
    z, targets, basis = transport(
        fused,
        triples[:, 2],
        prototypes,
        anchors,
        config.get("pca_rank", 12),
        config.get("projection_magnitude", 0.15),
    )
    root = Path(config.get("output") or config["log_dir"])
    root.mkdir(parents=True, exist_ok=True)
    path = root / "conditions.npz"
    np.savez(
        path,
        operator_z=z,
        target_row_ids=targets,
        source_row_ids=triples,
        original_fused_z=fused,
        modality_source_weights=weights,
        pca_basis=basis,
        visible_ids=anchors,
    )
    write_json(
        root / "conditions.json",
        dict(
            checkpoint_sha256=sha256(config["checkpoint"]),
            own_D_forbidden=True,
            rank=config.get("pca_rank", 12),
            beta=config.get("projection_magnitude", 0.15),
            weights_source=config.get("weights_path", "fixed_fg26_logistic_normal"),
            config=config,
        ),
    )
    return path


def train(config):
    """Select decoder epochs using held-out visible queries, reset, then refit."""
    if str(config.get("dataset", "")).lower() != "fg26":
        raise ValueError("Task 4 supports FG26 only")
    if config.get("phase", "decoder") not in ("decoder", "build"):
        raise ValueError("Task 4 supports decoder/build phases only; warmup is independent")
    import json
    import pandas as pd
    from torchmdnet.checkpoint import load_model, load_decoder
    from torchmdnet.data import ExactProcessedDataset
    from torchmdnet.utils import write_json

    device = config.get("device") or ("cuda:0" if torch.cuda.is_available() else "cpu")
    if config.get("phase") == "build":
        print(build_conditions(config, device=device))
        return
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    root = Path(config["log_dir"])
    root.mkdir(parents=True, exist_ok=True)
    with np.load(config["training_conditions"]) as c:
        z = c["operator_z"]
        target = c["target_row_ids"]
        triples = c["source_row_ids"]
        source = triples[:, 2]
    visible = pd.read_csv(config["known_reference"]).row_id.to_numpy(dtype=np.int64)
    if (
        set(source) != set(visible)
        or not set(target.ravel()).issubset(set(visible))
        or np.any(target == source[:, None])
    ):
        raise ValueError(
            "Training conditions must transport the disclosed anchors to other disclosed anchors"
        )
    data = ExactProcessedDataset(config["data_path"])
    canonical = [str(data[i].smiles) for i in range(len(data))]
    with np.load(config["splits"]) as split:
        official_train = set(split["idx_train"].tolist())
        official_test = set(split["idx_test"].tolist())
    if set(triples[:, :2].ravel().tolist()) & official_test:
        raise ValueError("Final test A/B queries cannot be used for decoder adaptation")
    tr = np.array([i for i, r in enumerate(source) if int(r) in official_train])
    va = np.array([i for i, r in enumerate(source) if int(r) not in official_train])
    if not len(tr) or not len(va):
        raise ValueError("Both selection and held-out visible query sets are required")
    args = SimpleNamespace(
        seed=seed,
        batch_size=config.get("batch_size", 64),
        random_smiles_probability=config.get("random_smiles_probability", 0.5),
        unfreeze_last_decoder_blocks=config.get("unfreeze_last_decoder_blocks", 3),
    )
    replay = None
    if config.get("replay"):
        replay = np.load(config["replay"])
    if config.get("replay_required", False) and replay is None:
        raise ValueError(
            "This selected protocol requires non-dual train/val replay arrays"
        )

    def values(indices, split):
        a, b = z[indices].reshape(-1, z.shape[-1]), target[indices].reshape(-1)
        if replay is not None:
            ids = replay[split + "_ids"]
            if set(ids.tolist()) & set(visible):
                raise ValueError(
                    "Replay must be non-dual; visible anchors belong to the explicit anchor branch"
                )
            a = np.concatenate([a, replay[split + "_z"]])
            b = np.concatenate([b, ids])
        return a, b

    def new_model():
        model = load_model(config["checkpoint"], args={"task": "A+B"}, device=device)
        if config.get("init_checkpoint"):
            load_decoder(model, config["init_checkpoint"])
        configure_decoder_trainability(model, args.unfreeze_last_decoder_blocks)
        optimizer = torch.optim.AdamW(
            [p for p in model.hopping_decoder.parameters() if p.requires_grad],
            lr=config.get("decoder_lr", 1.5e-6),
            weight_decay=config.get("weight_decay", 1e-4),
        )
        return model, optimizer

    model, opt = new_model()
    train_loader = loader(values(tr, "train"), canonical, model, args, True)
    val_loader = loader(values(va, "val"), canonical, model, args, False)
    history = []
    best = float("inf")
    best_epoch = 0
    stale = 0
    for epoch in range(1, config.get("num_epochs", 16) + 1):
        tl = run_epoch(model, train_loader, args, device, opt)
        vl = run_epoch(model, val_loader, args, device)
        if not np.isfinite(tl + vl):
            raise RuntimeError("Nonfinite decoder loss")
        history.append(dict(epoch=epoch, train_loss=tl, val_loss=vl))
        print(history[-1], flush=True)
        if vl < best:
            best, best_epoch, stale = vl, epoch, 0
        else:
            stale += 1
        if stale >= config.get("patience", 5):
            break
    del model, opt
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model, opt = new_model()
    final_loader = loader(
        values(np.arange(len(source)), "train"), canonical, model, args, True
    )
    for epoch in range(best_epoch):
        run_epoch(model, final_loader, args, device, opt)
    torch.save(
        dict(
            decoder_state_dict={
                k: v.detach().cpu()
                for k, v in model.hopping_decoder.state_dict().items()
            },
            config=config,
            selected_epochs=best_epoch,
        ),
        root / "decoder_final.pt",
    )
    pd.DataFrame(history).to_csv(root / "selection.csv", index=False)
    write_json(
        root / "selection.json",
        dict(
            selected_epochs=best_epoch,
            val_loss=best,
            train_queries=tr.tolist(),
            val_queries=va.tolist(),
        ),
    )


def sampled_weights(queries, draws, args, seed):
    rng = np.random.default_rng(seed)
    ratio = (args.initial_single_mass - args.min_single_mass) / (
        args.max_single_mass - args.min_single_mass
    )
    base = math.log(ratio / (1.0 - ratio))
    latent = rng.normal(size=(queries, draws, 3, 2)) * args.logistic_std
    total = args.min_single_mass + (args.max_single_mass - args.min_single_mass) / (
        1.0 + np.exp(-(base + latent[..., 0]))
    )
    fraction = args.min_source_fraction + (1.0 - 2.0 * args.min_source_fraction) / (
        1.0 + np.exp(-latent[..., 1])
    )
    weights = np.stack(
        [total * fraction, total * (1.0 - fraction), 1.0 - total], axis=-1
    ).astype(np.float32)
    mask = np.asarray(
        [name in args.source_subset for name in ("A", "B", "D")], dtype=np.float32
    )
    weights *= mask[None, None, None, :]
    denominator = weights.sum(axis=-1, keepdims=True)
    if np.any(denominator <= 0):
        raise RuntimeError(f"Source subset {args.source_subset} has zero mass")
    return weights / denominator


def normalized(values):
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-12)


from typing import List
import random
from torch.utils.data import Dataset, DataLoader
from rdkit import Chem


def configure_decoder_trainability(model, last_n: int) -> List[str]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    blocks = list(model.hopping_decoder.blocks)
    if not 1 <= last_n <= len(blocks):
        raise ValueError(
            f"--unfreeze-last-decoder-blocks must be in [1, {len(blocks)}]"
        )
    for block in blocks[-last_n:]:
        for parameter in block.parameters():
            parameter.requires_grad_(True)
    for parameter in model.hopping_decoder.ln_f.parameters():
        parameter.requires_grad_(True)
    return [
        name
        for name, parameter in model.hopping_decoder.named_parameters()
        if parameter.requires_grad
    ]


def set_decoder_training_mode(model, last_n: int, training: bool) -> None:
    model.eval()
    if training:
        for block in list(model.hopping_decoder.blocks)[-last_n:]:
            block.train()
        model.hopping_decoder.ln_f.train()


class ConditionTargetDataset(Dataset):

    def __init__(
        self, conditions, target_ids, canonical, model, training, random_probability
    ):
        self.conditions = np.asarray(conditions, dtype=np.float32)
        self.target_ids = np.asarray(target_ids, dtype=np.int64)
        self.canonical = canonical
        self.model = model
        self.training = bool(training)
        self.random_probability = float(random_probability)
        self.max_length = int(model.hopping_decoder.get_block_size()) + 1
        self.targets = {
            int(row_id): model._normalize_gpt_smiles_target(str(canonical[int(row_id)]))
            for row_id in np.unique(self.target_ids)
        }
        if any((value is None for value in self.targets.values())):
            raise RuntimeError("A decoder target is outside the SMILES vocabulary")

    def __len__(self):
        return len(self.target_ids)

    def __getitem__(self, index):
        row_id = int(self.target_ids[index])
        smiles = str(self.targets[row_id])
        if self.training and random.random() < self.random_probability:
            candidate = random_smiles(self.canonical[row_id])
            try:
                encoded = self.model.tokenizer(
                    [candidate],
                    padding=True,
                    truncation=False,
                    max_length=self.max_length,
                )
                if encoded["input_ids"].shape[1] <= self.max_length:
                    smiles = candidate
            except (TypeError, ValueError):
                pass
        return (torch.from_numpy(self.conditions[index].copy()), smiles)


def collate(rows):
    return {
        "z": torch.stack([row[0] for row in rows]),
        "smiles": [row[1] for row in rows],
    }


def loader(values, canonical, model, args, training):
    dataset = ConditionTargetDataset(
        *values,
        canonical,
        model,
        training,
        args.random_smiles_probability if training else 0.0,
    )
    generator = torch.Generator().manual_seed(args.seed + int(training))
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=training,
        generator=generator if training else None,
        num_workers=0,
        collate_fn=collate,
    )


def run_epoch(model, data_loader, args, device, optimizer=None):
    training = optimizer is not None
    set_decoder_training_mode(model, args.unfreeze_last_decoder_blocks, training)
    total_loss = total_examples = 0
    for batch in data_loader:
        if training:
            optimizer.zero_grad(set_to_none=True)
        decoder_input, labels = tokenized_targets(model, batch["smiles"], device)
        with torch.set_grad_enabled(training):
            _, loss = model.hopping_decoder(
                decoder_input, labels, batch["z"].to(device).unsqueeze(1).clone()
            )
            if training:
                loss.backward()
                trainable = [
                    p for p in model.hopping_decoder.parameters() if p.requires_grad
                ]
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
        total_loss += float(loss.detach()) * len(batch["smiles"])
        total_examples += len(batch["smiles"])
    return total_loss / max(total_examples, 1)


def random_smiles(value):
    molecule = Chem.MolFromSmiles(str(value))
    if molecule is None:
        return str(value)
    try:
        return Chem.MolToSmiles(
            molecule, canonical=False, doRandom=True, isomericSmiles=False
        )
    except Exception:
        return str(value)


def tokenized_targets(
    model, smiles: Sequence[str], device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    max_length = int(model.hopping_decoder.get_block_size()) + 1
    tokenized = model.tokenizer(
        list(smiles), padding=True, truncation=False, max_length=max_length
    )
    ids = tokenized["input_ids"]
    if ids.shape[1] > max_length:
        raise ValueError("Target SMILES exceeds decoder block size")
    return (ids[:, :-1].to(device), ids[:, 1:].to(device))
