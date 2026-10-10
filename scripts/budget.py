"""Single-stage additive budget conditioning with uniform DPO; legacy adapters supported."""

import copy
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F


def load_adapter(original, path, device, architecture=None):
    from torchmdnet.checkpoint import read_checkpoint

    extend_position_encoding(original.smiles_decoder)
    backbone = LegacyCrossBackbone(original).to(device).requires_grad_(False).eval()
    payload = read_checkpoint(path) if path else {}
    state = payload.get("adapter", payload)
    # Experimental additive checkpoints also contain unused region_projection
    # tensors. Check their actual budget parameterization before legacy detection.
    is_single = architecture == "single" or "budget_embedding.weight" in state
    is_patch = architecture == "patch" or any(
        k.startswith("region_projection.") for k in state
    )
    if is_single:
        if payload.get("arm") not in (None, "weighted", "unweighted", "preference"):
            raise ValueError("This experimental arm is not the additive pooled model")
        for key in ("scale.weight", "region_projection.weight"):
            if key in state and torch.count_nonzero(state[key]).item():
                raise ValueError(f"Additive-only loading cannot discard active {key}")
        adapter = AdditiveBudgetDecoder(backbone)
    else:
        adapter = PatchConditionDecoder(backbone) if is_patch else BudgetDecoder(backbone)
    adapter = adapter.to(device)
    if state:
        if not is_single and is_patch and not any(k.startswith("region_projection.") for k in state):
            adapter.base.load_state_dict(state, strict=True)
        else:
            adapter.load_state_dict(state, strict=True)
    return backbone, adapter.eval()


@torch.no_grad()
def prepare_features(model, payload, config, device, patches):
    from torchmdnet.data import ExactProcessedDataset
    from test.generate.conditions import encode

    encoder_hash = encoder_sha256(model.representation_spec_model)
    panel_hash = feature_signature(payload, config)
    if config.get("features") and Path(config["features"]).exists():
        saved = torch.load(
            config["features"], map_location="cpu", weights_only=False, mmap=True
        )
        expected = saved.get("encoder_sha256", config.get("features_encoder_sha256"))
        if expected == encoder_hash and saved.get("panel_sha256") == panel_hash:
            if patches and saved["features"].shape[1] <= 1:
                raise ValueError("Patch training requires pre-pooling tokens, not only pooled features")
            return saved
        print("Rebuilding features: encoder, sample order or input data changed, or cache provenance is missing.", flush=True)
    data = ExactProcessedDataset(config["data_path"])
    arrays = []
    offsets = {}
    cursor = 0
    panels = [(s, payload["sources"][s]) for s in ("train", "val")]
    # Preserve historical offsets without encoding the test panel for training.
    panels += [
        ("test", payload["test"]),
        ("replay", payload["bank"].iloc[payload["replay_ids"]]),
    ]
    for tag, panel in panels:
        offsets[tag] = cursor
        if tag == "test":
            tokens = (
                1 + sum(model.representation_spec_model.patch_nums) if patches else 1
            )
            arrays.append(torch.zeros(len(panel), tokens, 256))
            cursor += len(panel)
            continue
        ids = panel.source_row_id.tolist()
        for start in range(0, len(ids), 8):
            arrays.append(
                encode(
                    model,
                    [data[int(i)] for i in ids[start : start + 8]],
                    config,
                    device,
                    patches=patches,
                ).cpu()
            )
            cursor += len(arrays[-1])
    saved = dict(
        features=torch.cat(arrays), offsets=offsets, test_features_omitted=True,
        encoder_sha256=encoder_hash,
        panel_sha256=panel_hash,
    )
    torch.save(saved, Path(config["log_dir"]) / "features.pt")
    return saved


def feature_signature(payload, config):
    """Prevent reusing features after changing panels, their order, or raw data."""
    import hashlib

    panels = [payload["sources"][s] for s in ("train", "val")]
    panels += [payload["test"], payload["bank"].iloc[payload["replay_ids"]]]
    data = Path(config["data_path"]).resolve()
    stat = data.stat() if data.exists() else None
    record = dict(panels=[p[[c for c in ("source_row_id", "key") if c in p]].to_dict("list") for p in panels],
                  data=str(data), size=stat.st_size if stat else None,
                  mtime=stat.st_mtime_ns if stat else None,
                  normalization=config.get("input_data_norm_type", "log10"))
    return hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()


def encoder_sha256(encoder):
    """Bind cached latent features to the frozen encoder's actual tensors."""
    import hashlib

    digest = hashlib.sha256()
    for name, tensor in sorted(encoder.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(f"{name}:{value.dtype}:{tuple(value.shape)}".encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def validate_generation(adapter, saved, payload, config, device):
    from torchmdnet.data import ExactProcessedDataset
    from torchmdnet.tokenizer import SMILES_DICTIONARY
    from test.generate.decoding import completed_beams
    from test.evaluation.metrics import identity, cosine

    data = ExactProcessedDataset(config["data_path"])
    lookup = {}
    for i in range(len(data)):
        key = identity(str(data[i].smiles))["canonical_smiles"]
        if key:
            lookup.setdefault(key, i)
    records = []
    adapter.eval()
    for i, q in enumerate(payload["sources"]["val"].itertuples(index=False)):
        f = saved["features"][
            saved["offsets"]["val"] + i : saved["offsets"]["val"] + i + 1
        ].to(device)
        for budget in payload.get("budgets", BUDGETS):
            memory = adapter.memory(f, torch.tensor([budget], device=device))
            candidates = completed_beams(
                adapter.decode, memory, vocab=SMILES_DICTIONARY
            )
            seen = set()
            unique = []
            for c in candidates:
                if c["valid"] and c["canonical_smiles"] not in seen:
                    unique.append(c)
                    seen.add(c["canonical_smiles"])
            hit = next(
                (
                    c
                    for c in unique[:10]
                    if c["eligible"] and c["heavy_atoms"] <= budget
                ),
                None,
            )
            score = np.nan
            if hit and hit["canonical_smiles"] in lookup:
                a, b = data[int(q.source_row_id)], data[lookup[hit["canonical_smiles"]]]
                score = np.mean(
                    [
                        cosine(
                            getattr(a, m).reshape(-1).numpy(),
                            getattr(b, m).reshape(-1).numpy(),
                        )
                        for m in ("uv", "ir", "raman")
                    ]
                )
            records.append(
                dict(
                    query_id=int(q.source_row_id),
                    budget=budget,
                    cosine=score,
                    coverage=float(hit is not None),
                )
            )
    frame = pd.DataFrame(records)
    return (
        frame.groupby("budget").agg(
            cosine=("cosine", "mean"),
            n_spectral=("cosine", "count"),
            coverage=("coverage", "mean"),
        ),
        frame,
    )


def train(config):
    from torchmdnet.checkpoint import load_model, sha256
    from torchmdnet.utils import write_json

    phase = config.get("phase", "single")
    if phase not in ("single", "base", "preference", "patch"):
        raise ValueError("budget phase must be single (or legacy base/preference/patch)")
    if phase == "single" and config.get("init_checkpoint"):
        raise ValueError("Single-stage adaptation starts from --checkpoint warmup; omit --init-checkpoint")
    device = config.get("device") or ("cuda:0" if torch.cuda.is_available() else "cpu")
    root = Path(config["log_dir"])
    root.mkdir(parents=True, exist_ok=True)
    payload = torch.load(config["prepared"], map_location="cpu", weights_only=False)
    check_partitions(payload)
    if "budgets" in payload and list(config.get("budgets", payload["budgets"])) != list(payload["budgets"]):
        raise ValueError("Configured budgets differ from prepared data; use the generated config.yaml")
    original = load_model(
        config["checkpoint"], args={"task": "smiles"}, device=device
    ).eval()
    if phase in ("preference", "patch") and not config.get("init_checkpoint"):
        raise ValueError(
            "Preference/patch phases require their preceding adapter checkpoint"
        )
    backbone, adapter = load_adapter(
        original,
        config.get("init_checkpoint"),
        device,
        architecture=phase if phase in ("single", "patch") else None,
    )
    saved = prepare_features(
        backbone,
        payload,
        dict(
            original.public_config,
            **{
                k: v
                for k, v in config.items()
                if k in ("features", "features_encoder_sha256", "log_dir", "data_path")
            },
        ),
        device,
        phase == "patch",
    )
    if phase != "patch" and saved["features"].shape[1] > 1:
        saved["features"] = saved["features"][:, :1]
    features = saved["features"]
    tokens = payload["target_tokens"]
    pairs = None
    reference = None
    if phase != "base":
        pairs = torch.load(config["pairs"], map_location="cpu", weights_only=False)
        lo, hi = saved["offsets"]["train"], saved["offsets"]["val"]
        if ((pairs["source_index"] < lo) | (pairs["source_index"] >= hi)).any():
            raise ValueError(
                "Preference sources must belong to the training feature partition"
            )
        sizes = torch.as_tensor(payload["bank"].ha.to_numpy())
        if not torch.equal(sizes[pairs["positive"]], sizes[pairs["negative"]]):
            raise ValueError("Preference targets must have equal heavy-atom counts")
        if (sizes[pairs["positive"]] > pairs["budget"]).any():
            raise ValueError("Preference target exceeds its budget")
        pos = []
        neg = []
        adapter.eval()
        with torch.no_grad():
            for start in range(0, len(pairs["gap"]), 32):
                a, b = logs_for_pairs(
                    adapter,
                    features,
                    pairs,
                    tokens,
                    torch.arange(start, min(start + 32, len(pairs["gap"]))),
                    device,
                )
                pos.append(a.cpu())
                neg.append(b.cpu())
        reference = dict(positive=torch.cat(pos), negative=torch.cat(neg))
    base = adapter.base if phase == "patch" else adapter
    groups = [
        dict(params=base.decoder.parameters(), lr=1e-5),
        dict(params=base.projector.parameters(), lr=3e-5),
        dict(params=(list(base.budget_embedding.parameters()) + list(base.shift.parameters()))
             if phase == "single" else base.budget.parameters(), lr=1e-3),
    ]
    if phase == "patch":
        groups.append(dict(params=adapter.region_projection.parameters(), lr=1e-4))
    optimizer = torch.optim.AdamW(groups, weight_decay=0.0)
    rng = np.random.default_rng(config["seed"])
    pair_rng = np.random.default_rng(config["seed"] + 1)
    best_score = (-float("inf"), -float("inf"))
    selected = None
    history = []
    baseline = (
        pd.read_csv(config["baseline"], index_col="budget")
        if config.get("baseline")
        else None
    )
    if phase in ("preference", "patch"):
        if baseline is None:
            raise ValueError(
                "Preference/patch selection requires the fixed original validation baseline summary"
            )
        summary, frame = validate_generation(adapter, saved, payload, config, device)
        best_score = selection_score(summary, baseline)
        selected = config["init_checkpoint"]
    if phase == "single":
        # Independent of reference-cache batching and optional diagnostics.
        torch.manual_seed(config["seed"] + 100000)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config["seed"] + 100000)
    for step in range(1, config.get("steps", 8000) + 1):
        adapter.train()
        optimizer.zero_grad(set_to_none=True)
        ids, budgets, tids = choose_batch(
            payload, saved["offsets"], rng, config.get("batch_size", 16)
        )
        ce = adapter.loss(
            features[ids].to(device),
            torch.tensor(budgets, device=device),
            tokens[tids].to(device),
        )
        pref = ce.new_zeros(())
        if pairs is not None:
            adapter.eval()
            idx = torch.tensor(
                pair_rng.integers(len(pairs["gap"]), size=config.get("batch_size", 16))
            )
            plus, minus = logs_for_pairs(adapter, features, pairs, tokens, idx, device)
            loss_fn = preference_loss if phase == "single" else legacy_weighted_preference_loss
            pref = loss_fn(
                plus,
                minus,
                reference["positive"][idx].to(device),
                reference["negative"][idx].to(device),
                pairs["gap"][idx].to(device),
                beta=config.get("preference_beta", 0.1),
            )
        loss = (
            ce
            + config.get("preference_weight", 0.3 if phase in ("single", "patch") else 1.0) * pref
        )
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite budget loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % config.get("log_every", 200) == 0:
            history.append(
                dict(step=step, ce=float(ce.detach()), preference=float(pref.detach()))
            )
            print(history[-1], flush=True)
        if (config.get("validate_every", 4000) > 0 and step % config.get("validate_every", 4000) == 0) or step == config.get(
            "steps", 8000
        ):
            path = root / f"step{step}.pt"
            torch.save(
                dict(
                    adapter={
                        k: v.detach().cpu() for k, v in adapter.state_dict().items()
                    },
                    step=step,
                    architecture=phase,
                    config=config,
                ),
                path,
            )
            if phase == "single":
                # Adopted manuscript runs report fixed step8000, not best-val.
                selected = str(path)
                if config.get("validate_every", 4000) > 0:
                    summary, frame = validate_generation(adapter, saved, payload, config, device)
                    summary.to_csv(root / f"validation{step}.csv")
                    frame.to_csv(root / f"validation_queries{step}.csv", index=False)
                continue
            if phase == "base":
                score = (
                    -validation(adapter, features, saved["offsets"], payload, device),
                    0.0,
                )
            else:
                summary, frame = validate_generation(
                    adapter, saved, payload, config, device
                )
                summary.to_csv(root / f"validation{step}.csv")
                frame.to_csv(root / f"validation_queries{step}.csv", index=False)
                score = selection_score(summary, baseline)
            if score > best_score:
                best_score = score
                selected = str(path)
    if selected is None:
        raise RuntimeError("No eligible checkpoint")
    write_json(
        root / "selected.json",
        dict(
            checkpoint=selected,
            sha256=sha256(selected),
            score=None if phase == "single" else best_score,
            selection="fixed final step" if phase == "single" else "validation only; includes initialization fallback",
            config=config,
        ),
    )
    pd.DataFrame(history).to_csv(root / "metrics.csv", index=False)


class BudgetDecoder(nn.Module):

    def __init__(self, model, method=None):
        super().__init__()
        self.cross = True
        self.projector = model.spec_projector
        self.decoder = model.smiles_decoder if self.cross else model.hopping_decoder
        self.projector.requires_grad_(True)
        self.decoder.requires_grad_(True)
        self.budget = nn.Embedding(31, 512)
        nn.init.zeros_(self.budget.weight)

    def memory(self, features, budgets):
        memory = self.projector(features)
        scale, shift = self.budget(budgets).chunk(2, -1)
        return memory * (1 + 0.1 * scale.tanh()[:, None]) + shift[:, None]

    def decode(self, tokens, memory):
        if not self.cross:
            return self.decoder(tokens, None, memory.clone())[0]
        mask = torch.ones(
            tokens.shape[1], tokens.shape[1], device=tokens.device, dtype=torch.bool
        ).triu(1)
        return self.decoder(tokens, memory, mask, tokens.eq(0))

    def loss(self, features, budgets, targets):
        logits = self.decode(targets[:, :-1], self.memory(features, budgets))
        return F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets[:, 1:].reshape(-1),
            ignore_index=0,
        )


class AdditiveBudgetDecoder(BudgetDecoder):
    """Pooled projector(z) + shift(embedding(B)); no scale or patch tokens."""

    def __init__(self, model):
        nn.Module.__init__(self)
        self.cross = True
        self.projector = model.spec_projector.requires_grad_(True)
        self.decoder = model.smiles_decoder.requires_grad_(True)
        self.budget_embedding = nn.Embedding(31, 64)
        self.shift = nn.Linear(64, 256, bias=False)
        # Frozen zero tensors preserve exact experimental state_dict and RNG
        # compatibility. Neither participates in memory() or optimization.
        self.scale = nn.Linear(64, 256, bias=False)
        self.region_norm = nn.LayerNorm(256, elementwise_affine=False)
        self.region_projection = nn.Linear(256, 256, bias=False)
        for module in (self.shift, self.scale, self.region_projection):
            nn.init.zeros_(module.weight)
        self.scale.requires_grad_(False)
        self.region_projection.requires_grad_(False)

    def memory(self, features, budgets):
        return self.projector(features[:, :1]) + self.shift(self.budget_embedding(budgets))[:, None]


def extend_position_encoding(decoder, length=79):
    import math

    pe_module = decoder.positional_encoding
    old = pe_module.pe
    if old.shape[1] >= length:
        return old.shape[1]
    dim = old.shape[-1]
    position = torch.arange(length).unsqueeze(1)
    factor = torch.exp(torch.arange(0, dim, 2) * -(math.log(10000.0) / dim))
    extended = torch.zeros(1, length, dim)
    extended[0, :, 0::2] = torch.sin(position * factor)
    extended[0, :, 1::2] = torch.cos(position * factor)
    extended = extended.to(old)
    extended[:, : old.shape[1]].copy_(old)
    pe_module.pe = extended
    assert torch.equal(pe_module.pe[:, : old.shape[1]], old)
    return old.shape[1]


class LegacyCrossBackbone(nn.Module):

    def __init__(self, original):
        super().__init__()
        self.representation_spec_model = original.representation_spec_model
        self.smiles_decoder = original.smiles_decoder
        self.spec_projector = nn.Linear(256, 256)
        nn.init.eye_(self.spec_projector.weight)
        nn.init.zeros_(self.spec_projector.bias)

    def condition(self, spectra):
        from test.generate.conditions import encode_spectra

        return encode_spectra(self, spectra)

    def decode(self, tokens, memory):
        mask = torch.ones(
            tokens.shape[1], tokens.shape[1], device=tokens.device, dtype=torch.bool
        ).triu(1)
        return self.smiles_decoder(tokens, memory, mask, tokens.eq(0))


class PatchConditionDecoder(nn.Module):

    def __init__(self, backbone):
        super().__init__()
        self.base = BudgetDecoder(backbone)
        self.region_norm = nn.LayerNorm(256, elementwise_affine=False)
        self.region_projection = nn.Linear(256, 256, bias=False)
        nn.init.zeros_(self.region_projection.weight)

    def memory(self, features, budgets):
        global_memory = self.base.memory(features[:, :1], budgets)
        residual = self.region_projection(self.region_norm(features[:, 1:]))
        return torch.cat([global_memory, global_memory + residual], dim=1)

    def decode(self, tokens, memory):
        return self.base.decode(tokens, memory)

    def loss(self, features, budgets, targets):
        logits = self.decode(targets[:, :-1], self.memory(features, budgets))
        return nn.functional.cross_entropy(
            logits.flatten(0, 1), targets[:, 1:].reshape(-1), ignore_index=0
        )


def choose_batch(payload, offset, rng, size=16):
    feature_ids, budgets, target_ids = ([], [], [])
    examples = payload["examples"]["train"]
    for _ in range(size):
        if rng.random() < 0.75:
            ex = examples[int(rng.integers(len(examples)))]
            target = int(
                rng.choice(
                    ex["target_bank_ids"],
                    p=np.array(ex["weights"]) / sum(ex["weights"]),
                )
            )
            feature_ids.append(offset["train"] + ex["source_index"])
            budgets.append(ex["budget"])
        else:
            pos = int(rng.integers(len(payload["replay_ids"])))
            target = payload["replay_ids"][pos]
            feature_ids.append(offset["replay"] + pos)
            budgets.append(int(payload["bank"].iloc[target].ha))
        target_ids.append(target)
    return (feature_ids, budgets, target_ids)


@torch.no_grad()
def validation(adapter, features, offset, payload, device):
    adapter.eval()
    examples = payload["examples"]["val"]
    flat = [
        (ex["source_index"] + offset["val"], ex["budget"], tid, w)
        for ex in examples
        for tid, w in zip(ex["target_bank_ids"], ex["weights"])
    ]
    total = 0.0
    for start in range(0, len(flat), 16):
        batch = flat[start : start + 16]
        f = features[[r[0] for r in batch]].to(device)
        b = torch.tensor([r[1] for r in batch], device=device)
        t = payload["target_tokens"][[r[2] for r in batch]].to(device)
        logits = adapter.decode(t[:, :-1], adapter.memory(f, b))
        losses = F.cross_entropy(
            logits.transpose(1, 2), t[:, 1:], ignore_index=0, reduction="none"
        )
        losses = losses.sum(1) / t[:, 1:].ne(0).sum(1)
        total += float(
            (losses * torch.tensor([r[3] for r in batch], device=device)).sum()
        )
    return total / len(examples)


def sequence_logp(logits, targets):
    token_logp = logits.log_softmax(-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return (token_logp * targets.ne(0)).sum(-1)


def preference_loss(positive, negative, ref_positive, ref_negative, gaps=None, beta=0.1):
    """Uniform DPO. Gaps are accepted for historical call compatibility only."""
    logits = beta * (positive - negative - ref_positive + ref_negative)
    return -F.logsigmoid(logits).mean()


def legacy_weighted_preference_loss(positive, negative, ref_positive, ref_negative, gaps, beta=0.1):
    logits = beta * (positive - negative - ref_positive + ref_negative)
    weights = (gaps / 0.1).clamp(0.5, 2.0).detach()
    return (-F.logsigmoid(logits) * weights).sum() / weights.sum()


def negative_ids(scores, sizes, positive, budget, rng, k=4, gap_min=0.05, gap_max=0.25, hard_pool=256):
    gap = scores[positive] - scores
    valid = torch.where(
        (sizes == sizes[positive]) & (sizes <= budget) & (gap >= gap_min) & (gap <= gap_max)
    )[0]
    if not len(valid):
        return []
    hard = (
        valid[torch.argsort(scores[valid], descending=True, stable=True)[:hard_pool]]
        .cpu()
        .numpy()
    )
    return rng.choice(hard, min(k, len(hard)), replace=False).tolist()


def logs_for_pairs(adapter, features, pairs, tokens, idx, device):
    idx = torch.as_tensor(idx, dtype=torch.long)
    src = pairs["source_index"][idx]
    budgets = pairs["budget"][idx]
    target = torch.cat(
        [tokens[pairs["positive"][idx]], tokens[pairs["negative"][idx]]]
    ).to(device)
    f = torch.cat([features[src], features[src]]).to(device)
    b = torch.cat([budgets, budgets]).to(device)
    logits = adapter.decode(target[:, :-1], adapter.memory(f, b))
    return sequence_logp(logits, target[:, 1:]).chunk(2)


def selection_score(summary, baseline):
    s = summary.reindex(BUDGETS)
    b = baseline.reindex(BUDGETS)
    eligible = (
        s.cosine.notna().all()
        and (s.n_spectral >= np.maximum(10, 0.75 * b.n_spectral)).all()
        and (s.coverage >= 0.9 * b.coverage).all()
    )
    return (
        (float((s.cosine - b.cosine).min()), float(s.cosine.mean()))
        if eligible
        else (-float("inf"), -float("inf"))
    )


BUDGETS = (9, 11, 13)


def check_partitions(payload):
    bank = payload["bank"]
    panels = [
        bank,
        payload["sources"]["train"],
        payload["sources"]["val"],
        payload["test"],
    ]
    keys = []
    for panel in panels:
        actual = [key_and_size(smiles) for smiles in panel.smiles]
        if any(
            (
                not ok or key != row.key or size != row.ha
                for (key, size, ok), row in zip(actual, panel.itertuples(index=False))
            )
        ):
            raise ValueError("Identity, size or chemical eligibility mismatch")
        keyset = {key for key, _, _ in actual}
        if len(keyset) != len(panel):
            raise ValueError("Duplicate molecular identity in panel")
        keys.append(keyset)
    if any((keys[i] & keys[j] for i in range(4) for j in range(i))):
        raise ValueError("Cross-panel molecular identity leakage")
    limits = payload.get("size_limits", dict(small_min=1, small_max=13, large_min=18, large_max=0))
    if not 1 <= limits["small_min"] <= limits["small_max"] < limits["large_min"] or limits["small_max"] > 30:
        raise ValueError("Invalid source/target size limits or unsupported budget embedding range")
    if ((bank.ha < limits["small_min"]) | (bank.ha > limits["small_max"])).any() or any(
        ((panel.ha < limits["large_min"]) | ((panel.ha > limits["large_max"]) if limits["large_max"] else False)).any()
        for panel in panels[1:]
    ):
        raise ValueError("Unexpected source/target size range")
    if any(not 1 <= b <= 30 for b in payload.get("budgets", BUDGETS)):
        raise ValueError("Budget must be within the embedding range 1..30")


from rdkit import Chem


def key_and_size(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ("", 0, False)
    return (
        Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False),
        mol.GetNumHeavyAtoms(),
        Chem.GetFormalCharge(mol) == 0
        and (not any((a.GetNumRadicalElectrons() for a in mol.GetAtoms())))
        and (len(Chem.GetMolFrags(mol)) == 1),
    )
