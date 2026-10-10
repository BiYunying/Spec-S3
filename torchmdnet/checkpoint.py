"""Explicit legacy tensor mapping and validated checkpoint loading."""

import hashlib
import re
from pathlib import Path

import torch


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_checkpoint(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def map_state_dict(state):
    mapped = {}
    for name, value in state.items():
        name = re.sub(r"^(?:module\.)?(?:model\.)?", "", name)
        name = name.replace("output_model_noise.0.", "output_model_noise.")
        if name in mapped:
            raise ValueError(f"Duplicate mapped checkpoint key: {name}")
        mapped[name] = value
    return mapped


def load_structure_weights(model, filepath):
    """Initialize the 3D encoder with an explicit, checked legacy mapping.

    Property heads, denoising heads, label statistics and all spectrum/SMILES
    tensors belong to the new task and are intentionally outside this transfer.
    """
    encoder = getattr(model, "representation_model", None)
    if encoder is None:
        raise ValueError("This model has no 3D representation_model to initialize")
    payload = read_checkpoint(filepath)
    state = map_state_dict(payload.get("state_dict", payload))
    prefix = "representation_model."
    source = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
    target = encoder.state_dict()
    missing = sorted(set(target) - set(source))
    # The retained project ET adds per-layer scalar/vector normalization to
    # the upstream denoising ET. Preserve that computation and its default
    # affine initialization; only these known additions may be absent.
    new_norms = [k for k in missing if re.fullmatch(
        r"(?:x_norms\.\d+\.(?:weight|bias)|vec_norms\.\d+\.weight)", k
    )]
    required_missing = sorted(set(missing) - set(new_norms))
    unexpected = sorted(set(source) - set(target))
    mismatch = [k for k in target if k in source and target[k].shape != source[k].shape]
    if required_missing or unexpected or mismatch or not source:
        raise RuntimeError(
            f"Incompatible 3D checkpoint: missing={required_missing}, unexpected={unexpected}, "
            f"shape_mismatch={mismatch}"
        )
    encoder.load_state_dict(source, strict=not new_norms)
    report = dict(
        mode="external_3d_initialization",
        checkpoint=str(filepath),
        sha256=sha256(filepath),
        loaded_tensors=len(source),
        missing_required=[], unexpected=[], shape_mismatch=[],
        initialized_project_norms=new_norms,
        ignored_non_encoder_tensors=len(state) - len(source),
    )
    model.structure_load_report = report
    return report


def infer_config(payload, overrides=None):
    from torchmdnet.tokenizer import get_smiles_dictionary
    from torchmdnet.utils import DEFAULTS

    config = dict(DEFAULTS)
    config.update(payload.get("hyper_parameters", payload.get("config", {})))
    # Legacy QM9S property checkpoints store the target name but not its index.
    # Recover the saved target before applying transfer-learning overrides.
    if (config.get("target_index") is None
            and str(config.get("dataset", "")).lower() in ("qm9", "qm9s", "qm9sp")
            and config.get("dataset_arg") is not None):
        from torch_geometric.nn.models.schnet import qm9_target_dict

        aliases = {"alpha": "isotropic_polarizability", "U": "energy_U", "U0": "energy_U0"}
        target_name = aliases.get(config["dataset_arg"], config["dataset_arg"])
        target_indices = {name: index for index, name in qm9_target_dict.items()}
        if target_name not in target_indices:
            raise ValueError(f"Unknown saved QM9S property target: {target_name}")
        config["target_index"] = target_indices[target_name]
    state = map_state_dict(payload.get("state_dict", payload))
    if config.get("prior_model") == "Atomref" and not config.get("prior_args"):
        if "prior_model.atomref" not in state:
            raise ValueError("Atomref checkpoint is missing its reference values")
        config["prior_args"] = {"max_z": state["prior_model.atomref"].shape[0]}
    # Historical cross-attention checkpoints omit this hyperparameter.
    # Recover their actual width so both old 1024 and new 512 models reload.
    ffn_key = "smiles_decoder.transformer_decoder.layers.0.linear1.weight"
    if ffn_key in state:
        config["smiles_d_ff"] = int(state[ffn_key].shape[0])
    if "hopping_decoder.pos_emb" in state:
        config["gpt_block_size"] = state["hopping_decoder.pos_emb"].shape[1]
        config["gpt_n_embd"] = state["hopping_decoder.pos_emb"].shape[2]
        layers = [
            int(m.group(1))
            for k in state
            if (m := re.match(r"hopping_decoder.blocks\.(\d+)\.", k))
        ]
        if layers:
            config["gpt_n_layer"] = max(layers) + 1
    decoder_sizes = {state[key].shape[0] for key in (
        "smiles_decoder.token_embedding.weight", "hopping_decoder.tok_emb.weight"
    ) if key in state}
    if len(decoder_sizes) > 1:
        raise ValueError("Checkpoint decoders have inconsistent vocabulary sizes")
    decoder_size = next(iter(decoder_sizes), None)
    source_vocab = get_smiles_dictionary(config.get("smiles_dictionary"), vocab_size=decoder_size)
    config["smiles_dictionary"] = source_vocab
    if any(k.startswith("representation_spec_model.backbone_uv.") for k in state):
        config["modalities"] = "uv_ir_raman"
    elif any(k.startswith("representation_spec_model.backbone_ir.") for k in state):
        config["modalities"] = "ir"
    config.update(overrides or {})
    config["smiles_dictionary"] = get_smiles_dictionary(config.get("smiles_dictionary"))
    if decoder_size is not None and config["smiles_dictionary"] != source_vocab:
        raise ValueError("Cannot change token IDs while loading decoder weights")
    config["pretrained_model"] = None
    config["load_model"] = None
    config["pretrained_smiles_decoder"] = None
    # Full Spec-S3 initialization/resume preserves its saved 3D branch. An
    # external initialization must never overwrite that branch while loading.
    config["structure_checkpoint"] = None
    return config


def load_model(
    filepath, args=None, device="cpu", mean=None, std=None, partial=False, **kwargs
):
    from torchmdnet.models.model import create_model

    payload = read_checkpoint(filepath)
    config = infer_config(payload, dict(args) if args is not None else None)
    model = create_model(
        config,
        mean=mean,
        std=std,
        **{k: v for k, v in kwargs.items() if k in ("prior_model", "var_uv", "var_ir", "var_raman")},
    )
    state = map_state_dict(payload.get("state_dict", payload))
    property_task = config["task"] in ("regression", "property_prediction")
    if partial and property_task:
        # New target/dataset: inherit representations, not source calibration.
        state = {k: v for k, v in state.items()
                 if k not in ("mean", "std") and not k.startswith("prior_model.")}
    if property_task:
        for key in ("mean", "std"):
            if key in state and state[key].numel() == 1:
                state[key] = state[key].reshape(model.state_dict()[key].shape)
    modality_transfer = None
    if partial and config.get("modalities") == "ir" and any(
        k.startswith("representation_spec_model.backbone_uv.") for k in state
    ):
        # The trimodal module stores IR at index 1; IR-only stores it at 0.
        # Transfer the actual IR patch embedding/reconstruction tensors, never UV.
        for component in ("encoders", "reconstruct_heads"):
            prefix = f"representation_spec_model.{component}."
            for name, value in list(state.items()):
                if name.startswith(prefix + "1."):
                    state[prefix + "0." + name[len(prefix + "1."):]] = value
        modality_transfer = "trimodal IR branch (index 1) -> IR-only (index 0)"
    # Task5 extends this deterministic buffer; preserve all saved positions.
    key = "smiles_decoder.positional_encoding.pe"
    if (
        key in state
        and hasattr(model, "smiles_decoder")
        and model.smiles_decoder is not None
    ):
        model.smiles_decoder.positional_encoding.pe = torch.empty_like(state[key])
    current = model.state_dict()
    mismatched = [
        k for k, v in state.items() if k in current and v.shape != current[k].shape
    ]
    usable = {
        k: v for k, v in state.items() if k in current and v.shape == current[k].shape
    }
    missing, unexpected = model.load_state_dict(usable, strict=False)
    task = config["task"]
    required = ["representation_spec_model."]
    if not partial:
        if task in ("smiles", "translation", "budget"):
            required += ["smiles_decoder."]
        elif task in (
            "A+B",
            "hopping",
            "multiobjective",
            "scaffold_editing",
            "ir_editing",
        ):
            required += ["hopping_decoder.", "spec_projector."]
        elif task in ("regression", "property_prediction"):
            required += ["output_model.", "representation_model."]
            if config.get("standardize"):
                required += ["mean", "std"]
            if config.get("prior_model"):
                required += ["prior_model."]
        else:
            required = [""]
    bad = [k for k in missing if k.startswith(tuple(required))]
    if bad or unexpected:
        raise RuntimeError(
            f"Incomplete checkpoint: missing={bad[:12]}, shape_mismatch={mismatched[:12]}"
        )
    if not usable:
        raise RuntimeError("No checkpoint tensors loaded")
    model.load_report = {
        "loaded_tensors": len(usable),
        "missing": missing,
        "shape_mismatch": mismatched,
        "unused": sorted(set(state) - set(usable)),
        "partial_initialization": partial,
        "modality_transfer": modality_transfer,
    }
    source_config = payload.get("hyper_parameters", payload.get("config", {}))
    source_dataset = str(source_config.get("dataset", "")).lower()
    target_dataset = str(config.get("dataset", "")).lower()
    adaptation = dict(source_dataset=source_dataset, target_dataset=target_dataset,
                      mode="preserved", reset_tensors=[])
    # Only a fresh FG26 -> QM9S pretraining initialization takes this branch.
    # Full loading, same-domain initialization, downstream tasks and resume
    # must retain the trained adapters/heads.
    if (partial and config.get("task") in ("pretrain", "pretraining")
            and target_dataset in ("qm9s", "qm9sp", "qm9")
            and not config.get("resume_checkpoint")):
        if source_dataset not in ("fg26", "qm9s", "qm9sp", "qm9"):
            raise ValueError("QM9S pretraining initialization requires a checkpoint with known FG26/QM9S dataset metadata")
        if source_dataset == "fg26":
            encoder = model.representation_spec_model
            if not hasattr(encoder, "reset_domain_adaptation"):
                raise ValueError("FG26 -> QM9S adaptation requires the trimodal spectral encoder")
            reset = encoder.reset_domain_adaptation()
            adaptation.update(mode="fg26_to_qm9s", reset_tensors=[
                "representation_spec_model." + name for name in reset
            ])
    model.domain_adaptation_report = adaptation
    model.load_report["domain_adaptation"] = adaptation
    model.public_config = config
    if getattr(model, "representation_model", None) is not None:
        model.structure_load_report = dict(
            mode="inherited_from_model_checkpoint", checkpoint=str(filepath),
            loaded_tensors=sum(k.startswith("representation_model.") for k in usable),
        )
    return model.to(device)


def load_decoder(model, path):
    payload = read_checkpoint(path)
    state = payload.get("decoder_state_dict", payload)
    state = {
        re.sub(r"^(?:model\.)?(?:hopping_decoder\.|decoder\.)", "", k): v
        for k, v in state.items()
    }
    model.hopping_decoder.load_state_dict(state, strict=True)


def parameter_counts(model):
    encoder = model.representation_spec_model
    return {
        "total": sum(p.numel() for p in model.parameters()),
        "spectral_encoder_excluding_reconstruction": sum(
            p.numel()
            for n, p in encoder.named_parameters()
            if not n.startswith("reconstruct_heads.")
        ),
    }
