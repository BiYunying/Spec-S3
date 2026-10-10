import re
from typing import Optional, List, Tuple
import torch
from torch.autograd import grad
from torch import nn
from torch_scatter import scatter
from pytorch_lightning.utilities import rank_zero_warn
from torchmdnet.models import heads as output_modules
from torchmdnet.models.heads import AtomFilter
from torchmdnet.models import heads as priors
import warnings
import torch.nn.functional as F
from torchmdnet.models.layers import generate_square_subsequent_mask, create_masks
from torch_geometric.utils import to_dense_batch
import math
from torchmdnet.utils import CustomTokenizer
from rdkit import Chem
import os
from rdkit.Chem import Draw
from transformers import AutoTokenizer, AutoModel
from pathlib import Path
from torchmdnet.tokenizer import get_smiles_dictionary, CanonicalNonIsomericFG26Tokenizer
from torchmdnet.models.heads import (
    OutputModel,
    Scalar,
    EquivariantScalar,
    DipoleMoment,
    EquivariantDipoleMoment,
    ElectronicSpatialExtent,
    EquivariantElectronicSpatialExtent,
    EquivariantVectorOutput,
    BaseWrapper,
    BasePrior,
    Atomref,
    MLPProjector,
    InfoNCELoss,
    SpectraFusion,
    CrossAttentionFusion,
    MultiModalFusion,
    PropertyPredictor,
    AccumulatedNormalization,
    PretrainMLPDecoder,
)
from torchmdnet.models.decoder import SMILES_Decoder, GPT, GPTConfig


def _get_arg(args, key, default=None):
    try:
        return args[key]
    except (KeyError, TypeError):
        return getattr(args, key, default)


def create_model(
    args,
    prior_model=None,
    mean=None,
    std=None,
    var_uv=None,
    var_ir=None,
    var_raman=None,
):
    """Assemble the selected task without changing historical tensor names."""
    from torchmdnet.utils import DEFAULTS

    config = dict(DEFAULTS)
    config.update(dict(args))
    task = config["task"]
    mapping = {
        "pretrain": "pretraining",
        "property_prediction": "regression",
        "translation": "smiles",
        "scaffold_editing": "hopping",
        "ir_editing": "hopping",
        "multiobjective": "A+B",
        "budget": "smiles",
    }
    config["task"] = mapping.get(task, task)
    if config["task"] == "regression" and config.get("property_mode", "structure_spectrum") != "structure_spectrum":
        raise ValueError("Spectrum-only property prediction was removed; use the 3D + spectrum model")
    if task == "ir_editing":
        config["modalities"] = "ir"
    if config["model"] not in ("equivariant-transformer", None):
        raise ValueError("Only the retained equivariant structure encoder is supported")
    if config.get("ab_decoder_type", "gpt") != "gpt":
        raise ValueError("Multi-objective generation uses the project GPT decoder")
    config["smiles_dictionary"] = get_smiles_dictionary(config.get("smiles_dictionary"))
    factory = (
        create_pretrain_model
        if config["task"] == "pretraining"
        else create_generation_model
    )
    model = factory(config, prior_model, mean, std, var_uv, var_ir, var_raman)
    if config.get("structure_checkpoint"):
        if config.get("resume_checkpoint"):
            model.structure_load_report = dict(
                mode="deferred_to_resume_checkpoint", checkpoint=config["resume_checkpoint"]
            )
        else:
            from torchmdnet.checkpoint import load_structure_weights

            # Load after constructors/reset_parameters have finished.
            load_structure_weights(model, config["structure_checkpoint"])
    return model


def create_pretrain_model(
    args,
    prior_model=None,
    mean=None,
    std=None,
    var_uv=None,
    var_ir=None,
    var_raman=None,
):
    shared_args = dict(
        hidden_channels=args["embedding_dimension"],
        num_layers=args["num_layers"],
        num_rbf=args["num_rbf"],
        rbf_type=args["rbf_type"],
        trainable_rbf=args["trainable_rbf"],
        activation=args["activation"],
        neighbor_embedding=args["neighbor_embedding"],
        cutoff_lower=args["cutoff_lower"],
        cutoff_upper=args["cutoff_upper"],
        max_z=args["max_z"],
        max_num_neighbors=args["max_num_neighbors"],
    )
    smiles_dictionary = get_smiles_dictionary(args.get("smiles_dictionary"))
    representation_model = None
    is_equivariant = True
    if args["model"] == "equivariant-transformer":
        from torchmdnet.models.structure import TorchMD_ET

        is_equivariant = True
        representation_model = TorchMD_ET(
            attn_activation=args["attn_activation"],
            num_heads=args["num_heads"],
            distance_influence=args["distance_influence"],
            layernorm_on_vec=args["layernorm_on_vec"],
            use_dataset_md17=args["use_dataset_md17"],
            **shared_args,
        )
    representation_spec_model = None
    if args["spectra_model"] == "SpecFormer":
        from torchmdnet.models.spectra import make_encoder

        representation_spec_model = make_encoder(
            modalities=args.get("modalities", "uv_ir_raman"),
            patch_len=args["patch_len"],
            stride=args["stride"],
            output_dim=args["embedding_dimension"],
            input_norm_type=args["input_data_norm_type"],
            spec_consistency_weight=_get_arg(args, "spec_consistency_weight", 0.0),
            var_raman=var_raman,
            var_uv=var_uv,
            var_ir=var_ir,
        )
    representation_smiles_model = None
    if args["smiles_model"] == "ChemBERTa":
        from torchmdnet.models.smiles import ChemBERTa_SMILESEncoder

        local_model_path = os.environ.get(
            "SPEC_S3_CHEMBERTA", str(Path(__file__).resolve().parents[2] / "datasets/ChemBERTa-77M-MTR-safetensors")
        )
        smiles_encoder_base = AutoModel.from_pretrained(local_model_path)
        representation_smiles_model = ChemBERTa_SMILESEncoder(
            base_model=smiles_encoder_base, projection_dim=256
        )
    representation_smiles_decoder = None
    if args["smiles_decoder"] == True:
        representation_smiles_decoder = SMILES_Decoder(
            vocab_size=len(smiles_dictionary),
            d_model=256,
            n_head=8,
            n_layers=6,
            d_ff=int(args.get("smiles_d_ff", 1024)),
            max_len=50,
        )
    if not args["derivative"] and args["atom_filter"] > -1:
        representation_model = AtomFilter(representation_model, args["atom_filter"])
    elif args["atom_filter"] > -1:
        raise ValueError("Derivative and atom filter can't be used together")
    if args["prior_model"] and prior_model is None:
        assert (
            "prior_args" in args
        ), f"""Requested prior model {args['prior_model']} but the arguments are lacking the key "prior_args"."""
        assert hasattr(
            priors, args["prior_model"]
        ), f"Unknown prior model {args['prior_model']}. Available models are {', '.join(priors.__all__)}"
        prior_model = getattr(priors, args["prior_model"])(**args["prior_args"])
    output_prefix = "Equivariant" if is_equivariant else ""
    output_model = getattr(output_modules, output_prefix + args["output_model"])(
        args["embedding_dimension"], args["activation"]
    )
    output_model_noise = None
    if args["output_model_noise"] is not None:
        output_model_noise = getattr(
            output_modules, output_prefix + args["output_model_noise"]
        )(args["embedding_dimension"], args["activation"])
    output_model_spec = None
    if args["output_model_spec"] is not None:
        output_model_spec = getattr(
            output_modules, output_prefix + args["output_model_spec"]
        )(args["embedding_dimension"], args["activation"])
    output_model_mol = None
    if args["output_model_mol"] is not None:
        output_model_mol = getattr(
            output_modules, output_prefix + args["output_model_mol"]
        )(args["embedding_dimension"], args["activation"])
    model = PretrainModel(
        representation_model,
        output_model,
        prior_model=prior_model,
        reduce_op=args["reduce_op"],
        mean=mean,
        std=std,
        derivative=args["derivative"],
        output_model_noise=output_model_noise,
        position_noise_scale=args["position_noise_scale"],
        representation_spec_model=representation_spec_model,
        output_model_spec=output_model_spec,
        output_model_mol=output_model_mol,
        d_model=args["embedding_dimension"],
        training_stage=args["training_stage"],
        representation_smiles_model=representation_smiles_model,
        representation_smiles_decoder=representation_smiles_decoder,
        smiles_dictionary=smiles_dictionary,
        task=args["task"],
    )
    return model


class PretrainModel(nn.Module):

    def __init__(
        self,
        representation_model,
        output_model,
        prior_model=None,
        reduce_op="add",
        mean=None,
        std=None,
        derivative=False,
        output_model_noise=None,
        position_noise_scale=0.0,
        representation_spec_model=None,
        output_model_spec=None,
        output_model_mol=None,
        fusion_model=None,
        d_model=256,
        training_stage="spectra",
        batch_size=100,
        representation_smiles_model=None,
        representation_smiles_decoder=None,
        task="pretraining",
        hidden_dim=1024,
        smiles_dictionary=None,
    ):
        super(PretrainModel, self).__init__()
        self.representation_model = representation_model
        self.output_model = output_model
        self.representation_spec_model = representation_spec_model
        self.batch_size = batch_size
        self.representation_smiles_model = representation_smiles_model
        self.task = task
        self.prior_model = prior_model
        if not output_model.allow_prior_model and prior_model is not None:
            self.prior_model = None
            rank_zero_warn(
                "Prior model was given but the output model does not allow prior models. Dropping the prior model."
            )
        self.reduce_op = reduce_op
        self.derivative = derivative
        self.output_model_noise = output_model_noise
        self.position_noise_scale = position_noise_scale
        self.output_model_spec = output_model_spec
        self.output_model_mol = output_model_mol
        self.training_stage = training_stage
        self.spec_projector1 = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
        )
        self.mol_projector1 = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
        )
        self.smiles_projector1 = nn.Sequential(
            nn.Linear(384, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
        )
        self.SMILES_DICTIONARY = get_smiles_dictionary(smiles_dictionary)
        local_model_path = os.environ.get(
            "SPEC_S3_CHEMBERTA", str(Path(__file__).resolve().parents[2] / "datasets/ChemBERTa-77M-MTR-safetensors")
        )
        self.smiles_tokenizer = (
            AutoTokenizer.from_pretrained(local_model_path)
            if representation_smiles_model is not None
            else None
        )
        self.decode_tokenizer = CustomTokenizer(vocab_dict=self.SMILES_DICTIONARY)
        mean = torch.scalar_tensor(0) if mean is None else mean
        self.register_buffer("mean", mean)
        std = torch.scalar_tensor(1) if std is None else std
        self.register_buffer("std", std)
        self.smiles_decoder = representation_smiles_decoder
        if self.position_noise_scale > 0:
            self.pos_normalizer = AccumulatedNormalization(accumulator_shape=(3,))
        self.reset_parameters()

    def reset_parameters(self):
        self.output_model.reset_parameters()
        if self.output_model_noise is not None:
            self.output_model_noise.reset_parameters()
        if self.representation_spec_model is not None:
            self.representation_spec_model.reset_parameters()
        if self.output_model_spec is not None:
            self.output_model_spec.reset_parameters()
        if self.output_model_mol is not None:
            self.output_model_mol.reset_parameters()
        if self.prior_model is not None:
            self.prior_model.reset_parameters()

    def forward(
        self,
        z=None,
        pos=None,
        spec_list=None,
        patch_masks=None,
        smiles=None,
        batch: Optional[torch.Tensor] = None,
        spec_clean_list=None,
    ):
        assert z.dim() == 1 and z.dtype == torch.long
        batch = torch.zeros_like(z) if batch is None else batch
        if self.derivative:
            pos.requires_grad_(True)
        mol_feature = None
        mol_feature_proj = None
        if self.representation_model is not None and self.task != "smiles":
            x, v, z, pos, batch = self.representation_model(z, pos, batch=batch)
            mol_feature = scatter(x, batch, dim=0, reduce=self.reduce_op)
            mol_feature_proj = self.mol_projector1(mol_feature)
        spec_feature_proj = None
        spec_feature = None
        loss_reconstruct = None
        smiles_feature = None
        smiles_feature_proj = None
        loss_reconstruct1 = None
        loss_reconstruct2 = None
        unique_abs_indices, relative_batch_indices = torch.unique(
            batch, return_inverse=True
        )
        batch_size = len(unique_abs_indices)
        if self.representation_spec_model is not None:
            if spec_list is not None:
                loss_reconstruct, spec_feature, loss_reconstruct1, loss_reconstruct2 = (
                    self.representation_spec_model(
                        spec_list,
                        patch_masks,
                        batch_size,
                        target_spectra=spec_clean_list,
                    )
                )
                spec_feature_proj = self.spec_projector1(spec_feature)
        if self.representation_smiles_model is not None:
            inputs = self.smiles_tokenizer(
                smiles,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=70,
            )
            smiles_ids, padding_mask = (inputs["input_ids"], inputs["attention_mask"])
            smiles_feature = self.representation_smiles_model(
                smiles_ids.to(z.device), attention_mask=padding_mask.to(z.device)
            )
            smiles_feature_proj = self.smiles_projector1(smiles_feature)
        noise_pred = None
        if (
            self.task == "pretraining"
            and self.representation_model is not None
            and (self.output_model_noise is not None)
        ):
            noise_pred = self.output_model_noise.pre_reduce(x, v, z, pos, batch)
        out = None
        smiles_logits = None
        smiles_labels = None
        if self.task == "regression":
            x = self.output_model.pre_reduce(x, v, z, pos, batch, spec_feature)
            if self.std is not None:
                x = x * self.std
            out = scatter(x, batch, dim=0, reduce=self.reduce_op)
            if self.mean is not None:
                out = out + self.mean
        elif self.task == "smiles":
            smiles_ids, padding_mask = self.decode_tokenizer(
                smiles,
                add_special_tokens=True,
                max_length=50,
                padding="max_length",
                truncation=True,
            )
            decoder_input = smiles_ids[:, :-1]
            smiles_labels = smiles_ids[:, 1:]
            tgt_mask, tgt_key_padding_mask = create_masks(decoder_input, pad_idx=0)
            memory = spec_feature.unsqueeze(1)
            smiles_logits = self.smiles_decoder(
                tgt=decoder_input,
                memory=memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
            )
        if self.derivative:
            grad_outputs: List[Optional[torch.Tensor]] = [torch.ones_like(out)]
            dy = grad(
                [out],
                [pos],
                grad_outputs=grad_outputs,
                create_graph=True,
                retain_graph=True,
            )[0]
            if dy is None:
                raise RuntimeError("Autograd returned None for the force prediction.")
            return (
                out,
                noise_pred,
                -dy,
                spec_feature_proj,
                mol_feature_proj,
                smiles_feature,
                smiles_logits,
                smiles_labels,
                loss_reconstruct,
                smiles_feature_proj,
                loss_reconstruct1,
                loss_reconstruct2,
            )
        return (
            out,
            noise_pred,
            None,
            spec_feature_proj,
            mol_feature_proj,
            smiles_feature,
            smiles_logits,
            smiles_labels,
            loss_reconstruct,
            smiles_feature_proj,
            loss_reconstruct1,
            loss_reconstruct2,
            spec_feature,
            mol_feature,
        )


def create_generation_model(
    args,
    prior_model=None,
    mean=None,
    std=None,
    var_uv=None,
    var_ir=None,
    var_raman=None,
):
    shared_args = dict(
        hidden_channels=args["embedding_dimension"],
        num_layers=args["num_layers"],
        num_rbf=args["num_rbf"],
        rbf_type=args["rbf_type"],
        trainable_rbf=args["trainable_rbf"],
        activation=args["activation"],
        neighbor_embedding=args["neighbor_embedding"],
        cutoff_lower=args["cutoff_lower"],
        cutoff_upper=args["cutoff_upper"],
        max_z=args["max_z"],
        max_num_neighbors=args["max_num_neighbors"],
    )
    if args["model"] == "equivariant-transformer":
        from torchmdnet.models.structure import TorchMD_ET

        is_equivariant = True
        representation_model = TorchMD_ET(
            attn_activation=args["attn_activation"],
            num_heads=args["num_heads"],
            distance_influence=args["distance_influence"],
            layernorm_on_vec=args["layernorm_on_vec"],
            use_dataset_md17=args["use_dataset_md17"],
            **shared_args,
        )
    else:
        raise ValueError(f"Unknown architecture: {args['model']}")
    representation_spec_model = None
    if args["spectra_model"] == "SpecFormer":
        from torchmdnet.models.spectra import make_encoder

        representation_spec_model = make_encoder(
            modalities=args.get("modalities", "uv_ir_raman"),
            patch_len=args["patch_len"],
            stride=args["stride"],
            output_dim=args["embedding_dimension"],
            input_norm_type=args["input_data_norm_type"],
            var_raman=var_raman,
            var_uv=var_uv,
            var_ir=var_ir,
        )
    fusion_model = None
    if not args["derivative"] and args["atom_filter"] > -1:
        representation_model = AtomFilter(representation_model, args["atom_filter"])
    elif args["atom_filter"] > -1:
        raise ValueError("Derivative and atom filter can't be used together")
    if args["prior_model"] and prior_model is None:
        assert (
            "prior_args" in args
        ), f"""Requested prior model {args['prior_model']} but the arguments are lacking the key "prior_args"."""
        assert hasattr(
            priors, args["prior_model"]
        ), f"Unknown prior model {args['prior_model']}. Available models are {', '.join(priors.__all__)}"
        prior_model = getattr(priors, args["prior_model"])(**args["prior_args"])
    output_prefix = "Equivariant" if is_equivariant else ""
    output_model = getattr(output_modules, output_prefix + args["output_model"])(
        args["embedding_dimension"], args["activation"]
    )
    output_model_noise = None
    if args["output_model_noise"] is not None:
        output_model_noise = getattr(
            output_modules, output_prefix + args["output_model_noise"]
        )(args["embedding_dimension"], args["activation"])
    output_model_spec = None
    if args["output_model_spec"] is not None:
        output_model_spec = getattr(
            output_modules, output_prefix + args["output_model_spec"]
        )(args["embedding_dimension"], args["activation"])
    output_model_mol = None
    if args["output_model_mol"] is not None:
        output_model_mol = getattr(
            output_modules, output_prefix + args["output_model_mol"]
        )(args["embedding_dimension"], args["activation"])
    model_class = (
        JointPropertyModel
        if args["task"] == "regression"
        else GenerationModel
    )
    model = model_class(
        representation_model,
        output_model,
        prior_model=prior_model,
        reduce_op=args["reduce_op"],
        mean=mean,
        std=std,
        derivative=args["derivative"],
        output_model_noise=output_model_noise,
        position_noise_scale=args["position_noise_scale"],
        representation_spec_model=representation_spec_model,
        output_model_spec=output_model_spec,
        output_model_mol=output_model_mol,
        fusion_model=fusion_model,
        d_model=args["embedding_dimension"],
        training_stage=args["training_stage"],
        args=args,
    )
    return model


class GenerationModel(nn.Module):

    def __init__(
        self,
        representation_model,
        output_model,
        prior_model=None,
        reduce_op="add",
        mean=None,
        std=None,
        derivative=False,
        output_model_noise=None,
        position_noise_scale=0.0,
        representation_spec_model=None,
        output_model_spec=None,
        output_model_mol=None,
        fusion_model=None,
        d_model=128,
        training_stage="spectra",
        batch_size=128,
        args=None,
    ):
        super(GenerationModel, self).__init__()
        self.args = args
        self.representation_model = representation_model
        self.output_model = output_model
        self.representation_spec_model = representation_spec_model
        self.batch_size = batch_size
        self.prior_model = prior_model
        if not output_model.allow_prior_model and prior_model is not None:
            self.prior_model = None
            rank_zero_warn(
                "Prior model was given but the output model does not allow prior models. Dropping the prior model."
            )
        self.reduce_op = reduce_op
        self.derivative = derivative
        self.output_model_noise = output_model_noise
        self.position_noise_scale = position_noise_scale
        self.output_model_spec = output_model_spec
        self.output_model_mol = output_model_mol
        self.training_stage = training_stage
        self.scaffold_maxlen = 0
        self.fusion_model = fusion_model
        self.contrastive_loss_fn = InfoNCELoss(temperature=0.1)
        self.projection_head = nn.Sequential(
            nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 128)
        )
        self.SMILES_DICTIONARY = get_smiles_dictionary(
            self._get_arg("smiles_dictionary", None)
        )
        self.tokenizer = (
            CanonicalNonIsomericFG26Tokenizer(self.SMILES_DICTIONARY)
            if self._get_arg("public_task", None) == "translation"
            else CustomTokenizer(vocab_dict=self.SMILES_DICTIONARY)
        )
        mean = torch.scalar_tensor(0) if mean is None else mean
        self.register_buffer("mean", mean)
        std = torch.scalar_tensor(1) if std is None else std
        self.register_buffer("std", std)
        self.smiles_decoder = SMILES_Decoder(
            vocab_size=len(self.SMILES_DICTIONARY),
            d_model=256,
            n_head=8,
            n_layers=6,
            d_ff=int(self._get_arg("smiles_d_ff", 1024)),
            max_len=70,
        )
        gpt_n_embd = int(self._get_arg("gpt_n_embd", None) or 256)
        mconf = GPTConfig(
            vocab_size=len(self.SMILES_DICTIONARY),
            block_size=int(self._get_arg("gpt_block_size", None) or 49),
            num_props=1,
            n_layer=int(self._get_arg("gpt_n_layer", None) or 8),
            n_head=int(self._get_arg("gpt_n_head", None) or 8),
            n_embd=gpt_n_embd,
        )
        self.hopping_decoder = GPT(mconf)
        if self.position_noise_scale > 0:
            self.pos_normalizer = AccumulatedNormalization(accumulator_shape=(3,))
        self.spec_projector = nn.Linear(d_model, gpt_n_embd)
        self.ab_decoder_type = str(self._get_arg("ab_decoder_type", "gpt")).lower()
        self.chem_gpt_tokenizer = None
        self.chem_gpt_decoder = None
        self.chem_gpt_spec_projector = None
        self.reset_parameters()
        self.mlp = MLPProjector(input_dim=256, output_dim=256)

    def _get_arg(self, key, default=None):
        if self.args is None:
            return default
        if isinstance(self.args, dict):
            return self.args.get(key, default)
        return getattr(self.args, key, default)

    def _normalize_gpt_smiles_target(self, smiles):
        from rdkit import Chem

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        try:
            Chem.Kekulize(mol, clearAromaticFlags=True)
        except Exception:
            pass
        normalized = Chem.MolToSmiles(
            mol, canonical=True, isomericSmiles=False, kekuleSmiles=True
        )
        tokens = self.tokenizer._tokenize(normalized)
        if "".join(tokens) != normalized:
            return None
        if any((token not in self.SMILES_DICTIONARY for token in tokens)):
            return None
        max_smiles_tokens = self.hopping_decoder.get_block_size() - 1
        if len(tokens) > max_smiles_tokens:
            return None
        return normalized

    def _prepare_gpt_smiles_targets(self, spec_feature, smiles):
        if isinstance(smiles, str):
            smiles = [smiles]
        target_smiles = []
        spec_indices = []
        for index, smi in enumerate(smiles):
            fragments = [fragment for fragment in str(smi).split(".") if fragment]
            if not fragments:
                fragments = [str(smi)]
            for fragment in fragments:
                normalized = self._normalize_gpt_smiles_target(fragment)
                if normalized is None:
                    continue
                target_smiles.append(normalized)
                spec_indices.append(index)
        if not target_smiles:
            raise ValueError(
                "No SMILES targets can be represented by the GPT decoder vocabulary."
            )
        spec_indices = torch.tensor(
            spec_indices, dtype=torch.long, device=spec_feature.device
        )
        return (spec_feature.index_select(0, spec_indices), target_smiles)

    def _gpt_smiles_loss_from_spectrum(self, spec_feature, smiles):
        spec_feature, smiles = self._prepare_gpt_smiles_targets(spec_feature, smiles)
        if isinstance(smiles, str):
            smiles = [smiles]
        max_length = self.hopping_decoder.get_block_size() + 1
        tokenized = self.tokenizer(
            smiles, max_length=max_length, padding="max_length", truncation=False
        )
        smiles_ids = tokenized["input_ids"]
        decoder_input = smiles_ids[:, :-1].to(spec_feature.device)
        smiles_labels = smiles_ids[:, 1:].to(spec_feature.device)
        spec_emb = self.spec_projector(spec_feature).unsqueeze(1)
        smiles_logits, smiles_loss = self.hopping_decoder(
            decoder_input, smiles_labels, spec_emb
        )
        return (smiles_logits, smiles_labels, smiles_loss)

    def reset_parameters(self):
        self.representation_model.reset_parameters()
        self.output_model.reset_parameters()
        if self.output_model_noise is not None:
            self.output_model_noise.reset_parameters()
        if self.output_model_spec is not None:
            self.output_model_spec.reset_parameters()
        if self.output_model_mol is not None:
            self.output_model_mol.reset_parameters()
        if self.prior_model is not None:
            self.prior_model.reset_parameters()

    def forward(
        self,
        z=None,
        pos=None,
        spec_list=None,
        patch_masks=None,
        smiles=None,
        task=None,
        batch: Optional[torch.Tensor] = None,
        epoch: Optional[int] = None,
        stage: Optional[str] = None,
    ):
        assert z.dim() == 1 and z.dtype == torch.long
        batch = torch.zeros_like(z) if batch is None else batch
        if self.derivative:
            pos.requires_grad_(True)
        mol_feature_normed = None
        if task != "smiles":
            x, v, z, pos, batch = self.representation_model(z, pos, batch=batch)
            mol_feature = scatter(x, batch, dim=0, reduce=self.reduce_op)
            mol_feature_normed = F.normalize(mol_feature, p=2, dim=1)
        spec_feature = None
        loss_reconstruct = None
        unique_abs_indices, relative_batch_indices = torch.unique(
            batch, return_inverse=True
        )
        batch_size = len(unique_abs_indices)
        if self.representation_spec_model is not None:
            if spec_list is not None:
                loss_reconstruct, spec_feature, _, _ = self.representation_spec_model(
                    spec_list, patch_masks, batch_size
                )
        if self.output_model_mol is not None:
            mol_feature = self.output_model_mol.pre_reduce(x, v, z, pos, batch)
            mol_feature = scatter(mol_feature, batch, dim=0, reduce=self.reduce_op)
        noise_pred = None
        if task != "smiles" and self.output_model_noise is not None:
            noise_pred = self.output_model_noise.pre_reduce(x, v, z, pos, batch)
        fusion_embed = None
        present_indices = [0, 1, 2]
        contrastive_loss = torch.tensor(0.0, device=z.device)
        out = None
        smiles_logits = None
        smiles_labels = None
        smiles_loss = None
        if task == "regression":
            x = self.output_model.pre_reduce(x, v, z, pos, batch, spec_feature)
            if self.std is not None:
                x = x * self.std
            out = scatter(x, batch, dim=0, reduce=self.reduce_op)
            if self.mean is not None:
                out = out + self.mean
        elif task == "smiles":
            target_options = {}
            if self._get_arg("public_task", None) == "translation":
                target_options["randomize"] = self.training and stage == "train"
            output = self.tokenizer(
                smiles, max_length=70, padding="max_length", truncation=True,
                **target_options,
            )
            smiles_ids, padding_mask = (output["input_ids"], output["attention_mask"])
            decoder_input = smiles_ids[:, :-1].to(spec_feature.device)
            smiles_labels = smiles_ids[:, 1:].to(spec_feature.device)
            tgt_mask, tgt_key_padding_mask = create_masks(decoder_input, pad_idx=0)
            memory = spec_feature.unsqueeze(1)
            smiles_logits = self.smiles_decoder(
                tgt=decoder_input,
                memory=memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                epoch=epoch,
                stage=stage,
            )
        elif task == "hopping":
            smiles_logits, smiles_labels, smiles_loss = (
                self._gpt_smiles_loss_from_spectrum(spec_feature, smiles)
            )
        elif task == "A+B":
            if self.ab_decoder_type == "chemgpt":
                smiles_logits, smiles_labels, smiles_loss = (
                    self._chem_gpt_loss_from_spectrum(spec_feature, smiles)
                )
            else:
                smiles_logits, smiles_labels, smiles_loss = (
                    self._gpt_smiles_loss_from_spectrum(spec_feature, smiles)
                )
        if self.derivative:
            grad_outputs: List[Optional[torch.Tensor]] = [torch.ones_like(out)]
            dy = grad(
                [out],
                [pos],
                grad_outputs=grad_outputs,
                create_graph=True,
                retain_graph=True,
            )[0]
            if dy is None:
                raise RuntimeError("Autograd returned None for the force prediction.")
            return (out, noise_pred, -dy, spec_feature, mol_feature, loss_reconstruct)
        return (
            out,
            noise_pred,
            None,
            spec_feature,
            mol_feature_normed,
            smiles_logits,
            smiles_labels,
            loss_reconstruct,
            contrastive_loss,
            smiles_loss,
        )


def _joint_property_forward(
    self, z, pos, spec_list, patch_masks, batch: Optional[torch.Tensor] = None
):
    assert z.dim() == 1 and z.dtype == torch.long
    batch = torch.zeros_like(z) if batch is None else batch
    if self.derivative:
        pos.requires_grad_(True)
    x, v, z, pos, batch = self.representation_model(z, pos, batch=batch)
    mol_feature = scatter(x, batch, dim=0, reduce=self.reduce_op)
    mol_feature_normed = F.normalize(mol_feature, p=2, dim=1)
    batch_size = mol_feature.shape[0]
    spec_feature = None
    loss_reconstruct = None
    if self.representation_spec_model is not None:
        if spec_list is not None:
            loss_reconstruct, spec_feature, loss_reconstruct1, loss_reconstruct2 = (
                self.representation_spec_model(spec_list, patch_masks, batch_size)
            )
    if self.output_model_mol is not None:
        mol_feature = self.output_model_mol.pre_reduce(x, v, z, pos, batch)
        mol_feature = scatter(mol_feature, batch, dim=0, reduce=self.reduce_op)
    noise_pred = None
    if self.output_model_noise is not None:
        noise_pred = self.output_model_noise.pre_reduce(x, v, z, pos, batch)
    out = None
    x = self.output_model.pre_reduce(x, v, z, pos, batch, spec_feature)
    if self.std is not None:
        x = x * self.std
    if self.prior_model is not None:
        x = self.prior_model(x, z, pos, batch)
    out = scatter(x, batch, dim=0, reduce=self.reduce_op)
    if self.mean is not None:
        out = out + self.mean
    if self.derivative:
        grad_outputs: List[Optional[torch.Tensor]] = [torch.ones_like(out)]
        dy = grad(
            [out],
            [pos],
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
        )[0]
        if dy is None:
            raise RuntimeError("Autograd returned None for the force prediction.")
        return (
            out,
            noise_pred,
            -dy,
            spec_feature,
            mol_feature_normed,
            loss_reconstruct,
        )
    return (
        out,
        noise_pred,
        None,
        spec_feature,
        mol_feature_normed,
        loss_reconstruct,
        loss_reconstruct1,
        loss_reconstruct2,
    )


class JointPropertyModel(GenerationModel):
    """Property prediction with the original structure-plus-spectrum head."""

    def forward(
        self,
        z,
        pos,
        spec_list,
        patch_masks,
        smiles=None,
        task="regression",
        batch=None,
        epoch=None,
        stage=None,
        spec_clean_list=None,
    ):
        result = _joint_property_forward(
            self, z, pos, spec_list, patch_masks, batch=batch
        )
        out, noise, derivative, features, molecule, reconstruction = result[:6]
        return (
            out,
            noise,
            derivative,
            features,
            molecule,
            None,
            None,
            reconstruction,
            features.new_zeros(()),
            None,
        )
