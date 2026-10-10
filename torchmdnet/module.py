import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    ReduceLROnPlateau,
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
)
from torch.nn.functional import mse_loss, l1_loss, smooth_l1_loss
from pytorch_lightning import LightningModule
from math import inf
from torchmdnet.utils import LabelSmoothingCrossEntropy, ExactMatchAccuracy
from torchmdnet.models.model import create_model
from torchmdnet.checkpoint import load_model


class PlateauScheduler(ReduceLROnPlateau):

    def __init__(self, factor, patience):
        self.factor = factor
        self.patience = patience
        self.threshold = 0.0001
        self.mode = "min"
        self.threshold_mode = "rel"
        self.best = inf
        self.num_bad_epochs = None
        self.eps = 1e-08
        self.last_epoch = 0

    def step(self, metrics, epoch=None):
        current = float(metrics)
        self.last_epoch += 1
        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
        if self.num_bad_epochs > self.patience:
            self.num_bad_epochs = 0
            return self.factor
        return 1.0


class PretrainModule(LightningModule):

    def __init__(
        self,
        hparams,
        prior_model=None,
        mean=None,
        std=None,
        var_uv=None,
        var_ir=None,
        var_raman=None,
    ):
        super(PretrainModule, self).__init__()
        self.save_hyperparameters(hparams, ignore=["var_uv", "var_ir", "var_raman"])
        if self.hparams.get("resume_checkpoint"):
            # Lightning restores the complete saved state later; do not first
            # reinitialize from the launcher's default FG26 parent.
            self.model = create_model(
                self.hparams, prior_model, mean, std, var_uv, var_ir, var_raman
            )
            self.model.domain_adaptation_report = dict(
                mode="deferred_to_resume_checkpoint",
                checkpoint=self.hparams.resume_checkpoint, reset_tensors=[],
            )
        elif self.hparams.load_model:
            self.model = load_model(
                self.hparams.load_model,
                args=self.hparams,
                var_uv=var_uv,
                var_ir=var_ir,
                var_raman=var_raman,
            )
        elif self.hparams.pretrained_model:
            self.model = load_model(
                self.hparams.pretrained_model,
                args=self.hparams,
                partial=True,
                prior_model=prior_model,
                mean=mean,
                std=std,
                var_uv=var_uv,
                var_ir=var_ir,
                var_raman=var_raman,
            )
        else:
            self.model = create_model(
                self.hparams, prior_model, mean, std, var_uv, var_ir, var_raman
            )
        self.hparams["smiles_dictionary"] = dict(self.model.SMILES_DICTIONARY)
        self.ema = None
        self._reset_ema_dict()
        self.losses = None
        self._reset_losses_dict()
        self.last_epoch = 0
        self.lr_gen_scheduler = PlateauScheduler(
            factor=self.hparams.lr_factor, patience=self.hparams.lr_patience
        )
        self.val_loss = None
        self.last_train_loss = None

    def configure_optimizers(self):
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        if self.hparams.lr_schedule == "cosine":
            scheduler = CosineAnnealingLR(
                optimizer, self.hparams.lr_cosine_length, eta_min=1e-07
            )
            lr_scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        elif self.hparams.lr_schedule == "cosine_warmup":
            scheduler = CosineAnnealingWarmRestarts(
                optimizer, T_0=self.hparams.lr_cosine_length, T_mult=2, eta_min=1e-07
            )
            lr_scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        elif self.hparams.lr_schedule == "reduce_on_plateau":
            scheduler = ReduceLROnPlateau(
                optimizer,
                "min",
                factor=self.hparams.lr_factor,
                patience=self.hparams.lr_patience,
                min_lr=self.hparams.lr_min,
            )
            lr_scheduler = {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            }
        else:
            raise ValueError(f"Unknown lr_schedule: {self.hparams.lr_schedule}")
        return ([optimizer], [lr_scheduler])

    def forward(self, z, pos, spec, patch_masks, smiles, batch=None, spec_clean=None):
        return self.model(
            z, pos, spec, patch_masks, smiles, batch=batch, spec_clean_list=spec_clean
        )

    def training_step(self, batch, batch_idx):
        return self.step(batch, mse_loss, "train")

    def validation_step(self, batch, batch_idx, *args):
        if len(args) == 0 or (len(args) > 0 and args[0] == 0):
            val_loss = self.step(batch, mse_loss, "val")
            return val_loss
        return self.step(batch, l1_loss, "test")

    def test_step(self, batch, batch_idx):
        loss = self.step(batch, l1_loss, "test")
        return loss

    def step(self, batch, loss_fn, stage):
        with torch.set_grad_enabled(stage == "train" or self.hparams.derivative):
            if "spectra" in batch:
                spectra_clean = (
                    batch.spectra_clean if "spectra_clean" in batch else None
                )
                (
                    pred,
                    noise_pred,
                    deriv,
                    sp_feature,
                    molecule_feature,
                    smiles_feature,
                    smiles_logits,
                    smiles_labels,
                    loss_reconstruct,
                    smiles_contrastive_feature,
                    loss_reconstruct1,
                    loss_reconstruct2,
                    spec_emb,
                    mol_emb,
                ) = self(
                    batch.z,
                    batch.pos,
                    batch.spectra,
                    batch.patch_masks,
                    batch.smiles,
                    batch=batch.batch,
                    spec_clean=spectra_clean,
                )
        if loss_reconstruct is not None and self.hparams.reconstruct_weight > 0:
            self.losses[stage + "_reconstruct"].append(loss_reconstruct.detach())
        else:
            loss_reconstruct = 0
        if self.hparams.reconstruct_weight > 0:
            if loss_reconstruct1 is not None:
                self.losses[stage + "_reconstruct1"].append(loss_reconstruct1.detach())
            if loss_reconstruct2 is not None:
                self.losses[stage + "_reconstruct2"].append(loss_reconstruct2.detach())
        denoising_is_on = (
            "pos_target" in batch
            and self.hparams.denoising_weight > 0
            and (noise_pred is not None)
        )
        contrastive_is_on = (
            "spectra" in batch
            and self.hparams.sp_struct_contrastive_weight > 0
            and (sp_feature is not None)
        )
        loss_y, loss_dy, loss_pos = (0, 0, 0)
        loss_ctr_sp_smiles, loss_ctr_struct_smiles, loss_ctr_sp_struct = (0, 0, 0)
        if self.hparams.derivative:
            if "y" not in batch:
                deriv = deriv + pred.sum() * 0
            loss_dy = loss_fn(deriv, batch.dy)
            if stage in ["train", "val"] and self.hparams.ema_alpha_dy < 1:
                if self.ema[stage + "_dy"] is None:
                    self.ema[stage + "_dy"] = loss_dy.detach()
                loss_dy = (
                    self.hparams.ema_alpha_dy * loss_dy
                    + (1 - self.hparams.ema_alpha_dy) * self.ema[stage + "_dy"]
                )
                self.ema[stage + "_dy"] = loss_dy.detach()
            if self.hparams.force_weight > 0:
                self.losses[stage + "_dy"].append(loss_dy.detach())
        if "y" in batch and pred is not None:
            if noise_pred is not None and (not denoising_is_on):
                pred = pred + noise_pred.sum() * 0
            if batch.y.ndim == 1:
                batch.y = batch.y.unsqueeze(1)
            loss_y = loss_fn(pred, batch.y)
            if stage in ["train", "val"] and self.hparams.ema_alpha_y < 1:
                if self.ema[stage + "_y"] is None:
                    self.ema[stage + "_y"] = loss_y.detach()
                loss_y = (
                    self.hparams.ema_alpha_y * loss_y
                    + (1 - self.hparams.ema_alpha_y) * self.ema[stage + "_y"]
                )
                self.ema[stage + "_y"] = loss_y.detach()
            if self.hparams.energy_weight > 0:
                self.losses[stage + "_y"].append(loss_y.detach())
        if denoising_is_on:
            normalized_pos_target = self.model.pos_normalizer(batch.pos_target)
            loss_pos = loss_fn(noise_pred, normalized_pos_target)
            self.losses[stage + "_pos"].append(loss_pos.detach())
        if contrastive_is_on:
            loss_ctr_sp_struct = self.ctr_loss_fn(molecule_feature, sp_feature)
            self.losses[stage + "_sp_struct_contrast"].append(
                loss_ctr_sp_struct.detach()
            )
        if self.hparams.sp_smiles_contrastive_weight > 0:
            loss_ctr_sp_smiles = self.ctr_loss_fn(
                sp_feature, smiles_contrastive_feature
            )
            self.losses[stage + "_sp_smiles_contrast"].append(
                loss_ctr_sp_smiles.detach()
            )
        if self.hparams.struct_smiles_contrastive_weight > 0:
            loss_ctr_struct_smiles = self.ctr_loss_fn(
                molecule_feature.detach(), smiles_contrastive_feature
            )
            self.losses[stage + "_3d_smiles_contrast"].append(
                loss_ctr_struct_smiles.detach()
            )
        loss = (
            loss_y * self.hparams.energy_weight
            + loss_dy * self.hparams.force_weight
            + loss_pos * self.hparams.denoising_weight
            + loss_ctr_sp_struct * self.hparams.sp_struct_contrastive_weight
            + loss_reconstruct * self.hparams.reconstruct_weight
            + loss_ctr_sp_smiles * self.hparams.sp_smiles_contrastive_weight
            + loss_ctr_struct_smiles * self.hparams.struct_smiles_contrastive_weight
        )
        self.losses[stage].append(loss.detach())
        if stage == "train":
            train_metrics = {
                k + "_per_step": v[-1]
                for k, v in self.losses.items()
                if k.startswith("train") and len(v) > 0
            }
            train_metrics["lr_per_step"] = self.trainer.optimizers[0].param_groups[0][
                "lr"
            ]
            train_metrics["step"] = self.trainer.global_step
            train_metrics["batch_pos_mean"] = batch.pos.mean().item()
            self.log_dict(train_metrics, sync_dist=True)
        return loss

    def optimizer_step(self, *args, **kwargs):
        epoch = kwargs["epoch"] if "epoch" in kwargs else args[0]
        bach_idx = kwargs["batch_idx"] if "batch_idx" in kwargs else args[1]
        optimizer = kwargs["optimizer"] if "optimizer" in kwargs else args[2]
        if self.trainer.global_step < self.hparams.lr_warmup_steps:
            lr_scale = min(
                1.0,
                float(self.trainer.global_step + 1)
                / float(self.hparams.lr_warmup_steps),
            )
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.hparams.lr
        elif self.hparams.reduce_lr_when_bad:
            if (
                self.val_loss is not None
                and optimizer.param_groups[0]["lr"] > self.hparams.lr * 0.1
            ):
                lr_scale = self.lr_gen_scheduler.step(self.val_loss.item())
                self.val_loss = None
            else:
                lr_scale = 1.0
            for pg in optimizer.param_groups:
                pg["lr"] *= lr_scale
        super().optimizer_step(*args, **kwargs)
        optimizer.zero_grad()

    def on_train_epoch_end(self):
        dm = self.trainer.datamodule
        if hasattr(dm, "test_dataset") and len(dm.test_dataset) > 0:
            should_reset = (
                self.current_epoch % self.hparams.test_interval == 0
                or (self.current_epoch - 1) % self.hparams.test_interval == 0
            )

    def _safe_mean(self, losses):
        return torch.stack(losses).mean() if losses else torch.tensor(float("nan"))

    def on_validation_epoch_end(self):
        if not self.trainer.sanity_checking:
            result_dict = {
                "epoch": self.current_epoch,
                "lr": self.trainer.optimizers[0].param_groups[0]["lr"],
                "train_loss": self._safe_mean(self.losses.get("train", [])),
                "val_loss": self._safe_mean(self.losses.get("val", [])),
            }
            self.val_loss = result_dict["val_loss"]
            if len(self.losses["test"]) > 0:
                result_dict["test_loss"] = torch.stack(self.losses["test"]).mean()
            if len(self.losses["train_y"]) > 0 and len(self.losses["train_dy"]) > 0:
                result_dict["train_loss_y"] = torch.stack(self.losses["train_y"]).mean()
                result_dict["train_loss_dy"] = torch.stack(
                    self.losses["train_dy"]
                ).mean()
                result_dict["val_loss_y"] = torch.stack(self.losses["val_y"]).mean()
                result_dict["val_loss_dy"] = torch.stack(self.losses["val_dy"]).mean()
                if len(self.losses["test"]) > 0:
                    result_dict["test_loss_y"] = torch.stack(
                        self.losses["test_y"]
                    ).mean()
                    result_dict["test_loss_dy"] = torch.stack(
                        self.losses["test_dy"]
                    ).mean()
            if len(self.losses["train_y"]) > 0:
                result_dict["train_loss_y"] = torch.stack(self.losses["train_y"]).mean()
            if len(self.losses["val_y"]) > 0:
                result_dict["val_loss_y"] = torch.stack(self.losses["val_y"]).mean()
            if len(self.losses["test_y"]) > 0:
                result_dict["test_loss_y"] = torch.stack(self.losses["test_y"]).mean()
            if len(self.losses["train_pos"]) > 0:
                result_dict["train_loss_pos"] = torch.stack(
                    self.losses["train_pos"]
                ).mean()
            if len(self.losses["val_pos"]) > 0:
                result_dict["val_loss_pos"] = torch.stack(self.losses["val_pos"]).mean()
            if len(self.losses["test_pos"]) > 0:
                result_dict["test_loss_pos"] = torch.stack(
                    self.losses["test_pos"]
                ).mean()
            if len(self.losses["train_sp_struct_contrast"]) > 0:
                result_dict["train_loss_sp_struct_contrast"] = torch.stack(
                    self.losses["train_sp_struct_contrast"]
                ).mean()
            if len(self.losses["val_sp_struct_contrast"]) > 0:
                result_dict["val_loss_sp_struct_contrast"] = torch.stack(
                    self.losses["val_sp_struct_contrast"]
                ).mean()
            if len(self.losses["test_sp_struct_contrast"]) > 0:
                result_dict["test_loss_sp_struct_contrast"] = torch.stack(
                    self.losses["test_sp_struct_contrast"]
                ).mean()
            if len(self.losses["train_reconstruct"]) > 0:
                result_dict["train_loss_reconstruct"] = torch.stack(
                    self.losses["train_reconstruct"]
                ).mean()
            if len(self.losses["val_reconstruct"]) > 0:
                result_dict["val_loss_reconstruct"] = torch.stack(
                    self.losses["val_reconstruct"]
                ).mean()
            if len(self.losses["test_reconstruct"]) > 0:
                result_dict["test_loss_reconstruct"] = torch.stack(
                    self.losses["test_reconstruct"]
                ).mean()
            if len(self.losses.get("train_reconstruct1", [])) > 0:
                result_dict["train_loss_reconstruct1"] = torch.stack(
                    self.losses["train_reconstruct1"]
                ).mean()
            if len(self.losses.get("val_reconstruct1", [])) > 0:
                result_dict["val_loss_reconstruct1"] = torch.stack(
                    self.losses["val_reconstruct1"]
                ).mean()
            if len(self.losses.get("test_reconstruct1", [])) > 0:
                result_dict["test_loss_reconstruct1"] = torch.stack(
                    self.losses["test_reconstruct1"]
                ).mean()
            if len(self.losses.get("train_reconstruct2", [])) > 0:
                result_dict["train_loss_reconstruct2"] = torch.stack(
                    self.losses["train_reconstruct2"]
                ).mean()
            if len(self.losses.get("val_reconstruct2", [])) > 0:
                result_dict["val_loss_reconstruct2"] = torch.stack(
                    self.losses["val_reconstruct2"]
                ).mean()
            if len(self.losses.get("test_reconstruct2", [])) > 0:
                result_dict["test_loss_reconstruct2"] = torch.stack(
                    self.losses["test_reconstruct2"]
                ).mean()
            if len(self.losses.get("train_smiles_mlm", [])) > 0:
                result_dict["train_loss_smiles_mlm"] = torch.stack(
                    self.losses["train_smiles_mlm"]
                ).mean()
            if len(self.losses.get("val_smiles_mlm", [])) > 0:
                result_dict["val_loss_smiles_mlm"] = torch.stack(
                    self.losses["val_smiles_mlm"]
                ).mean()
            if len(self.losses.get("test_smiles_mlm", [])) > 0:
                result_dict["test_loss_smiles_mlm"] = torch.stack(
                    self.losses["test_smiles_mlm"]
                ).mean()
            if len(self.losses.get("train_sp_smiles_contrast", [])) > 0:
                result_dict["train_loss_sp_smiles_contrast"] = torch.stack(
                    self.losses["train_sp_smiles_contrast"]
                ).mean()
            if len(self.losses.get("val_sp_smiles_contrast", [])) > 0:
                result_dict["val_loss_sp_smiles_contrast"] = torch.stack(
                    self.losses["val_sp_smiles_contrast"]
                ).mean()
            if len(self.losses.get("test_sp_smiles_contrast", [])) > 0:
                result_dict["test_loss_sp_smiles_contrast"] = torch.stack(
                    self.losses["test_sp_smiles_contrast"]
                ).mean()
            if len(self.losses.get("train_3d_smiles_contrast", [])) > 0:
                result_dict["train_loss_3d_smiles_contrast"] = torch.stack(
                    self.losses["train_3d_smiles_contrast"]
                ).mean()
            if len(self.losses.get("val_3d_smiles_contrast", [])) > 0:
                result_dict["val_loss_3d_smiles_contrast"] = torch.stack(
                    self.losses["val_3d_smiles_contrast"]
                ).mean()
            if len(self.losses.get("test_3d_smiles_contrast", [])) > 0:
                result_dict["test_loss_3d_smiles_contrast"] = torch.stack(
                    self.losses["test_3d_smiles_contrast"]
                ).mean()
            self.log_dict(result_dict, sync_dist=True)
        self._reset_losses_dict()

    def _reset_losses_dict(self):
        self.losses = {
            "train": [],
            "val": [],
            "test": [],
            "train_y": [],
            "val_y": [],
            "test_y": [],
            "train_dy": [],
            "val_dy": [],
            "test_dy": [],
            "train_pos": [],
            "val_pos": [],
            "test_pos": [],
            "train_sp_struct_contrast": [],
            "val_sp_struct_contrast": [],
            "test_sp_struct_contrast": [],
            "train_reconstruct": [],
            "val_reconstruct": [],
            "test_reconstruct": [],
            "train_reconstruct1": [],
            "val_reconstruct1": [],
            "test_reconstruct1": [],
            "train_reconstruct2": [],
            "val_reconstruct2": [],
            "test_reconstruct2": [],
            "train_smiles_mlm": [],
            "val_smiles_mlm": [],
            "test_smiles_mlm": [],
            "train_sp_smiles_contrast": [],
            "val_sp_smiles_contrast": [],
            "test_sp_smiles_contrast": [],
            "train_3d_smiles_contrast": [],
            "val_3d_smiles_contrast": [],
            "test_3d_smiles_contrast": [],
        }

    def _reset_ema_dict(self):
        self.ema = {"train_y": None, "val_y": None, "train_dy": None, "val_dy": None}

    def ctr_loss_fn(self, molecule_feature, sp_feature, temperature=0.07):
        from torch.nn import functional as F

        cos_sim = F.cosine_similarity(
            molecule_feature[:, None, :], sp_feature[None, :, :], dim=-1
        )
        postive_mask = torch.eye(
            cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device
        )
        cos_sim = cos_sim / temperature
        nll = -cos_sim[postive_mask] + torch.logsumexp(cos_sim, dim=-1)
        nll = nll.mean()
        return nll


class DownstreamModule(LightningModule):

    def __init__(
        self,
        hparams,
        prior_model=None,
        mean=None,
        std=None,
        var_uv=None,
        var_ir=None,
        var_raman=None,
    ):
        super(DownstreamModule, self).__init__()
        self.save_hyperparameters(hparams, ignore=["var_uv", "var_ir", "var_raman"])
        if self.hparams.load_model:
            self.model = load_model(
                self.hparams.load_model,
                args=self.hparams,
                var_uv=var_uv,
                var_ir=var_ir,
                var_raman=var_raman,
            )
        elif self.hparams.pretrained_model and not self.hparams.get("resume_checkpoint"):
            self.model = load_model(
                self.hparams.pretrained_model,
                args=self.hparams,
                partial=True,
                prior_model=prior_model,
                mean=mean,
                std=std,
                var_uv=var_uv,
                var_ir=var_ir,
                var_raman=var_raman,
            )
        else:
            self.model = create_model(
                self.hparams, prior_model, mean, std, var_uv, var_ir, var_raman
            )
        self.hparams["smiles_dictionary"] = dict(self.model.SMILES_DICTIONARY)
        self.ema = None
        self._reset_ema_dict()
        self.losses = None
        self._reset_losses_dict()
        self.last_epoch = 0
        self.lr_gen_scheduler = PlateauScheduler(
            factor=self.hparams.lr_factor, patience=self.hparams.lr_patience
        )
        self.val_loss = None
        self.last_train_loss = None
        self.task = self.hparams.task
        if self.hparams.task == "smiles" and self.hparams.dataset == "FG26":
            from torchmdnet.tokenizer import CanonicalNonIsomericFG26Tokenizer

            self.model.tokenizer = CanonicalNonIsomericFG26Tokenizer(
                self.model.SMILES_DICTIONARY
            )
        self.criterion_smiles = LabelSmoothingCrossEntropy(smoothing=0.1, pad_idx=0)
        self.accuracy_fn = ExactMatchAccuracy(
            pad_idx=0, vocab_dict=getattr(self.model, "SMILES_DICTIONARY", None)
        )

    def configure_optimizers(self):
        if (
            self.hparams.task == "A+B"
            and getattr(self.hparams, "ab_decoder_type", "gpt") == "gpt"
            and getattr(self.hparams, "ab_finetune_gpt_decoder", False)
        ):
            decoder_params = [
                param
                for param in self.model.hopping_decoder.parameters()
                if param.requires_grad
            ]
            decoder_lr = self.hparams.lr * getattr(
                self.hparams, "ab_gpt_decoder_lr_scale", 0.1
            )
            decoder_param_ids = {id(param) for param in decoder_params}
            base_params = [
                param
                for param in self.model.parameters()
                if param.requires_grad and id(param) not in decoder_param_ids
            ]
            param_groups = []
            if base_params:
                param_groups.append(
                    {
                        "params": base_params,
                        "lr": self.hparams.lr,
                        "base_lr": self.hparams.lr,
                        "weight_decay": self.hparams.weight_decay,
                    }
                )
            if decoder_params:
                param_groups.append(
                    {
                        "params": decoder_params,
                        "lr": decoder_lr,
                        "base_lr": decoder_lr,
                        "weight_decay": self.hparams.weight_decay,
                    }
                )
            optimizer = AdamW(param_groups)
        else:
            optimizer = AdamW(
                self.model.parameters(),
                lr=self.hparams.lr,
                weight_decay=self.hparams.weight_decay,
            )
            for param_group in optimizer.param_groups:
                param_group.setdefault("base_lr", self.hparams.lr)
        if self.hparams.lr_schedule == "cosine":
            scheduler = CosineAnnealingLR(
                optimizer, self.hparams.lr_cosine_length, eta_min=1e-07
            )
            lr_scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        elif self.hparams.lr_schedule == "cosine_warmup":
            scheduler = CosineAnnealingWarmRestarts(
                optimizer, T_0=self.hparams.lr_cosine_length, T_mult=2, eta_min=1e-07
            )
            lr_scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        elif self.hparams.lr_schedule == "reduce_on_plateau":
            scheduler = ReduceLROnPlateau(
                optimizer,
                "min",
                factor=self.hparams.lr_factor,
                patience=self.hparams.lr_patience,
                min_lr=self.hparams.lr_min,
            )
            lr_scheduler = {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            }
        else:
            raise ValueError(f"Unknown lr_schedule: {self.hparams.lr_schedule}")
        return ([optimizer], [lr_scheduler])

    def forward(
        self,
        z,
        pos,
        spec,
        patch_masks,
        smiles_tokens,
        task,
        batch=None,
        epoch=None,
        stage=None,
    ):
        return self.model(
            z,
            pos,
            spec,
            patch_masks,
            smiles_tokens,
            task,
            batch=batch,
            epoch=epoch,
            stage=stage,
        )

    def training_step(self, batch, batch_idx):
        return self.step(batch, mse_loss, "train")

    def validation_step(self, batch, batch_idx, *args):
        if len(args) == 0 or (len(args) > 0 and args[0] == 0):
            val_loss = self.step(batch, mse_loss, "val")
            return val_loss
        return self.step(batch, l1_loss, "test")

    def test_step(self, batch, batch_idx):
        loss = self.step(batch, l1_loss, "test")
        return loss

    def step(self, batch, loss_fn, stage):
        smiles_model_loss = None
        with torch.set_grad_enabled(stage == "train" or self.hparams.derivative):
            if "spectra" in batch:
                (
                    pred,
                    noise_pred,
                    deriv,
                    sp_feature,
                    molecule_feature,
                    smiles_logits,
                    smiles_labels,
                    loss_reconstruct,
                    loss_ctr,
                    smiles_model_loss,
                ) = self(
                    batch.z,
                    batch.pos,
                    batch.spectra,
                    batch.patch_masks,
                    batch.smiles,
                    self.task,
                    batch.batch,
                    self.current_epoch,
                    stage,
                )
            elif "ir" in batch and "h_nmr" in batch and ("c_nmr" in batch):
                (
                    pred,
                    noise_pred,
                    deriv,
                    sp_feature,
                    molecule_feature,
                    loss_reconstruct,
                ) = self(
                    batch.z,
                    batch.pos,
                    [batch.ir, batch.h_nmr, batch.c_nmr],
                    batch.batch,
                )
            else:
                (
                    pred,
                    noise_pred,
                    deriv,
                    sp_feature,
                    molecule_feature,
                    loss_reconstruct,
                ) = self(batch.z, batch.pos, None, batch.batch)
        if loss_reconstruct is not None and self.hparams.reconstruct_weight > 0:
            self.losses[stage + "_reconstruct"].append(loss_reconstruct.detach())
        else:
            loss_reconstruct = 0
        denoising_is_on = (
            "pos_target" in batch
            and self.hparams.denoising_weight > 0
            and (noise_pred is not None)
        )
        contrastive_is_on = (
            "spectra" in batch
            and self.hparams.contrastive_weight > 0
            and (sp_feature is not None)
        )
        loss_y, loss_dy, loss_pos = (0, 0, 0)
        loss_smiles = 0
        if smiles_model_loss is not None:
            loss_smiles = smiles_model_loss
            self.losses[stage + "_smiles"].append(loss_smiles.detach())
            if smiles_logits is not None and smiles_labels is not None:
                chem_acc, str_acc = self._smiles_accuracy(smiles_logits, smiles_labels)
                self.losses[stage + "_smiles_accuracy"].append(str_acc)
                self.losses[stage + "_chem_accuracy"].append(chem_acc)
        elif smiles_logits is not None and smiles_labels is not None:
            loss_smiles = self.criterion_smiles(smiles_logits, smiles_labels)
            chem_acc, str_acc = self._smiles_accuracy(smiles_logits, smiles_labels)
            self.losses[stage + "_smiles"].append(loss_smiles.detach())
            self.losses[stage + "_smiles_accuracy"].append(str_acc)
            self.losses[stage + "_chem_accuracy"].append(chem_acc)
        if self.hparams.derivative:
            if "y" not in batch:
                deriv = deriv + pred.sum() * 0
            loss_dy = loss_fn(deriv, batch.dy)
            if stage in ["train", "val"] and self.hparams.ema_alpha_dy < 1:
                if self.ema[stage + "_dy"] is None:
                    self.ema[stage + "_dy"] = loss_dy.detach()
                loss_dy = (
                    self.hparams.ema_alpha_dy * loss_dy
                    + (1 - self.hparams.ema_alpha_dy) * self.ema[stage + "_dy"]
                )
                self.ema[stage + "_dy"] = loss_dy.detach()
            if self.hparams.force_weight > 0:
                self.losses[stage + "_dy"].append(loss_dy.detach())
        if "y" in batch and pred is not None:
            if noise_pred is not None and (not denoising_is_on):
                pred = pred + noise_pred.sum() * 0
            if batch.y.ndim == 1:
                batch.y = batch.y.unsqueeze(1)
            loss_y = loss_fn(pred, batch.y)
            if stage in ["train", "val"] and self.hparams.ema_alpha_y < 1:
                if self.ema[stage + "_y"] is None:
                    self.ema[stage + "_y"] = loss_y.detach()
                loss_y = (
                    self.hparams.ema_alpha_y * loss_y
                    + (1 - self.hparams.ema_alpha_y) * self.ema[stage + "_y"]
                )
                self.ema[stage + "_y"] = loss_y.detach()
            if self.hparams.energy_weight > 0:
                self.losses[stage + "_y"].append(loss_y.detach())
        if denoising_is_on:
            if "y" not in batch:
                noise_pred = noise_pred + pred.sum() * 0
            normalized_pos_target = self.model.pos_normalizer(batch.pos_target)
            loss_pos = loss_fn(noise_pred, normalized_pos_target)
            self.losses[stage + "_pos"].append(loss_pos.detach())
        if contrastive_is_on:
            loss_ctr = self.ctr_loss_fn(molecule_feature, sp_feature)
            self.losses[stage + "_contrast"].append(loss_ctr.detach())
        loss = (
            loss_y * self.hparams.energy_weight
            + loss_dy * self.hparams.force_weight
            + loss_pos * self.hparams.denoising_weight
            + loss_ctr * self.hparams.contrastive_weight
            + loss_reconstruct * self.hparams.reconstruct_weight
            + loss_smiles * self.hparams.smiles_weight
        )
        self.losses[stage].append(loss.detach())
        if stage == "train":
            train_metrics = {
                k + "_per_step": v[-1]
                for k, v in self.losses.items()
                if k.startswith("train") and len(v) > 0
            }
            train_metrics["lr_per_step"] = self.trainer.optimizers[0].param_groups[0][
                "lr"
            ]
            if len(self.trainer.optimizers[0].param_groups) > 1:
                train_metrics["lr_decoder_per_step"] = self.trainer.optimizers[
                    0
                ].param_groups[1]["lr"]
            train_metrics["step"] = self.trainer.global_step
            train_metrics["batch_pos_mean"] = batch.pos.mean().item()
            self.log_dict(train_metrics, sync_dist=True)
        return loss

    def optimizer_step(self, *args, **kwargs):
        epoch = kwargs["epoch"] if "epoch" in kwargs else args[0]
        bach_idx = kwargs["batch_idx"] if "batch_idx" in kwargs else args[1]
        optimizer = kwargs["optimizer"] if "optimizer" in kwargs else args[2]
        if self.trainer.global_step < self.hparams.lr_warmup_steps:
            lr_scale = min(
                1.0,
                float(self.trainer.global_step + 1)
                / float(self.hparams.lr_warmup_steps),
            )
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * pg.get("base_lr", self.hparams.lr)
        elif self.hparams.reduce_lr_when_bad:
            if (
                self.val_loss is not None
                and optimizer.param_groups[0]["lr"] > self.hparams.lr * 0.1
            ):
                lr_scale = self.lr_gen_scheduler.step(self.val_loss.item())
                self.val_loss = None
            else:
                lr_scale = 1.0
            for pg in optimizer.param_groups:
                pg["lr"] *= lr_scale
        super().optimizer_step(*args, **kwargs)
        optimizer.zero_grad()

    def on_train_epoch_end(self):
        dm = self.trainer.datamodule
        if hasattr(dm, "test_dataset") and len(dm.test_dataset) > 0:
            should_reset = (
                self.current_epoch % self.hparams.test_interval == 0
                or (self.current_epoch - 1) % self.hparams.test_interval == 0
            )

    def _safe_mean(self, losses):
        return torch.stack(losses).mean() if losses else torch.tensor(float("nan"))

    def on_validation_epoch_end(self):
        if not self.trainer.sanity_checking:
            result_dict = {
                "epoch": self.current_epoch,
                "lr": self.trainer.optimizers[0].param_groups[0]["lr"],
                "train_loss": self._safe_mean(self.losses.get("train", [])),
                "val_loss": self._safe_mean(self.losses.get("val", [])),
            }
            self.val_loss = result_dict["val_loss"]
            if len(self.losses["test"]) > 0:
                result_dict["test_loss"] = torch.stack(self.losses["test"]).mean()
            if len(self.losses["train_y"]) > 0 and len(self.losses["train_dy"]) > 0:
                result_dict["train_loss_y"] = torch.stack(self.losses["train_y"]).mean()
                result_dict["train_loss_dy"] = torch.stack(
                    self.losses["train_dy"]
                ).mean()
                result_dict["val_loss_y"] = torch.stack(self.losses["val_y"]).mean()
                result_dict["val_loss_dy"] = torch.stack(self.losses["val_dy"]).mean()
                if len(self.losses["test"]) > 0:
                    result_dict["test_loss_y"] = torch.stack(
                        self.losses["test_y"]
                    ).mean()
                    result_dict["test_loss_dy"] = torch.stack(
                        self.losses["test_dy"]
                    ).mean()
            if len(self.losses["train_y"]) > 0:
                result_dict["train_loss_y"] = torch.stack(self.losses["train_y"]).mean()
            if len(self.losses["val_y"]) > 0:
                result_dict["val_loss_y"] = torch.stack(self.losses["val_y"]).mean()
            if len(self.losses["test_y"]) > 0:
                result_dict["test_loss_y"] = torch.stack(self.losses["test_y"]).mean()
            if len(self.losses["train_smiles"]) > 0:
                result_dict["train_loss_smiles"] = torch.stack(
                    self.losses["train_smiles"]
                ).mean()
            if len(self.losses["val_smiles"]) > 0:
                result_dict["val_loss_smiles"] = torch.stack(
                    self.losses["val_smiles"]
                ).mean()
            if len(self.losses["test_smiles"]) > 0:
                result_dict["test_loss_smiles"] = torch.stack(
                    self.losses["test_smiles"]
                ).mean()
            if len(self.losses["train_pos"]) > 0:
                result_dict["train_loss_pos"] = torch.stack(
                    self.losses["train_pos"]
                ).mean()
            if len(self.losses["val_pos"]) > 0:
                result_dict["val_loss_pos"] = torch.stack(self.losses["val_pos"]).mean()
            if len(self.losses["test_pos"]) > 0:
                result_dict["test_loss_pos"] = torch.stack(
                    self.losses["test_pos"]
                ).mean()
            if len(self.losses["train_contrast"]) > 0:
                result_dict["train_loss_contrast"] = torch.stack(
                    self.losses["train_contrast"]
                ).mean()
            if len(self.losses["val_contrast"]) > 0:
                result_dict["val_loss_contrast"] = torch.stack(
                    self.losses["val_contrast"]
                ).mean()
            if len(self.losses["test_contrast"]) > 0:
                result_dict["test_loss_contrast"] = torch.stack(
                    self.losses["test_contrast"]
                ).mean()
            if len(self.losses["train_reconstruct"]) > 0:
                result_dict["train_loss_reconstruct"] = torch.stack(
                    self.losses["train_reconstruct"]
                ).mean()
            if len(self.losses["val_reconstruct"]) > 0:
                result_dict["val_loss_reconstruct"] = torch.stack(
                    self.losses["val_reconstruct"]
                ).mean()
            if len(self.losses["test_reconstruct"]) > 0:
                result_dict["test_loss_reconstruct"] = torch.stack(
                    self.losses["test_reconstruct"]
                ).mean()
            if len(self.losses.get("train_reconstruct1", [])) > 0:
                result_dict["train_loss_reconstruct1"] = torch.stack(
                    self.losses["train_reconstruct1"]
                ).mean()
            if len(self.losses.get("val_reconstruct1", [])) > 0:
                result_dict["val_loss_reconstruct1"] = torch.stack(
                    self.losses["val_reconstruct1"]
                ).mean()
            if len(self.losses.get("test_reconstruct1", [])) > 0:
                result_dict["test_loss_reconstruct1"] = torch.stack(
                    self.losses["test_reconstruct1"]
                ).mean()
            if len(self.losses.get("train_reconstruct2", [])) > 0:
                result_dict["train_loss_reconstruct2"] = torch.stack(
                    self.losses["train_reconstruct2"]
                ).mean()
            if len(self.losses.get("val_reconstruct2", [])) > 0:
                result_dict["val_loss_reconstruct2"] = torch.stack(
                    self.losses["val_reconstruct2"]
                ).mean()
            if len(self.losses.get("test_reconstruct2", [])) > 0:
                result_dict["test_loss_reconstruct2"] = torch.stack(
                    self.losses["test_reconstruct2"]
                ).mean()
            if len(self.losses.get("train_smiles_accuracy", [])) > 0:
                result_dict["train_smiles_accuracy"] = sum(
                    self.losses["train_smiles_accuracy"]
                ) / len(self.losses["train_smiles_accuracy"])
            if len(self.losses.get("val_smiles_accuracy", [])) > 0:
                result_dict["val_smiles_accuracy"] = sum(
                    self.losses["val_smiles_accuracy"]
                ) / len(self.losses["val_smiles_accuracy"])
            if len(self.losses.get("test_smiles_accuracy", [])) > 0:
                result_dict["test_smiles_accuracy"] = sum(
                    self.losses["test_smiles_accuracy"]
                ) / len(self.losses["test_smiles_accuracy"])
            if len(self.losses.get("train_chem_accuracy", [])) > 0:
                result_dict["train_chem_accuracy"] = sum(
                    self.losses["train_chem_accuracy"]
                ) / len(self.losses["train_chem_accuracy"])
            if len(self.losses.get("val_chem_accuracy", [])) > 0:
                result_dict["val_chem_accuracy"] = sum(
                    self.losses["val_chem_accuracy"]
                ) / len(self.losses["val_chem_accuracy"])
            if len(self.losses.get("test_chem_accuracy", [])) > 0:
                result_dict["test_chem_accuracy"] = sum(
                    self.losses["test_chem_accuracy"]
                ) / len(self.losses["test_chem_accuracy"])
            self.log_dict(result_dict, sync_dist=True)
        self._reset_losses_dict()

    def _reset_losses_dict(self):
        self.losses = {
            "train": [],
            "val": [],
            "test": [],
            "train_y": [],
            "val_y": [],
            "test_y": [],
            "train_dy": [],
            "val_dy": [],
            "test_dy": [],
            "train_smiles": [],
            "val_smiles": [],
            "test_smiles": [],
            "train_pos": [],
            "val_pos": [],
            "test_pos": [],
            "train_contrast": [],
            "val_contrast": [],
            "test_contrast": [],
            "train_reconstruct": [],
            "val_reconstruct": [],
            "test_reconstruct": [],
            "train_reconstruct1": [],
            "val_reconstruct1": [],
            "test_reconstruct1": [],
            "train_reconstruct2": [],
            "val_reconstruct2": [],
            "test_reconstruct2": [],
            "train_smiles_accuracy": [],
            "val_smiles_accuracy": [],
            "test_smiles_accuracy": [],
            "train_chem_accuracy": [],
            "val_chem_accuracy": [],
            "test_chem_accuracy": [],
        }

    def _reset_ema_dict(self):
        self.ema = {"train_y": None, "val_y": None, "train_dy": None, "val_dy": None}

    def _smiles_accuracy(self, logits, labels):
        from rdkit import Chem

        tokenizer = getattr(self.model, "chem_gpt_tokenizer", None)
        if tokenizer is not None:
            pred_tokens = torch.argmax(logits[:, :-1, :], dim=-1)
            target_tokens = labels[:, 1:]
            valid_mask = target_tokens != -100
            pad_token_id = tokenizer.pad_token_id

            def decode(batch_tokens):
                return tokenizer.batch_decode(
                    batch_tokens.detach().cpu(), skip_special_tokens=True
                )

        else:
            pred_tokens = torch.argmax(logits, dim=-1)
            target_tokens = labels
            pad_token_id = getattr(
                getattr(self.model, "tokenizer", None), "pad_token_id", 0
            )
            valid_mask = target_tokens != pad_token_id

            def decode(batch_tokens):
                project_tokenizer = getattr(self.model, "tokenizer", None)
                return [
                    (
                        project_tokenizer.decode(row.tolist())
                        if project_tokenizer is not None
                        else ""
                    )
                    for row in batch_tokens.detach().cpu()
                ]

        target_for_decode = target_tokens.masked_fill(~valid_mask, pad_token_id)
        token_correct = (pred_tokens == target_tokens) | ~valid_mask
        str_acc = (
            token_correct.all(dim=1).float().mean().detach()
            if token_correct.numel() > 0
            else torch.tensor(0.0, device=logits.device)
        )
        pred_for_decode = pred_tokens.masked_fill(~valid_mask, pad_token_id)
        pred_smiles = decode(pred_for_decode)
        target_smiles = decode(target_for_decode)
        chemical_correct = 0
        for pred_smi, target_smi in zip(pred_smiles, target_smiles):
            if pred_smi == target_smi:
                chemical_correct += 1
                continue
            mol_pred = Chem.MolFromSmiles(pred_smi)
            mol_target = Chem.MolFromSmiles(target_smi)
            if mol_pred is None or mol_target is None:
                continue
            try:
                if Chem.MolToInchiKey(mol_pred) == Chem.MolToInchiKey(mol_target):
                    chemical_correct += 1
            except Exception:
                continue
        chem_acc = torch.tensor(
            chemical_correct / max(len(target_smiles), 1), device=logits.device
        )
        return (chem_acc, str_acc)

    def ctr_loss_fn(self, molecule_feature, sp_feature, temperature=0.07):
        from torch.nn import functional as F

        cos_sim = F.cosine_similarity(
            molecule_feature[:, None, :], sp_feature[None, :, :], dim=-1
        )
        postive_mask = torch.eye(
            cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device
        )
        cos_sim = cos_sim / temperature
        nll = -cos_sim[postive_mask] + torch.logsumexp(cos_sim, dim=-1)
        nll = nll.mean()
        return nll
