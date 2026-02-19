import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, CosineAnnealingWarmRestarts
from torch.nn.functional import mse_loss, l1_loss, smooth_l1_loss

from pytorch_lightning import LightningModule
from torchmdnet.models.foundation_model_peak import create_model, load_model
from math import inf



class PlateauScheduler(ReduceLROnPlateau):
    def __init__(self, factor, patience):

        self.factor = factor
        self.patience = patience
        self.threshold = 1e-4
        self.mode = "min"
        self.threshold_mode = "rel"
        self.best = inf
        self.num_bad_epochs = None
        self.eps = 1e-8
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


class LNNP(LightningModule):
    def __init__(self, hparams, prior_model=None, mean=None, std=None, var_uv=None, var_ir=None, var_raman=None):

        super(LNNP, self).__init__()
        self.save_hyperparameters(hparams,ignore=["var_uv", "var_ir", "var_raman"])

        if self.hparams.load_model:
            self.model = load_model(self.hparams.load_model, args=self.hparams, var_uv=var_uv, var_ir=var_ir, var_raman=var_raman)
        elif self.hparams.pretrained_model:
            self.model = load_model(self.hparams.pretrained_model, args=self.hparams, mean=mean, std=std, var_uv=var_uv, var_ir=var_ir, var_raman=var_raman, prior_model=prior_model)
        else:
            self.model = create_model(self.hparams, prior_model, mean, std, var_uv, var_ir, var_raman)

        # initialize exponential smoothing
        self.ema = None
        self._reset_ema_dict()

        # initialize loss collection
        self.losses = None
        self._reset_losses_dict()

        self.last_epoch = 0
        self.lr_gen_scheduler = PlateauScheduler(
            factor=self.hparams.lr_factor,
            patience=self.hparams.lr_patience,
        )
        self.val_loss = None
        self.last_train_loss = None

    def configure_optimizers(self):
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        if self.hparams.lr_schedule == 'cosine':
            scheduler = CosineAnnealingLR(optimizer, self.hparams.lr_cosine_length, eta_min=1e-7)
            lr_scheduler = {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            }
        elif self.hparams.lr_schedule == 'cosine_warmup':
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=self.hparams.lr_cosine_length, T_mult=2, eta_min=1e-7)
            lr_scheduler = {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            }
        elif self.hparams.lr_schedule == 'reduce_on_plateau':
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
        return [optimizer], [lr_scheduler]

    def forward(self, z, pos, spec, patch_masks, batch=None):
        # for name, param in self.model.named_parameters():
        #     if param.grad is not None:
        #         print(f"Grad of {name}: {param.grad.norm().item():.6f}")
        #     else:
        #         print(f"Grad of {name} is None")
        return self.model(z, pos, spec, patch_masks, batch=batch)
    # def on_after_backward(self):
    #     for name, param in self.named_parameters():
    #         if param.grad is not None:
    #             print(f"Grad of {name}: {param.grad.norm().item():.6f}")
    #         else:
    #             print(f"Grad of {name} is None")

    def training_step(self, batch, batch_idx):
        # import matplotlib.pyplot as plt
        # train_specs = batch.raman.cpu().numpy()  # 先转CPU，再转numpy

        # plt.figure(figsize=(12, 6))
        # for i in range(50):
        #     plt.plot(train_specs[i], label=f'Train {i}', linestyle='-')

        # plt.title('Training and Validation Spectra Samples')
        # plt.legend()
        # plt.savefig("./figure_raman1.png")
        # plt.show()

        return self.step(batch, mse_loss, "train")


    def validation_step(self, batch, batch_idx, *args):
        if len(args) == 0 or (len(args) > 0 and args[0] == 0):
            val_loss = self.step(batch, mse_loss, "val")
            #     # --- 添加 embedding 收集 ---
            # z = batch['z']
            # pos = batch['pos']
            # spec = batch['spectra']
            # patch_masks = batch['patch_masks']
            # mol_batch = batch['batch']

            # with torch.no_grad():
            #     output = self(z, pos, spec, patch_masks, batch=mol_batch)
            #     mol_emb = output[6]     # [B, D]
            #     spec_emb0 = output[3]   # [B, D]
            #     spec_emb1 = output[4]   # [B, D]
            #     spec_emb2 = output[5]   # [B, D]

            # # 将 embedding 存储到模块属性（不用全局变量）
            # if not hasattr(self, "spec_emb_list0"):
            #     self.mol_emb_list = []
            #     self.spec_emb_list = []
            #     self.spec_emb_list0 = []
            #     self.spec_emb_list1 = []
            #     self.spec_emb_list2 = []

            # # self.mol_emb_list.append(mol_emb.cpu())
            # self.spec_emb_list0.append(spec_emb0.cpu())
            # self.spec_emb_list1.append(spec_emb1.cpu())
            # self.spec_emb_list2.append(spec_emb2.cpu())
            # self.mol_emb_list.append(mol_emb.cpu())
            return val_loss
        # test step
        return self.step(batch, l1_loss, "test")

    def test_step(self, batch, batch_idx):
        
        loss = self.step(batch, l1_loss, "test")
        # print(f"Batch {batch_idx} test loss: {loss.item()}")
        return loss

    def step(self, batch, loss_fn, stage):
        with torch.set_grad_enabled(stage == "train" or self.hparams.derivative):
            if ("spectra" in batch): 
                pred, noise_pred, deriv, sp_feature, molecule_feature, loss_reconstruct, loss_reconstruct1, loss_reconstruct2 = self(batch.z, batch.pos, batch.spectra, batch.patch_masks, batch.batch)
            elif ("ir" in batch) and ("h_nmr" in batch) and ("c_nmr" in batch):
                pred, noise_pred, deriv, sp_feature, molecule_feature, loss_reconstruct = self(batch.z, batch.pos, [batch.ir, batch.h_nmr, batch.c_nmr], batch.batch)
            else:
                pred, noise_pred, deriv, sp_feature, molecule_feature, loss_reconstruct = self(batch.z, batch.pos, None, batch.batch)

        if loss_reconstruct is not None and self.hparams.reconstruct_weight > 0:
            self.losses[stage + "_reconstruct"].append(loss_reconstruct.detach())
        else:
            loss_reconstruct = 0
        if self.hparams.reconstruct_weight > 0:
            loss_reconstruct1 = None
            if loss_reconstruct1 is not None:
                self.losses[stage + "_reconstruct1"].append(loss_reconstruct1.detach())
            if loss_reconstruct2 is not None:
                self.losses[stage + "_reconstruct2"].append(loss_reconstruct2.detach())

        denoising_is_on = ("pos_target" in batch) and (self.hparams.denoising_weight > 0) and (noise_pred is not None)
        contrastive_is_on = ("spectra" in batch) and (self.hparams.contrastive_weight > 0) and (sp_feature is not None)

        loss_y, loss_dy, loss_pos = 0, 0, 0
        loss_ctr = 0
        if self.hparams.derivative:
            if "y" not in batch:
                # "use" both outputs of the model's forward function but discard the first
                # to only use the derivative and avoid 'Expected to have finished reduction
                # in the prior iteration before starting a new one.', which otherwise get's
                # thrown because of setting 'find_unused_parameters=False' in the DDPPlugin
                deriv = deriv + pred.sum() * 0

            # force/derivative loss
            loss_dy = loss_fn(deriv, batch.dy)

            if stage in ["train", "val"] and self.hparams.ema_alpha_dy < 1:
                if self.ema[stage + "_dy"] is None:
                    self.ema[stage + "_dy"] = loss_dy.detach()
                # apply exponential smoothing over batches to dy
                loss_dy = (
                    self.hparams.ema_alpha_dy * loss_dy
                    + (1 - self.hparams.ema_alpha_dy) * self.ema[stage + "_dy"]
                )
                self.ema[stage + "_dy"] = loss_dy.detach()

            if self.hparams.force_weight > 0:
                self.losses[stage + "_dy"].append(loss_dy.detach())

        if "y" in batch and pred is not None:
            if (noise_pred is not None) and not denoising_is_on:
                # "use" both outputs of the model's forward (see comment above).
                pred = pred + noise_pred.sum() * 0

            if batch.y.ndim == 1:
                batch.y = batch.y.unsqueeze(1)

            # energy/prediction loss
            loss_y = loss_fn(pred, batch.y)

            if stage in ["train", "val"] and self.hparams.ema_alpha_y < 1:
                if self.ema[stage + "_y"] is None:
                    self.ema[stage + "_y"] = loss_y.detach()
                # apply exponential smoothing over batches to y
                loss_y = (
                    self.hparams.ema_alpha_y * loss_y
                    + (1 - self.hparams.ema_alpha_y) * self.ema[stage + "_y"]
                )
                self.ema[stage + "_y"] = loss_y.detach()

            if self.hparams.energy_weight > 0:
                self.losses[stage + "_y"].append(loss_y.detach())

        if denoising_is_on:
            # if "y" not in batch:
            #     # "use" both outputs of the model's forward (see comment above).
            #     noise_pred = noise_pred + pred.sum() * 0

            normalized_pos_target = self.model.pos_normalizer(batch.pos_target)
            loss_pos = loss_fn(noise_pred, normalized_pos_target)
            self.losses[stage + "_pos"].append(loss_pos.detach())

        # contrastive loss
        if contrastive_is_on:
            loss_ctr = self.ctr_loss_fn(molecule_feature, sp_feature)
            self.losses[stage + "_contrast"].append(loss_ctr.detach())

        # total loss
        loss = (
            loss_y * self.hparams.energy_weight \
            + loss_dy * self.hparams.force_weight \
            + loss_pos * self.hparams.denoising_weight \
            + loss_ctr * self.hparams.contrastive_weight \
            + loss_reconstruct * self.hparams.reconstruct_weight
        )
        
        self.losses[stage].append(loss.detach())

        # Frequent per-batch logging for training
        if stage == 'train':
            train_metrics = {k + "_per_step": v[-1] for k, v in self.losses.items() if (k.startswith("train") and len(v) > 0)}
            train_metrics['lr_per_step'] = self.trainer.optimizers[0].param_groups[0]["lr"]
            train_metrics['step'] = self.trainer.global_step   
            train_metrics['batch_pos_mean'] = batch.pos.mean().item()
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
            if self.val_loss is not None and optimizer.param_groups[0]["lr"] > self.hparams.lr * 0.1:
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
            # if should_reset:
            #     self.trainer.validate(self, dataloaders=self.trainer.datamodule.val_dataloader())

    def _safe_mean(self, losses):
        return torch.stack(losses).mean() if losses else torch.tensor(float('nan'))

    def on_validation_epoch_end(self):
        # if hasattr(self, "mol_emb_list"):
        #     mol_emb_all = torch.cat(self.mol_emb_list, dim=0)
            
        #     spec_emb_all0 = torch.cat(self.spec_emb_list0, dim=0)
        #     spec_emb_all1 = torch.cat(self.spec_emb_list1, dim=0)
        #     spec_emb_all2 = torch.cat(self.spec_emb_list2, dim=0)

        #     # 保存到文件或画图
        #     torch.save(mol_emb_all, "mol_emb.pt")
        #     torch.save(spec_emb_all0, "spec_emb0.pt")
        #     torch.save(spec_emb_all1, "spec_emb1.pt")
        #     torch.save(spec_emb_all2, "spec_emb2.pt")

        #     # 清空，避免下次 val 冲突
        #     del self.mol_emb_list
        #     del self.spec_emb_list0
        #     del self.spec_emb_list1
        #     del self.spec_emb_list2
            # del self.spec_emb_list
        if not self.trainer.sanity_checking:
            result_dict = {
                "epoch": self.current_epoch,
                "lr": self.trainer.optimizers[0].param_groups[0]["lr"],
                "train_loss": self._safe_mean(self.losses.get("train", [])),
                "val_loss": self._safe_mean(self.losses.get("val", [])),
            }
            self.val_loss = result_dict["val_loss"]

            # add test loss if available
            if len(self.losses["test"]) > 0:
                result_dict["test_loss"] = torch.stack(self.losses["test"]).mean()

            # if prediction and derivative are present, also log them separately
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
            if len(self.losses['val_y']) > 0:
                result_dict["val_loss_y"] = torch.stack(self.losses["val_y"]).mean()
            if len(self.losses["test_y"]) > 0:
                result_dict["test_loss_y"] = torch.stack(
                    self.losses["test_y"]
                ).mean()

            # if denoising is present, also log it
            if len(self.losses["train_pos"]) > 0:
                result_dict["train_loss_pos"] = torch.stack(
                    self.losses["train_pos"]
                ).mean()

            if len(self.losses["val_pos"]) > 0:
                result_dict["val_loss_pos"] = torch.stack(
                    self.losses["val_pos"]
                ).mean()

            if len(self.losses["test_pos"]) > 0:
                result_dict["test_loss_pos"] = torch.stack(
                    self.losses["test_pos"]
                ).mean()

            # if contrast is present, also log it
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

            # if reconstruct is present, also log it
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
        }

    def _reset_ema_dict(self):
        self.ema = {"train_y": None, "val_y": None, "train_dy": None, "val_dy": None}

    def ctr_loss_fn(self, molecule_feature, sp_feature, temperature=0.07):
        from torch.nn import functional as F

        # Calculate cosine similarity
        cos_sim = F.cosine_similarity(molecule_feature[:, None, :], sp_feature[None, :, :], dim=-1)

        postive_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
        # InfoNCE loss
        cos_sim = cos_sim / temperature
        nll = -cos_sim[postive_mask] + torch.logsumexp(cos_sim, dim=-1)
        nll = nll.mean()

        return nll
