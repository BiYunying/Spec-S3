from os.path import join
import os
from tqdm import tqdm
import torch
from torch.utils.data import Subset
from torch_geometric.data import DataLoader
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities import rank_zero_warn
from torchmdnet import datasets
from torchmdnet.utils import (
    make_splits,
    MissingEnergyException,
    get_peak_regions,
    probability_peak_masking,
    smooth_and_normalize_spectrum,
    uv_smart_masking,
)
from torch_scatter import scatter
from functools import partial
from torch.utils.data import Dataset
from torch_geometric.data import Data
import numpy as np
import matplotlib.pyplot as plt
import random
from rdkit import Chem
from torch.utils.data import ConcatDataset


class SpectraPreprocessedDataset(Dataset):

    def __init__(self, original_dataset, hparams, is_train=False, peak_mask=False):
        """
        Dataset wrapper for on-the-fly spectral preprocessing and SMILES enumeration.
        """
        self.original_dataset = original_dataset
        self.hparams = hparams
        self.is_train = is_train
        self.peak_mask = peak_mask
        self._spec_aug_vis_saved = 0
        if type(hparams) is dict:
            self.patch_len = hparams["patch_len"]
            self.stride = hparams["stride"]
        else:
            self.patch_len = hparams.patch_len
            self.stride = hparams.stride
        if peak_mask:
            self.peak_prominence = [0.01, 20.0, 20.0]
            self.width_scale = [2, 3, 3]
        else:
            self.peak_prominence = [0.01, 20.0, 20.0]
            self.width_scale = [0, 0, 0]
        self.p_peak = self._hparam("p_peak", 0.5)
        self.p_noise = self._hparam("p_noise", 0.1)
        for name in ("p_peak", "p_noise"):
            value = getattr(self, name)
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be a finite probability in [0, 1]")

    def _hparam(self, name, default):
        if isinstance(self.hparams, dict):
            return self.hparams.get(name, default)
        return getattr(self.hparams, name, default)

    def _sample_spectral_augment_recipe(self):
        if not self.is_train or not self._hparam("spec_augment", False):
            return "clean"
        r = random.random()
        clean_prob = self._hparam("spec_aug_clean_prob", 0.1)
        global_only_prob = self._hparam("spec_aug_global_only_prob", 0.3)
        global_intensity_prob = self._hparam("spec_aug_global_intensity_prob", 0.25)
        global_broadening_prob = self._hparam("spec_aug_global_broadening_prob", 0.15)
        global_local_shift_prob = self._hparam("spec_aug_global_local_shift_prob", 0.2)
        if r < clean_prob:
            return "clean"
        r -= clean_prob
        if r < global_only_prob:
            return "global"
        r -= global_only_prob
        if r < global_intensity_prob:
            return "global_intensity"
        r -= global_intensity_prob
        if r < global_broadening_prob:
            return "global_broadening"
        r -= global_broadening_prob
        if r < global_local_shift_prob:
            return "global_local_shift"
        return "global"

    def _sample_global_affine_params(self):
        ir_raman_scale = random.uniform(
            self._hparam("spec_aug_ir_raman_scale_min", 0.96),
            self._hparam("spec_aug_ir_raman_scale_max", 1.05),
        )
        return {
            "uv": (
                random.uniform(
                    self._hparam("spec_aug_uv_scale_min", 0.985),
                    self._hparam("spec_aug_uv_scale_max", 1.015),
                ),
                random.uniform(
                    -self._hparam("spec_aug_uv_shift_max", 60.0),
                    self._hparam("spec_aug_uv_shift_max", 60.0),
                ),
            ),
            "ir": (
                ir_raman_scale,
                random.uniform(
                    -self._hparam("spec_aug_ir_raman_shift_max", 35.0),
                    self._hparam("spec_aug_ir_raman_shift_max", 35.0),
                ),
            ),
            "raman": (
                ir_raman_scale,
                random.uniform(
                    -self._hparam("spec_aug_ir_raman_shift_max", 35.0),
                    self._hparam("spec_aug_ir_raman_shift_max", 35.0),
                ),
            ),
        }

    def _apply_global_affine(self, spectrum_np, scale, shift):
        if abs(scale - 1.0) < 1e-08 and abs(shift) < 1e-08:
            return spectrum_np
        x = np.arange(len(spectrum_np), dtype=np.float32)
        source_x = (x - shift) / scale
        return np.interp(source_x, x, spectrum_np, left=0.0, right=0.0).astype(
            np.float32
        )

    def _apply_peak_intensity_jitter(self, spectrum_np, peak_regions, sigma=0.1):
        if not peak_regions:
            return spectrum_np
        augmented = spectrum_np.copy()
        for start, end in peak_regions:
            if end <= start:
                continue
            width = end - start
            window = (
                np.hanning(width) if width > 2 else np.ones(width, dtype=np.float32)
            )
            multiplier = float(np.random.lognormal(mean=0.0, sigma=sigma))
            region = augmented[start:end]
            augmented[start:end] = region * (1.0 + (multiplier - 1.0) * window)
        return augmented.astype(np.float32)

    def _apply_broadening(self, spectrum_np, sigma):
        if sigma <= 0:
            return spectrum_np
        radius = max(1, int(round(3.0 * sigma)))
        x = np.arange(-radius, radius + 1, dtype=np.float32)
        kernel = np.exp(-(x**2) / (2.0 * sigma**2))
        kernel = kernel / kernel.sum()
        return np.convolve(spectrum_np, kernel, mode="same").astype(np.float32)

    def _apply_local_peak_shift(self, spectrum_np, peak_regions, spec_name):
        if not peak_regions:
            return spectrum_np
        if spec_name == "uv":
            frac = random.uniform(0.25, 0.5)
            max_delta = self._hparam("spec_aug_uv_local_shift_max", 45.0)
        else:
            frac = random.uniform(0.25, 0.4)
            max_delta = self._hparam("spec_aug_ir_raman_local_shift_max", 25.0)
        num_shift = max(1, int(round(len(peak_regions) * frac)))
        selected = random.sample(peak_regions, k=min(num_shift, len(peak_regions)))
        augmented = spectrum_np.copy()
        x = np.arange(len(spectrum_np), dtype=np.float32)
        for start, end in selected:
            if end <= start:
                continue
            center = 0.5 * (start + end)
            half_width = max(1.0, 0.5 * (end - start))
            blend_start = max(0, int(round(center - 1.5 * half_width)))
            blend_end = min(len(spectrum_np), int(round(center + 1.5 * half_width)))
            if blend_end <= blend_start:
                continue
            delta = random.uniform(-max_delta, max_delta)
            region_x = x[blend_start:blend_end]
            shifted = np.interp(region_x - delta, x, spectrum_np, left=0.0, right=0.0)
            width = blend_end - blend_start
            window = (
                np.hanning(width) if width > 2 else np.ones(width, dtype=np.float32)
            )
            augmented[blend_start:blend_end] = (
                augmented[blend_start:blend_end] * (1.0 - window) + shifted * window
            )
        return augmented.astype(np.float32)

    def augment_dft_like_spectrum(
        self, spectrum_np, peak_regions, spec_name, recipe, affine_params
    ):
        if recipe == "clean":
            return spectrum_np.copy()
        augmented = spectrum_np.copy()
        if recipe == "global_local_shift":
            augmented = self._apply_local_peak_shift(augmented, peak_regions, spec_name)
        if recipe == "global_intensity":
            sigma = (
                self._hparam("spec_aug_uv_intensity_sigma", 0.18)
                if spec_name == "uv"
                else self._hparam("spec_aug_ir_raman_intensity_sigma", 0.16)
            )
            augmented = self._apply_peak_intensity_jitter(
                augmented, peak_regions, sigma=sigma
            )
        scale, shift = affine_params[spec_name]
        augmented = self._apply_global_affine(augmented, scale=scale, shift=shift)
        if recipe == "global_broadening":
            maximum = (self._hparam("spec_aug_uv_broadening_max", 3.5)
                       if spec_name == "uv" else
                       self._hparam("spec_aug_ir_raman_broadening_max", 3.0))
            # A zero upper bound disables broadening; never sample above it.
            sigma = random.uniform(min(0.8, maximum), maximum)
            augmented = self._apply_broadening(augmented, sigma=sigma)
        return np.maximum(augmented, 0.0).astype(np.float32)

    def _log10_like_model(self, spectrum_np, spec_name):
        spectrum_np = np.maximum(spectrum_np, 0.0)
        if spec_name == "uv":
            return np.log10(spectrum_np * 10.0 + 1.0)
        return np.log10(spectrum_np + 1.0)

    def _maybe_visualize_spectral_augmentation(
        self, clean_np_list, augmented_np_list, idx, recipe
    ):
        if recipe == "clean":
            return
        if not self.is_train or not self._hparam("spec_aug_vis", False):
            return
        max_plots = int(self._hparam("spec_aug_vis_max", 8))
        if self._spec_aug_vis_saved >= max_plots:
            return
        vis_prob = float(self._hparam("spec_aug_vis_prob", 1.0))
        if random.random() > vis_prob:
            return
        vis_dir = self._hparam("spec_aug_vis_dir", "vis_results_spec_aug")
        os.makedirs(vis_dir, exist_ok=True)
        modality_names = ["UV", "IR", "Raman"]
        spec_names = ["uv", "ir", "raman"]
        fig, axes = plt.subplots(3, 2, figsize=(14, 10))
        for i, (modality, spec_name) in enumerate(zip(modality_names, spec_names)):
            clean = clean_np_list[i]
            augmented = augmented_np_list[i]
            x_axis = np.arange(len(clean))
            ax_raw = axes[i, 0]
            ax_raw.plot(
                x_axis,
                clean,
                color="tab:blue",
                linewidth=1.2,
                alpha=0.85,
                label="clean",
            )
            ax_raw.plot(
                x_axis,
                augmented,
                color="tab:orange",
                linewidth=1.1,
                alpha=0.8,
                label="augmented",
            )
            ax_raw.set_title(f"{modality} raw")
            ax_raw.legend(loc="upper right", fontsize=8)
            clean_log = self._log10_like_model(clean, spec_name)
            augmented_log = self._log10_like_model(augmented, spec_name)
            ax_log = axes[i, 1]
            ax_log.plot(
                x_axis,
                clean_log,
                color="tab:blue",
                linewidth=1.2,
                alpha=0.85,
                label="clean",
            )
            ax_log.plot(
                x_axis,
                augmented_log,
                color="tab:orange",
                linewidth=1.1,
                alpha=0.8,
                label="augmented",
            )
            ax_log.set_title(f"{modality} log10")
            ax_log.legend(loc="upper right", fontsize=8)
        fig.suptitle(f"Spectral augmentation: idx={idx}, recipe={recipe}", fontsize=12)
        fig.tight_layout(rect=[0, 0.02, 1, 0.96])
        safe_recipe = str(recipe).replace("/", "_")
        out_path = os.path.join(
            vis_dir,
            f"spec_aug_idx{idx}_pid{os.getpid()}_{self._spec_aug_vis_saved:03d}_{safe_recipe}.png",
        )
        fig.savefig(out_path, dpi=160)
        plt.close(fig)
        self._spec_aug_vis_saved += 1

    def augment_spectrum(
        self, tensor, max_shift=20, noise_std=0.02, scale_range=(0.9, 1.1)
    ):
        """
        Augment a spectral tensor with zero-padded shifts and intensity scaling.
        tensor: (1, L) or (L,)
        """
        if scale_range:
            scale = random.uniform(scale_range[0], scale_range[1])
            tensor = tensor * scale
        if max_shift > 0:
            shift = random.randint(-max_shift, max_shift)
            if shift != 0:
                if tensor.dim() == 1:
                    zeros = torch.zeros((abs(shift),), dtype=tensor.dtype)
                    if shift > 0:
                        tensor = torch.cat((zeros, tensor[:-shift]), dim=-1)
                    else:
                        tensor = torch.cat((tensor[-shift:], zeros), dim=-1)
                else:
                    zeros = torch.zeros(
                        (tensor.shape[0], abs(shift)), dtype=tensor.dtype
                    )
                    if shift > 0:
                        tensor = torch.cat((zeros, tensor[..., :-shift]), dim=-1)
                    else:
                        tensor = torch.cat((tensor[..., -shift:], zeros), dim=-1)
        return tensor

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        data_sample = self.original_dataset[idx]
        new_data_sample = data_sample.clone()
        canonical_smiles = data_sample.smiles
        mol = Chem.MolFromSmiles(canonical_smiles)
        if mol is not None:
            # Pretraining augments the ChemBERTa input only. Atom/coordinate
            # order stays unchanged; PyTorch's worker seed controls enumeration.
            task = self._hparam("public_task", None) or self._hparam("task", "")
            randomize = self.is_train and task in ("pretrain", "pretraining")
            if randomize:
                mol = Chem.RenumberAtoms(mol, torch.randperm(mol.GetNumAtoms()).tolist())
            processed_smiles = Chem.MolToSmiles(
                mol, kekuleSmiles=True, canonical=not randomize
            )
            new_data_sample.smiles = processed_smiles
        processed_spectra = []
        clean_spectra = []
        patch_masks = []
        clean_np_list = []
        augmented_np_list = []
        augment_recipe = self._sample_spectral_augment_recipe()
        affine_params = self._sample_global_affine_params()
        for i, spec_name in enumerate(["uv", "ir", "raman"]):
            spectrum_tensor = getattr(data_sample, spec_name)
            spectrum_np = (
                spectrum_tensor.detach().cpu().reshape(-1).numpy().astype(np.float32)
            )
            peak_regions = get_peak_regions(
                spectrum_np,
                prominence=self.peak_prominence[i],
                width_scale=self.width_scale[i],
            )
            patch_mask = probability_peak_masking(
                spectrum_len=len(spectrum_np),
                peak_regions=peak_regions,
                patch_len=self.patch_len[i],
                stride=self.stride[i],
                p_peak=self.p_peak if self.peak_mask else 0.0,
                p_noise=self.p_noise if self.peak_mask else 0.0,
            )
            augmented_spectrum_np = self.augment_dft_like_spectrum(
                spectrum_np,
                peak_regions=peak_regions,
                spec_name=spec_name,
                recipe=augment_recipe,
                affine_params=affine_params,
            )
            processed_spectra.append(torch.from_numpy(augmented_spectrum_np).float())
            clean_spectra.append(torch.from_numpy(spectrum_np.copy()).float())
            clean_np_list.append(spectrum_np.copy())
            augmented_np_list.append(augmented_spectrum_np.copy())
            patch_masks.append(torch.from_numpy(patch_mask).bool())
        self._maybe_visualize_spectral_augmentation(
            clean_np_list, augmented_np_list, idx=idx, recipe=augment_recipe
        )
        new_data_sample.spectra = processed_spectra
        new_data_sample.spectra_clean = clean_spectra
        new_data_sample.patch_masks = patch_masks
        new_data_sample.original_index = idx
        return new_data_sample


class DataModule(LightningDataModule):

    def __init__(self, hparams, dataset=None):
        super().__init__()
        if hasattr(hparams, "__dict__"):
            self.hparams_dict = hparams.__dict__
        else:
            self.hparams_dict = hparams
        self.save_hyperparameters(hparams)
        self._mean, self._std = (None, None)
        self._saved_dataloaders = dict()
        self.dataset = dataset
        self.dataset_maybe_noisy = dataset

    def setup(self, stage=None):
        if self.dataset is None:
            if self.hparams.dataset == "Custom":
                self.dataset = datasets.Custom(
                    self.hparams.coord_files,
                    self.hparams.embed_files,
                    self.hparams.energy_files,
                    self.hparams.force_files,
                )
            else:
                if self.hparams.position_noise_scale > 0.0:

                    def transform(data):
                        noise = (
                            torch.randn_like(data.pos)
                            * self.hparams.position_noise_scale
                        )
                        data.pos_target = noise
                        data.pos = data.pos + noise
                        return data

                else:
                    transform = None
                dataset_factory = lambda t: getattr(datasets, self.hparams.dataset)(
                    self.hparams.dataset_root,
                    dataset_arg=self.hparams.dataset_arg,
                    transform=t,
                )
                self.dataset_maybe_noisy = dataset_factory(transform)
                self.dataset = dataset_factory(None)
        self.idx_train, self.idx_val, self.idx_test = make_splits(
            len(self.dataset),
            self.hparams.train_size,
            self.hparams.val_size,
            self.hparams.test_size,
            self.hparams.seed,
            join(self.hparams.log_dir, "splits.npz"),
            self.hparams.splits,
        )
        print(
            f"train {len(self.idx_train)}, val {len(self.idx_val)}, test {len(self.idx_test)}"
        )
        original_train_dataset = Subset(
            self.dataset_maybe_noisy or self.dataset, self.idx_train
        )
        if self.hparams.denoising_only:
            original_val_dataset = Subset(
                self.dataset_maybe_noisy or self.dataset, self.idx_val
            )
            original_test_dataset = Subset(
                self.dataset_maybe_noisy or self.dataset, self.idx_test
            )
        else:
            original_val_dataset = Subset(self.dataset, self.idx_val)
            original_test_dataset = Subset(self.dataset, self.idx_test)
        self.train_dataset = SpectraPreprocessedDataset(
            original_train_dataset,
            self.hparams,
            is_train=True,
            peak_mask=self.hparams.peak_mask,
        )
        self.val_dataset = SpectraPreprocessedDataset(
            original_val_dataset,
            self.hparams,
            is_train=False,
            peak_mask=self.hparams.peak_mask,
        )
        self.test_dataset = SpectraPreprocessedDataset(
            original_test_dataset,
            self.hparams,
            is_train=False,
            peak_mask=self.hparams.peak_mask,
        )
        if getattr(self.hparams, "standardize", False):
            self._standardize()

    def train_dataloader(self):
        return self._get_dataloader(self.train_dataset, "train")

    def val_dataloader(self):
        loaders = [self._get_dataloader(self.val_dataset, "val")]
        if (
            len(self.test_dataset) > 0
            and self.trainer.current_epoch % self.hparams["test_interval"] == 0
        ):
            loaders.append(self._get_dataloader(self.test_dataset, "test"))
        return loaders

    def test_dataloader(self):
        return self._get_dataloader(self.test_dataset, "test")

    @property
    def atomref(self):
        if hasattr(self.dataset, "get_atomref"):
            return self.dataset.get_atomref()
        return None

    @property
    def mean(self):
        return self._mean

    @property
    def std(self):
        return self._std

    def _get_dataloader(self, dataset, stage, store_dataloader=True):
        store_dataloader = store_dataloader
        if stage in self._saved_dataloaders and store_dataloader:
            return self._saved_dataloaders[stage]
        if stage == "train":
            batch_size = self.hparams["batch_size"]
            shuffle = True
        elif stage in ["val", "test"]:
            batch_size = self.hparams["inference_batch_size"]
            shuffle = False
        dl = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.hparams["num_workers"],
            pin_memory=True,
        )
        if store_dataloader:
            self._saved_dataloaders[stage] = dl
        return dl

    def _standardize(self):

        def get_energy(batch, atomref):
            if batch.y is None:
                raise MissingEnergyException()
            if atomref is None:
                return batch.y.clone()
            atomref_energy = scatter(atomref[batch.z], batch.batch, dim=0)
            return (batch.y.reshape(-1, 1) - atomref_energy.reshape(-1, 1)).clone()

        data = tqdm(
            self._get_dataloader(Subset(self.dataset, self.idx_train), "val", store_dataloader=False),
            desc="computing mean and std",
        )
        try:
            atomref = self.atomref if self.hparams["prior_model"] == "Atomref" else None
            ys = torch.cat([get_energy(batch, atomref) for batch in data])
        except MissingEnergyException:
            rank_zero_warn(
                "Standardize is true but failed to compute dataset mean and standard deviation. Maybe the dataset only contains forces."
            )
            return
        self._mean = ys.mean(dim=0)
        self._std = ys.std(dim=0)
        if not torch.isfinite(self._mean).all() or not torch.isfinite(self._std).all() or (self._std <= 0).any():
            raise ValueError("Training residuals must have finite, positive sample standard deviation (at least two samples)")
        print(f"y mean: {self.mean}; y std: {self.std}")



# Compatibility import for generation/cache tools.
from torchmdnet.datasets.base import ExactProcessedDataset
