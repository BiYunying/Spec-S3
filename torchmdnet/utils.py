import yaml
import argparse
import numpy as np
import torch
from os.path import dirname, join, exists
from pytorch_lightning.utilities import rank_zero_warn
from scipy.signal import find_peaks, peak_widths
import scipy.ndimage as ndimage
from torch import nn
import torch.nn.functional as F
import json
import re
from typing import List, Dict
from rdkit import Chem
from torchmdnet.tokenizer import SMILES_DICTIONARY


def train_val_test_split(dset_len, train_size, val_size, test_size, seed, order=None):
    assert (train_size is None) + (val_size is None) + (
        test_size is None
    ) <= 1, "Only one of train_size, val_size, test_size is allowed to be None."
    is_float = (
        isinstance(train_size, float),
        isinstance(val_size, float),
        isinstance(test_size, float),
    )
    train_size = round(dset_len * train_size) if is_float[0] else train_size
    val_size = round(dset_len * val_size) if is_float[1] else val_size
    test_size = round(dset_len * test_size) if is_float[2] else test_size
    if train_size is None:
        train_size = dset_len - val_size - test_size
    elif val_size is None:
        val_size = dset_len - train_size - test_size
    elif test_size is None:
        test_size = dset_len - train_size - val_size
    if train_size + val_size + test_size > dset_len:
        if is_float[2]:
            test_size -= 1
        elif is_float[1]:
            val_size -= 1
        elif is_float[0]:
            train_size -= 1
    assert (
        train_size >= 0 and val_size >= 0 and (test_size >= 0)
    ), f"One of training ({train_size}), validation ({val_size}) or testing ({test_size}) splits ended up with a negative size."
    total = train_size + val_size + test_size
    assert (
        dset_len >= total
    ), f"The dataset ({dset_len}) is smaller than the combined split sizes ({total})."
    if total < dset_len:
        rank_zero_warn(f"{dset_len - total} samples were excluded from the dataset")
    idxs = np.arange(dset_len, dtype=int)
    if order is None:
        idxs = np.random.default_rng(seed).permutation(idxs)
    idx_train = idxs[:train_size]
    idx_val = idxs[train_size : train_size + val_size]
    idx_test = idxs[train_size + val_size : total]
    if order is not None:
        idx_train = [order[i] for i in idx_train]
        idx_val = [order[i] for i in idx_val]
        idx_test = [order[i] for i in idx_test]
    return (np.array(idx_train), np.array(idx_val), np.array(idx_test))


def make_splits(
    dataset_len,
    train_size,
    val_size,
    test_size,
    seed,
    filename=None,
    splits=None,
    order=None,
):
    if splits is not None:
        splits = np.load(splits)
        idx_train = splits.get("idx_train", splits.get("train_idx"))
        idx_val = splits.get("idx_val", splits.get("val_idx"))
        idx_test = splits.get("idx_test", splits.get("test_idx"))
    else:
        idx_train, idx_val, idx_test = train_val_test_split(
            dataset_len, train_size, val_size, test_size, seed, order
        )
    if filename is not None:
        np.savez(filename, idx_train=idx_train, idx_val=idx_val, idx_test=idx_test)
    return (
        torch.from_numpy(idx_train),
        torch.from_numpy(idx_val),
        torch.from_numpy(idx_test),
    )


class LoadFromFile(argparse.Action):

    def __call__(self, parser, namespace, values, option_string=None):
        if values.name.endswith("yaml") or values.name.endswith("yml"):
            with values as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
            for key in config.keys():
                if key not in namespace:
                    raise ValueError(f"Unknown argument in config file: {key}")
            namespace.__dict__.update(config)
        else:
            raise ValueError("Configuration file must end with yaml or yml")


class LoadFromCheckpoint(argparse.Action):

    def __call__(self, parser, namespace, values, option_string=None):
        hparams_path = join(dirname(values), "hparams.yaml")
        if not exists(hparams_path):
            print(
                "Failed to locate the checkpoint's hparams.yaml file. Relying on command line args."
            )
            return
        with open(hparams_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        for key in config.keys():
            if key not in namespace and key != "prior_args":
                raise ValueError(f"Unknown argument in the model checkpoint: {key}")
        namespace.__dict__.update(config)
        namespace.__dict__.update(load_model=values)


def save_argparse(args, filename, exclude=None):
    if filename.endswith("yaml") or filename.endswith("yml"):
        if isinstance(exclude, str):
            exclude = [exclude]
        args = args.__dict__.copy()
        for exl in exclude:
            del args[exl]
        yaml.dump(args, open(filename, "w"))
    else:
        raise ValueError("Configuration file should end with yaml or yml")


def number(text):
    if text is None or text == "None":
        return None
    try:
        num_int = int(text)
    except ValueError:
        num_int = None
    num_float = float(text)
    if num_int == num_float:
        return num_int
    return num_float


class MissingEnergyException(Exception):
    pass


def get_peak_regions(
    spectrum: np.ndarray, prominence: float = 0.01, width_scale: float = 3.0
):
    """
    Detect spectral peaks and return their region boundaries.

    Args:
        spectrum: One-dimensional NumPy array containing the spectral signal.
        prominence: Minimum peak prominence relative to normalized spectral intensity.
        width_scale: Peak-width expansion factor used to cover the peak base.

    Returns:
        A list of (start_index, end_index) pairs describing peak regions.
    """
    if prominence < 0.001:
        prominence = 0.001
    peaks, properties = find_peaks(spectrum, prominence=prominence)
    if len(peaks) == 0:
        return []
    widths, _, left_ips, right_ips = peak_widths(spectrum, peaks, rel_height=0.85)
    peak_regions = []
    for i, peak_idx in enumerate(peaks):
        start_idx = int(left_ips[i])
        end_idx = int(right_ips[i])
        center = peak_idx
        half_width = max(center - start_idx, end_idx - center)
        start_idx = int(center - half_width * width_scale)
        end_idx = int(center + half_width * width_scale)
        start_idx = max(0, start_idx)
        end_idx = min(len(spectrum), end_idx)
        if start_idx < end_idx:
            peak_regions.append((start_idx, end_idx))
    return peak_regions


def intelligent_masking(
    spectrum_len: int,
    peak_regions: list,
    patch_len: int,
    stride: int,
    num_peaks_to_mask: int = 1,
    baseline_mask_prob: float = 0.2,
):
    """
    Combine peak masking with baseline masking.

    Args:
        spectrum_len: Total number of spectral samples.
        peak_regions: Peak boundaries returned by get_peak_regions.
        patch_len: Number of samples per patch.
        stride: Step size between patches.
        num_peaks_to_mask: Number of peaks to mask per call.
        baseline_mask_prob: Probability of selecting baseline masking.

    Returns:
        mask: Boolean NumPy array of shape [num_patches]; True marks a masked patch.
    """
    num_patches = (spectrum_len - patch_len) // stride + 1
    patch_mask = np.zeros(num_patches, dtype=bool)
    if np.random.rand() < baseline_mask_prob and len(peak_regions) > 0:
        peak_patch_indices = set()
        for start, end in peak_regions:
            start_patch = start // stride
            end_patch = end // stride
            for i in range(start_patch, end_patch + 1):
                peak_patch_indices.add(i)
        baseline_patches = [
            i for i in range(num_patches) if i not in peak_patch_indices
        ]
        if len(baseline_patches) > 5:
            mask_center = np.random.choice(baseline_patches)
            mask_width = len(peak_regions[0]) // stride if peak_regions else 5
            start_mask = max(0, mask_center - mask_width // 2)
            end_mask = min(num_patches, start_mask + mask_width)
            patch_mask[start_mask:end_mask] = True
    elif len(peak_regions) > 0:
        peaks_to_mask_indices = np.random.choice(
            len(peak_regions),
            size=min(num_peaks_to_mask, len(peak_regions)),
            replace=False,
        )
        for idx in peaks_to_mask_indices:
            start, end = peak_regions[idx]
            start_patch = start // stride
            end_patch = end // stride
            patch_mask[start_patch : end_patch + 1] = True
    if not np.any(patch_mask):
        mask_start = np.random.randint(0, num_patches - 5)
        patch_mask[mask_start : mask_start + 5] = True
    return patch_mask


def probability_peak_masking(spectrum_len, peak_regions, patch_len, stride,
                             p_peak=0.5, p_noise=0.1):
    """Bernoulli per peak, plus independent Bernoulli per non-peak patch.

    Peak intervals are half-open [start, end). Every overlapping patch belongs
    to that peak, including patches beginning before its left boundary. A patch
    shared by multiple peaks is masked if any of those peaks is selected.
    Background sampling never masks a patch overlapping an unselected peak.
    Zero/all masks are valid outcomes; no forced minimum or fixed mask count.
    Uses NumPy's worker-seeded RNG for reproducible DataLoader sampling.
    """
    for name, value in (("p_peak", p_peak), ("p_noise", p_noise)):
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{name} must be a finite probability in [0, 1]")
    if patch_len <= 0 or stride <= 0 or spectrum_len < patch_len:
        raise ValueError("Require positive patch_len/stride and spectrum_len >= patch_len")
    starts = np.arange((spectrum_len - patch_len) // stride + 1) * stride
    mask = np.zeros(len(starts), dtype=bool)
    if p_peak == 0 and p_noise == 0:
        return mask
    peak_patches = np.zeros_like(mask)
    for start, end in peak_regions:
        overlap = (starts < end) & (starts + patch_len > start)
        peak_patches |= overlap
        if np.random.random() < p_peak:
            mask |= overlap
    background = np.flatnonzero(~peak_patches)
    mask[background] = np.random.random(len(background)) < p_noise
    return mask


def intelligent_masking_v2(
    spectrum_len: int,
    peak_regions: list,
    patch_len: int,
    stride: int,
    num_peaks_to_mask: int = 1,
    num_baseline_masks: int = 3,
    baseline_mask_width: int = 5,
):
    """
    Combine a fixed number of peak masks with additional baseline masks:
    1. Mask the requested number of peaks, limited by the available peaks.
    2. Add random masks in baseline regions.

    Args:
        spectrum_len: Total number of spectral samples.
        peak_regions: Peak boundaries returned by get_peak_regions.
        patch_len: Number of samples per patch.
        stride: Step size between patches.
        num_peaks_to_mask: Number of peaks to mask.
        num_baseline_masks: Number of additional baseline regions to mask.
        baseline_mask_width: Width of each baseline mask in patches.

    Returns:
        mask: Boolean NumPy array of shape [num_patches]; True marks a masked patch.
    """
    num_patches = (spectrum_len - patch_len) // stride + 1
    patch_mask = np.zeros(num_patches, dtype=bool)
    peak_patch_indices = set()
    if peak_regions:
        for start, end in peak_regions:
            start_patch = start // stride
            end_patch = end // stride
            for i in range(start_patch, end_patch + 1):
                if 0 <= i < num_patches:
                    peak_patch_indices.add(i)
    if num_peaks_to_mask > 0 and peak_regions:
        peaks_to_mask_indices = np.random.choice(
            len(peak_regions),
            size=min(num_peaks_to_mask, len(peak_regions)),
            replace=False,
        )
        masked_peak_patches = set()
        for idx in peaks_to_mask_indices:
            start, end = peak_regions[idx]
            start_patch = start // stride
            end_patch = end // stride
            for i in range(start_patch, end_patch + 1):
                if 0 <= i < num_patches:
                    patch_mask[i] = True
                    masked_peak_patches.add(i)
    available_baseline_patches = [
        i for i in range(num_patches) if i not in peak_patch_indices
    ]
    if available_baseline_patches and num_baseline_masks > 0:
        num_to_sample = min(num_baseline_masks, len(available_baseline_patches))
        mask_centers = np.random.choice(
            available_baseline_patches, size=num_to_sample, replace=False
        )
        for center in mask_centers:
            start_mask = max(0, center - baseline_mask_width // 2)
            end_mask = min(num_patches, start_mask + baseline_mask_width)
            patch_mask[start_mask:end_mask] = True
    return patch_mask


def smooth_and_normalize_spectrum(
    spectrum: np.ndarray, global_min: float, global_max: float, sigma: float = 2.0
):
    """
    Apply Gaussian smoothing and global normalization to a spectrum.
    """
    if sigma > 0:
        smoothed = ndimage.gaussian_filter1d(spectrum, sigma=sigma)
    else:
        smoothed = spectrum
    normalized = (smoothed - global_min) / (global_max - global_min + 1e-08)
    return normalized


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Apply label smoothing using PyTorch CrossEntropyLoss.
    Padding targets are excluded from the mean loss.
    """

    def __init__(self, smoothing=0.1, pad_idx=0):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=smoothing, ignore_index=pad_idx, reduction="mean"
        )

    def forward(self, x, target):
        x_flat = x.reshape(-1, x.size(-1))
        target_flat = target.reshape(-1)
        return self.criterion(x_flat, target_flat)


class ExactMatchAccuracy(nn.Module):
    """
    Compute exact sequence and molecular-identity accuracy.
    Report two metrics:
    1. String accuracy: exact token-sequence matches to the supplied SMILES targets.
    2. Chemical accuracy: molecular-identity matches, allowing equivalent SMILES.
    """

    def __init__(self, pad_idx=0, vocab_dict=None):
        """
        Args:
            pad_idx: Padding token ID.
            vocab_dict: Token-to-ID mapping; defaults to the shared SMILES dictionary.
        """
        super().__init__()
        self.pad_idx = pad_idx
        if vocab_dict is None:
            vocab_dict = SMILES_DICTIONARY
        self.id2char = {v: k for k, v in vocab_dict.items()}
        self.bos_idx = vocab_dict.get("BOS", 1)
        self.eos_idx = vocab_dict.get("EOS", 2)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        """
        Returns:
            chem_acc (float): Chemical-equivalence accuracy (exact SMILES or InChIKey match).
            str_acc (float): Exact token-sequence accuracy, excluding padding.
        """
        pred_tokens = torch.argmax(logits, dim=-1)
        batch_size = labels.size(0)
        non_pad_mask = labels != self.pad_idx
        correct_tokens_mask = (pred_tokens == labels) & non_pad_mask
        is_string_match = correct_tokens_mask.sum(dim=1) == non_pad_mask.sum(dim=1)
        str_acc = is_string_match.sum().item() / batch_size
        if self.id2char is None:
            return (str_acc, str_acc)
        chemical_correct_count = 0
        pred_list = pred_tokens.detach().cpu().tolist()
        label_list = labels.detach().cpu().tolist()
        for pred_seq, label_seq in zip(pred_list, label_list):
            try:
                pred_smi = self._decode(pred_seq)
                label_smi = self._decode(label_seq)
                if pred_smi == label_smi:
                    chemical_correct_count += 1
                    continue
                mol_pred = Chem.MolFromSmiles(pred_smi)
                mol_label = Chem.MolFromSmiles(label_smi)
                if mol_pred and mol_label:
                    key_pred = Chem.MolToInchiKey(mol_pred)
                    key_label = Chem.MolToInchiKey(mol_label)
                    if key_pred == key_label:
                        chemical_correct_count += 1
            except:
                pass
        chem_acc = chemical_correct_count / batch_size
        return (chem_acc, str_acc)

    def _decode(self, token_ids):
        """Convert token IDs to a SMILES string."""
        chars = []
        for t in token_ids:
            if t == self.pad_idx:
                continue
            if t == self.eos_idx:
                break
            if t == self.bos_idx:
                continue
            char = self.id2char.get(t, "")
            chars.append(char)
        return "".join(chars)


def uv_smart_masking(spectrum_np, spectrum_len, patch_len, stride, peak_mask):
    """
    Mask UV peak regions using partial-peak and flank masks.

    Args:
        spectrum_np: Raw spectral array of shape (L,).
        spectrum_len: Number of samples in the raw spectrum.
        patch_len: Number of samples per patch.
        stride: Step size between patches.
        peak_mask: Whether to enable masking; False returns an all-false mask.
    Returns:
        patch_mask: (Num_Patches,) Bool Tensor
    """
    num_patches = (spectrum_len - patch_len) // stride + 1
    patch_mask = np.zeros(num_patches, dtype=bool)
    if not peak_mask:
        return patch_mask
    peaks, properties = find_peaks(spectrum_np, prominence=0.01)
    if len(peaks) == 0:
        mask_start = np.random.randint(0, num_patches - 5)
        patch_mask[mask_start : mask_start + 5] = True
        return patch_mask
    widths, width_heights, left_ips, right_ips = peak_widths(
        spectrum_np, peaks, rel_height=0.95
    )
    for i in range(len(peaks)):
        peak_idx = peaks[i]
        left_bound = int(left_ips[i])
        right_bound = int(right_ips[i])
        p_start = max(0, int((left_bound - patch_len / 2) / stride))
        p_end = min(num_patches, int((right_bound - patch_len / 2) / stride) + 1)
        peak_patches = np.arange(p_start, p_end)
        if len(peak_patches) < 2:
            continue
        choice = np.random.choice(["left", "right", "top", "gap"])
        if choice == "left":
            cut_idx = int(len(peak_patches) * 0.5)
            mask_indices = peak_patches[:cut_idx]
        elif choice == "right":
            cut_idx = int(len(peak_patches) * 0.5)
            mask_indices = peak_patches[cut_idx:]
        elif choice == "top":
            start_cut = int(len(peak_patches) * 0.3)
            end_cut = int(len(peak_patches) * 0.7)
            mask_indices = peak_patches[start_cut:end_cut]
        else:
            num_to_mask = max(1, int(len(peak_patches) * 0.4))
            mask_indices = np.random.choice(peak_patches, num_to_mask, replace=False)
        if len(mask_indices) > 0:
            patch_mask[mask_indices] = True
    mask_len = 5
    start_idx = np.random.randint(0, num_patches - mask_len + 1)
    noise_indices = np.arange(start_idx, start_idx + mask_len)
    patch_mask[noise_indices] = True
    return patch_mask


from torchmdnet.tokenizer import CustomTokenizer


TASKS = (
    "pretrain",
    "property_prediction",
    "translation",
    "scaffold_editing",
    "ir_editing",
    "multiobjective",
    "budget",
)
TASK_INTERNAL = dict(
    pretrain="pretraining",
    property_prediction="regression",
    translation="smiles",
    scaffold_editing="hopping",
    ir_editing="hopping",
    multiobjective="A+B",
    budget="smiles",
)


def read_config(path, overrides=None):
    """YAML paths are relative to the public project root, independent of cwd."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    config = dict(DEFAULTS)
    # Public .yaml configs use JSON syntax. PyYAML treats JSON numbers such as
    # 3e-06 as strings; parse JSON first so optimizer settings remain numeric.
    import json
    content = Path(path).read_text()
    try:
        supplied = json.loads(content)
    except json.JSONDecodeError:
        supplied = yaml.safe_load(content) or {}

    if not isinstance(supplied, dict):
        raise ValueError("Configuration must be a mapping")
    config.update(supplied)
    config.update({k: v for k, v in (overrides or {}).items() if v is not None})
    public_task = config.get("public_task", config.get("task"))
    if config.get("task") in TASKS:
        public_task = config["task"]
    config["public_task"] = public_task
    task_profiles = config.pop("task_profiles", {})
    config.update(task_profiles.get(public_task, {}))
    config.update({k: v for k, v in (overrides or {}).items() if v is not None})
    config["task"] = TASK_INTERNAL.get(public_task, config["task"])
    dataset = config.get("dataset", "FG26").lower()
    if dataset in ("qm9sp", "qm9"):
        dataset = "qm9s"
    if public_task == "multiobjective":
        if dataset != "fg26":
            raise ValueError("Task 4 supports FG26 only")
        if config.get("phase") == "flow":
            raise ValueError("Task 4 learned-flow support has been removed")
        if config.get("property_source", "proxy") not in ("proxy", "dft"):
            raise ValueError("Task 4 property_source must be proxy or dft")
    config["dataset"] = "FG26" if dataset == "fg26" else "QM9SP"
    dataset_profiles = config.pop("dataset_profiles", {})
    if dataset_profiles:
        if dataset not in dataset_profiles:
            raise ValueError(f"No dataset profile for {dataset}")
        config.update(dataset_profiles[dataset])
        config.update({k: v for k, v in (overrides or {}).items() if v is not None})
        config["dataset"] = "FG26" if dataset == "fg26" else "QM9SP"
        config["task"] = TASK_INTERNAL.get(public_task, config["task"])
    groups = config.pop("protocols", {})
    if groups:
        key = f'{dataset}_{config.get("objective", "large_gap")}'
        if key not in groups:
            raise ValueError(f"No protocol config for {key}")
        config.update(groups[key])
        config.update(
            {
                k: v
                for k, v in (overrides or {}).items()
                if v is not None and k not in ("task", "dataset")
            }
        )
    config["modalities"] = (
        "ir" if public_task == "ir_editing" else config.get("modalities", "uv_ir_raman")
    )
    config["decoder_type"] = (
        "crossattn" if public_task in ("translation", "budget") else "gpt"
    )
    strength = float(config.get("spec_augment_strength", 1.0))
    if not np.isfinite(strength) or strength < 0:
        raise ValueError("spec_augment_strength must be finite and nonnegative")
    recipe_keys = ["spec_aug_clean_prob", "spec_aug_global_only_prob",
                   "spec_aug_global_intensity_prob", "spec_aug_global_broadening_prob",
                   "spec_aug_global_local_shift_prob"]
    for key in recipe_keys:
        if not np.isfinite(config[key]) or not 0 <= config[key] <= 1:
            raise ValueError(f"{key} must be a finite probability in [0, 1]")
    if not np.isclose(sum(config[k] for k in recipe_keys), 1.0, rtol=0, atol=1e-8):
        raise ValueError("The five spectral augmentation recipe probabilities must sum to 1")
    for modality in ("uv", "ir_raman"):
        prefix = "spec_aug_" + modality
        low, high = config[prefix + "_scale_min"], config[prefix + "_scale_max"]
        if not np.isfinite([low, high]).all() or not 0 < low <= high:
            raise ValueError(f"{prefix}: require 0 < scale_min <= scale_max")
        for suffix in ("shift_max", "local_shift_max", "intensity_sigma", "broadening_max"):
            value = config[prefix + "_" + suffix]
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{prefix}_{suffix} must be finite and nonnegative")
    config["spec_augment"] = bool(config.get("spec_augment", True)) and strength > 0
    for key in list(config):
        if key.startswith("spec_aug_") and key.endswith(
            ("_shift_max", "_intensity_sigma", "_broadening_max")
        ):
            config[key] *= strength
        elif key.startswith("spec_aug_") and key.endswith(("_scale_min", "_scale_max")):
            config[key] = 1 + (config[key] - 1) * strength
    if any(config[f"spec_aug_{m}_scale_min"] <= 0 for m in ("uv", "ir_raman")):
        raise ValueError("spec_augment_strength makes an affine scale nonpositive")
    path_keys = {
        "dataset_root",
        "data_path",
        "splits",
        "log_dir",
        "checkpoint",
        "init_checkpoint",
        "structure_checkpoint",
        "resume_checkpoint",
        "decoder_checkpoint",
        "adapter_checkpoint",
        "input",
        "output",
        "protocol_dir",
        "conditions",
        "prepared",
        "features",
        "pairs",
        "baseline",
        "selection",
        "dft_results",
        "proxy_path",
        "proxy_dir",
        "known_reference",
        "replay",
        "chemberta_path",
        "validation_manifest",
        "queries",
        "weights_path",
        "training_conditions",
    }
    for key in path_keys:
        value = config.get(key)
        if value:
            p = Path(value).expanduser()
            config[key] = str(p if p.is_absolute() else root / p)
    for name in ("p_peak", "p_noise"):
        value = config[name]
        if not isinstance(value, (int, float)) or not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{name} must be a finite probability in [0, 1]")
    return config


def write_json(path, value):
    from pathlib import Path
    import math

    def clean(obj):
        if isinstance(obj, dict):
            return {k: clean(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [clean(v) for v in obj]
        if isinstance(obj, (float, np.floating)) and not math.isfinite(obj):
            return None
        return obj

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            clean(value), indent=2, ensure_ascii=False, default=str, allow_nan=False
        )
        + "\n"
    )


DEFAULTS = {
    "patch_len": [20, 50, 50],
    "stride": [10, 25, 25],
    "mask_ratios": [0.1, 0.1, 0.1],
    "smiles_mask_ratio": 0.0,
    "peak_mask": True,
    "p_peak": 0.5,
    "p_noise": 0.1,
    "distributed_backend": "None",
    "denoising_only": False,
    "reconstruct_weight": 5.0,
    "spec_augment": True,
    "spec_aug_clean_prob": 0.25,
    "spec_aug_global_only_prob": 0.3,
    "spec_aug_global_intensity_prob": 0.2,
    "spec_aug_global_broadening_prob": 0.1,
    "spec_aug_global_local_shift_prob": 0.15,
    "spec_aug_ir_raman_scale_min": 0.97,
    "spec_aug_ir_raman_scale_max": 1.03,
    "spec_aug_ir_raman_shift_max": 20.0,
    "spec_aug_uv_scale_min": 0.94,
    "spec_aug_uv_scale_max": 1.06,
    "spec_aug_uv_shift_max": 30.0,
    "spec_aug_ir_raman_local_shift_max": 15.0,
    "spec_aug_uv_local_shift_max": 25.0,
    "spec_aug_ir_raman_intensity_sigma": 0.5,
    "spec_aug_uv_intensity_sigma": 0.55,
    "spec_aug_ir_raman_broadening_max": 4.0,
    "spec_aug_uv_broadening_max": 10.0,
    "spec_consistency_weight": 0.0,
    "spec_aug_vis": False,
    "spec_aug_vis_dir": "vis_results_spec_aug",
    "spec_aug_vis_max": 8,
    "spec_aug_vis_prob": 1.0,
    "energy_weight": 0.0,
    "force_weight": 0.0,
    "position_noise_scale": 0.0,
    "denoising_weight": 1.0,
    "sp_struct_contrastive_weight": 1.0,
    "sp_smiles_contrastive_weight": 0.1,
    "struct_smiles_contrastive_weight": 0.1,
    "smiles_mlm_weight": 0.0,
    "training_stage": "spectra",
    "task": "pretraining",
    "standardize": False,
    "dataset": "FG26",
    "dataset_root": "datasets/FG26",
    "dataset_arg": "alpha",
    "job_id": "spec-s3",
    "pretrained_model": None,
    "model": "equivariant-transformer",
    "output_model": "Scalar",
    "output_model_noise": "VectorOutput",
    "smiles_model": "ChemBERTa",
    "spectra_model": "SpecFormer",
    "smiles_decoder": False,
    "smiles_d_ff": 1024,
    "num_epochs": 500,
    "num_steps": 600000,
    "batch_size": 100,
    "inference_batch_size": 100,
    "lr": 0.0003,
    "lr_schedule": "cosine",
    "lr_patience": 150,
    "lr_min": 5e-07,
    "lr_factor": 0.8,
    "lr_warmup_steps": 10000,
    "lr_cosine_length": 600000,
    "early_stopping_patience": 150,
    "weight_decay": 0.0,
    "ema_alpha_y": 1.0,
    "ema_alpha_dy": 1.0,
    "ngpus": 1,
    "num_nodes": 1,
    "precision": 32,
    "log_dir": "experiments/run",
    "splits": None,
    "train_size": 0.8,
    "val_size": 0.1,
    "test_size": None,
    "test_interval": 1000000,
    "save_interval": 10,
    "seed": 2026,
    "num_workers": 0,
    "redirect": False,
    "wandb_notes": None,
    "prior_model": None,
    "coord_files": None,
    "embed_files": None,
    "energy_files": None,
    "force_files": None,
    "use_dataset_md17": False,
    "embedding_dimension": 256,
    "num_layers": 8,
    "num_rbf": 64,
    "activation": "silu",
    "rbf_type": "expnorm",
    "trainable_rbf": False,
    "neighbor_embedding": True,
    "aggr": "add",
    "distance_influence": "both",
    "attn_activation": "silu",
    "num_heads": 8,
    "layernorm_on_vec": "whitened",
    "derivative": False,
    "cutoff_lower": 0.0,
    "cutoff_upper": 5.0,
    "atom_filter": -1,
    "max_z": 100,
    "max_num_neighbors": 32,
    "reduce_op": "add",
    "output_model_spec": None,
    "output_model_mol": None,
    "input_data_norm_type": "log10",
    "fusion_model": "self_attention",
    "spec_num": 3,
    "reduce_lr_when_bad": False,
    "accumulate_grad_batches": 1,
    "pretrained_smiles_decoder": None,
    "smiles_weight": 1.0,
    "contrastive_weight": 0.0,
    "entire_spectrum_mask_ratio": 0,
    "fusion_reconstruct_weight": 0.0,
    "ab_decoder_type": "gpt",
    "ab_finetune_gpt_decoder": True,
    "ab_gpt_decoder_lr_scale": 0.5,
    "ab_finetune_spectrum_encoder": False,
    "ab_spectrum_encoder_lr_scale": 0.1,
    "smiles_train_all": False,
    "gpt_n_layer": 3,
    "gpt_n_head": 8,
    "gpt_n_embd": 256,
    "gpt_block_size": 79,
    "chem_gpt_model_name": "datasets/ChemGPT",
    "chem_gpt_local_files_only": True,
    "chem_gpt_max_length": 128,
    "chem_gpt_use_lora": True,
    "chem_gpt_lora_r": 8,
    "chem_gpt_lora_alpha": 16,
    "chem_gpt_lora_dropout": 0.05,
    "chem_gpt_lora_target_modules": None,
    "load_model": None,
    "modalities": "uv_ir_raman",
    "spec_augment_strength": 1.0,
    "load_epoch": 0,
    "freeze_encoder": False,
    "decoder_type": "crossattn",
    "smiles_loss_type": "ce",
}
