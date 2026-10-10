"""Unified training entry point for Spec-S3."""

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--task",
        required=True,
        choices=(
            "pretrain",
            "property_prediction",
            "translation",
            "scaffold_editing",
            "ir_editing",
            "multiobjective",
            "budget",
        ),
    )
    p.add_argument("--conf", required=True)
    for flag in (
        "init-checkpoint",
        "structure-checkpoint",
        "resume-checkpoint",
        "checkpoint",
        "dataset",
        "data-path",
        "splits",
        "log-dir",
        "objective",
        "phase",
        "device",
        "conditions",
        "training-conditions",
        "features",
        "prepared",
        "pairs",
        "baseline",
        "dataset-arg",
    ):
        p.add_argument("--" + flag)
    for flag in ("num-epochs", "batch-size", "num-workers", "ngpus", "steps", "seed", "validate-every", "target-index"):
        p.add_argument("--" + flag, type=int)
    for flag in (
        "reconstruct-weight",
        "denoising-weight",
        "sp-struct-contrastive-weight",
        "sp-smiles-contrastive-weight",
        "struct-smiles-contrastive-weight",
        "position-noise-scale",
        "spec-augment-strength",
        "spec-aug-clean-prob",
        "spec-aug-global-only-prob",
        "spec-aug-global-intensity-prob",
        "spec-aug-global-broadening-prob",
        "spec-aug-global-local-shift-prob",
        "spec-aug-uv-scale-min",
        "spec-aug-uv-scale-max",
        "spec-aug-uv-shift-max",
        "spec-aug-uv-local-shift-max",
        "spec-aug-uv-intensity-sigma",
        "spec-aug-uv-broadening-max",
        "spec-aug-ir-raman-scale-min",
        "spec-aug-ir-raman-scale-max",
        "spec-aug-ir-raman-shift-max",
        "spec-aug-ir-raman-local-shift-max",
        "spec-aug-ir-raman-intensity-sigma",
        "spec-aug-ir-raman-broadening-max",
        "p-peak",
        "p-noise",
        "lr",
        "preference-weight",
        "preference-beta",
    ):
        p.add_argument("--" + flag, type=float)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--fast-dev-run", action="store_true")
    return p


def main():
    args = parser().parse_args()
    from torchmdnet.utils import read_config, write_json

    config = read_config(args.conf, vars(args))
    if args.dry_run:
        print(json.dumps(config, indent=2))
        return
    import pytorch_lightning as pl

    pl.seed_everything(config["seed"], workers=True)
    if config.get("chemberta_path"):
        os.environ["SPEC_S3_CHEMBERTA"] = config["chemberta_path"]
    output = Path(config["log_dir"])
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "config.json", config)
    if args.task in ("multiobjective", "budget") and config.get("phase") != "warmup":
        if config.get("resume_checkpoint"):
            raise ValueError(
                "Task 4/5 custom trainers do not support Lightning --resume-checkpoint. Task 5 single-stage uses --checkpoint for its warmup."
            )
        from scripts import multiobjective, budget

        (multiobjective if args.task == "multiobjective" else budget).train(config)
        return
    if config.get("phase") == "warmup":
        config.update(energy_weight=0.0, smiles_weight=1.0, reconstruct_weight=0.0,
                      denoising_weight=0.0, contrastive_weight=0.0, peak_mask=False)
        write_json(output / "config.json", config)
    if not config.get("data_path") or not config.get("splits"):
        raise ValueError(
            "Set data_path and the fixed splits file explicitly; training does not invent a new split"
        )
    from torchmdnet.data import DataModule
    from torchmdnet.datasets import FG26, QM9SP
    from torchmdnet.models.heads import Atomref
    from torchmdnet.module import PretrainModule, DownstreamModule
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.loggers import CSVLogger
    import torch
    import numpy as np

    dataset_cls = {"fg26": FG26, "qm9s": QM9SP, "qm9sp": QM9SP, "qm9": QM9SP}.get(config["dataset"].lower())
    if dataset_cls is None:
        raise ValueError("Supported datasets are FG26 and QM9S (QM9SP/QM9 aliases)")
    property_task = args.task == "property_prediction"
    if property_task and args.dataset_arg is not None and args.target_index is None:
        config.pop("target_index", None)
    if property_task and args.target_index is not None and args.dataset_arg is None:
        config["dataset_arg"] = None
    dataset_args = dict(processed_path=config["data_path"])
    if property_task:
        dataset_args.update(dataset_arg=config.get("dataset_arg"), target_index=config.get("target_index"))
    data = dataset_cls(**dataset_args)
    if property_task:
        if data.label_idx is None:
            raise ValueError("Property prediction requires dataset_arg or target_index")
        config["target_index"] = data.label_idx
        data[0]  # Fail before initialization if measured labels are absent.
    with np.load(config["splits"]) as split:
        sets = [np.asarray(split["idx_" + name]) for name in ("train", "val", "test")]
        if any(len(set(x.tolist())) != len(x) for x in sets) or any(
            set(sets[i]) & set(sets[j]) for i, j in ((0, 1), (0, 2), (1, 2))
        ):
            raise ValueError(
                "Split indices contain duplicates or overlapping partitions"
            )
        if any((x < 0).any() or (x >= len(data)).any() for x in sets):
            raise ValueError("Split index is outside the supplied processed dataset")

    class Transform:
        def __init__(self, noise):
            self.noise = noise

        def __call__(self, sample):
            sample = sample.clone()
            if self.noise > 0:
                noise = torch.randn_like(sample.pos) * self.noise
                sample.pos_target = noise
                sample.pos = sample.pos + noise
            return sample

    data.transform = Transform(0.0)
    config["pretrained_model"] = config.get("init_checkpoint")
    config["load_model"] = None
    dm = DataModule(config, dataset=data)
    dm.dataset_maybe_noisy = dataset_cls(**dataset_args, transform=Transform(config["position_noise_scale"]))
    dm.setup()
    prior = None
    if property_task and config.get("prior_model") == "Atomref":
        prior = Atomref(dataset=data)
        config["prior_args"] = dict(max_z=prior.atomref.shape[0])
    write_json(output / "config.json", config)
    cls = PretrainModule if args.task == "pretrain" else DownstreamModule
    module = cls(config, prior_model=prior, mean=dm.mean, std=dm.std)
    if hasattr(module.model, "domain_adaptation_report"):
        report = module.model.domain_adaptation_report
        write_json(output / "domain_adaptation.json", report)
        print("Dataset adaptation:", report, flush=True)
    if hasattr(module.model, "structure_load_report"):
        report = module.model.structure_load_report
        write_json(output / "structure_initialization.json", report)
        print("3D initialization:", report, flush=True)
    if config.get("freeze_encoder"):
        module.model.representation_spec_model.requires_grad_(False)
    checkpoint = ModelCheckpoint(
        dirpath=output,
        filename="epoch{epoch:03d}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )
    use_gpu = (
        torch.cuda.is_available()
        and config["ngpus"] > 0
        and config.get("device") != "cpu"
    )
    trainer = pl.Trainer(
        default_root_dir=output,
        max_epochs=config["num_epochs"],
        accelerator="gpu" if use_gpu else "cpu",
        devices=config["ngpus"] if use_gpu else 1,
        strategy=(
            "ddp_find_unused_parameters_true"
            if use_gpu and config["ngpus"] > 1
            else "auto"
        ),
        precision=config["precision"],
        callbacks=[checkpoint],
        logger=CSVLogger(output, "logs"),
        fast_dev_run=args.fast_dev_run,
        gradient_clip_val=config.get("gradient_clip", 1.0),
    )
    trainer.fit(module, datamodule=dm, ckpt_path=config.get("resume_checkpoint"))
    if args.fast_dev_run:
        metrics = {k: float(v.detach().cpu()) for k, v in trainer.callback_metrics.items()}
        if any(not np.isfinite(v) for v in metrics.values()):
            raise RuntimeError(f"Nonfinite smoke-test metrics: {metrics}")
        trainer.save_checkpoint(output / "smoke.ckpt")
        write_json(output / "smoke.json", dict(metrics=metrics,
                   initialization=getattr(module.model, "load_report", None),
                   scope="one real-data train/validation batch; not a trained scientific model"))


if __name__ == "__main__":
    main()
