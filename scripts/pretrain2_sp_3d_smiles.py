import numpy as np  # sometimes needed to avoid mkl-service error
import os
import wandb
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ["WANDB_API_KEY"] = "49a3046a7d5f40f6b7495ef3563cb0eb5480b824"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
##========================================不要忘了改冻结参数和rec_adapter===============================
# 初始化 W&B
wandb.login()

import sys

sys.path.append(sys.path[0]+'/..')
import argparse
import logging
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from pytorch_lightning.utilities import rank_zero_only
from torchmdnet.module_FM_sp_3d_smiles import LNNP
from torchmdnet import datasets, priors, models
from torchmdnet.data_peak import DataModule
from torchmdnet.models import output_modules
from torchmdnet.models.utils import rbf_class_mapping, act_class_mapping
from torchmdnet.utils import LoadFromFile, LoadFromCheckpoint, save_argparse, number
from pathlib import Path
from pytorch_lightning.strategies import DDPStrategy
import torch

def get_args():
    # fmt: off
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--patch-len', type=int, nargs=3, default=[20,50,50], help='Patch lengths')
    parser.add_argument('--stride', type=int, nargs=3, default=[10,25,25], help='Strides')
    parser.add_argument('--mask-ratios', type=float, nargs=3, default=[0.1,0.1,0.1], help='Mask ratios')
    parser.add_argument('--smiles-mask-ratio', type=float, default=0.0, help='smiles mask ratio')
    parser.add_argument('--peak-mask', type=bool, default=True, help='peak mask for spectra training')
    parser.add_argument('--distributed-backend', default='None', help='Distributed backend: dp, ddp, ddp2')
    parser.add_argument('--denoising-only', type=bool, default=False, help='Denoising only')

    parser.add_argument('--reconstruct-weight', default=1.0, type=float, help='Reconstruct weight')
    parser.add_argument('--energy-weight', default=0.0, type=float, help='Energy weight')
    parser.add_argument('--force-weight', default=0.0, type=float, help='Force weight')
    parser.add_argument('--position-noise-scale', default=0.0, type=float, help='Position noise scale')
    parser.add_argument('--denoising-weight', default=0.0, type=float, help='Denoising weight')
    parser.add_argument('--sp_struct_contrastive-weight', default=0.0, type=float, help='Contrastive weight')
    parser.add_argument('--sp-smiles-contrastive-weight', default=0.0, type=float, help='Contrastive weight')
    parser.add_argument('--struct-smiles-contrastive-weight', default=0.0, type=float, help='Contrastive weight')
    parser.add_argument('--smiles-mlm-weight', default=0.0, type=float, help='Smiles MLM weight')
    
    parser.add_argument('--training-stage', type=str, default="spectra", choices=["fusion","spectra", "structure"], help='training stage')
    parser.add_argument('--task', type=str, default="pretraining", choices=["regression","smiles","pretraining"], help='tasks')
    parser.add_argument('--standardize', type=bool, default=False, help='Standardize')
    parser.add_argument('--dataset', default="QM9SP", type=str, choices=datasets.__all__, help='Dataset name')
    parser.add_argument('--dataset-root', default='./datasets/qm9sp', type=str, help='Data root')
    parser.add_argument('--dataset-arg', default="alpha", type=str, help='Target property for QM9')
    
    parser.add_argument('--job-id', default="pretrain2_spec_chemfg_qm9sp_lr_2e", type=str, help='Job ID')
    parser.add_argument('--pretrained-model', default="experiments/pretrain2_spec_chem_pcq_fg26/step=200399-epoch=119-val_loss=0.3318-test_loss=0.3353-train_per_step=0.4527.ckpt", type=str, help='Pretrained checkpoint')
    parser.add_argument('--model', type=str, default='equivariant-transformer', choices=models.__all__, help='Model')
    # parser.add_argument('--model', type=str, default=None, choices=models.__all__, help='Model')
    
    parser.add_argument('--output-model', type=str, default='Scalar', choices=output_modules.__all__, help='Output model')
    # parser.add_argument('--output-model', type=str, default=None, choices=output_modules.__all__, help='Output model')

    parser.add_argument('--output-model-noise', type=str, default='VectorOutput', choices=output_modules.__all__ + ['VectorOutput'], help='Output noise model')
    parser.add_argument('--smiles-model', type=str, default="ChemBERTa", choices=models.__all__, help='SMILES model')
    # parser.add_argument('--smiles-model', type=str, default=None, choices=models.__all__, help='SMILES model')
    parser.add_argument('--spectra-model', type=str, default="SpecFormer", choices=models.__all__, help='Spectra model')
    parser.add_argument('--smiles-decoder', type=bool, default=False, help='SMILES Decoder')

    parser.add_argument('--load-model', action=LoadFromCheckpoint, help='Restart training using a model checkpoint')  # keep first
    parser.add_argument('--conf', '-c', type=open, action=LoadFromFile, help='Configuration yaml file')  # keep second
    parser.add_argument('--num-epochs', default=1000, type=int, help='number of epochs')
    parser.add_argument('--num-steps', default=800000, type=int, help='Maximum number of gradient steps.')
    parser.add_argument('--batch-size', default=128, type=int, help='batch size')
    parser.add_argument('--inference-batch-size', default=128, type=int, help='Batchsize for validation and tests.')
    parser.add_argument('--lr', default=2e-4, type=float, help='learning rate')
    parser.add_argument('--lr-schedule', default="cosine", type=str, choices=['cosine', 'reduce_on_plateau'], help='Learning rate schedule.')
    parser.add_argument('--lr-patience', type=int, default=150, help='Patience for lr-schedule. Patience per eval-interval of validation')
    parser.add_argument('--lr-min', type=float, default=1.0e-07, help='Minimum learning rate before early stop')
    parser.add_argument('--lr-factor', type=float, default=0.8, help='LR factor for scheduler')
    parser.add_argument('--lr-warmup-steps', type=int, default=10000, help='Warm-up steps')
    parser.add_argument('--lr-cosine-length', type=int, default=600000, help='Cosine length if lr_schedule is cosine.')
    parser.add_argument('--early-stopping-patience', type=int, default=150, help='Stop training after this many epochs without improvement')
    parser.add_argument('--weight-decay', type=float, default=0.0, help='Weight decay strength')
    parser.add_argument('--ema-alpha-y', type=float, default=1.0, help='EMA alpha y')
    parser.add_argument('--ema-alpha-dy', type=float, default=1.0, help='EMA alpha dy')
    parser.add_argument('--ngpus', type=int, default=1, help='Number of GPUs')
    parser.add_argument('--num-nodes', type=int, default=1, help='Number of nodes')
    parser.add_argument('--precision', type=int, default=32, choices=[16, 32], help='Floating point precision')
    parser.add_argument('--log-dir', '-l', default='experiments/', help='log file')
    parser.add_argument('--splits', default=None, help='Npz with splits idx_train, idx_val, idx_test')
    parser.add_argument('--train-size', type=number, default=110000, help='Training set size')
    parser.add_argument('--val-size', type=number, default=10000, help='Validation set size')
    parser.add_argument('--test-size', type=number, default=None, help='Test set size')
    parser.add_argument('--test-interval', type=int, default=10, help='Test interval')
    parser.add_argument('--save-interval', type=int, default=10, help='Save interval')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    # parser.add_argument('--distributed-backend', default='None', help='Distributed backend')
    parser.add_argument('--num-workers', type=int, default=6, help='Number of workers')
    parser.add_argument('--redirect', type=bool, default=False, help='Redirect stdout/stderr')
    parser.add_argument('--wandb-notes', default="", type=str, help='Notes passed to wandb')
    parser.add_argument('--prior-model', type=str, default=None, choices=priors.__all__, help='Prior model')

    # dataset specific
    parser.add_argument('--coord-files', default=None, type=str, help='Custom coordinate files')
    parser.add_argument('--embed-files', default=None, type=str, help='Custom embedding files')
    parser.add_argument('--energy-files', default=None, type=str, help='Custom energy files')
    parser.add_argument('--force-files', default=None, type=str, help='Custom force files')   
    parser.add_argument('--use-dataset-md17', type=bool, default=False, help='use md17 as the eval dataset.')

    # architectural args
    parser.add_argument('--embedding-dimension', type=int, default=256, help='Embedding dimension')
    parser.add_argument('--num-layers', type=int, default=8, help='Number of layers')
    parser.add_argument('--num-rbf', type=int, default=64, help='Number of RBFs')
    parser.add_argument('--activation', type=str, default='silu', choices=list(act_class_mapping.keys()), help='Activation')
    parser.add_argument('--rbf-type', type=str, default='expnorm', choices=list(rbf_class_mapping.keys()), help='RBF type')
    parser.add_argument('--trainable-rbf', type=bool, default=False, help='Trainable RBF')
    parser.add_argument('--neighbor-embedding', type=bool, default=True, help='Neighbor embedding')
    parser.add_argument('--aggr', type=str, default='add', help='Aggregation op')

    # Transformer specific
    parser.add_argument('--distance-influence', type=str, default='both', choices=['keys', 'values', 'both', 'none'], help='Distance influence')
    parser.add_argument('--attn-activation', default='silu', choices=list(act_class_mapping.keys()), help='Attention activation')
    parser.add_argument('--num-heads', type=int, default=8, help='Number of heads')
    parser.add_argument('--layernorm-on-vec', type=str, default="whitened", choices=['whitened'], help='Layernorm on vec')
    parser.add_argument('--derivative', default=False, type=bool, help='Derivative')
    parser.add_argument('--cutoff-lower', type=float, default=0.0, help='Cutoff lower')
    parser.add_argument('--cutoff-upper', type=float, default=5.0, help='Cutoff upper')
    parser.add_argument('--atom-filter', type=int, default=-1, help='Atom filter')
    parser.add_argument('--max-z', type=int, default=100, help='Max Z')
    parser.add_argument('--max-num-neighbors', type=int, default=32, help='Max neighbors')
    parser.add_argument('--reduce-op', type=str, default='add', choices=['add', 'mean'], help='Reduce op')
    parser.add_argument('--output-model-spec', type=str, default=None, choices=output_modules.__all__ + ['VectorOutput'], help='Output model spec')
    parser.add_argument('--output-model-mol', type=str, default=None, choices=output_modules.__all__ + ['VectorOutput'], help='Output model mol')
    parser.add_argument('--input-data-norm-type', type=str, default='log10', choices=['minmax', 'log', 'log10', 'None'], help='Input data norm type')
    parser.add_argument('--fusion-model', type=str, default="self_attention", choices=models.__all__, help='Fusion model')
    parser.add_argument('--spec-num', type=int, default=3, help='Number of spetra')
    parser.add_argument('--reduce-lr-when-bad', type=bool, default=False, help='Reduce lr when bad')


    args = parser.parse_args()

    if args.job_id == "auto":
        assert len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')) == 1, "Might be problematic with DDP."
        if Path(args.log_dir).exists() and len(os.listdir(args.log_dir)) > 0:        
            next_job_id = str(max([int(x.name) for x in Path(args.log_dir).iterdir() if x.name.isnumeric()])+1)
        else:
            next_job_id = "1"
        args.job_id = next_job_id

    args.log_dir = str(Path(args.log_dir, args.job_id))
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    if args.redirect:
        sys.stdout = open(os.path.join(args.log_dir, "log"), "w")
        sys.stderr = sys.stdout
        logging.getLogger("pytorch_lightning").addHandler(
            logging.StreamHandler(sys.stdout)
        )

    if args.inference_batch_size is None:
        args.inference_batch_size = args.batch_size

    save_argparse(args, os.path.join(args.log_dir, "input.yaml"), exclude=["conf"])

    return args


def main():
    args = get_args()
    pl.seed_everything(args.seed, workers=True)
    print(args)

    # initialize data module
    data = DataModule(args)
    data.prepare_data()
    data.setup("fit")

    prior = None
    if args.prior_model:
        assert hasattr(priors, args.prior_model), (
            f"Unknown prior model {args['prior_model']}. "
            f"Available models are {', '.join(priors.__all__)}"
        )
        # initialize the prior model
        prior = getattr(priors, args.prior_model)(dataset=data.dataset)
        args.prior_args = prior.get_init_args()

    # initialize lightning module
    # var_uv = np.load("./datasets/qm9sp/processed/variance_weights_uv.npy")
    # var_ir = np.load("./datasets/qm9sp/processed/variance_weights_ir.npy")
    # var_raman = np.load("./datasets/qm9sp/processed/variance_weights_raman.npy")

    model = LNNP(args, prior_model=prior, mean=data.mean, std=data.std)
    # model = model.to("cuda:0") 
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.log_dir,
        monitor="val_loss",
        save_top_k=10,
        every_n_epochs=args.save_interval,  # 替代 period
        filename="{step}-{epoch}-{val_loss:.4f}-{test_loss:.4f}-{train_per_step:.4f}",
        save_last=True,
    )
    early_stopping = EarlyStopping("val_loss", patience=args.early_stopping_patience)

    tb_logger = pl.loggers.TensorBoardLogger(
        args.log_dir, name="tensorbord", version="", default_hp_metric=False
    )
    csv_logger = CSVLogger(args.log_dir, name="", version="")
    wandb_logger = WandbLogger(
        name=args.job_id,
        project="MolSpectra-0122",
        notes=args.wandb_notes,
        settings=wandb.Settings(start_method="fork", code_dir="."),
    )

    @rank_zero_only
    def log_code():
        wandb_logger.experiment # runs wandb.init, so then code can be logged next
        wandb.run.log_code(".", include_fn=lambda path: path.endswith(".py") or path.endswith(".yaml"))

    log_code()
    is_ddp = 'LOCAL_RANK' in os.environ

    if is_ddp:
        # --- DDP 环境 (由 torchrun 启动) ---
        # `devices` 会被 Lightning 自动从环境变量中解析
        # `strategy` 必须是 'ddp' 或 DDPStrategy 对象
        print("Detected DDP environment. Initializing Trainer for DDP.")
        trainer = pl.Trainer(
            max_epochs=args.num_epochs,
            max_steps=args.num_steps,
            accelerator="gpu",
            devices="auto",  # 'auto' 或 -1 都可以
            num_nodes=args.num_nodes,
            strategy="ddp",
            default_root_dir=args.log_dir,
            callbacks=[checkpoint_callback],
            logger=[tb_logger, csv_logger, wandb_logger],
            # reload_dataloaders_every_epoch=False,
            check_val_every_n_epoch=None,
            precision=args.precision,
        )
    else:
        # --- 单进程调试环境 (由 IDE Debug Runner 启动) ---
        # 明确指定只使用一张GPU，策略设为 'auto'
        print("Detected single-process environment. Initializing Trainer for debugging.")
        trainer = pl.Trainer(
            max_epochs=args.num_epochs,
            max_steps=args.num_steps,
            accelerator="gpu",
            devices=[0],  # 调试时固定使用GPU 0
            num_nodes=1,
            strategy="auto",
            default_root_dir=args.log_dir,
            callbacks=[checkpoint_callback],
            logger=[tb_logger, csv_logger, wandb_logger],
            # reload_dataloaders_every_epoch=False,
            check_val_every_n_epoch=None,
            precision=args.precision,
    )
    for param in model.model.representation_spec_model.reconstruct_heads1.parameters():
        param.requires_grad = False
    for param in model.model.representation_spec_model.fused_recon_heads1.parameters():
        param.requires_grad = False
    for param in model.model.representation_model.parameters():
        param.requires_grad = False
    for param in model.model.representation_smiles_model.parameters():
        param.requires_grad = False
    # for param in model.model.representation_spec_model.pool.parameters():
    #     param.requires_grad = True   
    trainer.fit(model, datamodule=data, ckpt_path=args.load_model)
    trainer.test(ckpt_path="last", datamodule=data)
    # model = model.load_from_checkpoint(ckpt_path)
    # trainer.test(datamodule=data)


if __name__ == "__main__":
    main()
