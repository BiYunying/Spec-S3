from os.path import join
from tqdm import tqdm
import torch
from torch.utils.data import Subset
from torch_geometric.data import DataLoader
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities import rank_zero_warn
from torchmdnet import datasets
from torchmdnet.utils import make_splits, MissingEnergyException, get_peak_regions, intelligent_masking_v2, smooth_and_normalize_spectrum, uv_smart_masking
from torch_scatter import scatter
from functools import partial # 用于向collate_fn传递额外参数
from torch.utils.data import Dataset
from torch_geometric.data import Data
import numpy as np
import matplotlib.pyplot as plt
import random
from rdkit import Chem  # --- NEW: 导入RDKit库 ---
from torch.utils.data import ConcatDataset


class SpectraPreprocessedDataset(Dataset):
    def __init__(self, original_dataset, hparams, is_train=False, peak_mask=False):
        """
        一个包装器数据集，用于动态地进行光谱预处理和SMILES枚举。
        """
        self.original_dataset = original_dataset
        self.hparams = hparams
        self.is_train = is_train  # 保存训练状态
        self.peak_mask = peak_mask
        
        # 从hparams中提取配置
        if type(hparams) is dict:
            self.patch_len = hparams['patch_len']
            self.stride = hparams['stride']
        else:
            self.patch_len = hparams.patch_len
            self.stride = hparams.stride

        if peak_mask:
            self.peak_prominence = [0.01, 20.0, 20.0]
            self.num_peaks_to_mask = [1, 2, 2]
            self.width_scale = [2, 3, 3]
            self.num_baseline_masks = [1, 2, 2]
        else:
            self.peak_prominence = [0.01, 20.0, 20.0]
            self.num_peaks_to_mask = [0, 0, 0]
            self.width_scale = [0, 0, 0]
            self.num_baseline_masks = [0, 0, 0]

    def augment_spectrum(self, tensor, max_shift=20, noise_std=0.02, scale_range=(0.9, 1.1)):
        """
        对光谱Tensor进行增强：平移(补零) + 缩放 + 高斯噪声
        tensor: (1, L) or (L,)
        """
        # 1. 随机幅度缩放 (模拟浓度/光程差异)
        if scale_range:
            scale = random.uniform(scale_range[0], scale_range[1])
            tensor = tensor * scale

        # 2. 平移并补零 (模拟仪器校准漂移)
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
                    zeros = torch.zeros((tensor.shape[0], abs(shift)), dtype=tensor.dtype)
                    if shift > 0:
                        tensor = torch.cat((zeros, tensor[..., :-shift]), dim=-1)
                    else:
                        tensor = torch.cat((tensor[..., -shift:], zeros), dim=-1)
            
        return tensor

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        # 1. 从原始数据集中获取一个样本
        data_sample = self.original_dataset[idx]
        
        # 克隆样本，避免修改原始数据集
        new_data_sample = data_sample.clone()
        if hasattr(data_sample, 'smiles_tokens'):
            new_data_sample.smiles_tokens = data_sample.smiles_tokens
        mol = None
        if hasattr(data_sample, 'smiles'):
            canonical_smiles = data_sample.smiles
            mol = Chem.MolFromSmiles(canonical_smiles)

        if mol is not None:
            # 决定是否进行随机化增强
            # 注意：我把注释的 50% 改成了代码实际写的 0.2 (20%)，保持注释与代码一致
            do_random = (self.is_train and random.random() < 0.3)
            
            # 生成 SMILES
            # 【核心修改】：无论是否 random，都强制开启 kekuleSmiles=True
            # 这样保证了“格式”统一，只是“原子顺序”不同
            processed_smiles = Chem.MolToSmiles(
                mol, 
                doRandom=do_random,    # 只有训练且触发概率时才随机化
                kekuleSmiles=True,     # 【关键】始终保持凯库勒式
                canonical=not do_random # 如果不随机，就保持规范化（Canonical）
            )
            
            new_data_sample.smiles = processed_smiles
        # --- NEW: SMILES ENUMERATION (SMILES枚举) ---
        # 核心逻辑：只在训练时 (self.is_train is True) 对SMILES进行随机化处理
        # if self.is_train and random.random() < 0.2:  # 50% 概率进行SMILES随机化
        #     canonical_smiles = data_sample.smiles
            
        #     # (1) 将SMILES字符串转换为RDKit的分子对象
        #     mol = Chem.MolFromSmiles(canonical_smiles)
            
        #     # (2) 检查分子对象是否有效，然后生成一个随机SMILES
        #     if mol is not None:
        #         # 【关键修改】显式指定生成模式
        #         random_smiles = Chem.MolToSmiles(
        #             mol, 
        #             doRandom=True,        # 开启随机枚举 (增强)
        #             kekuleSmiles=True   
        #         )
        #         new_data_sample.smiles = random_smiles
        
        # 对于验证集/测试集 (is_train=False)，我们不进入if块，
        # new_data_sample.smiles 会保持其原始的、规范的SMILES，以确保评估的一致性。
        # --- END NEW ---

        processed_spectra = []
        patch_masks = []

        for i, spec_name in enumerate(["uv", "ir", "raman"]):
            spectrum_tensor = getattr(data_sample, spec_name)
            
            # # # --- 光谱增强逻辑 (您的原有代码) ---
            # if True:
            #     shift_val = 6 if spec_name != 'uv' else 2
            #     noise_val = 0.01 if spec_name != 'uv' else 0.005
            #     spectrum_tensor = self.augment_spectrum(
            #         spectrum_tensor, 
            #         max_shift=shift_val, 
            #         noise_std=noise_val,
            #         scale_range=(0.95, 1.05)
            #     )

            spectrum_np = spectrum_tensor.squeeze(0).numpy()
            # if spec_name == "uv" and self.peak_mask: # 仅对 UV 且在训练时使用新策略
            #     patch_mask = uv_smart_masking(
            #         spectrum_np, 
            #         spectrum_len=len(spectrum_np),
            #         patch_len=self.patch_len[i], 
            #         stride=self.stride[i],
            #         peak_mask=self.peak_mask
            #     )
            # else:
            # 2. 寻峰
            peak_regions = get_peak_regions(spectrum_np, prominence=self.peak_prominence[i], width_scale=self.width_scale[i])
            
            # 3. 生成 Mask
            patch_mask = intelligent_masking_v2(
                spectrum_len=len(spectrum_np),
                peak_regions=peak_regions,
                patch_len=self.patch_len[i],
                stride=self.stride[i],
                num_peaks_to_mask=self.num_peaks_to_mask[i],
                num_baseline_masks=self.num_baseline_masks[i],
                baseline_mask_width=5
            )

            processed_spectra.append(torch.from_numpy(spectrum_np).float())
            patch_masks.append(torch.from_numpy(patch_mask).bool())

        new_data_sample.spectra = processed_spectra
        new_data_sample.patch_masks = patch_masks
        new_data_sample.original_index = idx

        return new_data_sample


    
class DataModule(LightningDataModule):
    def __init__(self, hparams, dataset=None):
        # super(DataModule, self).__init__()
        super().__init__()
        # 如果 hparams 是 argparse.Namespace，直接保存为普通属性
        if hasattr(hparams, "__dict__"):
            self.hparams_dict = hparams.__dict__  # 保存原参数字典
        else:
            self.hparams_dict = hparams

        # 如果你希望用 PL 的 save_hyperparameters 功能，可以直接保存整个对象
        self.save_hyperparameters(hparams)  # dataset 不保存为超参
        self._mean, self._std = None, None
        self._saved_dataloaders = dict()
        self.dataset = dataset
        # self.hparams = self.hparams_dict

    def setup(self, stage=None):
        # 如果没有传入 dataset，按 hparams 初始化
        if self.dataset is None:
            if self.hparams.dataset == "Custom":
                self.dataset = datasets.Custom(
                    self.hparams.coord_files,
                    self.hparams.embed_files,
                    self.hparams.energy_files,
                    self.hparams.force_files,
                )
            else:
                # 如果需要加位置噪声
                if self.hparams.position_noise_scale > 0.0:
                    def transform(data):
                        noise = torch.randn_like(data.pos) * self.hparams.position_noise_scale
                        data.pos_target = noise
                        data.pos = data.pos + noise
                        return data
                else:
                    transform = None

                dataset_factory = lambda t: getattr(datasets, self.hparams.dataset)(
                    self.hparams.dataset_root,
                    dataset_arg=self.hparams.dataset_arg,
                    transform=t
                )

                # 带噪声版本
                self.dataset_maybe_noisy = dataset_factory(transform)
                # 干净版本
                self.dataset = dataset_factory(None)

        # 划分 train/val/test
        self.idx_train, self.idx_val, self.idx_test = make_splits(
            len(self.dataset),
            self.hparams.train_size,
            self.hparams.val_size,
            self.hparams.test_size,
            self.hparams.seed,
            join(self.hparams.log_dir, "splits.npz"),
            self.hparams.splits,
        )
        print(f"train {len(self.idx_train)}, val {len(self.idx_val)}, test {len(self.idx_test)}")


        # 构建 Subset 数据集
        original_train_dataset = Subset(self.dataset_maybe_noisy or self.dataset, self.idx_train)

        if self.hparams.denoising_only:
            original_val_dataset = Subset(self.dataset_maybe_noisy or self.dataset, self.idx_val)
            original_test_dataset = Subset(self.dataset_maybe_noisy or self.dataset, self.idx_test)
        else:
            original_val_dataset = Subset(self.dataset, self.idx_val)
            original_test_dataset = Subset(self.dataset, self.idx_test)

        # 【【【 关键修改 】】】
        # 用我们的包装器将它们“包”起来
        self.train_dataset = SpectraPreprocessedDataset(original_train_dataset, self.hparams, is_train=True, peak_mask=self.hparams.peak_mask)
        self.val_dataset = SpectraPreprocessedDataset(original_val_dataset, self.hparams, is_train=False, peak_mask=self.hparams.peak_mask)
        self.test_dataset = SpectraPreprocessedDataset(original_test_dataset, self.hparams, is_train=False, peak_mask=self.hparams.peak_mask)

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
        store_dataloader = (
            store_dataloader
        )
        if stage in self._saved_dataloaders and store_dataloader:
            # storing the dataloaders like this breaks calls to trainer.reload_train_val_dataloaders
            # but makes it possible that the dataloaders are not recreated on every testing epoch
            return self._saved_dataloaders[stage]

        if stage == "train":
            batch_size = self.hparams["batch_size"]
            shuffle = True
        elif stage in ["val", "test"]:
            batch_size = self.hparams["inference_batch_size"]
            shuffle = False
        # collate_fn_with_hparams = partial(spectra_collate_fn, hparams=self.hparams)

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

            # remove atomref energies from the target energy
            atomref_energy = scatter(atomref[batch.z], batch.batch, dim=0)
            return (batch.y.squeeze() - atomref_energy.squeeze()).clone()

        data = tqdm(
            self._get_dataloader(self.train_dataset, "val", store_dataloader=False),
            desc="computing mean and std",
        )
        try:
            # only remove atomref energies if the atomref prior is used
            atomref = self.atomref if self.hparams["prior_model"] == "Atomref" else None
            # extract energies from the data
            ys = torch.cat([get_energy(batch, atomref) for batch in data])
        except MissingEnergyException:
            rank_zero_warn(
                "Standardize is true but failed to compute dataset mean and "
                "standard deviation. Maybe the dataset only contains forces."
            )
            return

        # compute mean and standard deviation
        self._mean = ys.mean(dim=0)
        self._std = ys.std(dim=0)
        print(f"y mean: {self.mean}; y std: {self.std}")
