import yaml
import argparse
import numpy as np
import torch
from os.path import dirname, join, exists
from pytorch_lightning.utilities import rank_zero_warn
import numpy as np
import torch
from scipy.signal import find_peaks, peak_widths
import scipy.ndimage as ndimage
from torch import nn
import torch.nn.functional as F
import json
import re
from typing import List, Dict
from scipy.signal import find_peaks, peak_widths
from rdkit import Chem


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

    assert train_size >= 0 and val_size >= 0 and test_size >= 0, (
        f"One of training ({train_size}), validation ({val_size}) or "
        f"testing ({test_size}) splits ended up with a negative size."
    )

    total = train_size + val_size + test_size
    assert dset_len >= total, (
        f"The dataset ({dset_len}) is smaller than the "
        f"combined split sizes ({total})."
    )
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

    return np.array(idx_train), np.array(idx_val), np.array(idx_test)


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
        idx_val   = splits.get("idx_val",   splits.get("val_idx"))
        idx_test  = splits.get("idx_test",  splits.get("test_idx"))     
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
    # parser.add_argument('--file', type=open, action=LoadFromFile)
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
    # parser.add_argument('--file', type=open, action=LoadFromFile)
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



def get_peak_regions(spectrum: np.ndarray, prominence: float = 0.01, width_scale: float = 3.0):
    """
    从光谱中检测峰值并返回它们的区域边界。

    Args:
        spectrum: 一维Numpy数组的光谱信号。
        prominence: 峰的突出程度阈值，用于过滤小峰。值应相对于归一化后的光谱强度。
        width_scale: 将峰宽扩大的倍数，以确保覆盖整个峰的基座。

    Returns:
        一个列表，每个元素是 (start_index, end_index) 代表一个峰的区域。
    """
    # 确保prominence至少是一个很小的值，避免检测所有噪声
    if prominence < 1e-3: 
        prominence = 1e-3
    # print(spectrum)    
    # 使用scipy的find_peaks进行峰值检测
    peaks, properties = find_peaks(spectrum, prominence=prominence)
    
    if len(peaks) == 0:
        return []

    # 计算峰宽以确定区域边界
    widths, _, left_ips, right_ips = peak_widths(spectrum, peaks, rel_height=0.85)
    
    peak_regions = []
    for i, peak_idx in enumerate(peaks):
        # 使用插值得到的边界，并进行取整
        start_idx = int(left_ips[i])
        end_idx = int(right_ips[i])
        
        # 扩大区域以更好地覆盖峰的基座
        center = peak_idx
        half_width = max(center - start_idx, end_idx - center)
        
        # 确保我们覆盖了完整的宽度
        start_idx = int(center - half_width * width_scale)
        end_idx = int(center + half_width * width_scale)

        # 确保边界不越界
        start_idx = max(0, start_idx)
        end_idx = min(len(spectrum), end_idx)

        if start_idx < end_idx:
            peak_regions.append((start_idx, end_idx))
    
    return peak_regions

def intelligent_masking(spectrum_len: int, peak_regions: list, 
                        patch_len: int, stride: int, 
                        num_peaks_to_mask: int = 1, baseline_mask_prob: float = 0.2):
    """
    实现智能掩码策略，混合掩盖峰和基线。

    Args:
        spectrum_len: 光谱总长度。
        peak_regions: get_peak_regions的输出。
        patch_len: patch的长度。
        stride: patch的步长。
        num_peaks_to_mask: 每次要掩盖的峰的数量。
        baseline_mask_prob: 掩盖基线区域的概率。

    Returns:
        mask: 一个布尔型Numpy数组，形状为[num_patches]，True代表该patch被掩码。
    """
    num_patches = (spectrum_len - patch_len) // stride + 1
    patch_mask = np.zeros(num_patches, dtype=bool)

    # 决定这次是掩盖峰还是掩盖基线
    if np.random.rand() < baseline_mask_prob and len(peak_regions) > 0:
        # --- 迷惑题：掩盖一片基线 ---
        
        # 找出所有非峰值的patch
        peak_patch_indices = set()
        for start, end in peak_regions:
            start_patch = start // stride
            end_patch = end // stride
            for i in range(start_patch, end_patch + 1):
                peak_patch_indices.add(i)
        
        baseline_patches = [i for i in range(num_patches) if i not in peak_patch_indices]
        
        if len(baseline_patches) > 5: # 确保有足够的基线区域
            # 随机选择一个基线patch作为中心，并掩盖其周围的一片区域
            mask_center = np.random.choice(baseline_patches)
            mask_width = len(peak_regions[0]) // stride if peak_regions else 5 # 掩盖一个典型峰宽的区域
            start_mask = max(0, mask_center - mask_width // 2)
            end_mask = min(num_patches, start_mask + mask_width)
            patch_mask[start_mask:end_mask] = True
            
    elif len(peak_regions) > 0:
        # --- 难题：掩盖一或多个峰 ---
        
        # 随机选择要掩盖的峰
        peaks_to_mask_indices = np.random.choice(len(peak_regions), size=min(num_peaks_to_mask, len(peak_regions)), replace=False)
        
        for idx in peaks_to_mask_indices:
            start, end = peak_regions[idx]
            # 找到所有与该峰区域重叠的patches
            start_patch = start // stride
            end_patch = end // stride
            
            # 标记这些patches需要被掩码
            patch_mask[start_patch:end_patch+1] = True
            
    # 如果没有任何峰，或者roll到了基线但基线不够长，就fallback到随机块掩码
    if not np.any(patch_mask):
        mask_start = np.random.randint(0, num_patches - 5)
        patch_mask[mask_start:mask_start+5] = True

    return patch_mask

def intelligent_masking_v2(spectrum_len: int, peak_regions: list, 
                           patch_len: int, stride: int, 
                           num_peaks_to_mask: int = 1, 
                           num_baseline_masks: int = 3, 
                           baseline_mask_width: int = 5):
    """
    实现一个更高级的混合掩码策略：
    1. 总是掩盖掉指定数量的峰。
    2. 在此基础上，额外掩盖掉几片随机的基线区域。

    Args:
        spectrum_len: 光谱总长度。
        peak_regions: get_peak_regions的输出。
        patch_len: patch的长度。
        stride: patch的步长。
        num_peaks_to_mask: 要掩盖的峰的数量。
        num_baseline_masks: 要额外掩盖的基线区域的数量。
        baseline_mask_width: 每个基线掩码区域的宽度（以patch为单位）。

    Returns:
        mask: 一个布尔型Numpy数组，形状为[num_patches]，True代表该patch被掩码。
    """
    num_patches = (spectrum_len - patch_len) // stride + 1
    patch_mask = np.zeros(num_patches, dtype=bool)
    
    # --- 步骤一：标记出所有峰所在的patches，作为“不可用于基线掩码”的区域 ---
    peak_patch_indices = set()
    if peak_regions:
        for start, end in peak_regions:
            start_patch = start // stride
            end_patch = end // stride
            for i in range(start_patch, end_patch + 1):
                # 确保索引在范围内
                if 0 <= i < num_patches:
                    peak_patch_indices.add(i)

    # --- 步骤二：掩盖一或多个峰 (如果存在峰) ---
    if num_peaks_to_mask > 0 and peak_regions:
        # 随机选择要掩盖的峰
        peaks_to_mask_indices = np.random.choice(
            len(peak_regions), 
            size=min(num_peaks_to_mask, len(peak_regions)), 
            replace=False
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

    # --- 步骤三：额外掩盖几处随机的基线区域 ---
    
    # 找出所有可用于掩码的基线patches
    # (即，所有patches中，去掉所有是峰的patches)
    available_baseline_patches = [i for i in range(num_patches) if i not in peak_patch_indices]
    
    if available_baseline_patches and num_baseline_masks > 0:
        # 随机选择N个基线区域的起始点
        # 我们要确保选择的区域不重叠，并且不超出边界
        # 为简化起见，我们先随机选择中心点，然后应用宽度
        
        num_to_sample = min(num_baseline_masks, len(available_baseline_patches))
        
        # 随机选择中心点
        mask_centers = np.random.choice(
            available_baseline_patches, 
            size=num_to_sample,
            replace=False # 确保中心点不重复
        )
        
        for center in mask_centers:
            start_mask = max(0, center - baseline_mask_width // 2)
            end_mask = min(num_patches, start_mask + baseline_mask_width)
            patch_mask[start_mask:end_mask] = True

    # --- 步骤四：Fallback 机制 ---
    # 如果经过以上操作，仍然没有任何patch被掩盖
    # (例如，光谱完全平坦，没有任何峰)
    # if not np.any(patch_mask):
    #     num_fallback_masks = num_baseline_masks if num_baseline_masks > 0 else 1
    #     for _ in range(num_fallback_masks):
    #         mask_start = np.random.randint(0, num_patches - baseline_mask_width)
    #         patch_mask[mask_start : mask_start + baseline_mask_width] = True

    return patch_mask

def smooth_and_normalize_spectrum(spectrum: np.ndarray, global_min: float, global_max: float, sigma: float = 2.0):
    """
    对光谱进行高斯平滑和全局归一化。
    """
    # 1. 高斯平滑
    if sigma > 0:
        smoothed = ndimage.gaussian_filter1d(spectrum, sigma=sigma)
    else:
        smoothed = spectrum
        
    # 2. 全局Min-Max归一化
    normalized = (smoothed - global_min) / (global_max - global_min + 1e-8)
    
    return normalized

class LabelSmoothingCrossEntropy(nn.Module):
    """
    使用 PyTorch 官方优化的 CrossEntropyLoss 实现标签平滑。
    比手动实现更稳定、更快。
    """
    def __init__(self, smoothing=0.1, pad_idx=0):
        super().__init__()
        # PyTorch 1.10+ 支持 label_smoothing 参数
        # ignore_index=pad_idx 会自动忽略 PAD token 的损失
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=smoothing, 
            ignore_index=pad_idx,
            reduction='mean' 
        )

    def forward(self, x, target):
        # x: [bs, seq_len, vocab_size] -> 模型输出
        # target: [bs, seq_len] -> 真实标签
        
        # CrossEntropyLoss 需要输入的形状为:
        # Input: [N, C] 或 [N, C, d1, d2...]
        # Target: [N] 或 [N, d1, d2...]
        
        # 为了高效计算，我们将 batch 和 seq_len 展平
        # x_flat: [bs * seq_len, vocab_size]
        x_flat = x.view(-1, x.size(-1))
        
        # target_flat: [bs * seq_len]
        target_flat = target.view(-1)
        
        return self.criterion(x_flat, target_flat)
    
class ExactMatchAccuracy(nn.Module):
    """
    计算化学分子的真实准确率。
    包含两个指标：
    1. String Accuracy (字符级): 用于观察模型是否拟合了当前的随机SMILES（训练时参考）。
    2. Chemical Accuracy (化学级): 用于评估模型是否学会了正确的分子结构（验证/测试时核心指标）。
    """
    def __init__(self, pad_idx=0):
        """
        Args:
            pad_idx: padding token 的索引
            id2char: (dict) id到字符的映射表。如果不提供，只能计算字符准确率。
        """
        super().__init__()
        self.pad_idx = pad_idx
        SMILES_DICTIONARY = {'PAD': 0, 'BOS': 1, 'EOS': 2, '#': 3, '(': 4, ')': 5, '1': 6, '2': 7, 
                     '3': 8, '4': 9, '5': 10, '=': 11, 'C': 12, 'F': 13, 'N': 14, 'O': 15, 
                     '[C-]': 16, '[CH-]': 17, '[N+]': 18, '[N-]': 19, '[NH+]': 20, 
                     '[NH2+]': 21, '[NH3+]': 22, '[O-]': 23,'[MASK]': 24, '[UNK]': 25}
    #     self.id2char = {
    #     0: 'PAD',1: 'BOS', 2: 'EOS',3: '#',4: '(',5: ')', 6: '-',  7: '1',  8: '2',
    #     9: '3',  10: '4', 11: '5', 12: '=', 13: 'C', 14: 'F', 15: 'N', 16: 'O', 17: '[C-]',  18: '[CH-]', 19: '[N+]',
    #     20: '[N-]', 21: '[NH+]', 22: '[NH2+]',  23: '[NH3+]', 24: '[O-]', 25: '[c-]', 26: '[cH-]',
    #     27: '[n-]',  28: '[nH+]', 29: '[nH]', 30: 'c', 31: 'n', 32: 'o', 33: '@', 34: '@@', 35: '/', 36: '\\', 37: '[UNK]', 38: '[MASK]'
    # }
        self.id2char = {v: k for k, v in SMILES_DICTIONARY.items()}

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        """
        返回:
            chem_acc (float): 化学等价准确率 (InChIKey匹配)
            str_acc (float): 字符串完全匹配准确率 (Token匹配)
        """
        # 1. 获取预测 Token
        pred_tokens = torch.argmax(logits, dim=-1) # [bs, seq_len]
        
        batch_size = labels.size(0)
        
        # --- 指标 A: 字符串完全匹配 (你原来的逻辑，保留用于监控过拟合) ---
        non_pad_mask = (labels != self.pad_idx)
        correct_tokens_mask = (pred_tokens == labels) & non_pad_mask
        # 检查是否该样本所有非PAD位都对
        is_string_match = (correct_tokens_mask.sum(dim=1) == non_pad_mask.sum(dim=1))
        str_acc = is_string_match.sum().item() / batch_size

        # --- 指标 B: 化学等价性匹配 (核心修改) ---
        # 如果没有提供字典，无法进行化学评估，直接返回字符串准确率
        if self.id2char is None:
            return str_acc, str_acc
        
        chemical_correct_count = 0
        
        # 将 Tensor 转回 CPU list 以便解码
        pred_list = pred_tokens.detach().cpu().tolist()
        label_list = labels.detach().cpu().tolist()
        
        for pred_seq, label_seq in zip(pred_list, label_list):
            try:
                # 1. 解码预测序列
                pred_smi = self._decode(pred_seq)
                # 2. 解码标签序列
                label_smi = self._decode(label_seq)
                
                # 3. 快速检查：如果字符串一样，化学肯定一样
                if pred_smi == label_smi:
                    chemical_correct_count += 1
                    continue
                
                # 4. RDKit 化学等价性检查
                mol_pred = Chem.MolFromSmiles(pred_smi)
                mol_label = Chem.MolFromSmiles(label_smi)
                
                if mol_pred and mol_label:
                    # 使用 InChIKey 是最严格且不受写法影响的比对方式
                    key_pred = Chem.MolToInchiKey(mol_pred)
                    key_label = Chem.MolToInchiKey(mol_label)
                    
                    if key_pred == key_label:
                        chemical_correct_count += 1
            except:
                # 解析失败视为错误
                pass
                
        chem_acc = chemical_correct_count / batch_size
        
        # 返回两个准确率，建议在训练进度条里打印 chem_acc
        return chem_acc, str_acc

    def _decode(self, token_ids):
        """辅助函数：将ID转为SMILES字符串"""
        chars = []
        for t in token_ids:
            if t == self.pad_idx: continue
            if t == 2: break # 假设 2 是 EOS
            if t == 1: continue # 假设 1 是 BOS
            # 获取字符，跳过未知
            char = self.id2char.get(t, '')
            chars.append(char)
        return "".join(chars)
    
class CustomTokenizer:
    def __init__(self, vocab_dict: dict, add_missing_special_tokens: bool = True):
        """
        CHANGED: 从一个Python字典直接初始化，而不是文件路径。
        """
        self.vocab = vocab_dict.copy() # 使用副本以避免修改原始字典

        # # NEW: 自动检查并添加缺失但必要的特殊token
        # if add_missing_special_tokens:
        #     self._add_missing_tokens()
        
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

        # --- 根据你的词表定义特殊token ---
        # 假设你的词表用BOS/EOS, 而不是CLS/SEP
        self.bos_token = 'BOS'
        self.eos_token = 'EOS'
        
        self.bos_token_id = self.vocab[self.bos_token]
        self.eos_token_id = self.vocab[self.eos_token]
        self.pad_token_id = self.vocab['PAD']
        # self.unk_token_id = self.vocab['[UNK]']
        # self.mask_token_id = self.vocab['[MASK]']

    # def _add_missing_tokens(self):
    #     """检查并添加 [UNK] 和 [MASK] token (如果不存在)。"""
    #     for token in ['[UNK]', '[MASK]']:
    #         if token not in self.vocab:
    #             new_id = len(self.vocab)
    #             self.vocab[token] = new_id
    #             print(f"注意: 特殊token '{token}' 不在你的词表中，已自动添加，ID为 {new_id}。")

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def _tokenize(self, text: str) -> List[str]:
        """
        核心分词逻辑。
        这个正则表达式可以很好地处理你的词表中的特殊项如 [N+] 和 [O-]
        """
        # 正则表达式：优先匹配中括号内的内容，然后匹配单个字符
        pattern = r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
        
        regex = re.compile(pattern)
        
        # findall 会返回所有匹配的子串列表
        tokens = [token for token in regex.findall(text)]
        
        return tokens

    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        """将token列表转换为ID列表，处理未知token。"""
        return [self.vocab.get(token) for token in tokens]
        
    def decode(self, ids: List[int]) -> str:
        # 过滤掉特殊token，让解码结果更干净
        special_ids = {self.bos_token_id, self.eos_token_id, self.pad_token_id}
        tokens = [self.inv_vocab.get(id) for id in ids if id not in special_ids and id in self.inv_vocab]
        return "".join(tokens)

    def __call__(self, texts: List[str], padding: bool = True, truncation: bool = True, max_length: int = 512, patch_max = False) -> Dict[str, torch.Tensor]:
        """
        模拟Hugging Face tokenizer的行为。
        """
        batch_ids = []
        for text in texts:
            tokens = self._tokenize(text)
            
            if truncation and len(tokens) > max_length - 2: # -2 for BOS and EOS
                tokens = tokens[:max_length - 2]
            
            # CHANGED: 使用BOS和EOS
            ids = [self.bos_token_id] + self.convert_tokens_to_ids(tokens) + [self.eos_token_id]
            batch_ids.append(ids)

        if padding:
            if patch_max:
                max_len_in_batch = max_length
            else:
                max_len_in_batch = max(len(ids) for ids in batch_ids)
            padded_batch_ids = []
            attention_masks = []
            for ids in batch_ids:
                padding_len = max_len_in_batch - len(ids)
                padded_ids = ids + ([self.pad_token_id] * padding_len)
                attention_mask = ([1] * len(ids)) + ([0] * padding_len)
                
                padded_batch_ids.append(padded_ids)
                attention_masks.append(attention_mask)
            
            return {
                "input_ids": torch.tensor(padded_batch_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_masks, dtype=torch.long)
            }
        else:
             return {"input_ids": [torch.tensor(ids, dtype=torch.long) for ids in batch_ids]}

    def get_special_tokens_mask(self, token_ids_0: List[int]) -> List[int]:
        """
        为MLM任务提供一个辅助函数，用于识别特殊token。
        """
        # CHANGED: 使用BOS, EOS, PAD
        special_tokens = {self.bos_token_id, self.eos_token_id, self.pad_token_id}
        return [1 if token_id in special_tokens else 0 for token_id in token_ids_0]
    


def uv_smart_masking(spectrum_np, spectrum_len, patch_len, stride, peak_mask):
    """
    专门为 UV 设计的掩码机制：半峰掩码 / 侧翼掩码
    
    Args:
        spectrum_np: (L,) 原始光谱数据
        spectrum_len: (L,) 原始光谱数据的长度
        patch_len: Patch 长度
        stride: 步长
        mask_ratio: 这里的 ratio 控制的是掩码覆盖峰面积的比例，而不是全谱比例
    Returns:
        patch_mask: (Num_Patches,) Bool Tensor
    """
    
    # 1. 计算 Patch 的总数
    num_patches = (spectrum_len - patch_len) // stride + 1
    patch_mask = np.zeros(num_patches, dtype=bool)
    if not peak_mask:
        return patch_mask
    # L = len(spectrum_np)
    # num_patches = int((L - patch_len) / stride + 1)
    # mask = np.zeros(num_patches, dtype=bool) # False = Visible, True = Masked

    # 2. 找到主峰 (UV通常只有1-2个大峰)
    # distance=50 防止找太密，prominence 设大一点只找主峰
    peaks, properties = find_peaks(spectrum_np, prominence=0.01)
    
    # 如果没找到峰 (基线数据)，回退到随机 Mask
    if len(peaks) == 0:
        mask_start = np.random.randint(0, num_patches - 5)
        patch_mask[mask_start:mask_start+5] = True
        return patch_mask

    # 3. 对每个峰应用“半峰掩码”策略
    # 我们使用 peak_widths 来找到峰的左右边界 (rel_height=0.95 表示几乎覆盖整个峰底)
    widths, width_heights, left_ips, right_ips = peak_widths(spectrum_np, peaks, rel_height=0.95)
    
    for i in range(len(peaks)):
        peak_idx = peaks[i]
        # 获取峰的左右边界索引 (光谱上的索引)
        left_bound = int(left_ips[i])
        right_bound = int(right_ips[i])
        
        # 将光谱索引转换为 Patch 索引
        # Patch i 覆盖范围: [i*stride, i*stride + patch_len]
        # 简单估算：Patch中心落在峰范围内就算相关 Patch
        
        # 找到覆盖这个峰的所有 Patch 的索引范围 [p_start, p_end]
        p_start = max(0, int((left_bound - patch_len/2) / stride))
        p_end = min(num_patches, int((right_bound - patch_len/2) / stride) + 1)
        
        peak_patches = np.arange(p_start, p_end)
        
        if len(peak_patches) < 2:
            continue # 峰太窄了，覆盖不到几个 Patch，跳过

        # === 核心策略：随机选择一种遮挡模式 ===
        choice = np.random.choice(['left', 'right', 'top', 'gap'])
        
        if choice == 'left':
            # 遮挡左坡 (Mask 左边 50% 的 Patch)
            cut_idx = int(len(peak_patches) * 0.5)
            mask_indices = peak_patches[:cut_idx]
            
        elif choice == 'right':
            # 遮挡右坡 (Mask 右边 50% 的 Patch)
            cut_idx = int(len(peak_patches) * 0.5)
            mask_indices = peak_patches[cut_idx:]
            
        elif choice == 'top':
            # 遮挡山顶 (保留两边的山脚) - 这是为了让模型通过底宽推测高度
            # Mask 中间 40%
            start_cut = int(len(peak_patches) * 0.3)
            end_cut = int(len(peak_patches) * 0.7)
            mask_indices = peak_patches[start_cut:end_cut]
            
        else: # 'gap'
            # 随机挖空中间的一块，但保留顶点
            # 比如：保留 [Apex]，Mask [Apex-1, Apex+1]
            # 这种比较少用在 UV，但在宽峰中可以增加难度
            num_to_mask = max(1, int(len(peak_patches) * 0.4))
            mask_indices = np.random.choice(peak_patches, num_to_mask, replace=False)

        # 应用 Mask
        if len(mask_indices) > 0:
            patch_mask[mask_indices] = True
    
    mask_len = 5

    # 随机选择起始位置（保证不越界）
    start_idx = np.random.randint(0, num_patches - mask_len + 1)

    # 连续索引
    noise_indices = np.arange(start_idx, start_idx + mask_len)

    patch_mask[noise_indices] = True
    # 4. (可选) 基线区域也随机 Mask 一点点，防止模型认为没 Mask 的就是基线
    # # 找到非峰值区域
    # all_indices = np.arange(num_patches)
    # # 简单的逻辑：不在任何峰范围内的就是基线
    # # 这里为了性能简单处理，再随机叠加 5% 的全局随机 Mask
    # baseline_noise_num = int(num_patches * 0.1)
    # noise_indices = np.random.choice(all_indices, baseline_noise_num, replace=False)
    # patch_mask[noise_indices] = True

    return patch_mask