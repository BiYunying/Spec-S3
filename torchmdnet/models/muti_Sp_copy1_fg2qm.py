from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
import sys
import matplotlib.pyplot as plt
import os
from torch.nn.init import trunc_normal_ # 需要引入这个初始化方法


sys.path.append(sys.path[0] + "/..")
from torchmdnet.models.CBAM import CBAMBlock
from torchmdnet.models.SpecFormer_layers import *

class AttentionPooling(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        # 一个简单的线性层，将d_model维的嵌入映射到一个单一的“分数”
        self.attention_net = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, patch_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patch_embeddings: Transformer编码器的输出, 形状 [bs, num_patches, d_model]。
        
        Returns:
            final_embedding: 经过注意力池化后的分子嵌入, 形状 [bs, d_model]。
        """
        # 1. 计算每个patch的“重要性分数”
        #    [bs, num_patches, d_model] -> [bs, num_patches, 1]
        attn_logits = self.attention_net(patch_embeddings)
        
        # 2. 将分数转换为权重 (通过Softmax)
        #    这会使得所有patch的权重总和为1
        attn_weights = F.softmax(attn_logits, dim=1) # Shape: [bs, num_patches, 1]
        
        # 3. 进行加权平均
        #    (patch_embeddings * attn_weights) -> 逐元素相乘
        #    .sum(dim=1) -> 在patch维度上求和
        final_embedding = (patch_embeddings * attn_weights).sum(dim=1)
        
        return final_embedding
    
class SpectralAdapter(nn.Module):
    def __init__(self, d_model, reduction_factor=4):
        super().__init__()
        # 瓶颈层维度 (e.g., 256 -> 64 -> 256)
        # 这种 "沙漏型" 结构既能减少参数量，又能强迫模型提取核心特征
        bottleneck_dim = d_model // reduction_factor
        
        self.net = nn.Sequential(
            # 1. 降维 (Down-projection)
            nn.Linear(d_model, bottleneck_dim),
            
            # 2. 非线性激活 (必须有！建议用 GELU 或 ReLU)
            # 这赋予了 Adapter 处理复杂形变和“抹除多余峰”的能力
            nn.GELU(), 
            
            # 3. 升维 (Up-projection)
            nn.Linear(bottleneck_dim, d_model)
        )
        
        # 初始化：通常把最后一个 Linear 初始化为接近 0
        # 这样在训练刚开始时，Adapter 输出为 0，不干扰预训练好的特征
        nn.init.zeros_(self.net[2].weight)
        nn.init.zeros_(self.net[2].bias)

    def forward(self, x):
        # 残差连接 (Residual Connection) 是必须的
        # 也就是：保留原来的特征，只学习“修正量”
        return x + self.net(x)
    
class SpecFormer(nn.Module):
    def __init__(
        self,
        patch_len=[20,50,50], stride=[20,50,50], output_dim=256, mask_ratios=[0.1,0.1,0.1],var_uv=None, var_ir=None, var_raman=None,
        
        input_norm_type:str='log10',  # or 'log10', 'log'
        
        n_layers:int=3, d_model=256, n_heads=16, hidden_dim=512, d_k:Optional[int]=None, d_v:Optional[int]=None, d_ff:int=512, 
        attn_dropout:float=0.1, dropout:float=0.1, act:str="gelu", key_padding_mask:bool='auto', padding_var:Optional[int]=None, 
        attn_mask:Optional[Tensor]=None, res_attention:bool=True, pre_norm:bool=False, store_attn:bool=False,
        pe:str='exp1d', learn_pe:bool=True, verbose:bool=False,
        
        fc_dropout:float=0., head_dropout = 0,
        pretrain_head:bool=False, head_type = 'flatten', individual = False,
        **kwargs
    ):
        super(SpecFormer, self).__init__()

        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.mask_ratios = mask_ratios
        self.variance_weights = []

        list_len_spectrum = [601, 3501, 3501]

        patch_nums = [int((list_len_spectrum[i] - self.patch_len[i])/self.stride[i] + 1) for i in range(len(list_len_spectrum))]
        self.patch_nums = patch_nums
        all_patch_num = sum(patch_nums)

        self.input_norm_type = input_norm_type.lower()
        # Minmax_norm
        self.norm_eps = 1e-8
        self.spectra_min_vals = [0.0, 7.871089474065229e-5, 3.300115713500418e-5]                          # [uv, ir, raman]
        self.spectra_max_vals = [2.1593494415283203, 2029.77783203125, 19843.4453125]                      # [uv, ir, raman]

        # Backbone
        self.encoders = nn.ModuleList()

        for i, modality in enumerate(["uv", "ir", "raman"]):
            encoder = TSTiEncoder(patch_num=patch_nums[i], patch_len=self.patch_len[i],stride=self.stride[i],
                                n_layers=n_layers, d_model=d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff,
                                attn_dropout=attn_dropout, dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var,
                                attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                pe=pe, learn_pe=learn_pe, verbose=verbose, **kwargs)
            setattr(self, f"backbone_{modality}", encoder)
            self.encoders.append(encoder)
        self.adapters = nn.ModuleList()
        # 融合编码器可以是一个标准的 nn.TransformerEncoder
        fusion_encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, activation=act, batch_first=True
        )
        self.fusion_encoder = nn.TransformerEncoder(
            fusion_encoder_layer, num_layers=3
        )
        # 为融合编码器的输入添加模态嵌入
        self.modality_embeddings = nn.Embedding(3, d_model)
        # self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model)) 

        # === 4. 融合后重建头 (用于第二重损失) ===
        self.fused_recon_heads1 = nn.ModuleList([
            nn.Linear(d_model, patch_len[i]) for i in range(3)
        ])
        self.fused_recon_adapter =  nn.ModuleList([
            SpectralAdapter(d_model) for i in range(3)
        ])  

        self.recon_adapter =  nn.ModuleList([
            SpectralAdapter(d_model) for i in range(3)
        ]) 
        self.reconstruct_heads1 = nn.ModuleList([nn.Linear(d_model, self.patch_len[i]) for i in range(len(self.patch_len))])
        self.pool = AttentionPooling(d_model)


        self.reset_parameters()

    def reset_parameters(self):
        for encoder in self.encoders:
            encoder.reset_parameters()

        # for h in self.reconstruct_heads:
        #     nn.init.xavier_uniform_(h.weight)
        #     h.bias.data.fill_(0)

    def forward(self, spectra, patch_masks, batch_size):  # spectra is a list

        # uv, ir, raman = x[0], x[1], x[2]

        # patching
        patched_spectra = []
        patched_spectra_masked = []
        masks = []
        loss_reconstruct1 = 0
        output = []
        weights = []
        # import random
        # rand_int = random.choice([0, 1, 2])
        for i, spec in enumerate(spectra):
            spec = spec.reshape(batch_size,-1)
            patch_mask = patch_masks[i].reshape(batch_size,-1).to(spec.device)  # [bs, num_patches]            
            if self.input_norm_type == 'minmax':
                # spec_min = spec.amin(dim=-1, keepdim=True)
                # spec_max = spec.amax(dim=-1, keepdim=True)
                # spec = (spec - spec_min) / (spec_max - spec_min + self.norm_eps)
                spec = (spec - self.spectra_min_vals[i] + self.norm_eps) / (self.spectra_max_vals[i] - self.spectra_min_vals[i] + self.norm_eps)
            elif self.input_norm_type == 'log10':
                if i == 0:  # 假设 0 是 UV
                    # 放大系数可以是 100, 1000, 取决于你的数据 min/max
                    # 目标是让 spec * scale 的值在 1 ~ 100 左右
                    scale_factor = 10.0 
                    spec = torch.log10(spec * scale_factor + 1)
                else:
                    spec = torch.log10(spec + 1)
                # spec = torch.log10(spec + 1)
            elif self.input_norm_type == 'log':
                spec = torch.log(spec + 1)
            original_spectra = spec.clone()
            spec = spec.unfold(dimension=-1, size=self.patch_len[i], step=self.stride[i])
            spec_masked = spec * (~patch_mask.unsqueeze(-1))
            # spec_masked, _, mask, _ = random_masking(spec, self.mask_ratios[i])
            # spec_masked = self.proj[i](spec_masked)
            # masks.append(mask)
            masks.append(patch_mask)
            spec = spec
            patched_spectra.append(spec)  
            patched_spectra_masked.append(spec_masked) 

            # model
            # if i == 0:
            z = self.encoders[i](spec_masked, i)          # list -> z: [bs x patch_num x d_model]
            # elif i == 1:
            # z = self.backbone(spec_masked, i)
            # elif i == 2:
            #     z = self.backbone_raman(spec_masked, i)
            z = self.recon_adapter[i](z)
            cur_reconstructed_patch = self.reconstruct_heads1[i](z)
            cur_orginal_patch = spec
            loss1 = compute_reconstruct_loss(cur_reconstructed_patch, cur_orginal_patch, patch_mask)
            # if i == 0:
            #     loss1 = loss1 * 100  # UV 权重调小一些
            loss_reconstruct1 += loss1
            # loss = (cur_reconstructed_patch - cur_orginal_patch) ** 2
            # variance_weights_patched = self.variance_weights[i].unfold(dimension=-1, size=self.patch_len[i], step=self.stride[i]) # [all_num_patches, patch_len]
            # loss = loss * variance_weights_patched.unsqueeze(0).to(loss.device) # 应用权重
            # loss_reconstruct += (loss.mean(dim=-1) * mask).sum() / mask.sum() # 只在被mask的patch上计算loss
            # weights.append(variance_weights_patched)
            # fig, axes = plt.subplots(3, 1, figsize=(10, 12))
            
            # # 确保目录存在
            # os.makedirs("vis_results0", exist_ok=True)
            # ax = axes[i]
            
            # # 1. 获取参数
            # orig_len = original_spectra.shape[-1] # 原始长度
            # stride = self.stride[i]
            
            # # 2. 获取 Patch 并还原
            # # 注意：patched_spectra_list 里已经是归一化后的数据了
            # # 我们对比的是：归一化后的真值 vs 归一化后的重建值
            
            # # 真值 Patch -> 还原
            # gt_patches = cur_orginal_patch
            # rec_full_norm = merge_patches(gt_patches, orig_len, stride)
            
            # # 预测 Patch -> 还原
            # pred_patches = cur_reconstructed_patch
            # pred_full_norm = merge_patches(pred_patches, orig_len, stride)
            
            # # Mask Patch -> 还原 (为了知道哪里被 Mask 了)
            # # Mask 是 Bool 型，转 float 计算覆盖次数
            # mask_patches = patch_mask.float().unsqueeze(-1).expand_as(gt_patches)
            # mask_full = merge_patches(mask_patches, orig_len, stride)
            # # mask_full > 0 的地方说明该位置至少在一个 patch 里被 mask 了
            
            # # 3. 提取单个样本的数据转 numpy
            # gt_plot = rec_full_norm[0].detach().cpu().numpy()
            # pred_plot = pred_full_norm[0].detach().cpu().numpy()
            # mask_plot = mask_full[0].detach().cpu().numpy()
            
            # x_axis = np.arange(len(gt_plot))
            
            # # 4. 绘图
            # # 画真值 (蓝色)
            # ax.plot(x_axis, gt_plot, label="Ground Truth (Norm)", color='blue', alpha=0.6, linewidth=1.5)
            # # 画重建值 (红色虚线)
            # ax.plot(x_axis, pred_plot, label="Reconstructed", color='red', linestyle='--', alpha=0.8, linewidth=1.5)
            
            # # 5. 高亮 Mask 区域
            # # mask_plot > 0 的区域标记为灰色背景
            # # 为了美观，使用 fill_between
            # ax.fill_between(x_axis, gt_plot.min(), gt_plot.max(), 
            #                 where=(mask_plot > 0), 
            #                 color='gray', alpha=0.3, label="Masked Regions")
            
            # ax.set_title(f"{[i]} Reconstruction")
            # ax.legend(loc='upper right')
            # ax.grid(True, linestyle=':', alpha=0.6)
        
            # plt.tight_layout()
            # # 保存图片，文件名带随机数防止覆盖，或者带 step
            # save_path = f"vis_results0/recon_vis_{np.random.randint(0, 1000)}.png"
            # plt.savefig(save_path)
            # plt.close()
            output.append(z)
        # === 阶段二：融合编码与重建 ===
        # a. 准备融合编码器的输入
        #    为每个模态的序列添加模态嵌入
        fused_input_list = []

        for i, emb_seq in enumerate(output):
            modality_emb = self.modality_embeddings(torch.tensor(i, device=emb_seq.device))
            fused_input_list.append(emb_seq + modality_emb)
        fused_patch_sequence = torch.cat(fused_input_list, dim=1)

        fused_emb_seq = self.fusion_encoder(fused_patch_sequence)

        # cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        # fused_sequence = torch.cat((cls_tokens, fused_patch_sequence), dim=1)
        # fused_emb_seq = self.fusion_encoder(fused_sequence)
        # spec_global_emb = fused_emb_seq[:, 0, :] 

        total_loss_recon_2 = 0.0
        
        # [修改点 7] 切片索引必须偏移！
        # 因为第 0 位是 CLS，Patch 从第 1 位开始
        start_idx = 0
        
        for i in range(3):
            num_p = self.patch_nums[i]
            # 取出对应模态的 Patch 特征 (跳过 CLS)
            modality_fused_seq = fused_emb_seq[:, start_idx : start_idx + num_p]
            start_idx += num_p
            modality_fused_seq = self.fused_recon_adapter[i](modality_fused_seq)
            recon_patches_2 = self.fused_recon_heads1[i](modality_fused_seq)
            loss_2 = compute_reconstruct_loss(recon_patches_2, patched_spectra[i], masks[i])
            # if i == 0:
            #     loss_2 = loss_2 * 100  # UV 权重调小一些
            total_loss_recon_2 += loss_2

        full_mask = torch.cat(masks, dim=1) # True代表被掩码
        
        # 在池化前，将被掩码位置的特征置为0，避免它们对结果产生影响
        fused_emb_seq_unmasked = fused_emb_seq * (~full_mask.unsqueeze(-1))
        
        # 使用 AttentionPooling
        spec_global_emb = self.pool(fused_emb_seq_unmasked)
        # spec_global_emb = self.spec_projector(spec_global_emb)
        # spec_global_emb = F.normalize(spec_global_emb, p=2, dim=1)

        loss_reconstruct = loss_reconstruct1 + total_loss_recon_2
        # loss_reconstruct = total_loss_recon_2
        
        # 返回 CLS token 作为全局特征
        return loss_reconstruct, spec_global_emb, loss_reconstruct1, total_loss_recon_2        #     loss_reconstruct += compute_reconstruct_loss(cur_reconstructed_patch, cur_orginal_patch, mask) 
        #     output.append(z)

        # return output, loss_reconstruct

def merge_patches(patches, original_length, stride):
    """
    将重叠的 patch 还原为完整光谱。
    
    Args:
        patches: [Batch, Num_Patches, Patch_Len] 重建出的 patch
        original_length: 原始光谱的长度
        stride: 切片时的步长
    Returns:
        reconstructed_spectrum: [Batch, Original_Length]
    """
    batch_size, num_patches, patch_len = patches.shape
    device = patches.device
    
    # 初始化输出容器和计数器
    reconstructed = torch.zeros((batch_size, original_length), device=device)
    count_map = torch.zeros((batch_size, original_length), device=device)
    
    for i in range(num_patches):
        start_idx = i * stride
        end_idx = start_idx + patch_len
        
        # 累加 Patch 值
        # 注意边界保护，防止计算长度稍微溢出
        valid_end = min(end_idx, original_length)
        # 对应的 patch 部分也要截断
        patch_valid_len = valid_end - start_idx
        
        reconstructed[:, start_idx:valid_end] += patches[:, i, :patch_valid_len]
        count_map[:, start_idx:valid_end] += 1.0
        
    # 防止除以0
    count_map[count_map == 0] = 1.0
    return reconstructed / count_map

def compute_reconstruct_loss(preds, target, mask):        
        """
        preds:   [bs x num_patch x patch_len]
        targets: [bs x num_patch x patch_len] 
        """
        loss = (preds - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum()
        return loss

def random_masking(xb, mask_ratio):
    # xb: [bs x num_patch x n_vars x patch_len]
    bs, L, D = xb.shape
    x = xb.clone()
    
    len_keep = int(L * (1 - mask_ratio))
        
    noise = torch.rand(bs, L,device=xb.device)  # noise in [0, 1], bs x L
        
    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep]
    x_kept = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

    # removed x
    x_removed = torch.zeros(bs, L-len_keep, D, device=xb.device)
    x_ = torch.cat([x_kept, x_removed], dim=1)

    # combine the kept part and the removed one
    x_masked = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1,1,D))

    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([bs, L], device=x.device)
    mask[:, :len_keep] = 0
    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)
    return x_masked, x_kept, mask, ids_restore

class TSTiEncoder(nn.Module):  #i means channel-independent
    def __init__(self, patch_num, patch_len,
                    n_layers=3, d_model=256, n_heads=16, d_k=None, d_v=None, d_ff=256, 
                    norm='BatchNorm', attn_dropout=0., dropout=0., act="gelu", key_padding_mask='auto', padding_var=None, 
                    attn_mask=None, res_attention=True, pre_norm=False, store_attn=False,
                    pe='zeros', learn_pe=True, verbose=False,**kwargs):

        super().__init__()

        self.patch_num = patch_num
        self.patch_len = patch_len
        self.adapter1 =  SpectralAdapter(d_model)


        # Input encoding
        self.W_P = nn.Linear(patch_len, d_model)
        self.W_pos = positional_encoding(pe, learn_pe, patch_num, d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

        # Encoder
        # all_patch_nums = sum(patch_nums)
        self.encoder = TSTEncoder(patch_num, d_model, n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout, dropout=dropout,
                                    pre_norm=pre_norm, activation=act, res_attention=res_attention, n_layers=n_layers, store_attn=store_attn)

    def reset_parameters(self):
        # for w in self.W_P:
        #     nn.init.xavier_uniform_(w.weight)
        #     w.bias.data.fill_(0)
        self.W_P.reset_parameters()
        self.encoder.reset_parameters()

    def forward(self, patched_spec, i) -> Tensor:                                              # x: [bs x patch_len x patch_num]
        batch_size = patched_spec.size(0)      # x: [bs x patch_num x patch_len]
        patched_spec = self.W_P(patched_spec)           # x: [bs x patch_num x d_model]
        
        patched_spec = self.dropout(patched_spec + 0.05*self.W_pos)
        patched_spec = self.adapter1(patched_spec)

        # Encoder
        z = self.encoder(patched_spec)                                                      # z: [bs x patch_num x d_model] -> [bs x patch_num x d_model]

        return z  


# Cell
class TSTEncoder(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=None, 
                        norm='BatchNorm', attn_dropout=0., dropout=0., activation='gelu',
                        res_attention=False, n_layers=1, pre_norm=False, store_attn=False):
        super().__init__()

        self.layers = nn.ModuleList([TSTEncoderLayer(q_len, d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                                                        attn_dropout=attn_dropout, dropout=dropout,
                                                        activation=activation, res_attention=res_attention,
                                                        pre_norm=pre_norm, store_attn=store_attn) for i in range(n_layers)])
        self.res_attention = res_attention

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()
    
    def forward(self, src:Tensor, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):
        output = src
        scores = None
        if self.res_attention:
            for mod in self.layers: output, scores = mod(output, prev=scores, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            return output
        else:
            for mod in self.layers: output = mod(output, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            return output


class TSTEncoderLayer(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=256, store_attn=False,
                    norm='BatchNorm', attn_dropout=0, dropout=0., bias=True, activation="gelu", res_attention=False, pre_norm=False):
        super().__init__()
        assert not d_model%n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        # Multi-Head attention
        self.res_attention = res_attention
        self.self_attn = _MultiheadAttention(d_model, n_heads, d_k, d_v, attn_dropout=attn_dropout, proj_dropout=dropout, res_attention=res_attention)

        # Add & Norm
        self.dropout_attn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_attn = nn.LayerNorm(d_model)

        # Position-wise Feed-Forward
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff, bias=bias),
                                get_activation_fn(activation),
                                nn.Dropout(dropout),
                                nn.Linear(d_ff, d_model, bias=bias))

        # Add & Norm
        self.dropout_ffn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_ffn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_ffn = nn.LayerNorm(d_model)

        self.pre_norm = pre_norm
        self.store_attn = store_attn

    def reset_parameters(self):
        self.self_attn.reset_parameters()
        if isinstance(self.norm_attn, nn.LayerNorm):
            self.norm_attn.reset_parameters()
        for name, param in self.ff.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, val=0)
        if isinstance(self.norm_ffn, nn.LayerNorm):
            self.norm_ffn.reset_parameters()

    def forward(self, src:Tensor, prev:Optional[Tensor]=None, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None) -> Tensor:

        # Multi-Head attention sublayer
        if self.pre_norm:
            src = self.norm_attn(src)
        ## Multi-Head attention
        if self.res_attention:
            src2, attn, scores = self.self_attn(src, src, src, prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        else:
            src2, attn = self.self_attn(src, src, src, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        if self.store_attn:
            self.attn = attn
        ## Add & Norm
        src = src + self.dropout_attn(src2) # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_attn(src)

        # Feed-forward sublayer
        if self.pre_norm:
            src = self.norm_ffn(src)
        ## Position-wise Feed-Forward
        src2 = self.ff(src)
        ## Add & Norm
        src = src + self.dropout_ffn(src2) # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_ffn(src)

        if self.res_attention:
            return src, scores
        else:
            return src


class _MultiheadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k=None, d_v=None, res_attention=False, attn_dropout=0., proj_dropout=0., qkv_bias=True, lsa=False):
        """Multi Head Attention Layer
        Input shape:
            Q:       [batch_size (bs) x max_q_len x d_model]
            K, V:    [batch_size (bs) x q_len x d_model]
            mask:    [q_len x q_len]
        """
        super().__init__()
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v

        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=qkv_bias)

        # Scaled Dot-Product Attention (multiple heads)
        self.res_attention = res_attention
        self.sdp_attn = _ScaledDotProductAttention(d_model, n_heads, attn_dropout=attn_dropout, res_attention=self.res_attention, lsa=lsa)

        # Poject output
        self.to_out = nn.Sequential(nn.Linear(n_heads * d_v, d_model), nn.Dropout(proj_dropout))
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_Q.weight)
        self.W_Q.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.W_K.weight)
        self.W_K.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.W_V.weight)
        self.W_V.bias.data.fill_(0)
        
    def forward(self, Q:Tensor, K:Optional[Tensor]=None, V:Optional[Tensor]=None, prev:Optional[Tensor]=None,
                key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):

        bs = Q.size(0)
        if K is None: K = Q
        if V is None: V = Q

        # Linear (+ split in multiple heads)
        q_s = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1,2)       # q_s    : [bs x n_heads x max_q_len x d_k]
        k_s = self.W_K(K).view(bs, -1, self.n_heads, self.d_k).permute(0,2,3,1)     # k_s    : [bs x n_heads x d_k x q_len] - transpose(1,2) + transpose(2,3)
        v_s = self.W_V(V).view(bs, -1, self.n_heads, self.d_v).transpose(1,2)       # v_s    : [bs x n_heads x q_len x d_v]

        # Apply Scaled Dot-Product Attention (multiple heads)
        if self.res_attention:
            output, attn_weights, attn_scores = self.sdp_attn(q_s, k_s, v_s, prev=prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        else:
            output, attn_weights = self.sdp_attn(q_s, k_s, v_s, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        # output: [bs x n_heads x q_len x d_v], attn: [bs x n_heads x q_len x q_len], scores: [bs x n_heads x max_q_len x q_len]

        # back to the original inputs dimensions
        output = output.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_v) # output: [bs x q_len x n_heads * d_v]
        output = self.to_out(output)

        if self.res_attention: return output, attn_weights, attn_scores
        else: return output, attn_weights


class _ScaledDotProductAttention(nn.Module):
    r"""Scaled Dot-Product Attention module (Attention is all you need by Vaswani et al., 2017) with optional residual attention from previous layer
    (Realformer: Transformer likes residual attention by He et al, 2020) and locality self sttention (Vision Transformer for Small-Size Datasets
    by Lee et al, 2021)"""

    def __init__(self, d_model, n_heads, attn_dropout=0., res_attention=False, lsa=False):
        super().__init__()
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.res_attention = res_attention
        head_dim = d_model // n_heads
        self.scale = nn.Parameter(torch.tensor(head_dim ** -0.5), requires_grad=lsa)
        self.lsa = lsa

    def forward(self, q:Tensor, k:Tensor, v:Tensor, prev:Optional[Tensor]=None, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):
        '''
        Input shape:
            q               : [bs x n_heads x max_q_len x d_k]
            k               : [bs x n_heads x d_k x seq_len]
            v               : [bs x n_heads x seq_len x d_v]
            prev            : [bs x n_heads x q_len x seq_len]
            key_padding_mask: [bs x seq_len]
            attn_mask       : [1 x seq_len x seq_len]
        Output shape:
            output:  [bs x n_heads x q_len x d_v]
            attn   : [bs x n_heads x q_len x seq_len]
            scores : [bs x n_heads x q_len x seq_len]
        '''

        # Scaled MatMul (q, k) - similarity scores for all pairs of positions in an input sequence
        attn_scores = torch.matmul(q, k) * self.scale      # attn_scores : [bs x n_heads x max_q_len x q_len]

        # Add pre-softmax attention scores from the previous layer (optional)
        if prev is not None: attn_scores = attn_scores + prev

        # Attention mask (optional)
        if attn_mask is not None:                                     # attn_mask with shape [q_len x seq_len] - only used when q_len == seq_len
            if attn_mask.dtype == torch.bool:
                attn_scores.masked_fill_(attn_mask, -np.inf)
            else:
                attn_scores += attn_mask

        # Key padding mask (optional)
        if key_padding_mask is not None:                              # mask with shape [bs x q_len] (only when max_w_len == q_len)
            attn_scores.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), -np.inf)

        # normalize the attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)                 # attn_weights   : [bs x n_heads x max_q_len x q_len]
        attn_weights = self.attn_dropout(attn_weights)

        # compute the new values given the attention weights
        output = torch.matmul(attn_weights, v)                        # output: [bs x n_heads x max_q_len x d_v]

        if self.res_attention: return output, attn_weights, attn_scores
        else: return output, attn_weights


def calculate_patch_coords(signal_coords, kernel_size: int, stride: int) -> torch.Tensor:
    """
    计算每个patch的中心物理坐标。
    
    Args:
        signal_coords: 包含物理坐标的一维Numpy数组 (例如,波数/eV)。
        kernel_size: 卷积核大小 (即 patch_size)。
        stride: 卷积步长。
        
    Returns:
        一个包含每个patch中心坐标的一维Tensor。
    """
    coords_tensor = signal_coords.float()
    # 使用unfold模拟分块操作
    coord_patches = coords_tensor.unfold(dimension=0, size=kernel_size, step=stride)
    # 计算每个patch坐标的平均值作为中心点
    center_coords = coord_patches.mean(dim=1)
    return center_coords.float()

class WavenumberPositionalEncoding(nn.Module):
    """
    基于物理坐标 (波数/eV) 的正弦位置编码。
    """
    def __init__(self, d_model: int, scale_factor: float = 1000.0):
        super().__init__()
        # 创建逆频率项，这是位置编码的核心
        self.inv_freq = 1.0 / (scale_factor ** (torch.arange(0, d_model, 2).float() / d_model))
        # self.register_buffer('inv_freq', inv_freq)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: 形状为 (batch_size, seq_len) 的张量，代表每个patch的物理坐标。
        
        Returns:
            pos_embedding: 形状为 (batch_size, seq_len, d_model) 的位置编码。
        """
        inv_freq = self.inv_freq.to(coords.device)
        pos_enc_input = coords.unsqueeze(-1) * inv_freq
        pe = torch.zeros(coords.shape[0], coords.shape[1], inv_freq.shape[0] * 2, device=coords.device)
        pe[:, :, 0::2] = torch.sin(pos_enc_input)
        pe[:, :, 1::2] = torch.cos(pos_enc_input)
        return pe
    

