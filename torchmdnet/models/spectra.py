from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
import sys
import matplotlib.pyplot as plt
import os
from torch.nn.init import trunc_normal_
from torchmdnet.models.layers import CBAMBlock
from torchmdnet.models.layers import positional_encoding, Transpose, get_activation_fn


class AttentionPooling(nn.Module):

    def __init__(self, d_model: int):
        super().__init__()
        self.attention_net = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.Tanh(), nn.Linear(d_model // 2, 1)
        )

    def forward(self, patch_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patch_embeddings: Transformer encoder output, [bs, num_patches, d_model].

        Returns:
            final_embedding: Attention-pooled molecular embedding, [bs, d_model].
        """
        attn_logits = self.attention_net(patch_embeddings)
        attn_weights = F.softmax(attn_logits, dim=1)
        final_embedding = (patch_embeddings * attn_weights).sum(dim=1)
        return final_embedding


class SpectralAdapter(nn.Module):

    def __init__(self, d_model, reduction_factor=4):
        super().__init__()
        bottleneck_dim = d_model // reduction_factor
        self.net = nn.Sequential(
            nn.Linear(d_model, bottleneck_dim),
            nn.GELU(),
            nn.Linear(bottleneck_dim, d_model),
        )
        nn.init.zeros_(self.net[2].weight)
        nn.init.zeros_(self.net[2].bias)

    def reset_parameters(self):
        self.net[0].reset_parameters()
        self.net[2].reset_parameters()
        nn.init.zeros_(self.net[2].weight)
        nn.init.zeros_(self.net[2].bias)

    def forward(self, x):
        return x + self.net(x)


class CrossModulatedVibrationalAttention(nn.Module):
    """
    IR-Raman cross-modulated attention at the pre-softmax logit level.

    IR and Raman keep their own Q/K/V projections and values. Each branch uses
    the other branch's head-shared relation prior plus a donor-head-averaged
    cross-affinity term, so an IR head can read all Raman heads and vice versa.
    """

    def __init__(self, modulation_alpha=0.1, cross_alpha=0.05):
        super().__init__()
        self.modulation_alpha = modulation_alpha
        self.cross_alpha = cross_alpha
        self.gamma_ir = nn.Parameter(torch.zeros(1))
        self.gamma_raman = nn.Parameter(torch.zeros(1))
        self.eta_ir = nn.Parameter(torch.zeros(1))
        self.eta_raman = nn.Parameter(torch.zeros(1))

    def reset_parameters(self):
        nn.init.zeros_(self.gamma_ir)
        nn.init.zeros_(self.gamma_raman)
        nn.init.zeros_(self.eta_ir)
        nn.init.zeros_(self.eta_raman)

    def _project_qkv(self, attn, x):
        bs = x.size(0)
        q = attn.W_Q(x).view(bs, -1, attn.n_heads, attn.d_k).transpose(1, 2)
        k = attn.W_K(x).view(bs, -1, attn.n_heads, attn.d_k).permute(0, 2, 3, 1)
        v = attn.W_V(x).view(bs, -1, attn.n_heads, attn.d_v).transpose(1, 2)
        return (q, k, v)

    def _apply_masks(self, scores, key_padding_mask=None, attn_mask=None):
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                scores = scores.masked_fill(attn_mask, -np.inf)
            else:
                scores = scores + attn_mask
        if key_padding_mask is not None:
            scores = scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), -np.inf
            )
        return scores

    def _attention_from_scores(self, attn, scores, value):
        weights = F.softmax(scores, dim=-1)
        weights = attn.sdp_attn.attn_dropout(weights)
        output = torch.matmul(weights, value)
        output = (
            output.transpose(1, 2)
            .contiguous()
            .view(value.size(0), -1, attn.n_heads * attn.d_v)
        )
        output = attn.to_out(output)
        return (output, weights)

    def forward(
        self,
        ir,
        raman,
        ir_attn,
        raman_attn,
        key_padding_mask_ir=None,
        key_padding_mask_raman=None,
        attn_mask_ir=None,
        attn_mask_raman=None,
    ):
        if ir.size(1) != raman.size(1):
            raise ValueError(
                f"CMVA requires aligned IR/Raman patch tokens, but got {ir.size(1)} IR tokens and {raman.size(1)} Raman tokens."
            )
        q_ir, k_ir, v_ir = self._project_qkv(ir_attn, ir)
        q_raman, k_raman, v_raman = self._project_qkv(raman_attn, raman)
        scores_ir = torch.matmul(q_ir, k_ir) * ir_attn.sdp_attn.scale
        scores_raman = torch.matmul(q_raman, k_raman) * raman_attn.sdp_attn.scale
        raman_prior = scores_raman.mean(dim=1, keepdim=True)
        ir_prior = scores_ir.mean(dim=1, keepdim=True)
        cross_ir_all = (
            torch.einsum("bhnd,bkdm->bhknm", q_ir, k_raman) * ir_attn.sdp_attn.scale
        )
        cross_raman_all = (
            torch.einsum("bhnd,bkdm->bhknm", q_raman, k_ir) * raman_attn.sdp_attn.scale
        )
        cross_ir = cross_ir_all.mean(dim=2)
        cross_raman = cross_raman_all.mean(dim=2)
        gamma_ir = self.modulation_alpha * torch.tanh(self.gamma_ir)
        gamma_raman = self.modulation_alpha * torch.tanh(self.gamma_raman)
        eta_ir = self.cross_alpha * torch.tanh(self.eta_ir)
        eta_raman = self.cross_alpha * torch.tanh(self.eta_raman)
        mod_scores_ir = scores_ir + gamma_ir * raman_prior + eta_ir * cross_ir
        mod_scores_raman = (
            scores_raman + gamma_raman * ir_prior + eta_raman * cross_raman
        )
        mod_scores_ir = self._apply_masks(
            mod_scores_ir, key_padding_mask_ir, attn_mask_ir
        )
        mod_scores_raman = self._apply_masks(
            mod_scores_raman, key_padding_mask_raman, attn_mask_raman
        )
        out_ir, weights_ir = self._attention_from_scores(ir_attn, mod_scores_ir, v_ir)
        out_raman, weights_raman = self._attention_from_scores(
            raman_attn, mod_scores_raman, v_raman
        )
        return (out_ir, weights_ir, out_raman, weights_raman)


class SpecFormer(nn.Module):

    def __init__(
        self,
        patch_len=[20, 50, 50],
        stride=[20, 50, 50],
        output_dim=256,
        mask_ratios=[0.1, 0.1, 0.1],
        var_uv=None,
        var_ir=None,
        var_raman=None,
        input_norm_type: str = "log10",
        n_layers: int = 3,
        d_model=256,
        n_heads=16,
        hidden_dim=512,
        d_k: Optional[int] = None,
        d_v: Optional[int] = None,
        d_ff: int = 512,
        attn_dropout: float = 0.1,
        dropout: float = 0.1,
        act: str = "gelu",
        key_padding_mask: bool = "auto",
        padding_var: Optional[int] = None,
        attn_mask: Optional[Tensor] = None,
        res_attention: bool = True,
        pre_norm: bool = False,
        store_attn: bool = False,
        pe: str = "exp1d",
        learn_pe: bool = True,
        verbose: bool = False,
        fc_dropout: float = 0.0,
        head_dropout=0,
        pretrain_head: bool = False,
        head_type="flatten",
        individual=False,
        spec_consistency_weight: float = 0.0,
        **kwargs,
    ):
        super(SpecFormer, self).__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.mask_ratios = mask_ratios
        self.variance_weights = []
        self.spec_consistency_weight = spec_consistency_weight
        list_len_spectrum = [601, 3501, 3501]
        patch_nums = [
            int((list_len_spectrum[i] - self.patch_len[i]) / self.stride[i] + 1)
            for i in range(len(list_len_spectrum))
        ]
        self.patch_nums = patch_nums
        all_patch_num = sum(patch_nums)
        self.input_norm_type = input_norm_type.lower()
        self.norm_eps = 1e-08
        self.spectra_min_vals = [0.0, 7.871089474065229e-05, 3.300115713500418e-05]
        self.spectra_max_vals = [2.1593494415283203, 2029.77783203125, 19843.4453125]
        self.encoders = nn.ModuleList()
        for i, modality in enumerate(["uv", "ir", "raman"]):
            encoder = TSTiEncoder(
                patch_num=patch_nums[i],
                patch_len=self.patch_len[i],
                stride=self.stride[i],
                n_layers=n_layers,
                d_model=d_model,
                n_heads=n_heads,
                d_k=d_k,
                d_v=d_v,
                d_ff=d_ff,
                attn_dropout=attn_dropout,
                dropout=dropout,
                act=act,
                key_padding_mask=key_padding_mask,
                padding_var=padding_var,
                attn_mask=attn_mask,
                res_attention=res_attention,
                pre_norm=pre_norm,
                store_attn=store_attn,
                pe=pe,
                learn_pe=learn_pe,
                verbose=verbose,
                **kwargs,
            )
            setattr(self, f"backbone_{modality}", encoder)
            self.encoders.append(encoder)
        self.vibrational_attn = nn.ModuleList(
            [CrossModulatedVibrationalAttention() for _ in range(n_layers)]
        )
        self.adapters = nn.ModuleList()
        fusion_encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation=act,
            batch_first=True,
        )
        self.fusion_encoder = nn.TransformerEncoder(fusion_encoder_layer, num_layers=3)
        self.modality_embeddings = nn.Embedding(3, d_model)
        self.fused_recon_heads1 = nn.ModuleList(
            [nn.Linear(d_model, patch_len[i]) for i in range(3)]
        )
        self.reconstruct_heads = nn.ModuleList(
            [nn.Linear(d_model, self.patch_len[i]) for i in range(len(self.patch_len))]
        )
        self.pool = AttentionPooling(d_model)
        self.reset_parameters()

    def reset_parameters(self):
        for encoder in self.encoders:
            encoder.reset_parameters()
        for attn in self.vibrational_attn:
            attn.reset_parameters()
        for h in self.reconstruct_heads:
            nn.init.xavier_uniform_(h.weight)
            h.bias.data.fill_(0)

    def reset_domain_adaptation(self):
        """Fresh QM9S adapters and both reconstruction heads; retain backbone."""
        for encoder in self.encoders:
            encoder.adapter.reset_parameters()
        for head in self.reconstruct_heads:
            nn.init.xavier_uniform_(head.weight)
            nn.init.zeros_(head.bias)
        for head in self.fused_recon_heads1:
            head.reset_parameters()
        # encoders and backbone_* share parameters; include both saved aliases.
        return [name for name in self.state_dict() if (
            ".adapter." in name or name.startswith(("reconstruct_heads.", "fused_recon_heads1."))
        )]

    def forward_vibrational_pair(self, ir_patches, raman_patches):
        ir = self.encoders[1].embed(ir_patches)
        raman = self.encoders[2].embed(raman_patches)
        ir_layers = self.encoders[1].encoder.layers
        raman_layers = self.encoders[2].encoder.layers
        for ir_layer, raman_layer, cmva in zip(
            ir_layers, raman_layers, self.vibrational_attn
        ):
            ir, raman = self._forward_cmva_layer(ir, raman, ir_layer, raman_layer, cmva)
        return (ir, raman)

    def _forward_cmva_layer(self, ir, raman, ir_layer, raman_layer, cmva):
        if ir_layer.pre_norm:
            ir = ir_layer.norm_attn(ir)
            raman = raman_layer.norm_attn(raman)
        ir2, ir_attn, raman2, raman_attn = cmva(
            ir=ir,
            raman=raman,
            ir_attn=ir_layer.self_attn,
            raman_attn=raman_layer.self_attn,
        )
        if ir_layer.store_attn:
            ir_layer.attn = ir_attn
        if raman_layer.store_attn:
            raman_layer.attn = raman_attn
        ir = ir + ir_layer.dropout_attn(ir2)
        raman = raman + raman_layer.dropout_attn(raman2)
        if not ir_layer.pre_norm:
            ir = ir_layer.norm_attn(ir)
            raman = raman_layer.norm_attn(raman)
        if ir_layer.pre_norm:
            ir = ir_layer.norm_ffn(ir)
            raman = raman_layer.norm_ffn(raman)
        ir2 = ir_layer.ff(ir)
        raman2 = raman_layer.ff(raman)
        ir = ir + ir_layer.dropout_ffn(ir2)
        raman = raman + raman_layer.dropout_ffn(raman2)
        if not ir_layer.pre_norm:
            ir = ir_layer.norm_ffn(ir)
            raman = raman_layer.norm_ffn(raman)
        return (ir, raman)

    def _normalize_spectrum(self, spec, modality_index):
        if self.input_norm_type == "minmax":
            return (spec - self.spectra_min_vals[modality_index] + self.norm_eps) / (
                self.spectra_max_vals[modality_index]
                - self.spectra_min_vals[modality_index]
                + self.norm_eps
            )
        if self.input_norm_type == "log10":
            if modality_index == 0:
                scale_factor = 10.0
                return torch.log10(spec * scale_factor + 1)
            return torch.log10(spec + 1)
        if self.input_norm_type == "log":
            return torch.log(spec + 1)
        return spec

    def _encode_patched_spectra(self, patched_spectra_masked, masks):
        output = [None, None, None]
        output[0] = self.encoders[0](patched_spectra_masked[0], 0)
        output[1], output[2] = self.forward_vibrational_pair(
            patched_spectra_masked[1], patched_spectra_masked[2]
        )
        fused_input_list = []
        for i, emb_seq in enumerate(output):
            modality_emb = self.modality_embeddings(
                torch.tensor(i, device=emb_seq.device)
            )
            fused_input_list.append(emb_seq + modality_emb)
        fused_patch_sequence = torch.cat(fused_input_list, dim=1)
        fused_emb_seq = self.fusion_encoder(fused_patch_sequence)
        full_mask = torch.cat(masks, dim=1)
        fused_emb_seq_unmasked = fused_emb_seq * ~full_mask.unsqueeze(-1)
        spec_global_emb = self.pool(fused_emb_seq_unmasked)
        return (output, fused_emb_seq, spec_global_emb)

    def forward(self, spectra, patch_masks, batch_size, target_spectra=None):
        target_patched_spectra = []
        patched_spectra_masked = []
        masks = []
        loss_reconstruct1 = 0
        target_spectra = spectra if target_spectra is None else target_spectra
        for i, spec in enumerate(spectra):
            spec = spec.reshape(batch_size, -1)
            target_spec = target_spectra[i].reshape(batch_size, -1).to(spec.device)
            patch_mask = patch_masks[i].reshape(batch_size, -1).to(spec.device)
            spec = self._normalize_spectrum(spec, i)
            target_spec = self._normalize_spectrum(target_spec, i)
            spec = spec.unfold(
                dimension=-1, size=self.patch_len[i], step=self.stride[i]
            )
            target_spec = target_spec.unfold(
                dimension=-1, size=self.patch_len[i], step=self.stride[i]
            )
            spec_masked = spec * ~patch_mask.unsqueeze(-1)
            masks.append(patch_mask)
            target_patched_spectra.append(target_spec)
            patched_spectra_masked.append(spec_masked)
        output, fused_emb_seq, spec_global_emb = self._encode_patched_spectra(
            patched_spectra_masked, masks
        )
        for i, z in enumerate(output):
            cur_reconstructed_patch = self.reconstruct_heads[i](z)
            cur_orginal_patch = target_patched_spectra[i]
            loss1 = compute_reconstruct_loss(
                cur_reconstructed_patch, cur_orginal_patch, masks[i]
            )
            loss_reconstruct1 += loss1
        total_loss_recon_2 = 0.0
        start_idx = 0
        for i in range(3):
            num_p = self.patch_nums[i]
            modality_fused_seq = fused_emb_seq[:, start_idx : start_idx + num_p]
            start_idx += num_p
            recon_patches_2 = self.fused_recon_heads1[i](modality_fused_seq)
            loss_2 = compute_reconstruct_loss(
                recon_patches_2, target_patched_spectra[i], masks[i]
            )
            total_loss_recon_2 += loss_2
        loss_reconstruct = loss_reconstruct1 + total_loss_recon_2
        if self.spec_consistency_weight > 0 and target_spectra is not spectra:
            clean_masked = [
                target * ~mask.unsqueeze(-1)
                for target, mask in zip(target_patched_spectra, masks)
            ]
            _, _, clean_global_emb = self._encode_patched_spectra(clean_masked, masks)
            consistency_loss = (
                1.0
                - F.cosine_similarity(
                    spec_global_emb, clean_global_emb.detach(), dim=-1
                ).mean()
            )
            loss_reconstruct = (
                loss_reconstruct + self.spec_consistency_weight * consistency_loss
            )
        return (
            loss_reconstruct,
            spec_global_emb,
            loss_reconstruct1,
            total_loss_recon_2,
        )


def merge_patches(patches, original_length, stride):
    """
    Reconstruct a full spectrum from overlapping patches.

    Args:
        patches: Reconstructed patches, [Batch, Num_Patches, Patch_Len].
        original_length: Number of samples in the original spectrum.
        stride: Step size used to extract patches.
    Returns:
        reconstructed_spectrum: [Batch, Original_Length]
    """
    batch_size, num_patches, patch_len = patches.shape
    device = patches.device
    reconstructed = torch.zeros((batch_size, original_length), device=device)
    count_map = torch.zeros((batch_size, original_length), device=device)
    for i in range(num_patches):
        start_idx = i * stride
        end_idx = start_idx + patch_len
        valid_end = min(end_idx, original_length)
        patch_valid_len = valid_end - start_idx
        reconstructed[:, start_idx:valid_end] += patches[:, i, :patch_valid_len]
        count_map[:, start_idx:valid_end] += 1.0
    count_map[count_map == 0] = 1.0
    return reconstructed / count_map


def _smooth_patches(patches, sigma):
    if sigma <= 0:
        return patches
    radius = max(1, int(round(3.0 * sigma)))
    x = torch.arange(-radius, radius + 1, device=patches.device, dtype=patches.dtype)
    kernel = torch.exp(-(x**2) / (2.0 * sigma**2))
    kernel = (kernel / kernel.sum()).view(1, 1, -1)
    flat = patches.reshape(-1, 1, patches.size(-1))
    flat = F.pad(flat, (radius, radius), mode="replicate")
    smoothed = F.conv1d(flat, kernel)
    return smoothed.reshape_as(patches)


def _masked_patch_mse(preds, target, mask):
    loss = (preds - target) ** 2
    loss = loss.mean(dim=-1)
    denom = mask.sum().clamp_min(1)
    return (loss * mask).sum() / denom


def compute_reconstruct_loss(preds, target, mask):
    """
    preds:   [bs x num_patch x patch_len]
    targets: [bs x num_patch x patch_len]
    """
    raw_loss = _masked_patch_mse(preds, target, mask)
    smooth_loss_3 = _masked_patch_mse(
        _smooth_patches(preds, 3.0), _smooth_patches(target, 3.0), mask
    )
    smooth_loss_8 = _masked_patch_mse(
        _smooth_patches(preds, 8.0), _smooth_patches(target, 8.0), mask
    )
    return 0.2 * raw_loss + 0.5 * smooth_loss_3 + 0.3 * smooth_loss_8


def random_masking(xb, mask_ratio):
    bs, L, D = xb.shape
    x = xb.clone()
    len_keep = int(L * (1 - mask_ratio))
    noise = torch.rand(bs, L, device=xb.device)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    ids_keep = ids_shuffle[:, :len_keep]
    x_kept = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
    x_removed = torch.zeros(bs, L - len_keep, D, device=xb.device)
    x_ = torch.cat([x_kept, x_removed], dim=1)
    x_masked = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, D))
    mask = torch.ones([bs, L], device=x.device)
    mask[:, :len_keep] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)
    return (x_masked, x_kept, mask, ids_restore)


class TSTiEncoder(nn.Module):

    def __init__(
        self,
        patch_num,
        patch_len,
        n_layers=3,
        d_model=256,
        n_heads=16,
        d_k=None,
        d_v=None,
        d_ff=256,
        norm="BatchNorm",
        attn_dropout=0.0,
        dropout=0.0,
        act="gelu",
        key_padding_mask="auto",
        padding_var=None,
        attn_mask=None,
        res_attention=True,
        pre_norm=False,
        store_attn=False,
        pe="zeros",
        learn_pe=True,
        verbose=False,
        **kwargs,
    ):
        super().__init__()
        self.patch_num = patch_num
        self.patch_len = patch_len
        self.adapter = SpectralAdapter(d_model)
        self.W_P = nn.Linear(patch_len, d_model)
        self.W_pos = positional_encoding(pe, learn_pe, patch_num, d_model)
        self.dropout = nn.Dropout(dropout)
        self.encoder = TSTEncoder(
            patch_num,
            d_model,
            n_heads,
            d_k=d_k,
            d_v=d_v,
            d_ff=d_ff,
            norm=norm,
            attn_dropout=attn_dropout,
            dropout=dropout,
            pre_norm=pre_norm,
            activation=act,
            res_attention=res_attention,
            n_layers=n_layers,
            store_attn=store_attn,
        )

    def reset_parameters(self):
        self.W_P.reset_parameters()
        self.encoder.reset_parameters()

    def embed(self, patched_spec) -> Tensor:
        patched_spec = self.W_P(patched_spec)
        patched_spec = self.dropout(patched_spec + 0.05 * self.W_pos)
        patched_spec = self.adapter(patched_spec)
        return patched_spec

    def forward(self, patched_spec, i) -> Tensor:
        batch_size = patched_spec.size(0)
        patched_spec = self.embed(patched_spec)
        z = self.encoder(patched_spec)
        return z


class TSTEncoder(nn.Module):

    def __init__(
        self,
        q_len,
        d_model,
        n_heads,
        d_k=None,
        d_v=None,
        d_ff=None,
        norm="BatchNorm",
        attn_dropout=0.0,
        dropout=0.0,
        activation="gelu",
        res_attention=False,
        n_layers=1,
        pre_norm=False,
        store_attn=False,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TSTEncoderLayer(
                    q_len,
                    d_model,
                    n_heads=n_heads,
                    d_k=d_k,
                    d_v=d_v,
                    d_ff=d_ff,
                    norm=norm,
                    attn_dropout=attn_dropout,
                    dropout=dropout,
                    activation=activation,
                    res_attention=res_attention,
                    pre_norm=pre_norm,
                    store_attn=store_attn,
                )
                for i in range(n_layers)
            ]
        )
        self.res_attention = res_attention

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(
        self,
        src: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ):
        output = src
        scores = None
        if self.res_attention:
            for mod in self.layers:
                output, scores = mod(
                    output,
                    prev=scores,
                    key_padding_mask=key_padding_mask,
                    attn_mask=attn_mask,
                )
            return output
        else:
            for mod in self.layers:
                output = mod(
                    output, key_padding_mask=key_padding_mask, attn_mask=attn_mask
                )
            return output


class TSTEncoderLayer(nn.Module):

    def __init__(
        self,
        q_len,
        d_model,
        n_heads,
        d_k=None,
        d_v=None,
        d_ff=256,
        store_attn=False,
        norm="BatchNorm",
        attn_dropout=0,
        dropout=0.0,
        bias=True,
        activation="gelu",
        res_attention=False,
        pre_norm=False,
    ):
        super().__init__()
        assert (
            not d_model % n_heads
        ), f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v
        self.res_attention = res_attention
        self.self_attn = _MultiheadAttention(
            d_model,
            n_heads,
            d_k,
            d_v,
            attn_dropout=attn_dropout,
            proj_dropout=dropout,
            res_attention=res_attention,
        )
        self.dropout_attn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(
                Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2)
            )
        else:
            self.norm_attn = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=bias),
            get_activation_fn(activation),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model, bias=bias),
        )
        self.dropout_ffn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_ffn = nn.Sequential(
                Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2)
            )
        else:
            self.norm_ffn = nn.LayerNorm(d_model)
        self.pre_norm = pre_norm
        self.store_attn = store_attn

    def reset_parameters(self):
        self.self_attn.reset_parameters()
        if isinstance(self.norm_attn, nn.LayerNorm):
            self.norm_attn.reset_parameters()
        for name, param in self.ff.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.constant_(param, val=0)
        if isinstance(self.norm_ffn, nn.LayerNorm):
            self.norm_ffn.reset_parameters()

    def forward(
        self,
        src: Tensor,
        prev: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        if self.pre_norm:
            src = self.norm_attn(src)
        if self.res_attention:
            src2, attn, scores = self.self_attn(
                src,
                src,
                src,
                prev,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask,
            )
        else:
            src2, attn = self.self_attn(
                src, src, src, key_padding_mask=key_padding_mask, attn_mask=attn_mask
            )
        if self.store_attn:
            self.attn = attn
        src = src + self.dropout_attn(src2)
        if not self.pre_norm:
            src = self.norm_attn(src)
        if self.pre_norm:
            src = self.norm_ffn(src)
        src2 = self.ff(src)
        src = src + self.dropout_ffn(src2)
        if not self.pre_norm:
            src = self.norm_ffn(src)
        if self.res_attention:
            return (src, scores)
        else:
            return src


class _MultiheadAttention(nn.Module):

    def __init__(
        self,
        d_model,
        n_heads,
        d_k=None,
        d_v=None,
        res_attention=False,
        attn_dropout=0.0,
        proj_dropout=0.0,
        qkv_bias=True,
        lsa=False,
    ):
        """Multi Head Attention Layer
        Input shape:
            Q:       [batch_size (bs) x max_q_len x d_model]
            K, V:    [batch_size (bs) x q_len x d_model]
            mask:    [q_len x q_len]
        """
        super().__init__()
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v
        self.n_heads, self.d_k, self.d_v = (n_heads, d_k, d_v)
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=qkv_bias)
        self.res_attention = res_attention
        self.sdp_attn = _ScaledDotProductAttention(
            d_model,
            n_heads,
            attn_dropout=attn_dropout,
            res_attention=self.res_attention,
            lsa=lsa,
        )
        self.to_out = nn.Sequential(
            nn.Linear(n_heads * d_v, d_model), nn.Dropout(proj_dropout)
        )

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_Q.weight)
        self.W_Q.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.W_K.weight)
        self.W_K.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.W_V.weight)
        self.W_V.bias.data.fill_(0)

    def forward(
        self,
        Q: Tensor,
        K: Optional[Tensor] = None,
        V: Optional[Tensor] = None,
        prev: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ):
        bs = Q.size(0)
        if K is None:
            K = Q
        if V is None:
            V = Q
        q_s = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1, 2)
        k_s = self.W_K(K).view(bs, -1, self.n_heads, self.d_k).permute(0, 2, 3, 1)
        v_s = self.W_V(V).view(bs, -1, self.n_heads, self.d_v).transpose(1, 2)
        if self.res_attention:
            output, attn_weights, attn_scores = self.sdp_attn(
                q_s,
                k_s,
                v_s,
                prev=prev,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask,
            )
        else:
            output, attn_weights = self.sdp_attn(
                q_s, k_s, v_s, key_padding_mask=key_padding_mask, attn_mask=attn_mask
            )
        output = (
            output.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_v)
        )
        output = self.to_out(output)
        if self.res_attention:
            return (output, attn_weights, attn_scores)
        else:
            return (output, attn_weights)


class _ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention module (Attention is all you need by Vaswani et al., 2017) with optional residual attention from previous layer
    (Realformer: Transformer likes residual attention by He et al, 2020) and locality self sttention (Vision Transformer for Small-Size Datasets
    by Lee et al, 2021)"""

    def __init__(
        self, d_model, n_heads, attn_dropout=0.0, res_attention=False, lsa=False
    ):
        super().__init__()
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.res_attention = res_attention
        head_dim = d_model // n_heads
        self.scale = nn.Parameter(torch.tensor(head_dim ** (-0.5)), requires_grad=lsa)
        self.lsa = lsa

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        prev: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ):
        """
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
        """
        attn_scores = torch.matmul(q, k) * self.scale
        if prev is not None:
            attn_scores = attn_scores + prev
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_scores.masked_fill_(attn_mask, -np.inf)
            else:
                attn_scores += attn_mask
        if key_padding_mask is not None:
            attn_scores.masked_fill_(
                key_padding_mask.unsqueeze(1).unsqueeze(2), -np.inf
            )
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        output = torch.matmul(attn_weights, v)
        if self.res_attention:
            return (output, attn_weights, attn_scores)
        else:
            return (output, attn_weights)


def calculate_patch_coords(
    signal_coords, kernel_size: int, stride: int
) -> torch.Tensor:
    """
    Compute the physical coordinate at the center of each patch.

    Args:
        signal_coords: One-dimensional NumPy array of physical coordinates (wavenumber or eV).
        kernel_size: Convolution kernel size (patch size).
        stride: Convolution stride.

    Returns:
        A one-dimensional tensor containing the center coordinate of each patch.
    """
    coords_tensor = signal_coords.float()
    coord_patches = coords_tensor.unfold(dimension=0, size=kernel_size, step=stride)
    center_coords = coord_patches.mean(dim=1)
    return center_coords.float()


class WavenumberPositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding based on physical coordinates (wavenumber or eV).
    """

    def __init__(self, d_model: int, scale_factor: float = 1000.0):
        super().__init__()
        self.inv_freq = 1.0 / scale_factor ** (
            torch.arange(0, d_model, 2).float() / d_model
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: Physical coordinates of patches, shaped (batch_size, seq_len).

        Returns:
            pos_embedding: Positional encodings, shaped (batch_size, seq_len, d_model).
        """
        inv_freq = self.inv_freq.to(coords.device)
        pos_enc_input = coords.unsqueeze(-1) * inv_freq
        pe = torch.zeros(
            coords.shape[0],
            coords.shape[1],
            inv_freq.shape[0] * 2,
            device=coords.device,
        )
        pe[:, :, 0::2] = torch.sin(pos_enc_input)
        pe[:, :, 1::2] = torch.cos(pos_enc_input)
        return pe


class IRSpecFormer(nn.Module):

    def __init__(
        self,
        patch_len=[50],
        stride=[25],
        output_dim=256,
        mask_ratios=[0.1],
        var_uv=None,
        var_ir=None,
        var_raman=None,
        input_norm_type: str = "log10",
        n_layers: int = 3,
        d_model=256,
        n_heads=16,
        hidden_dim=512,
        d_k: Optional[int] = None,
        d_v: Optional[int] = None,
        d_ff: int = 512,
        attn_dropout: float = 0.1,
        dropout: float = 0.1,
        act: str = "gelu",
        key_padding_mask: bool = "auto",
        padding_var: Optional[int] = None,
        attn_mask: Optional[Tensor] = None,
        res_attention: bool = True,
        pre_norm: bool = False,
        store_attn: bool = False,
        pe: str = "exp1d",
        learn_pe: bool = True,
        verbose: bool = False,
        fc_dropout: float = 0.0,
        head_dropout=0,
        pretrain_head: bool = False,
        head_type="flatten",
        individual=False,
        spec_consistency_weight: float = 0.0,
        **kwargs,
    ):
        super(IRSpecFormer, self).__init__()
        if len(patch_len) > 1:
            patch_len = [patch_len[1]]
        if len(stride) > 1:
            stride = [stride[1]]
        if len(mask_ratios) > 1:
            mask_ratios = [mask_ratios[1]]
        self.patch_len = patch_len
        self.stride = stride
        self.mask_ratios = mask_ratios
        self.variance_weights = []
        self.spec_consistency_weight = spec_consistency_weight
        self.source_modality_index = 1
        list_len_spectrum = [3501]
        patch_nums = [
            int((list_len_spectrum[i] - self.patch_len[i]) / self.stride[i] + 1)
            for i in range(len(list_len_spectrum))
        ]
        self.patch_nums = patch_nums
        all_patch_num = sum(patch_nums)
        self.input_norm_type = input_norm_type.lower()
        self.norm_eps = 1e-08
        self.spectra_min_vals = [7.871089474065229e-05]
        self.spectra_max_vals = [2029.77783203125]
        self.encoders = nn.ModuleList()
        for i, modality in enumerate(["ir"]):
            encoder = TSTiEncoder(
                patch_num=patch_nums[i],
                patch_len=self.patch_len[i],
                stride=self.stride[i],
                n_layers=n_layers,
                d_model=d_model,
                n_heads=n_heads,
                d_k=d_k,
                d_v=d_v,
                d_ff=d_ff,
                attn_dropout=attn_dropout,
                dropout=dropout,
                act=act,
                key_padding_mask=key_padding_mask,
                padding_var=padding_var,
                attn_mask=attn_mask,
                res_attention=res_attention,
                pre_norm=pre_norm,
                store_attn=store_attn,
                pe=pe,
                learn_pe=learn_pe,
                verbose=verbose,
                **kwargs,
            )
            setattr(self, f"backbone_{modality}", encoder)
            self.encoders.append(encoder)
        self.adapters = nn.ModuleList()
        self.reconstruct_heads = nn.ModuleList(
            [nn.Linear(d_model, self.patch_len[i]) for i in range(len(self.patch_len))]
        )
        self.pool = AttentionPooling(d_model)
        self.reset_parameters()

    def reset_parameters(self):
        for encoder in self.encoders:
            encoder.reset_parameters()
        for h in self.reconstruct_heads:
            nn.init.xavier_uniform_(h.weight)
            h.bias.data.fill_(0)

    def _normalize_spectrum(self, spec, modality_index):
        if self.input_norm_type == "minmax":
            return (spec - self.spectra_min_vals[modality_index] + self.norm_eps) / (
                self.spectra_max_vals[modality_index]
                - self.spectra_min_vals[modality_index]
                + self.norm_eps
            )
        if self.input_norm_type == "log10":
            return torch.log10(spec + 1)
        if self.input_norm_type == "log":
            return torch.log(spec + 1)
        return spec

    def _encode_patched_spectra(self, patched_spectra_masked, masks):
        output = [self.encoders[0](patched_spectra_masked[0], 0)]
        ir_emb_seq_unmasked = output[0] * ~masks[0].unsqueeze(-1)
        spec_global_emb = self.pool(ir_emb_seq_unmasked)
        return (output, output[0], spec_global_emb)

    def _select_ir_only(self, values):
        if values is None:
            return None
        if len(values) == 1:
            return values
        return [values[self.source_modality_index]]

    def forward(self, spectra, patch_masks, batch_size, target_spectra=None):
        target_patched_spectra = []
        patched_spectra_masked = []
        masks = []
        loss_reconstruct1 = 0
        target_spectra = spectra if target_spectra is None else target_spectra
        spectra = self._select_ir_only(spectra)
        patch_masks = self._select_ir_only(patch_masks)
        target_spectra = self._select_ir_only(target_spectra)
        for i, spec in enumerate(spectra):
            spec = spec.reshape(batch_size, -1)
            target_spec = target_spectra[i].reshape(batch_size, -1).to(spec.device)
            patch_mask = patch_masks[i].reshape(batch_size, -1).to(spec.device)
            spec = self._normalize_spectrum(spec, i)
            target_spec = self._normalize_spectrum(target_spec, i)
            spec = spec.unfold(
                dimension=-1, size=self.patch_len[i], step=self.stride[i]
            )
            target_spec = target_spec.unfold(
                dimension=-1, size=self.patch_len[i], step=self.stride[i]
            )
            spec_masked = spec * ~patch_mask.unsqueeze(-1)
            masks.append(patch_mask)
            target_patched_spectra.append(target_spec)
            patched_spectra_masked.append(spec_masked)
        output, ir_emb_seq, spec_global_emb = self._encode_patched_spectra(
            patched_spectra_masked, masks
        )
        for i, z in enumerate(output):
            cur_reconstructed_patch = self.reconstruct_heads[i](z)
            cur_orginal_patch = target_patched_spectra[i]
            loss1 = compute_reconstruct_loss(
                cur_reconstructed_patch, cur_orginal_patch, masks[i]
            )
            loss_reconstruct1 += loss1
        total_loss_recon_2 = torch.zeros(
            (), device=spec_global_emb.device, dtype=spec_global_emb.dtype
        )
        loss_reconstruct = loss_reconstruct1
        if self.spec_consistency_weight > 0 and target_spectra is not spectra:
            clean_masked = [
                target * ~mask.unsqueeze(-1)
                for target, mask in zip(target_patched_spectra, masks)
            ]
            _, _, clean_global_emb = self._encode_patched_spectra(clean_masked, masks)
            consistency_loss = (
                1.0
                - F.cosine_similarity(
                    spec_global_emb, clean_global_emb.detach(), dim=-1
                ).mean()
            )
            loss_reconstruct = (
                loss_reconstruct + self.spec_consistency_weight * consistency_loss
            )
        return (
            loss_reconstruct,
            spec_global_emb,
            loss_reconstruct1,
            total_loss_recon_2,
        )


def make_encoder(*, modalities="uv_ir_raman", **kwargs):
    cls = IRSpecFormer if modalities == "ir" else SpecFormer
    if modalities not in ("ir", "uv_ir_raman"):
        raise ValueError(f"Unsupported modalities: {modalities}")
    return cls(**kwargs)
