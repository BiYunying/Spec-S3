import re
from typing import Optional, List, Tuple
import torch
from torch.autograd import grad
from torch import nn
from torch_scatter import scatter
from pytorch_lightning.utilities import rank_zero_warn
from torchmdnet.models import output_modules
from torchmdnet.models.wrappers import AtomFilter
from torchmdnet import priors
import warnings
import torch.nn.functional as F

def count_parameters(model, trainable_only=True):
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())

def create_model(args, prior_model=None, mean=None, std=None, var_uv=None, var_ir=None, var_raman=None):
    shared_args = dict(
        hidden_channels=args["embedding_dimension"],
        num_layers=args["num_layers"],
        num_rbf=args["num_rbf"],
        rbf_type=args["rbf_type"],
        trainable_rbf=args["trainable_rbf"],
        activation=args["activation"],
        neighbor_embedding=args["neighbor_embedding"],
        cutoff_lower=args["cutoff_lower"],
        cutoff_upper=args["cutoff_upper"],
        max_z=args["max_z"],
        max_num_neighbors=args["max_num_neighbors"],
    )

    # representation network
    if args["model"] == "graph-network":
        from torchmdnet.models.torchmd_gn import TorchMD_GN

        is_equivariant = False
        representation_model = TorchMD_GN(
            num_filters=args["embedding_dimension"], aggr=args["aggr"], **shared_args
        )
    elif args["model"] == "transformer":
        from torchmdnet.models.torchmd_t import TorchMD_T

        is_equivariant = False
        representation_model = TorchMD_T(
            attn_activation=args["attn_activation"],
            num_heads=args["num_heads"],
            distance_influence=args["distance_influence"],
            **shared_args,
        )
    elif args["model"] == "equivariant-transformer":
        from torchmdnet.models.torchmd_et import TorchMD_ET

        is_equivariant = True
        representation_model = TorchMD_ET(
            attn_activation=args["attn_activation"],
            num_heads=args["num_heads"],
            distance_influence=args["distance_influence"],
            layernorm_on_vec=args["layernorm_on_vec"],
            use_dataset_md17=args["use_dataset_md17"],
            **shared_args,
        )
    else:
        raise ValueError(f'Unknown architecture: {args["model"]}')

    representation_spec_model = None
    if args["spectra_model"] == "CNN-AM":
        from torchmdnet.models import CNN_AM

        input_dim = 1500
        in_channel = 1
        representation_spec_model = CNN_AM(
            input_dim=input_dim,
            in_channel=in_channel,
            output_channel=args["embedding_dimension"],
        )
    elif args["spectra_model"] == "SpecFormer":
        from torchmdnet.models.muti_Sp_copy1_fg2qm import SpecFormer

        representation_spec_model = SpecFormer(
            patch_len=args["patch_len"],
            stride=args["stride"],
            output_dim=args["embedding_dimension"],
            input_norm_type=args["input_data_norm_type"],
            var_raman=var_raman,
            var_uv=var_uv,
            var_ir=var_ir,
            # n_heads=n_heads,
            # n_layers=n_layers,
        )
        num_params = count_parameters(representation_spec_model)
        print(f"Spectra encoder trainable params: {num_params:,}")
    
    fusion_model = None
    # if args["fusion_model"] == "self_attention":
    #     from torchmdnet.models.foundation_model_copy import MultiModalFusion
    #     fusion_model = MultiModalFusion(
    #         output_dim=args["embedding_dimension"], 
    #         patch_len=args["patch_len"],
    #         stride=args["stride"],
    #         # n_heads=8, 
    #         # n_layers=2, 
    #         # dropout=0.1, 
    #         # max_spec_tokens_per_modality=16, 
    #         num_modalities=args['spec_num'],
    #     )

    # atom filter
    if not args["derivative"] and args["atom_filter"] > -1:
        representation_model = AtomFilter(representation_model, args["atom_filter"])
    elif args["atom_filter"] > -1:
        raise ValueError("Derivative and atom filter can't be used together")

    # prior model
    if args["prior_model"] and prior_model is None:

        assert hasattr(priors, args["prior_model"]), (
            f'Unknown prior model {args["prior_model"]}. '
            f'Available models are {", ".join(priors.__all__)}'
        )
        # instantiate prior model if it was not passed to create_model (i.e. when loading a model)
        prior_model = getattr(priors, args["prior_model"])(**args["prior_args"])

    # create output network
    output_prefix = "Equivariant" if is_equivariant else ""
    output_model = getattr(output_modules, output_prefix + args["output_model"])(
        args["embedding_dimension"], args["activation"]
    )

    # create the denoising output network
    output_model_noise = None
    if args['output_model_noise'] is not None:
        output_model_noise = getattr(output_modules, output_prefix + args["output_model_noise"])(
            args["embedding_dimension"], args["activation"],
        )

    # create the spec feature output network
    output_model_spec = None
    if args['output_model_spec'] is not None:
        output_model_spec = getattr(output_modules, output_prefix + args["output_model_spec"])(
            args["embedding_dimension"], args["activation"],
        )

    # create the mol feature output network
    output_model_mol = None
    if args['output_model_mol'] is not None:
        output_model_mol = getattr(output_modules, output_prefix + args["output_model_mol"])(
            args["embedding_dimension"], args["activation"],
        )

    # combine representation and output network
    model = TorchMD_Net(
        representation_model,
        output_model,
        prior_model=prior_model,
        reduce_op=args["reduce_op"],
        mean=mean,
        std=std,
        derivative=args["derivative"],
        output_model_noise=output_model_noise,
        position_noise_scale=args["position_noise_scale"],
        representation_spec_model=representation_spec_model,
        output_model_spec=output_model_spec,
        output_model_mol=output_model_mol,
        fusion_model=fusion_model,
        d_model=args["embedding_dimension"],
        training_stage=args["training_stage"],
    )
    return model


def load_model(filepath, args=None, device="cpu", mean=None, std=None,var_uv=None, var_ir=None, var_raman=None, prior_model=None, **kwargs):
    ckpt = torch.load(filepath, map_location=torch.device("cuda"))
    if args is None:
        args = ckpt["hyper_parameters"]

    for key, value in kwargs.items():
        if not key in args:
            warnings.warn(f'Unknown hyperparameter: {key}={value}')
        args[key] = value

    model = create_model(args, var_uv=var_uv, var_ir=var_ir, var_raman=var_raman, prior_model=prior_model)

    print("---------Total params (manual)---------", sum(p.numel() for p in model.representation_model.parameters()))
    # for name, _ in model.representation_model.named_parameters():
    #     print(name)

    state_dict = {re.sub(r"^model\.", "", k): v for k, v in ckpt["state_dict"].items()}

    # NOTE for debug
    new_state_dict = {}
    for k, v in state_dict.items():
        # if 'pos_normalizer' not in k:
        if "output_model_noise.0" in k:
            k = k.replace("output_model_noise.0", "output_model_noise")
        if "head.2" in k:
            continue
        new_state_dict[k] = v

    current_model_dict = model.state_dict()
    # ommit mismatching shape
    new_state_dict2 = {}
    for k in current_model_dict:
        if k in new_state_dict:
            # print(k, current_model_dict[k].size(), new_state_dict[k].size())
            if current_model_dict[k].size() == new_state_dict[k].size():
                new_state_dict2[k] = new_state_dict[k]
            else:
                print(f"warning {k} shape mismatching, not loaded")
                new_state_dict2[k] = current_model_dict[k]

    # loading_return = model.load_state_dict(state_dict, strict=False)
    loading_return = model.load_state_dict(new_state_dict2, strict=False)

    if len(loading_return.unexpected_keys) > 0:
        # Should only happen if not applying denoising during fine-tuning.
        assert all(
            (
                "output_model_noise" in k
                or "pos_normalizer" in k
                or "representation_spec_model" in k
                or "output_model_spec" in k
                or "output_model_mol" in k
            )
            for k in loading_return.unexpected_keys
        )
    # assert len(loading_return.missing_keys) == 0, f"Missing keys: {loading_return.missing_keys}"

    if mean:
        model.mean = mean
    if std:
        model.std = std

    return model.to(device)


class TorchMD_Net(nn.Module):

    def __init__(
        self,
        representation_model,
        output_model,
        prior_model=None,
        reduce_op="add",
        mean=None,
        std=None,
        derivative=False,
        output_model_noise=None,
        position_noise_scale=0.0,
        representation_spec_model=None,
        output_model_spec=None,
        output_model_mol=None,
        fusion_model=None,
        d_model=128,
        training_stage="spectra",
    ):
        super(TorchMD_Net, self).__init__()
        self.representation_model = representation_model
        self.output_model = output_model
        self.representation_spec_model = representation_spec_model

        self.prior_model = prior_model
        if not output_model.allow_prior_model and prior_model is not None:
            self.prior_model = None
            rank_zero_warn(
                (
                    "Prior model was given but the output model does "
                    "not allow prior models. Dropping the prior model."
                )
            )

        self.reduce_op = reduce_op
        self.derivative = derivative
        self.output_model_noise = output_model_noise        
        self.position_noise_scale = position_noise_scale

        self.output_model_spec = output_model_spec
        self.output_model_mol = output_model_mol

        self.training_stage = training_stage

        self.fusion_model = fusion_model
        self.contrastive_loss_fn = InfoNCELoss(temperature=0.1)
        # self.property_head = PropertyPredictor(input_dim=d_model, hidden_dim=256, output_dim=1)
        self.projection_head = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

        mean = torch.scalar_tensor(0) if mean is None else mean
        self.register_buffer("mean", mean)
        std = torch.scalar_tensor(1) if std is None else std
        self.register_buffer("std", std)
        self.predictor = MolecularPropertyPredictor()
        self.spectra_decoder = MLPDecoder( output_dim=1)
        if self.position_noise_scale > 0:
            self.pos_normalizer = AccumulatedNormalization(accumulator_shape=(3,))

        self.reset_parameters()

    def reset_parameters(self):
        self.representation_model.reset_parameters()
        self.output_model.reset_parameters()
        if self.output_model_noise is not None:
            self.output_model_noise.reset_parameters()
        if self.representation_spec_model is not None:
            self.representation_spec_model.reset_parameters()
        if self.output_model_spec is not None:
            self.output_model_spec.reset_parameters()
        if self.output_model_mol is not None:
            self.output_model_mol.reset_parameters()
        # if self.prior_model is not None:
        #     self.prior_model.reset_parameters()
        # if self.fusion_model is not None:
        #     self.fusion_model.reset_parameters()

    def forward(self, z, pos, spec_list, patch_masks, batch: Optional[torch.Tensor] = None):
        assert z.dim() == 1 and z.dtype == torch.long
        batch = torch.zeros_like(z) if batch is None else batch

        if self.derivative:
            pos.requires_grad_(True)

        # run the potentially wrapped representation model
        x, v, z, pos, batch = self.representation_model(z, pos, batch=batch)

        # construct spectra feature
        # construct molecule feature
        mol_feature = scatter(x, batch, dim=0, reduce=self.reduce_op)
        mol_feature_normed = F.normalize(mol_feature, p=2, dim=1)
        batch_size = mol_feature.shape[0]
        spec_feature = None
        loss_reconstruct = None
        if self.representation_spec_model is not None:
            if spec_list is not None:
                loss_reconstruct,spec_feature, loss_reconstruct1, loss_reconstruct2  = self.representation_spec_model(spec_list, patch_masks, batch_size)
                # spec_feature_normed = F.normalize(spec_feature, p=2, dim=1)
        if self.output_model_mol is not None:
            mol_feature = self.output_model_mol.pre_reduce(x, v, z, pos, batch)
            mol_feature = scatter(mol_feature, batch, dim=0, reduce=self.reduce_op)

        # predict noise
        noise_pred = None
        if self.output_model_noise is not None:
            noise_pred = self.output_model_noise.pre_reduce(x, v, z, pos, batch) 

        out = None
        x = self.output_model.pre_reduce(x, v, z, pos, batch, spec_feature)
        # x = self.smlies2prop_head(smiles_feature)
        if self.std is not None:
            x = x * self.std
        if self.prior_model is not None:
            x = self.prior_model(x, z, pos, batch)
        out = scatter(x, batch, dim=0, reduce=self.reduce_op)
        # out = x
        if self.mean is not None:
            out = out + self.mean

        # compute gradients with respect to coordinates
        if self.derivative:
            grad_outputs: List[Optional[torch.Tensor]] = [torch.ones_like(out)]
            dy = grad(
                [out],
                [pos],
                grad_outputs=grad_outputs,
                create_graph=True,
                retain_graph=True,
            )[0]
            if dy is None:
                raise RuntimeError("Autograd returned None for the force prediction.")
            return out, noise_pred, -dy, spec_feature, mol_feature_normed, loss_reconstruct
        # TODO: return only `out` once Union typing works with TorchScript (https://github.com/pytorch/pytorch/pull/53180)
        return out, noise_pred, None, spec_feature, mol_feature_normed, loss_reconstruct,loss_reconstruct1, loss_reconstruct2

class MLPDecoder(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256, output_dim=1, dropout=0.05):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            # nn.BatchNorm1d(hidden_dim), # 或 LayerNorm
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            # nn.BatchNorm1d(hidden_dim // 2),
            nn.GELU(),
            
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x):
        # x: [B, D]
        return self.net(x)

class MolecularPropertyPredictor(nn.Module):
    # def __init__(self, d_model=128, hidden_dim=512, output_dim=1, dropout=0.1):
    #     """
    #     Args:
    #         uv_encoder, ir_encoder, raman_encoder: 预训练好的编码器实例。
    #         d_model: 每个编码器输出的[CLS]令牌的维度。
    #         hidden_dim: 预测头MLP的隐藏层维度。
    #         output_dim: 最终要预测的属性数量 (例如，1表示预测一个标量)。
    #         dropout: MLP中的dropout率。
    #     """
    #     super().__init__()
        
    #     # # 1. 加载并（可选地）冻结预训练的编码器
    #     # self.uv_encoder = uv_encoder
    #     # self.ir_encoder = ir_encoder
    #     # self.raman_encoder = raman_encoder
        
    #     # # --- 微调策略：选择以下一种 ---
    #     # # a) 冻结所有编码器 (只训练预测头，速度快，适合小数据集)
    #     # for param in self.uv_encoder.parameters():
    #     #     param.requires_grad = False
    #     # for param in self.ir_encoder.parameters():
    #     #     param.requires_grad = False
    #     # for param in self.raman_encoder.parameters():
    #     #     param.requires_grad = False
        
    #     # b) 不做任何事，让它们参与微调 (适合大数据集)
    #     # ------------------------------------

    #     # 2. 定义预测头 (一个简单的多层感知机)
    #     # 输入维度是 3 * d_model，因为我们拼接了三个向量
    #     self.prediction_head = nn.Sequential(
    #         # nn.LayerNorm(3 * d_model), # 在输入前做一次LayerNorm非常重要！
    #         nn.Linear(3 * d_model, hidden_dim),
    #         nn.ReLU(),
    #         nn.Dropout(dropout),
    #         nn.Linear(hidden_dim, hidden_dim // 2),
    #         nn.ReLU(),
    #         nn.Dropout(dropout),
    #         nn.Linear(hidden_dim // 2, output_dim)
    #     )

    # def forward(self, uv_spec, ir_spec, raman_spec):
    #     # # 确保模型处于评估模式，以关闭预训练编码器中的dropout
    #     # self.uv_encoder.eval()
    #     # self.ir_encoder.eval()
    #     # self.raman_encoder.eval()

    #     # # 1. 使用编码器提取高级特征 ([CLS] 令牌)
    #     # with torch.no_grad(): # 如果编码器被冻结，这可以节省计算
    #     #     uv_cls = self.uv_encoder(uv_spec)[:, 0]
    #     #     ir_cls = self.ir_encoder(ir_spec)[:, 0]
    #     #     raman_cls = self.raman_encoder(raman_spec)[:, 0]
        
    #     # 2. 【关键】对每个表征进行归一化，以平衡它们的尺度
    #     #    这是防止某个模态“霸凌”其他模态的关键步骤
    #     uv_spec = F.layer_norm(uv_spec, (uv_spec.shape[-1],))
    #     ir_spec = F.layer_norm(ir_spec, (ir_spec.shape[-1],))
    #     raman_spec = F.layer_norm(raman_spec, (raman_spec.shape[-1],))
        
    #     # 3. 简单拼接 (Concatenate)
    #     fused_features = torch.cat([uv_spec, ir_spec, raman_spec], dim=1)
        
    #     # 4. 通过预测头得到最终输出
    #     prediction = self.prediction_head(fused_features)
        
    #     return prediction
    def __init__(self,  
                 d_model=128, n_head_fusion=8, n_layers_fusion=2,
                 hidden_dim=512, output_dim=1, dropout=0.1):
        super().__init__()
        
        fusion_encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_head_fusion,
            dim_feedforward=hidden_dim, dropout=dropout,
            activation='gelu', batch_first=True
        )
        self.fusion_transformer = nn.TransformerEncoder(
            fusion_encoder_layer, num_layers=n_layers_fusion
        )
        
        # 3. 定义最终的预测头
        self.prediction_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, uv_spec, ir_spec, raman_spec):

        uv_spec = F.layer_norm(uv_spec, (uv_spec.shape[-1],))
        ir_spec = F.layer_norm(ir_spec, (ir_spec.shape[-1],))
        raman_spec = F.layer_norm(raman_spec, (raman_spec.shape[-1],))

        # 3. 【关键】构建新的短序列并融合
        #    堆叠成 [batch_size, 3, d_model] 的序列
        stacked_features = torch.stack([uv_spec, uv_spec, raman_spec], dim=1)
        
        #    通过融合Transformer进行深度交互
        fused_output = self.fusion_transformer(stacked_features)
        
        #    提取融合后的表征 (可以取平均，也可以只取第一个)
        final_representation = torch.mean(fused_output, dim=1)
        
        # 4. 通过预测头得到最终输出
        prediction = self.prediction_head(final_representation)
        
        return prediction

class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, z_i, z_j):
        """
        计算两个模态特征 z_i 和 z_j 之间的对比损失。
        """
        batch_size = z_i.shape[0]
        device = z_i.device
        
        # 计算相似度矩阵
        sim_matrix = torch.matmul(z_i, z_j.T) / self.temperature
        
        # 目标标签: 对角线上的元素 (i, i) 是正样本对
        labels = torch.arange(batch_size, device=device)
        
        # 对称地计算损失，增强稳定性
        loss_i_j = self.criterion(sim_matrix, labels)
        loss_j_i = self.criterion(sim_matrix.T, labels)
        
        return (loss_i_j + loss_j_i) / 2
    
class SpectraFusion(nn.Module):
    """
    融合多光谱 embedding 的 Transformer
    - 支持模态缺失（通过 attention mask 屏蔽）
    - 输出一个通用光谱表征 embedding
    """
    def __init__(self, 
                 patch_nums,
                 d_model=128, 
                 n_heads=8, 
                 n_layers=2, 
                 dropout=0.1, 
                 max_spec_tokens_per_modality=16, 
                 num_modalities=3,
                 output_dim=256,
                 ):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=2 * d_model,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.patch_nums = patch_nums
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # learnable modality embeddings (用于区分UV/IR/Raman/结构)
        self.modality_embed = nn.Parameter(torch.randn(num_modalities + 1, d_model))

        # 输出一个全局embedding
        self.out_proj = nn.Linear(d_model, d_model)

        self.num_modalities = num_modalities
        self.max_spec_tokens_per_modality = max_spec_tokens_per_modality
        self.head = nn.Linear(d_model, output_dim)


    def forward(self, spec_feature_list, present_indices,  device, B, D):
        """
        mol_feature: [B, D]
        spec_feature_list: list of existing modalities, each [B, L_i, D]
        present_indices: list of indices for modalities that are present (e.g., [0, 2])
        """

        

        # # ---------- Step 1. 放入结构 embedding ----------
        # mol_feature = mol_feature.unsqueeze(1)  # [B, 1, D]
        # mol_feature = mol_feature + self.modality_embed[self.num_modalities]  # 加上结构模态 embedding

        # ---------- Step 2. 构建光谱 token ----------
        spec_features = []
        attn_mask = []

        for i in range(self.num_modalities):
            if i in present_indices:
                feat = spec_feature_list[present_indices.index(i)]  # [B, L_i, D]
                # L_i = feat.size(1)
                # # 加上模态 embedding
                # feat = feat + self.modality_embed[i].unsqueeze(0).unsqueeze(0)
                # padding 到统一长度
                # if L_i < self.max_spec_tokens_per_modality:
                #     pad_len = self.max_spec_tokens_per_modality - L_i
                #     feat = F.pad(feat, (0, 0, 0, pad_len))  # 在patch维度pad
                #     mask = torch.cat([
                #         torch.zeros(L_i, dtype=torch.bool),
                #         torch.ones(pad_len, dtype=torch.bool)
                #     ], dim=0)
                # else:
                #     feat = feat[:, :self.max_spec_tokens_per_modality, :]
                mask = torch.zeros(self.patch_nums[i], dtype=torch.bool)
            else:
                # 缺失模态 → 全padding + 全mask
                feat = torch.zeros(B, self.patch_nums[i], D, device=device)
                mask = torch.ones(self.patch_nums[i], dtype=torch.bool)
            spec_features.append(feat)
            attn_mask.append(mask)

        # ---------- Step 3. 拼接 ----------
        fused_spec = torch.cat(spec_features, dim=1).to(device)  # [B, num_modalities * max_spec_tokens, D]
        total_spec_tokens = sum(self.patch_nums)

        full_mask = torch.cat(attn_mask, dim=0).unsqueeze(0).repeat(B, 1).to(device)  # [B, total_spec_tokens]

        # Transformer融合
        fused_out = self.transformer(fused_spec, src_key_padding_mask=full_mask)  # [B, total_spec_tokens, D]

        # 池化输出融合表征
        # 这里可以用mean pooling或者masked mean pooling
        # valid_mask = (~full_mask).float()  # mask部分为0
        # global_embed = (fused_out * valid_mask.unsqueeze(-1)).sum(1) / valid_mask.sum(1, keepdim=True)  # [B, D]
        fused_out = self.head(fused_out)
        # global_embed = self.out_proj(global_embed)
        return fused_out
    
class CrossAttentionFusion(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query, key_value, key_padding_mask=None):
        # query: [query_len, batch, d_model]  (e.g., molecule embedding tokens)
        # key_value: [kv_len, batch, d_model] (e.g., spectra tokens)
        residual = query
        attn_output, _ = self.cross_attn(query, key_value, key_value, key_padding_mask=key_padding_mask)
        x = residual + self.dropout(attn_output)
        x = self.norm(x)
        return x  # [query_len, batch, d_model]

class MultiModalFusion(nn.Module):
    def __init__(self,patch_len, stride,  d_model=128, n_heads=8, dropout=0.0, num_modalities=3, output_dim=256):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.num_modalities = num_modalities
        list_len_spectrum = [701, 3501, 3501]
        self.patch_nums = [
            int((list_len_spectrum[i] - patch_len[i]) / stride[i] + 1)
            for i in range(len(list_len_spectrum))
        ]
        self.spectra_encoder = SpectraFusion(d_model=self.d_model, n_heads=self.n_heads,num_modalities=self.num_modalities, patch_nums= self.patch_nums,output_dim=output_dim)
        self.cross_attention = CrossAttentionFusion(d_model=output_dim, n_heads=self.n_heads)

    def forward(self, mol_emb, spectra_emb, present_indices, spectra_mask=None):
        # mol_emb: [B, 1, d_model]
        # spectra_emb: [B, seq_len, d_model]
        # convert to [seq_len, B, d_model] for nn.MultiheadAttention
        if len(present_indices) == 0:
            return mol_emb
        B, D = mol_emb.shape
        device = mol_emb.device
        mol_emb = mol_emb.unsqueeze(1).permute(1, 0, 2)
        
        # Step 1: spectra self-attention
        spectra_encoded = self.spectra_encoder(spectra_emb, present_indices=present_indices, B=B, D=D, device=device)
        spec = spectra_encoded.mean(dim=1)
        spectra_encoded = spectra_encoded.permute(1, 0, 2)
        # Step 2: cross-attention: molecule queries spectra
        fused = self.cross_attention(mol_emb, spectra_encoded)

        # optionally: convert back to [B, seq_len, d_model] or [B, 1, d_model]
        fused = fused.permute(1, 0, 2)  # [B, 1, d_model]
        return fused.squeeze(1), spec  # [B, d_model]
    
class PropertyPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, output_dim=1, dropout=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            # nn.ReLU(),
            # nn.Dropout(dropout),
            # nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.mlp(x)

class AccumulatedNormalization(nn.Module):
    """Running normalization of a tensor."""
    def __init__(self, accumulator_shape: Tuple[int, ...], epsilon: float = 1e-8):
        super(AccumulatedNormalization, self).__init__()

        self._epsilon = epsilon
        self.register_buffer("acc_sum", torch.zeros(accumulator_shape))
        self.register_buffer("acc_squared_sum", torch.zeros(accumulator_shape))
        self.register_buffer("acc_count", torch.zeros((1,)))
        self.register_buffer("num_accumulations", torch.zeros((1,)))

    def update_statistics(self, batch: torch.Tensor):
        batch_size = batch.shape[0]
        self.acc_sum += batch.sum(dim=0)
        self.acc_squared_sum += batch.pow(2).sum(dim=0)
        self.acc_count += batch_size
        self.num_accumulations += 1

    @property
    def acc_count_safe(self):
        return self.acc_count.clamp(min=1)

    @property
    def mean(self):
        return self.acc_sum / self.acc_count_safe

    @property
    def std(self):
        return torch.sqrt(
            (self.acc_squared_sum / self.acc_count_safe) - self.mean.pow(2)
        ).clamp(min=self._epsilon)

    def forward(self, batch: torch.Tensor):
        if self.training:
            self.update_statistics(batch)
        return ((batch - self.mean) / self.std)
