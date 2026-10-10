from abc import abstractmethod, ABCMeta
from typing import Optional
from typing import Tuple
import ase
from torchmdnet.models.layers import act_class_mapping, GatedEquivariantBlock
from torch_scatter import scatter
import torch
from torch import nn

__all__ = ["Scalar", "DipoleMoment", "ElectronicSpatialExtent"]


class OutputModel(nn.Module, metaclass=ABCMeta):

    def __init__(self, allow_prior_model):
        super(OutputModel, self).__init__()
        self.allow_prior_model = allow_prior_model

    def reset_parameters(self):
        pass

    @abstractmethod
    def pre_reduce(self, x, v, z, pos, batch):
        return

    def post_reduce(self, x):
        return x


class Scalar(OutputModel):

    def __init__(self, hidden_channels, activation="silu", allow_prior_model=True):
        super(Scalar, self).__init__(allow_prior_model=allow_prior_model)
        act_class = act_class_mapping[activation]
        self.output_network = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            act_class(),
            nn.Linear(hidden_channels // 2, 1),
        )
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.output_network[0].weight)
        self.output_network[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.output_network[2].weight)
        self.output_network[2].bias.data.fill_(0)

    def pre_reduce(self, x, v: Optional[torch.Tensor], z, pos, batch):
        return self.output_network(x)


class EquivariantScalar(OutputModel):

    def __init__(self, hidden_channels, activation="silu", allow_prior_model=True):
        super().__init__(allow_prior_model=allow_prior_model)
        self.output_network = nn.ModuleList(
            [
                GatedEquivariantBlock(
                    hidden_channels,
                    hidden_channels // 2,
                    activation=activation,
                    scalar_activation=True,
                ),
                GatedEquivariantBlock(hidden_channels // 2, 1, activation=activation),
            ]
        )
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.gate_alpha = nn.Parameter(torch.tensor(0.01))
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.fusion_layer:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def pre_reduce(self, x, v, z, pos, batch, spec_emb):
        """
        Combine atomic features with the spectral embedding spec_emb.

        Args:
            x: Atomic scalar features, [num_atoms, hidden_channels].
            v: Atomic vector features, [num_atoms, num_vecs, hidden_channels].
            spec_emb: Fused spectral embeddings, [batch_size, spec_emb_dim].
        """
        spec_emb_per_atom = spec_emb[batch]
        fused_x = torch.cat([x, spec_emb_per_atom], dim=1)
        modulated_x = self.fusion_layer(fused_x)
        x = x + self.gate_alpha * modulated_x
        for layer in self.output_network:
            x, v = layer(x, v)
        return x + v.sum() * 0


class DipoleMoment(Scalar):

    def __init__(self, hidden_channels, activation="silu"):
        super(DipoleMoment, self).__init__(
            hidden_channels, activation, allow_prior_model=False
        )
        atomic_mass = torch.from_numpy(ase.data.atomic_masses).float()
        self.register_buffer("atomic_mass", atomic_mass)

    def pre_reduce(self, x, v: Optional[torch.Tensor], z, pos, batch):
        x = self.output_network(x)
        mass = self.atomic_mass[z].view(-1, 1)
        c = scatter(mass * pos, batch, dim=0) / scatter(mass, batch, dim=0)
        x = x * (pos - c[batch])
        return x

    def post_reduce(self, x):
        return torch.norm(x, dim=-1, keepdim=True)


class EquivariantDipoleMoment(EquivariantScalar):

    def __init__(self, hidden_channels, activation="silu"):
        super(EquivariantDipoleMoment, self).__init__(
            hidden_channels, activation, allow_prior_model=False
        )
        atomic_mass = torch.from_numpy(ase.data.atomic_masses).float()
        self.register_buffer("atomic_mass", atomic_mass)

    def pre_reduce(self, x, v, z, pos, batch):
        for layer in self.output_network:
            x, v = layer(x, v)
        mass = self.atomic_mass[z].view(-1, 1)
        c = scatter(mass * pos, batch, dim=0) / scatter(mass, batch, dim=0)
        x = x * (pos - c[batch])
        return x + v.squeeze()

    def post_reduce(self, x):
        return torch.norm(x, dim=-1, keepdim=True)


class ElectronicSpatialExtent(OutputModel):

    def __init__(self, hidden_channels, activation="silu"):
        super(ElectronicSpatialExtent, self).__init__(allow_prior_model=False)
        act_class = act_class_mapping[activation]
        self.output_network = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            act_class(),
            nn.Linear(hidden_channels // 2, 1),
        )
        atomic_mass = torch.from_numpy(ase.data.atomic_masses).float()
        self.register_buffer("atomic_mass", atomic_mass)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.output_network[0].weight)
        self.output_network[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.output_network[2].weight)
        self.output_network[2].bias.data.fill_(0)

    def pre_reduce(self, x, v: Optional[torch.Tensor], z, pos, batch):
        x = self.output_network(x)
        mass = self.atomic_mass[z].view(-1, 1)
        c = scatter(mass * pos, batch, dim=0) / scatter(mass, batch, dim=0)
        x = torch.norm(pos - c[batch], dim=1, keepdim=True) ** 2 * x
        return x


class EquivariantElectronicSpatialExtent(ElectronicSpatialExtent):
    pass


class EquivariantVectorOutput(EquivariantScalar):

    def __init__(self, hidden_channels, activation="silu"):
        super(EquivariantVectorOutput, self).__init__(
            hidden_channels, activation, allow_prior_model=False
        )

    def pre_reduce(self, x, v, z, pos, batch):
        for layer in self.output_network:
            x, v = layer(x, v)
        return v.squeeze()


class BaseWrapper(nn.Module, metaclass=ABCMeta):
    """Base class for model wrappers.

    Children of this class should implement the `forward` method,
    which calls `self.model(z, pos, batch=batch)` at some point.
    Wrappers that are applied before the REDUCE operation should return
    the model's output, `z`, `pos`, `batch` and potentially vector
    features`v`. Wrappers that are applied after REDUCE should only
    return the model's output.
    """

    def __init__(self, model):
        super(BaseWrapper, self).__init__()
        self.model = model

    def reset_parameters(self):
        self.model.reset_parameters()

    @abstractmethod
    def forward(self, z, pos, batch=None):
        return


class AtomFilter(BaseWrapper):

    def __init__(self, model, remove_threshold):
        super(AtomFilter, self).__init__(model)
        self.remove_threshold = remove_threshold

    def forward(self, z, pos, batch=None):
        x, v, z, pos, batch = self.model(z, pos, batch=batch)
        n_samples = len(batch.unique())
        atom_mask = z > self.remove_threshold
        x = x[atom_mask]
        if v is not None:
            v = v[atom_mask]
        z = z[atom_mask]
        pos = pos[atom_mask]
        batch = batch[atom_mask]
        assert (
            len(batch.unique()) == n_samples
        ), f"Some samples were completely filtered out by the atom filter. Make sure that at least one atom per sample exists with Z > {self.remove_threshold}."
        return (x, v, z, pos, batch)


from pytorch_lightning.utilities import rank_zero_warn

__all__ = ["Atomref"]


class BasePrior(nn.Module, metaclass=ABCMeta):
    """Base class for prior models.
    Derive this class to make custom prior models, which take some arguments and a dataset as input.
    As an example, have a look at the `torchmdnet.priors.Atomref` prior.
    """

    def __init__(self, dataset=None):
        super(BasePrior, self).__init__()

    @abstractmethod
    def get_init_args(self):
        """A function that returns all required arguments to construct a prior object.
        The values should be returned inside a dict with the keys being the arguments' names.
        All values should also be saveable in a .yaml file as this is used to reconstruct the
        prior model from a checkpoint file.
        """
        return

    @abstractmethod
    def forward(self, x, z, pos, batch):
        """Forward method of the prior model.

        Args:
            x (torch.Tensor): scalar atomwise predictions from the model.
            z (torch.Tensor): atom types of all atoms.
            pos (torch.Tensor): 3D atomic coordinates.
            batch (torch.Tensor): tensor containing the sample index for each atom.

        Returns:
            torch.Tensor: updated scalar atomwise predictions
        """
        return


class Atomref(BasePrior):
    """Atomref prior model.
    When using this in combination with some dataset, the dataset class must implement
    the function `get_atomref`, which returns the atomic reference values as a tensor.
    """

    def __init__(self, max_z=None, dataset=None):
        super(Atomref, self).__init__()
        if max_z is None and dataset is None:
            raise ValueError("Can't instantiate Atomref prior, all arguments are None.")
        if dataset is None:
            atomref = torch.zeros(max_z, 1)
        else:
            atomref = dataset.get_atomref()
            if atomref is None:
                rank_zero_warn(
                    "The atomref returned by the dataset is None, defaulting to zeros with max. atomic number 99. Maybe atomref is not defined for the current target."
                )
                atomref = torch.zeros(100, 1)
        if atomref.ndim == 1:
            atomref = atomref.view(-1, 1)
        self.register_buffer("atomref", atomref)
        self.dataset = dataset

    def get_init_args(self):
        return dict(max_z=self.atomref.size(0), dataset=self.dataset)

    def reset_parameters(self):
        # Atom references are fixed data, not randomly initialized weights.
        pass

    def forward(self, x, z, pos, batch):
        return x + self.atomref[z]


import math
import torch.nn.functional as F


class MLPProjector(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_dim=None, dropout=0.1):
        """
        Args:
            input_dim: Encoder output dimension (e.g., 256).
            output_dim: Decoder input dimension (e.g., 512 or 768).
            hidden_dim: MLP hidden dimension; defaults to twice output_dim.
            dropout: Dropout probability for regularization.
        """
        super().__init__()
        if hidden_dim is None:
            hidden_dim = output_dim * 2
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        return self.net(x)


class InfoNCELoss(nn.Module):

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, z_i, z_j):
        """
        Compute the contrastive loss between modality features z_i and z_j.
        """
        batch_size = z_i.shape[0]
        device = z_i.device
        sim_matrix = torch.matmul(z_i, z_j.T) / self.temperature
        labels = torch.arange(batch_size, device=device)
        loss_i_j = self.criterion(sim_matrix, labels)
        loss_j_i = self.criterion(sim_matrix.T, labels)
        return (loss_i_j + loss_j_i) / 2


class SpectraFusion(nn.Module):
    """
    Transformer for fusing embeddings from multiple spectral modalities.
    - Handles missing modalities through attention masking.
    - Produces a shared spectral embedding.
    """

    def __init__(
        self,
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
            activation="gelu",
            batch_first=True,
        )
        self.patch_nums = patch_nums
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.modality_embed = nn.Parameter(torch.randn(num_modalities + 1, d_model))
        self.out_proj = nn.Linear(d_model, d_model)
        self.num_modalities = num_modalities
        self.max_spec_tokens_per_modality = max_spec_tokens_per_modality
        self.head = nn.Linear(d_model, output_dim)

    def forward(self, spec_feature_list, present_indices, device, B, D):
        """
        mol_feature: [B, D]
        spec_feature_list: list of existing modalities, each [B, L_i, D]
        present_indices: list of indices for modalities that are present (e.g., [0, 2])
        """
        spec_features = []
        attn_mask = []
        for i in range(self.num_modalities):
            if i in present_indices:
                feat = spec_feature_list[present_indices.index(i)]
                mask = torch.zeros(self.patch_nums[i], dtype=torch.bool)
            else:
                feat = torch.zeros(B, self.patch_nums[i], D, device=device)
                mask = torch.ones(self.patch_nums[i], dtype=torch.bool)
            spec_features.append(feat)
            attn_mask.append(mask)
        fused_spec = torch.cat(spec_features, dim=1).to(device)
        total_spec_tokens = sum(self.patch_nums)
        full_mask = torch.cat(attn_mask, dim=0).unsqueeze(0).repeat(B, 1).to(device)
        fused_out = self.transformer(fused_spec, src_key_padding_mask=full_mask)
        fused_out = self.head(fused_out)
        return fused_out


class CrossAttentionFusion(nn.Module):

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key_value, key_padding_mask=None):
        residual = query
        attn_output, _ = self.cross_attn(
            query, key_value, key_value, key_padding_mask=key_padding_mask
        )
        x = residual + self.dropout(attn_output)
        x = self.norm(x)
        return x


class MultiModalFusion(nn.Module):

    def __init__(
        self,
        patch_len,
        stride,
        d_model=128,
        n_heads=8,
        dropout=0.0,
        num_modalities=3,
        output_dim=256,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.num_modalities = num_modalities
        list_len_spectrum = [701, 3501, 3501]
        self.patch_nums = [
            int((list_len_spectrum[i] - patch_len[i]) / stride[i] + 1)
            for i in range(len(list_len_spectrum))
        ]
        self.spectra_encoder = SpectraFusion(
            d_model=self.d_model,
            n_heads=self.n_heads,
            num_modalities=self.num_modalities,
            patch_nums=self.patch_nums,
            output_dim=output_dim,
        )
        self.cross_attention = CrossAttentionFusion(
            d_model=output_dim, n_heads=self.n_heads
        )

    def forward(self, mol_emb, spectra_emb, present_indices, spectra_mask=None):
        if len(present_indices) == 0:
            return mol_emb
        B, D = mol_emb.shape
        device = mol_emb.device
        mol_emb = mol_emb.unsqueeze(1).permute(1, 0, 2)
        spectra_encoded = self.spectra_encoder(
            spectra_emb, present_indices=present_indices, B=B, D=D, device=device
        )
        spec = spectra_encoded.mean(dim=1)
        spectra_encoded = spectra_encoded.permute(1, 0, 2)
        fused = self.cross_attention(mol_emb, spectra_encoded)
        fused = fused.permute(1, 0, 2)
        return (fused.squeeze(1), spec)


class PropertyPredictor(nn.Module):

    def __init__(self, input_dim, hidden_dim=256, output_dim=1, dropout=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.mlp(x)


class AccumulatedNormalization(nn.Module):
    """Running normalization of a tensor."""

    def __init__(self, accumulator_shape: Tuple[int, ...], epsilon: float = 1e-08):
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
            self.acc_squared_sum / self.acc_count_safe - self.mean.pow(2)
        ).clamp(min=self._epsilon)

    def forward(self, batch: torch.Tensor):
        if self.training:
            self.update_statistics(batch)
        return (batch - self.mean) / self.std


class PretrainMLPDecoder(nn.Module):

    def __init__(self, input_dim=256, hidden_dim=512, output_dim=1, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, x):
        return self.net(x)


