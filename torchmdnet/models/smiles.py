import torch
from torch import nn


class ChemBERTa_SMILESEncoder(nn.Module):

    def __init__(self, base_model, embedding_dim=384, projection_dim=256):
        """
        Args:
            base_model: Pretrained ChemBERTa model.
            embedding_dim: Output dimension of the supplied ChemBERTa model.
            projection_dim: Projection output dimension; should match the 3D model output.
        """
        super().__init__()
        self.encoder = base_model

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        return cls_embedding
