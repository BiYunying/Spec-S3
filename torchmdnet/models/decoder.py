import math
from typing import Optional
import torch
from torch import nn
import torch.nn.functional as F


class SMILES_Decoder(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_head: int,
        n_layers: int,
        d_ff: int,
        max_len: int,
        dropout: float = 0.1,
    ):
        """
        Transformer decoder that generates SMILES sequences from spectral embeddings.

        Args:
            vocab_size: Number of tokens in the SMILES vocabulary.
            d_model: Model width; must match the encoder output width.
            n_head: Number of decoder attention heads.
            n_layers: Number of decoder layers.
            d_ff: Feed-forward hidden dimension.
            max_len: Maximum SMILES sequence length.
            dropout: Dropout probability.
        """
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout, max_len)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=n_layers
        )
        self.generator = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor,
        tgt_key_padding_mask: torch.Tensor,
        epoch: Optional[int] = None,
        stage: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Run the decoder forward pass.

        Args:
            tgt (torch.Tensor): Right-shifted SMILES targets, [bs, seq_len].
            memory (torch.Tensor): Encoder spectral embeddings, [bs, memory_len, d_model].
            tgt_mask (torch.Tensor): Causal attention mask, [seq_len, seq_len].
            tgt_key_padding_mask (torch.Tensor): Padding mask, [bs, seq_len].

        Returns:
            torch.Tensor: Next-token logits, [bs, seq_len, vocab_size].
        """
        noisy_decoder_input = tgt.to(memory.device)
        tgt_emb = self.positional_encoding(
            self.token_embedding(noisy_decoder_input) * math.sqrt(self.d_model)
        )
        output = self.transformer_decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )
        logits = self.generator(output)
        return logits


def apply_token_dropout_or_replace(input_ids, vocab_size, replace_prob=0.1):
    """
    Simulate prediction errors by corrupting decoder inputs.
    Replace input tokens with random vocabulary tokens at the given probability
    to encourage recovery from incorrect prefixes using spectral memory.
    """
    corrupted_input = input_ids.clone()
    mask = torch.rand(input_ids.shape, device=input_ids.device) < replace_prob
    mask[:, 0] = False
    random_tokens = torch.randint(
        3, vocab_size, input_ids.shape, device=input_ids.device
    )
    corrupted_input[mask] = random_tokens[mask]
    return corrupted_input


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout, max_len=50):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        scaffold_maxlen = getattr(config, "scaffold_maxlen", 0)
        num = int(bool(config.num_props)) + int(scaffold_maxlen)
        self.register_buffer(
            "mask",
            torch.tril(
                torch.ones(config.block_size + num, config.block_size + num)
            ).view(1, 1, config.block_size + num, config.block_size + num),
        )
        self.n_head = config.n_head

    def forward(self, x, layer_past=None):
        B, T, C = x.size()
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        attn_save = att
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_drop(self.proj(y))
        return (y, attn_save)


class Block(nn.Module):
    """an unassuming Transformer block"""

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x):
        y, attn = self.attn(self.ln1(x))
        x = x + y
        x = x + self.mlp(self.ln2(x))
        return (x, attn)


class GPT(nn.Module):
    """the full GPT language model, with a context size of block_size"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.type_emb = nn.Embedding(2, config.n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.block_size = config.block_size
        self.apply(self._init_weights)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=train_config.learning_rate,
            betas=train_config.betas,
            weight_decay=train_config.weight_decay,
        )
        return optimizer

    def forward(self, idx, targets=None, p=None):
        b, t = idx.size()
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."
        token_embeddings = self.tok_emb(idx)
        position_embeddings = self.pos_emb[:, :t, :]
        type_embeddings = self.type_emb(
            torch.ones((b, t), dtype=torch.long, device=idx.device)
        )
        x = self.drop(token_embeddings + position_embeddings + type_embeddings)
        type_embd = self.type_emb(
            torch.zeros((b, 1), dtype=torch.long, device=idx.device)
        )
        p += type_embd
        x = torch.cat([p, x], 1)
        attn_maps = []
        for layer in self.blocks:
            x, attn = layer(x)
            attn_maps.append(attn)
        x = self.ln_f(x)
        logits = self.head(x)
        num = int(bool(self.config.num_props))
        logits = logits[:, num:, :]
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=0
            )
        return (logits, loss)


class GPTConfig:
    """base GPT config, params common to all GPT versions"""

    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.scaffold_maxlen = 0
        for k, v in kwargs.items():
            setattr(self, k, v)
