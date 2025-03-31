from typing import Optional, Any

import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoderLayer, TransformerEncoder


class OmicsEncoder(nn.Module):
    def __init__(
            self,
            num_embeddings: int,
            embedding_dim: int,
            padding_idx: Optional[int] = None,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings, embedding_dim, padding_idx=padding_idx
        )
        self.enc_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.embedding(x)  # (batch, seq_len, emb_size)
        x = self.enc_norm(x)
        return x


class TransformerModel(nn.Module):
    def __init__(
        self,
        ntoken: int,
        n_input_bins: int,
        n_sources: int,
        d_model: int,
        nhead: int,
        d_hid: int,
        nlayers: int,
        dropout: float = 0.5,
        pad_token_id: int = 0,
    ):
        super().__init__()
        self.d_model = d_model
        self.encoder = OmicsEncoder(ntoken, d_model, padding_idx=pad_token_id)
        self.value_encoder = OmicsEncoder(n_input_bins, d_model, padding_idx=pad_token_id)
        self.source_encoder = OmicsEncoder(n_sources, d_model, padding_idx=pad_token_id)

        encoder_layers = TransformerEncoderLayer(
            d_model, nhead, d_hid, dropout, batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.out_layer = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, batch) -> Tensor:
        device = self.encoder.embedding.weight.device

        values = batch['X_bin_input'].to(device)
        sources = batch['X_input_source'].to(device)
        x = batch['X_input_names'].to(device)
        src_key_padding_mask = batch['X_key_padding_mask'].type(torch.float32).to(device)

        x = self.encoder(x)
        sources = self.source_encoder(sources)
        values = self.value_encoder(values)
        x = x + sources + values
        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        x = self.out_layer(x)
        return x
