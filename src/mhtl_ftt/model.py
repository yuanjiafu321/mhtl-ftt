from __future__ import annotations

import torch
from torch import nn


class MultiTaskFTTransformer(nn.Module):
    def __init__(self, n_num_features, cat_cardinalities, d_model, n_heads,
                 n_layers, dropout, dim_feedforward=2048):
        super().__init__()
        self.num_weight = nn.Parameter(torch.randn(1, n_num_features, d_model))
        self.num_bias = nn.Parameter(torch.randn(1, n_num_features, d_model))
        self.cat_embeddings = nn.ModuleList([nn.Embedding(card, d_model) for card in cat_cardinalities])
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads,
                                           dim_feedforward=dim_feedforward, dropout=dropout,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head_landslide = self._make_head(d_model)
        self.head_rockfall = self._make_head(d_model)
        self.head_debris = self._make_head(d_model)
        self._init_weights()

    @staticmethod
    def _make_head(d_model):
        return nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model // 2),
                             nn.ReLU(), nn.Linear(d_model // 2, 1))

    def _init_weights(self):
        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)
        nn.init.normal_(self.num_weight, std=0.01)
        nn.init.zeros_(self.num_bias)

    def forward(self, x_num, x_cat, task_name):
        batch_size = x_num.shape[0] if x_num is not None else x_cat.shape[0]
        tokens = [self.cls_token.expand(batch_size, -1, -1)]
        if x_num is not None and x_num.shape[1] > 0:
            tokens.append(x_num.unsqueeze(-1) * self.num_weight + self.num_bias)
        if x_cat is not None and len(self.cat_embeddings) > 0:
            tokens.append(torch.cat([embedding(x_cat[:, i].long()).unsqueeze(1)
                                     for i, embedding in enumerate(self.cat_embeddings)], dim=1))
        representation = self.transformer(torch.cat(tokens, dim=1))[:, 0, :]
        if task_name == "landslide":
            return self.head_landslide(representation).squeeze(-1)
        if task_name == "rockfall":
            return self.head_rockfall(representation).squeeze(-1)
        if task_name == "debris":
            return self.head_debris(representation).squeeze(-1)
        raise ValueError(f"Unknown task name: {task_name}")


def build_model(config, cardinalities, device):
    return MultiTaskFTTransformer(
        len(config.data.numerical_columns), cardinalities, config.model.d_model,
        config.model.n_heads, config.model.n_layers, config.model.dropout,
        config.model.dim_feedforward,
    ).to(device)

