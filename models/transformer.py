import torch
from data.settings import device, deftype


def position_encoding(seq_len: int, features_dim: int) -> torch.Tensor:
    pos = torch.arange(seq_len, dtype=torch.float64, device=device).reshape(1, -1, 1)
    dim = torch.arange(features_dim, dtype=torch.float64, device=device).reshape(1, 1, -1)
    phase = pos / (1e4 ** torch.div(dim, features_dim, rounding_mode='floor'))

    return torch.where(dim.long() % 2 == 0, torch.sin(phase), torch.cos(phase))

def feed_forward(dim_input: int, feedforward_dim: int) -> torch.nn.Module:
    return torch.nn.Sequential(
        torch.nn.Linear(dim_input, feedforward_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(feedforward_dim, dim_input),
    )

def scaled_dot_product_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    query_key = query.bmm(key.transpose(1, 2)) # attention score matrix to each seq. step 50x50
    scale = query.size(-1) ** 0.5
    query_key_scaled = torch.nn.functional.softmax(query_key / scale, dim=-1) # normalized and scaled attention score
    return query_key_scaled.bmm(value) # summiere gewichtungen (scaled) mit den eigentlichen werten

class Residual(torch.nn.Module):
    def __init__(self, sublayer: torch.nn.Module, dimension: int, normalising: bool, dropout: float):
        super().__init__()
        self.sublayer = sublayer
        self.norm = torch.nn.LayerNorm(dimension) if normalising else torch.nn.Identity(dimension)
        self.dropout = torch.nn.Dropout(dropout)
    def forward(self, *tensors: torch.Tensor) -> torch.Tensor:
        return self.norm(tensors[0] + self.dropout(self.sublayer(*tensors)))

class AttentionHead(torch.nn.Module):
    def __init__(self, features_dim: int, dim_q: int, dim_k: int):
        super().__init__()
        self.q = torch.nn.Linear(features_dim, dim_q) # transformation to the last dimension of tensor
        self.k = torch.nn.Linear(features_dim, dim_k) # dim_k = feature_dim // heads
        self.v = torch.nn.Linear(features_dim, dim_k) # these linear layers are actually the W_q, W_k, W_v matrices (weight matrix)
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        return scaled_dot_product_attention(self.q(query), self.k(key), self.v(value))

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, heads_dim: int, features_dim: int, dim_q: int, dim_k: int):
        super().__init__()
        self.heads = torch.nn.ModuleList(
            [AttentionHead(features_dim, dim_q, dim_k) for _ in range(heads_dim)]
        )
        self.linear = torch.nn.Linear(heads_dim * dim_k, features_dim)
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        return self.linear(
            torch.cat([head(query, key, value) for head in self.heads], dim=-1)
        )

class TransformerEncoderLayer(torch.nn.Module):
    def __init__(self, features_dim: int, heads_dim: int, feedforward_dim: int, normalising: bool, dropout: float):
        super().__init__()
        dim_q = dim_k = max(features_dim // heads_dim, 1)
        self.attention = Residual(
            MultiHeadAttention(heads_dim, features_dim, dim_q, dim_k),
            dimension=features_dim,
            normalising=normalising,
            dropout=dropout,
        )
        self.feed_forward = Residual(
            feed_forward(features_dim, feedforward_dim),
            dimension=features_dim,
            normalising=normalising,
            dropout=dropout,
        )
    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = self.attention(tensor, tensor, tensor)
        return self.feed_forward(tensor)

class TransformerEncoder(torch.nn.Module):
    def __init__(self, param):
        super().__init__()
        self.dim_sequence = param["dim_sequence"]
        self.dim_features = param["dim_features"]
        self.dim_layers = param["transformer_dim_layers"]
        self.dim_heads = param["transformer_dim_heads"]
        self.dim_feedforward = param["transformer_dim_feedforward"]
        self.normalising = param["transformer_normalising"]
        self.dropout = param["transformer_dropout"]
        self.position_encoding = param["transformer_position_encoding"]
        self.layers = torch.nn.ModuleList(
            [
                TransformerEncoderLayer(self.dim_features, self.dim_heads, self.dim_feedforward, self.normalising, self.dropout)
                for _ in range(self.dim_layers)
            ]
        )
    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        seq_len, dimension = tensor.size(1), tensor.size(2)
        if self.position_encoding:
            tensor += position_encoding(seq_len, dimension)
        for layer in self.layers:
            tensor = layer(tensor)
        return tensor
