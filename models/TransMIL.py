"""TransMIL classification baseline and reusable WSI encoder.

The important public interface is ``encode(data)``:

    data      : [B, N_tiles, feat_dim]
    embedding : [B, 512]

The DAG model reuses this encoder but bypasses the classification head.
"""

from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn

try:
    from nystrom_attention import NystromAttention
except ImportError as exc:  # pragma: no cover - dependency error is clearer here
    raise ImportError(
        "TransMIL requires nystrom-attention. Install it with "
        "`pip install nystrom-attention`."
    ) from exc


class TransLayer(nn.Module):
    def __init__(self, dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = NystromAttention(
            dim=dim,
            dim_head=dim // 8,
            heads=8,
            num_landmarks=max(1, dim // 2),
            pinv_iterations=6,
            residual=True,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.attn(self.norm(x))


class PPEG(nn.Module):
    """Pyramid position encoding generator used by TransMIL."""

    def __init__(self, dim: int = 512):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7 // 2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5 // 2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3 // 2, groups=dim)

    def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        cls_token, feature_tokens = x[:, :1], x[:, 1:]
        batch, n_tokens, dim = feature_tokens.shape
        if n_tokens != height * width:
            raise ValueError(
                f"PPEG received {n_tokens} feature tokens but H*W={height * width}."
            )

        cnn_feat = feature_tokens.transpose(1, 2).reshape(batch, dim, height, width)
        cnn_feat = (
            cnn_feat
            + self.proj(cnn_feat)
            + self.proj1(cnn_feat)
            + self.proj2(cnn_feat)
        )
        feature_tokens = cnn_feat.flatten(2).transpose(1, 2)
        return torch.cat((cls_token, feature_tokens), dim=1)


class TransMILBackbone(nn.Module):
    """WSI tile aggregation backbone shared by all three tasks."""

    embedding_dim = 512

    def __init__(
        self,
        feat_dim: int = 1024,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.feat_dim = int(feat_dim)
        self.dropout = float(dropout)

        self.fc1 = nn.Sequential(
            nn.Linear(self.feat_dim, self.embedding_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embedding_dim))
        self.layer1 = TransLayer(self.embedding_dim, self.dropout)
        self.pos_layer = PPEG(self.embedding_dim)
        self.layer2 = TransLayer(self.embedding_dim, self.dropout)
        self.norm = nn.LayerNorm(self.embedding_dim)

    @staticmethod
    def _validate_input(data: torch.Tensor, feat_dim: int) -> torch.Tensor:
        if not torch.is_tensor(data):
            raise TypeError("data must be a torch.Tensor.")
        if data.ndim == 2:
            data = data.unsqueeze(0)
        if data.ndim != 3:
            raise ValueError(
                "Expected WSI features [B, N_tiles, feat_dim], "
                f"received {tuple(data.shape)}."
            )
        if data.shape[1] < 1:
            raise ValueError("Each WSI bag must contain at least one tile.")
        if data.shape[2] != feat_dim:
            raise ValueError(
                f"Expected feat_dim={feat_dim}, received {data.shape[2]}."
            )
        return data.float()

    @staticmethod
    def _square_pad(tokens: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        n_tokens = tokens.shape[1]
        side = int(math.ceil(math.sqrt(n_tokens)))
        target = side * side
        missing = target - n_tokens
        if missing:
            # Repetition is deterministic and also works when missing > n_tokens.
            repeats = int(math.ceil(missing / n_tokens))
            padding = tokens.repeat(1, repeats, 1)[:, :missing]
            tokens = torch.cat((tokens, padding), dim=1)
        return tokens, side, side

    def encode(self, data: torch.Tensor) -> torch.Tensor:
        data = self._validate_input(data, self.feat_dim)
        tokens = self.fc1(data)
        tokens, height, width = self._square_pad(tokens)

        cls_tokens = self.cls_token.expand(tokens.shape[0], -1, -1)
        tokens = torch.cat((cls_tokens, tokens), dim=1)
        tokens = self.layer1(tokens)
        tokens = self.pos_layer(tokens, height, width)
        tokens = self.layer2(tokens)
        return self.norm(tokens)[:, 0]


class TransMIL(nn.Module):
    """Original classification baseline; not used inside the DAG model."""

    def __init__(
        self,
        n_classes: int,
        feat_dim: int = 1024,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.n_classes = int(n_classes)
        self.backbone = TransMILBackbone(feat_dim=feat_dim, dropout=dropout)
        self._fc2 = nn.Linear(self.backbone.embedding_dim, self.n_classes)

    @property
    def feat_dim(self) -> int:
        return self.backbone.feat_dim

    def encode(self, data: torch.Tensor) -> torch.Tensor:
        return self.backbone.encode(data)

    def forward(self, data: torch.Tensor, **_: object) -> Dict[str, torch.Tensor]:
        embedding = self.encode(data)
        logits = self._fc2(embedding)
        return {"logits": logits, "embedding": embedding}


if __name__ == "__main__":
    model = TransMIL(n_classes=3, feat_dim=1024)
    output = model(torch.randn(2, 121, 1024))
    print({key: tuple(value.shape) for key, value in output.items()})
