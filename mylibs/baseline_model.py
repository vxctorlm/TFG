import math
import torch
import torch.nn as nn

from model.mylibs.resnet import resnet18
from model.mylibs.transformer import TransformerEncoder, TransformerEncoderLayer


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, L, D]
        self.register_buffer("pe", pe)

    def forward(self, x):  # x: [B, L, D]
        l = x.size(1)
        return x + self.pe[:, :l]


class BaselineResNetTransformer(nn.Module):
    """
    ResNet18 preentrenada + Transformer sobre múltiples tokens espaciales por frame.
    """
    def __init__(
        self,
        num_classes=2,
        d_model=256,
        nhead=4,
        num_layers=2,
        dim_feedforward=512,
        dropout=0.15,
        spatial_size=(2, 2),
    ):
        super().__init__()

        self.backbone = resnet18(num_classes=1000, pretrained=True)

        self.spatial_pool = nn.AdaptiveAvgPool2d(spatial_size)
        self.num_spatial_tokens = spatial_size[0] * spatial_size[1]

        self.proj = nn.Sequential(
            nn.Linear(512, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.pos_encoding = PositionalEncoding(d_model=d_model, max_len=2000)

        encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):  # [B, T, C, H, W]
        b, t, c, h, w = x.shape

        x = x.reshape(b * t, c, h, w)

        # [B*T, 512, 7, 7] aprox
        feats = self.backbone.forward_feature_map(x)

        # [B*T, 512, sh, sw] -> por defecto (2,2)
        feats = self.spatial_pool(feats)

        # [B*T, 512, N]
        feats = feats.flatten(2)

        # [B*T, N, 512]
        feats = feats.transpose(1, 2)

        # [B*T, N, d_model]
        feats = self.proj(feats)

        # [B, T, N, d_model]
        feats = feats.reshape(b, t, self.num_spatial_tokens, -1)

        # [B, T*N, d_model]
        feats = feats.reshape(b, t * self.num_spatial_tokens, -1)

        feats = self.pos_encoding(feats)
        feats = self.transformer(feats)

        pooled = feats.mean(dim=1)
        logits = self.classifier(self.dropout(pooled))
        return logits