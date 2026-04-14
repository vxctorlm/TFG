import math
import torch
import torch.nn as nn

from model.mylibs.resnet import resnet18
from model.mylibs.transformer import TransformerEncoder, TransformerEncoderLayer


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, T, D]
        self.register_buffer("pe", pe)

    def forward(self, x):  # x: [B, T, D]
        t = x.size(1)
        return x + self.pe[:, :t]


class BaselineResNetTransformer(nn.Module):
    """
    Modelo optimizado: ResNet18 + Transformer temporal
    - Mean pooling (recomendado aquí)
    - 2 capas de transformer
    - Dropout moderado (0.15)
    """
    def __init__(self, num_classes=2, d_model=256, nhead=4, num_layers=2,
                 dim_feedforward=512, dropout=0.15):
        super().__init__()

        #self.backbone = resnet18(num_classes=1000)
        self.backbone = resnet18(num_classes=1000, pretrained=True)

        self.proj = nn.Sequential(
            nn.Linear(512, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.pos_encoding = PositionalEncoding(d_model=d_model, max_len=200)

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
        feats = self.backbone.forward_features(x)          # [B*T, 512]
        feats = self.proj(feats)                           # [B*T, d_model]
        feats = feats.reshape(b, t, -1)                    # [B, T, d_model]

        feats = self.pos_encoding(feats)
        feats = self.transformer(feats)

        pooled = feats.mean(dim=1)                         # Mean pooling

        logits = self.classifier(self.dropout(pooled))
        return logits