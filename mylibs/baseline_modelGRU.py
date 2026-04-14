import torch
import torch.nn as nn
from model.mylibs.resnet import resnet18

class BaselineResNetGRUSimple(nn.Module):
    def __init__(
        self,
        num_classes=2,
        d_model=256,
        hidden_size=96,           # ← bajado un poco (128 → 96)
        num_gru_layers=1,
        bidirectional=True,       # ← cambiado a True (alto impacto esperado)
        dropout=0.45,             # ← subido significativamente
    ):
        super().__init__()

        self.backbone = resnet18(num_classes=1000)

        # Proyección
        self.proj = nn.Sequential(
            nn.Linear(512, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # GRU
        gru_dropout = dropout if num_gru_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=hidden_size,
            num_layers=num_gru_layers,
            batch_first=True,
            dropout=gru_dropout,
            bidirectional=bidirectional,
        )

        gru_output_dim = hidden_size * 2 if bidirectional else hidden_size

        self.post_gru_norm = nn.LayerNorm(gru_output_dim)

        # Atención temporal mejorada (menos agresiva)
        attn_hidden = max(gru_output_dim // 2, 16)
        self.attn = nn.Sequential(
            nn.Linear(gru_output_dim, attn_hidden),
            nn.Tanh(),
            nn.Dropout(0.2),               # ← dropout dentro de atención
            nn.Linear(attn_hidden, 1),
        )

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(gru_output_dim, num_classes)

    def forward(self, x):
        b, t, c, h, w = x.shape

        # Backbone
        x = x.reshape(b * t, c, h, w)
        feats = self.backbone.forward_features(x)      # [B*T, 512]
        feats = self.proj(feats)                       # [B*T, d_model]
        feats = feats.reshape(b, t, -1)                # [B, T, d_model]

        # GRU + norm
        gru_out, _ = self.gru(feats)
        gru_out = self.post_gru_norm(gru_out)

        # Atención temporal
        attn_scores = self.attn(gru_out)
        attn_weights = torch.softmax(attn_scores, dim=1)
        pooled = torch.sum(attn_weights * gru_out, dim=1)

        logits = self.classifier(self.dropout(pooled))
        return logits