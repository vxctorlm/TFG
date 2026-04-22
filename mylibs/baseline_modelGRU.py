import torch
import torch.nn as nn

from model.mylibs.resnet import resnet18


class TemporalAttentionPooling(nn.Module):
    def __init__(self, dim: int, attn_hidden: int = 64, dropout: float = 0.2):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(dim, attn_hidden),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(attn_hidden, 1),
        )

    def forward(self, x):
        """
        x: [B, T, D]
        returns:
            pooled: [B, D]
            weights: [B, T]
        """
        scores = self.attn(x).squeeze(-1)               # [B, T]
        weights = torch.softmax(scores, dim=1)          # [B, T]
        pooled = torch.sum(x * weights.unsqueeze(-1), dim=1)
        return pooled, weights


class BaselineResNetGRU(nn.Module):
    def __init__(
        self,
        num_classes=2,
        d_model=128,
        gru_hidden=128,
        gru_layers=1,
        dropout=0.4,
        pretrained=True,
        freeze_early=False,
        freeze_all=True,
        unfreeze_layer4=False,
        bidirectional=True,
    ):
        super().__init__()

        # Guardamos flags para el override de train()
        self.freeze_all = freeze_all
        self.unfreeze_layer4 = unfreeze_layer4

        # Backbone visual
        self.backbone = resnet18(
            pretrained=pretrained,
            freeze_early=freeze_early,
            freeze_all=freeze_all,
        )

        if unfreeze_layer4:
            #for p in self.backbone.layer4.parameters():
            for p in self.backbone.layer4[1].parameters():
                p.requires_grad = True
            print("[BaselineResNetGRU] layer4 descongelado.")

        # Proyección compacta y estable
        self.proj = nn.Sequential(
            nn.Linear(512, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # GRU temporal pequeña
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        gru_out_dim = gru_hidden * 2 if bidirectional else gru_hidden

        # Residual temporal
        self.residual_proj = (
            nn.Identity() if gru_out_dim == d_model else nn.Linear(d_model, gru_out_dim)
        )

        self.temporal_norm = nn.LayerNorm(gru_out_dim)
        self.temporal_dropout = nn.Dropout(dropout)

        # Attention pooling en lugar de mean pooling
        self.pool = TemporalAttentionPooling(
            dim=gru_out_dim,
            attn_hidden=max(32, gru_out_dim // 2),
            dropout=0.2,
        )

        # Head final
        self.classifier = nn.Sequential(
            nn.LayerNorm(gru_out_dim),
            nn.Dropout(dropout),
            nn.Linear(gru_out_dim, gru_out_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(gru_out_dim // 2, num_classes),
        )

    def train(self, mode: bool = True):
        """
        Override de train() para evitar que los BatchNorm del backbone
        congelado actualicen sus running_mean/running_var durante el
        entrenamiento.

        Sin este override, aunque freeze_all=True pone requires_grad=False
        en todos los params, los BN siguen acumulando estadísticas del
        dataset nuevo cada vez que se llama model.train(). En evaluación
        esas stats contaminadas se usan y las features del backbone salen
        distintas a las de training → val_loss diverge aunque train baje.

        Comportamiento:
          - freeze_all=True, unfreeze_layer4=False → todo el backbone en eval().
          - freeze_all=True, unfreeze_layer4=True  → backbone en eval() salvo
            layer4, que sigue el modo normal (porque sus params sí entrenan).
          - freeze_all=False → comportamiento estándar de PyTorch.
        """
        super().train(mode)

        if self.freeze_all:
            # Backbone entero en eval: BN usan running stats ImageNet tal cual
            self.backbone.eval()

            # Si layer4 se descongela, ella sí entra en modo normal
            if self.unfreeze_layer4:
                self.backbone.layer4.train(mode)

        return self

    def forward(self, x):
        """
        x: [B, T, C, H, W]
        returns:
            logits:       [B, num_classes]
            attn_weights: [B, T]
        """
        b, t, c, h, w = x.shape

        # Backbone frame a frame
        x = x.reshape(b * t, c, h, w)
        feats = self.backbone.forward_features(x)
        feats = self.proj(feats)
        feats = feats.reshape(b, t, -1)

        # Rama temporal
        gru_out, _ = self.gru(feats)                    # [B, T, gru_out_dim]

        # Residual desde las features proyectadas
        residual = self.residual_proj(feats)            # [B, T, gru_out_dim]
        temporal = self.temporal_norm(gru_out + residual)
        temporal = self.temporal_dropout(temporal)

        # Pooling temporal aprendido
        pooled, attn_weights = self.pool(temporal)      # [B, gru_out_dim], [B, T]

        logits = self.classifier(pooled)

        return logits, attn_weights