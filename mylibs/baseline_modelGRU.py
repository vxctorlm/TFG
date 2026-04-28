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

    def forward(self, x: torch.Tensor):
        scores = self.attn(x).squeeze(-1)          # [B, T]
        weights = torch.softmax(scores, dim=1)     # [B, T]
        pooled = torch.sum(x * weights.unsqueeze(-1), dim=1)
        return pooled, weights


class BaselineResNetGRU(nn.Module):
    """
    ResNet18 + GRU causal para anticipación de accidentes.

    Por defecto bidirectional=False para no depender de contexto futuro dentro de la secuencia.
    El modelo recibe una ventana ya observada [B,T,C,H,W] y predice riesgo futuro.
    """

    def __init__(
        self,
        num_classes: int = 2,
        d_model: int = 128,
        gru_hidden: int = 128,
        gru_layers: int = 1,
        dropout: float = 0.3,
        pretrained: bool = True,
        freeze_early: bool = False,
        freeze_all: bool = True,
        unfreeze_layer4: bool = True,
        unfreeze_mode: str = "full_layer4",
        bidirectional: bool = False,
        pooling: str = "attention",  # "attention" o "last"
    ):
        super().__init__()

        if pooling not in {"attention", "last"}:
            raise ValueError("pooling debe ser 'attention' o 'last'")

        self.freeze_all = freeze_all
        self.unfreeze_layer4 = unfreeze_layer4
        self.unfreeze_mode = unfreeze_mode
        self.bidirectional = bidirectional
        self.pooling = pooling

        self.backbone = resnet18(
            pretrained=pretrained,
            freeze_early=freeze_early,
            freeze_all=freeze_all,
        )
        self._apply_unfreeze_policy()

        self.proj = nn.Sequential(
            nn.Linear(512, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        gru_out_dim = gru_hidden * (2 if bidirectional else 1)
        self.residual_proj = nn.Identity() if gru_out_dim == d_model else nn.Linear(d_model, gru_out_dim)
        self.temporal_norm = nn.LayerNorm(gru_out_dim)
        self.temporal_dropout = nn.Dropout(dropout)

        self.attention_pool = TemporalAttentionPooling(
            dim=gru_out_dim,
            attn_hidden=max(32, gru_out_dim // 2),
            dropout=0.2,
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(gru_out_dim),
            nn.Dropout(dropout),
            nn.Linear(gru_out_dim, max(32, gru_out_dim // 2)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(32, gru_out_dim // 2), num_classes),
        )

    def _apply_unfreeze_policy(self) -> None:
        if not self.unfreeze_layer4:
            return

        if self.unfreeze_mode == "last_block":
            for p in self.backbone.layer4[1].parameters():
                p.requires_grad = True
            print("[BaselineResNetGRU] Descongelado: layer4[1].")
        elif self.unfreeze_mode == "full_layer4":
            for p in self.backbone.layer4.parameters():
                p.requires_grad = True
            print("[BaselineResNetGRU] Descongelado: layer4 completo.")
        elif self.unfreeze_mode == "layer3_layer4":
            for p in self.backbone.layer3.parameters():
                p.requires_grad = True
            for p in self.backbone.layer4.parameters():
                p.requires_grad = True
            print("[BaselineResNetGRU] Descongelado: layer3 + layer4.")
        else:
            raise ValueError(f"unfreeze_mode desconocido: {self.unfreeze_mode}")

    @staticmethod
    def _set_bn_eval(module: nn.Module) -> None:
        for m in module.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def train(self, mode: bool = True):
        super().train(mode)

        if self.freeze_all:
            self.backbone.eval()

            if self.unfreeze_layer4:
                if self.unfreeze_mode == "last_block":
                    self.backbone.layer4[1].train(mode)
                    self._set_bn_eval(self.backbone.layer4[1])
                elif self.unfreeze_mode == "full_layer4":
                    self.backbone.layer4.train(mode)
                    self._set_bn_eval(self.backbone.layer4)
                elif self.unfreeze_mode == "layer3_layer4":
                    self.backbone.layer3.train(mode)
                    self.backbone.layer4.train(mode)
                    self._set_bn_eval(self.backbone.layer3)
                    self._set_bn_eval(self.backbone.layer4)
        return self

    def forward(self, x: torch.Tensor):
        b, t, c, h, w = x.shape

        x = x.reshape(b * t, c, h, w)
        feats = self.backbone.forward_features(x)  # [B*T, 512]
        feats = self.proj(feats)                   # [B*T, d_model]
        feats = feats.reshape(b, t, -1)            # [B, T, d_model]

        gru_out, _ = self.gru(feats)
        temporal = self.temporal_norm(gru_out + self.residual_proj(feats))
        temporal = self.temporal_dropout(temporal)

        if self.pooling == "attention":
            pooled, attn_weights = self.attention_pool(temporal)
        else:
            pooled = temporal[:, -1, :]
            attn_weights = torch.zeros(b, t, device=x.device, dtype=temporal.dtype)
            attn_weights[:, -1] = 1.0

        logits = self.classifier(pooled)
        return logits, attn_weights
