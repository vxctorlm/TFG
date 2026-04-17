import torch
import torch.nn as nn

from model.mylibs.resnet import resnet18

"""
Arquitectura: ResNet-18 (pretrained) + GRU temporal

Historial de cambios:
v1 → v2: h_n[-1] sustituido por outputs.mean(dim=1). num_layers 1→2.
v2 → v3: freeze_all=True, num_layers=1, label smoothing, lr=1e-4.
v3 → v5: sin pesos de clase, toa_center_strength=0.3, lr=2e-5.

v5 → v6 (esta versión):
  - Unfreezing de layer4: el backbone está parcialmente descongelado
    para que las features de alto nivel se adapten al dominio de
    accidentes de tráfico. Las capas anteriores (conv1, layer1, layer2,
    layer3) siguen congeladas porque detectan features universales
    (bordes, texturas, formas simples) que no necesitan fine-tuning.
  - El train loop debe usar LRs diferenciados: muy bajo para layer4
    (~1e-5) y normal para el resto (2e-5) para no destruir los pesos
    preentrenados.
"""


class BaselineResNetGRU(nn.Module):
    def __init__(
        self,
        num_classes=2,
        d_model=256,
        num_layers=1,
        dropout=0.35,
        pretrained=True,
        freeze_early=False,
        freeze_all=False,
        unfreeze_layer4=False,    # Nuevo: descongela layer4 tras el freeze_all inicial
        bidirectional=False,
    ):
        super().__init__()

        # Backbone: primero aplicamos freeze_all (o freeze_early)
        # Luego, si unfreeze_layer4=True, descongelamos layer4 selectivamente.
        self.backbone = resnet18(
            pretrained=pretrained,
            freeze_early=freeze_early,
            freeze_all=freeze_all,
        )

        if unfreeze_layer4:
            for param in self.backbone.layer4.parameters():
                param.requires_grad = True
            print("[BaselineResNetGRU] layer4 descongelado para fine-tuning.")

        # Proyección de 512 -> d_model
        self.proj = nn.Sequential(
            nn.Linear(512, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # GRU temporal
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        out_dim = d_model * 2 if bidirectional else d_model

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(out_dim, num_classes)

    def forward(self, x):  # x: [B, T, C, H, W]
        b, t, c, h, w = x.shape

        # [B*T, C, H, W]
        x = x.reshape(b * t, c, h, w)

        # [B*T, 512]
        # Con layer4 descongelado el backbone calcula gradientes parcialmente.
        # Las capas congeladas no generan gradientes (requires_grad=False).
        feats = self.backbone.forward_features(x)

        # [B*T, d_model]
        feats = self.proj(feats)

        # [B, T, d_model]
        feats = feats.reshape(b, t, -1)

        # outputs: [B, T, d_model]
        outputs, _ = self.gru(feats)

        # Mean pooling sobre toda la secuencia temporal
        pooled = outputs.mean(dim=1)

        logits = self.classifier(self.dropout(pooled))
        return logits