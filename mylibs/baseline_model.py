import math
import torch
import torch.nn as nn

from model.mylibs.resnet import resnet18
from model.mylibs.transformer import TransformerEncoder, TransformerEncoderLayer

"""
1. Entrada: [B, T, C, H, W]
   - B: batch size, número de clips procesados a la vez
   - T: número de frames del clip
   - C: canales de la imagen (3 en RGB)
   - H, W: alto y ancho de la imagen

2. ResNet-18: extractor de características visuales
   - Se aplica frame a frame.
   - Cargado con pesos ImageNet preentrenados.
   - Las capas de bajo nivel (conv1, layer1, layer2) se congelan opcionalmente.

2.1 Forward de la parte espacial
   - Reorganiza la entrada para tratar cada frame como una imagen independiente:
     [B, T, C, H, W] -> [B*T, C, H, W]
   - Cada frame deja de ser una imagen cruda y pasa a representarse mediante un embedding visual de 512 dimensiones.

3. Proyección de características
   - El vector de 512 dimensiones se proyecta al espacio latente del Transformer (d_model = 256).
   - Esta adaptación incluye una transformación lineal, normalización, activación y dropout.

4. Recuperación de la estructura temporal
   - Tras extraer las features de cada frame, se reconstruye la secuencia temporal: [B*T, 256] -> [B, T, 256]

5. Positional Encoding
   - Añade información sobre la posición temporal de cada frame.
   - Es necesario porque el Transformer, por sí solo, no conoce el orden de la secuencia.

6. Transformer temporal
   - Modela las relaciones temporales entre los frames.
   - Aprende cómo evoluciona la escena a lo largo del clip y qué frames son más relevantes para la decisión final.

7. Pooling temporal
   - Resume toda la secuencia temporal en un único vector representativo del clip.
   - Se utiliza mean pooling sobre la dimensión temporal.

8. Clasificador final
   - A partir del vector global del clip, se realiza la clasificación binaria.

8.1 Dropout
   - Se utiliza como mecanismo de regularización para reducir el sobreajuste.

8.2 Capa lineal final
   - Convierte el vector global del clip en logits de tamaño [B, 2]:
     - clase 0: no accidente
     - clase 1: accidente
"""


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
    def __init__(
        self,
        num_classes=2,
        d_model=256,
        nhead=4,
        num_layers=2,
        dim_feedforward=512,
        dropout=0.3,
        pretrained=True,
        freeze_early=True,
    ):
        super().__init__()

        # Backbone espacial con pesos ImageNet preentrenados
        self.backbone = resnet18(pretrained=pretrained, freeze_early=freeze_early)

        # Proyección de 512 -> d_model con normalización, activación y dropout
        self.proj = nn.Sequential(
            nn.Linear(512, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # Positional encoding temporal
        self.pos_encoding = PositionalEncoding(d_model=d_model, max_len=200)

        # Transformer temporal
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        self.transformer = TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):  # x: [B, T, C, H, W]
        b, t, c, h, w = x.shape

        # Juntamos batch y tiempo para pasar frame a frame por ResNet
        x = x.reshape(b * t, c, h, w)

        # [B*T, 512]
        feats = self.backbone.forward_features(x)

        # [B*T, d_model]
        feats = self.proj(feats)

        # [B, T, d_model]
        feats = feats.reshape(b, t, -1)

        feats = self.pos_encoding(feats)
        feats = self.transformer(feats)

        # Mean pooling temporal para obtener una representación global del clip
        pooled = feats.mean(dim=1)

        logits = self.classifier(self.dropout(pooled))  # [B, 2]
        return logits