import torch
import torch.nn as nn

from model.mylibs.resnet import resnet18

"""
1. Entrada: [B, T, C, H, W]
   - B: batch size, número de clips procesados a la vez
   - T: número de frames del clip
   - C: canales de la imagen (3 en RGB)
   - H, W: alto y ancho de la imagen

2. ResNet-18: extractor de características visuales
   - Se aplica frame a frame.
   - Su función es transformar cada frame en un vector representativo de características visuales.

2.1 Forward de la parte espacial
   - Reorganiza la entrada para tratar cada frame como una imagen independiente:
     [B, T, C, H, W] -> [B*T, C, H, W]
   - Cada frame deja de ser una imagen cruda y pasa a representarse mediante un embedding visual de 512 dimensiones.

3. Proyección de características
   - El vector de 512 dimensiones se proyecta a un espacio latente de dimensión d_model = 256.
   - Esta adaptación incluye una transformación lineal, normalización, activación y dropout.

4. Recuperación de la estructura temporal
   - Tras extraer las features de cada frame, se reconstruye la secuencia temporal:
     [B*T, 256] -> [B, T, 256]

5. GRU temporal
   - La secuencia de embeddings visuales se introduce en una GRU para modelar la evolución temporal del clip.
   - La GRU procesa los frames en orden y va actualizando su estado oculto para capturar dependencias temporales entre ellos.
   - A diferencia del Transformer, aquí el orden temporal ya está implícito en el propio procesamiento secuencial, por lo que no hace falta positional encoding.

6. Salida de la GRU
   - La GRU genera una representación temporal para cada instante de la secuencia:
     [B, T, 256] -> [B, T, H_out]
   - Si la GRU es bidireccional, H_out = hidden_size * 2.
   - Si no es bidireccional, H_out = hidden_size.

7. Normalización tras la GRU
   - Se aplica LayerNorm sobre la salida de la GRU para estabilizar la representación temporal antes de clasificar.

8. Selección de la representación final del clip
   - En este baseline se utiliza el último estado temporal de la secuencia:
     [B, T, H_out] -> [B, H_out]
   - La idea es que este último vector resume la información acumulada de los frames anteriores.

9. Clasificador final
   - A partir del vector final del clip, se realiza la clasificación binaria.

9.1 Dropout
   - Se utiliza como mecanismo de regularización para reducir el sobreajuste.

9.2 Capa lineal final
   - Convierte el vector global del clip en logits de tamaño [B, 2]:
     - clase 0: no accidente
     - clase 1: accidente
"""

class BaselineResNetGRUSimple(nn.Module):
    def __init__(
        self,
        num_classes=2,
        d_model=256,
        hidden_size=128,
        num_gru_layers=1,
        bidirectional=False,
        dropout=0.33,
    ):
        super().__init__()

        self.backbone = resnet18(num_classes=1000)

        self.proj = nn.Sequential(
            nn.Linear(512, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

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

        # Atención temporal
        attn_hidden = max(gru_output_dim // 2, 16)
        self.attn = nn.Sequential(
            nn.Linear(gru_output_dim, attn_hidden),
            nn.Tanh(),
            nn.Linear(attn_hidden, 1),
        )

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(gru_output_dim, num_classes)

    def forward(self, x):
        # x: [B, T, C, H, W]
        b, t, c, h, w = x.shape

        x = x.reshape(b * t, c, h, w)

        # [B*T, 512]
        feats = self.backbone.forward_features(x)

        # [B*T, d_model]
        feats = self.proj(feats)

        # [B, T, d_model]
        feats = feats.reshape(b, t, -1)

        # [B, T, H_out]
        gru_out, _ = self.gru(feats)
        gru_out = self.post_gru_norm(gru_out)
        
        # Atención temporal: 
        # calcula un peso por cada frame de la secuencia y se obtiene una combinación ponderada de todas las as salidas temporales

        # [B, T, 1]
        attn_scores = self.attn(gru_out)

        # [B, T, 1]
        attn_weights = torch.softmax(attn_scores, dim=1)

        # [B, H_out]
        pooled = torch.sum(attn_weights * gru_out, dim=1)

        '''
        # Solo usa el último estado temporal de la GRU
        pooled = gru_out[:, -1, :]
        '''

        logits = self.classifier(self.dropout(pooled))
        return logits

        