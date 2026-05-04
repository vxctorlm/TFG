import torch
import torch.nn as nn


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
        x: [B, S, D]
        returns:
            pooled: [B, D]
            weights: [B, S]
        """
        scores = self.attn(x).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        pooled = torch.sum(x * weights.unsqueeze(-1), dim=1)
        return pooled, weights


class X3DGRU(nn.Module):
    """
    X3D-S + GRU + attention.

    Expected input from AccidentClipDataset:
        x: [B, T, C, H, W]

    Internally:
        [B, T, C, H, W]
        -> sliding subclips [B, S, L, C, H, W]
        -> X3D feature extractor per subclip
        -> [B, S, D]
        -> GRU + temporal attention
        -> logits [B, num_classes]

    Recommended for X3D-S:
        image_size = 160
        num_frames = 16 or 32
        subclip_len = 13
        subclip_stride = 1 for T=16, 3-6 for T=32/64
    """

    def __init__(
        self,
        num_classes=2,
        model_name="x3d_s",
        pretrained=True,
        subclip_len=13,
        subclip_stride=1,
        d_model=128,
        gru_hidden=128,
        gru_layers=1,
        dropout=0.4,
        bidirectional=True,
        freeze_backbone=True,
        unfreeze_last_n_blocks=1,
        use_aux_heads=False,
        num_types=61,
        num_weather=4,
        num_light=2,
        num_scenes=5,
        num_linear=5,
    ):
        super().__init__()

        self.model_name = model_name
        self.subclip_len = subclip_len
        self.subclip_stride = subclip_stride
        self.freeze_backbone = freeze_backbone
        self.unfreeze_last_n_blocks = unfreeze_last_n_blocks

        # Requires: pip install pytorchvideo
        x3d = torch.hub.load(
            "facebookresearch/pytorchvideo",
            model_name,
            pretrained=pretrained,
        )

        # PyTorchVideo X3D: blocks[-1] is the classification head.
        # We keep the convolutional/video blocks and remove the Kinetics classifier.
        self.backbone = nn.Sequential(*list(x3d.blocks[:-1]))

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

            if unfreeze_last_n_blocks > 0:
                for block in list(self.backbone.children())[-unfreeze_last_n_blocks:]:
                    for p in block.parameters():
                        p.requires_grad = True

        # Infer backbone feature dimension dynamically.
        feat_dim = self._infer_backbone_dim()
        self.backbone_feat_dim = feat_dim

        self.proj = nn.Sequential(
            nn.Linear(feat_dim, d_model),
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

        gru_out_dim = gru_hidden * 2 if bidirectional else gru_hidden

        self.residual_proj = (
            nn.Identity() if gru_out_dim == d_model else nn.Linear(d_model, gru_out_dim)
        )
        self.temporal_norm = nn.LayerNorm(gru_out_dim)
        self.temporal_dropout = nn.Dropout(dropout)

        self.pool = TemporalAttentionPooling(
            dim=gru_out_dim,
            attn_hidden=max(32, gru_out_dim // 2),
            dropout=0.2,
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(gru_out_dim),
            nn.Dropout(dropout),
            nn.Linear(gru_out_dim, gru_out_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(gru_out_dim // 2, num_classes),
        )

        self.use_aux_heads = use_aux_heads

        if self.use_aux_heads:
            self.type_head = nn.Sequential(
                nn.LayerNorm(gru_out_dim),
                nn.Dropout(dropout),
                nn.Linear(gru_out_dim, num_types),
            )

            self.weather_head = nn.Sequential(
                nn.LayerNorm(gru_out_dim),
                nn.Dropout(dropout),
                nn.Linear(gru_out_dim, num_weather),
            )

            self.light_head = nn.Sequential(
                nn.LayerNorm(gru_out_dim),
                nn.Dropout(dropout),
                nn.Linear(gru_out_dim, num_light),
            )

            self.scene_head = nn.Sequential(
                nn.LayerNorm(gru_out_dim),
                nn.Dropout(dropout),
                nn.Linear(gru_out_dim, num_scenes),
            )

            self.linear_head = nn.Sequential(
                nn.LayerNorm(gru_out_dim),
                nn.Dropout(dropout),
                nn.Linear(gru_out_dim, num_linear),
            )

    @torch.no_grad()
    def _infer_backbone_dim(self):
        was_training = self.backbone.training
        self.backbone.eval()
        dummy = torch.zeros(1, 3, self.subclip_len, 160, 160)
        out = self.backbone(dummy)
        out = self._pool_backbone_output(out)
        if was_training:
            self.backbone.train()
        return out.shape[-1]

    @staticmethod
    def _pool_backbone_output(x):
        # Typical output: [N, C, T, H, W]. Some variants may already be [N, C].
        if x.ndim == 5:
            x = x.mean(dim=(2, 3, 4))
        elif x.ndim == 4:
            x = x.mean(dim=(2, 3))
        elif x.ndim == 3:
            x = x.mean(dim=2)
        elif x.ndim != 2:
            raise ValueError(f"Unexpected X3D backbone output shape: {tuple(x.shape)}")
        return x

    def _make_subclips(self, x):
        """
        x: [B, T, C, H, W]
        returns: [B, S, L, C, H, W]
        """
        b, t, c, h, w = x.shape
        if t < self.subclip_len:
            raise ValueError(
                f"num_frames={t} menor que subclip_len={self.subclip_len}. "
                f"Aumenta num_frames o reduce subclip_len."
            )

        starts = list(range(0, t - self.subclip_len + 1, self.subclip_stride))
        if starts[-1] != t - self.subclip_len:
            starts.append(t - self.subclip_len)

        subclips = [x[:, s:s + self.subclip_len] for s in starts]
        return torch.stack(subclips, dim=1)

    def train(self, mode: bool = True):
        super().train(mode)

        if self.freeze_backbone:
            self.backbone.eval()

            # Keep unfrozen final blocks trainable, but freeze their BN statistics.
            if self.unfreeze_last_n_blocks > 0:
                for block in list(self.backbone.children())[-self.unfreeze_last_n_blocks:]:
                    block.train(mode)
                    for m in block.modules():
                        if isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d, nn.BatchNorm1d)):
                            m.eval()
        return self

    def forward(self, x):
        """
        x: [B, T, C, H, W]
        returns:
            logits: [B, num_classes]
            attn_weights: [B, S]
        """
        b, t, c, h, w = x.shape

        subclips = self._make_subclips(x)  # [B, S, L, C, H, W]
        b, s, l, c, h, w = subclips.shape

        subclips = subclips.reshape(b * s, l, c, h, w)
        subclips = subclips.permute(0, 2, 1, 3, 4).contiguous()  # [B*S, C, L, H, W]

        feats = self.backbone(subclips)
        feats = self._pool_backbone_output(feats)  # [B*S, D]
        feats = self.proj(feats)
        feats = feats.reshape(b, s, -1)  # [B, S, d_model]

        gru_out, _ = self.gru(feats)
        residual = self.residual_proj(feats)
        temporal = self.temporal_norm(gru_out + residual)
        temporal = self.temporal_dropout(temporal)

        pooled, attn_weights = self.pool(temporal)
        logits = self.classifier(pooled)

        if self.use_aux_heads:
            aux_logits = {
                "type": self.type_head(pooled),
                "weather": self.weather_head(pooled),
                "light": self.light_head(pooled),
                "scene": self.scene_head(pooled),
                "linear": self.linear_head(pooled),
            }
            return logits, attn_weights, aux_logits

        return logits, attn_weights
