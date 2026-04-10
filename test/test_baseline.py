import torch
from model.mylibs.baseline_model import BaselineResNetTransformer

model = BaselineResNetTransformer(
    num_classes=2,
    d_model=256,
    nhead=4,
    num_layers=2,
    dim_feedforward=512,
    dropout=0.1,
)

x = torch.randn(2, 16, 3, 224, 224)

y = model(x)

print("Output shape:", y.shape)
print(y)