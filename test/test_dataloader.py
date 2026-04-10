from torchvision import transforms
from torch.utils.data import DataLoader
from model.dataset import AccidentClipDataset

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

dataset = AccidentClipDataset(
    txt_path="/data-fast/data-server/vlopezmo/model/training/training.txt",
    rgb_root="/data-fast/data-server/vlopezmo/DADA2000",
    num_frames=16, #None
    transform=transform,
)

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
)

clips, labels = next(iter(loader))

print("clips shape:", clips.shape)
print("labels shape:", labels.shape)
print("labels:", labels)