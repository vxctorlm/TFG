from torchvision import transforms
from model.dataset import AccidentClipDataset

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

dataset = AccidentClipDataset(
    txt_path="/data-fast/data-server/vlopezmo/model/training/training.txt",
    rgb_root="/data-fast/data-server/vlopezmo/DADA2000",
    num_frames=16,   # None
    transform=transform,
)

clip, label = dataset[0]

print("clip shape:", clip.shape)
print("label:", label)