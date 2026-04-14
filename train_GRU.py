from torchvision.transforms import v2
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import os
import wandb
import random
import numpy as np
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score, average_precision_score, roc_auc_score

from model.dataset import AccidentClipDataset
from model.mylibs.baseline_modelGRU import BaselineResNetGRUSimple


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_preds = []
    all_scores = []

    for clips, labels in dataloader:
        clips = clips.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        outputs = model(clips)
        loss = criterion(outputs, labels)
        running_loss += loss.item()

        preds = torch.argmax(outputs, dim=1)
        probs = torch.softmax(outputs, dim=1)[:, 1]

        all_labels.extend(labels.cpu().numpy().tolist())
        all_preds.extend(preds.cpu().numpy().tolist())
        all_scores.extend(probs.cpu().numpy().tolist())

    val_loss = running_loss / len(dataloader)
    val_acc = accuracy_score(all_labels, all_preds)
    val_f1 = f1_score(all_labels, all_preds, zero_division=0)

    try:
        val_ap = average_precision_score(all_labels, all_scores)
    except ValueError:
        val_ap = float("nan")
    try:
        val_auc = roc_auc_score(all_labels, all_scores)
    except ValueError:
        val_auc = float("nan")

    return val_loss, val_acc, val_f1, val_ap, val_auc


def build_clip_transform(
    train=False,
    image_size=(224, 224),
    enable_augmentation=False,
    use_hflip=False,
    use_color_jitter=False,
    use_random_resized_crop=False,
    use_gaussian_blur=False,
):
    ops = []
    if train and enable_augmentation and use_random_resized_crop:
        ops.append(
            v2.RandomResizedCrop(
                size=image_size,
                scale=(0.90, 1.0),
                ratio=(0.95, 1.05),
                antialias=True,
            )
        )
    else:
        ops.append(v2.Resize(image_size, antialias=True))

    if train and enable_augmentation:
        if use_hflip:
            ops.append(v2.RandomHorizontalFlip(p=0.5))
        if use_color_jitter:
            ops.append(
                v2.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02)
            )
        if use_gaussian_blur:
            ops.append(
                v2.RandomApply([v2.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5))], p=0.3)
            )

    ops.extend([
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return v2.Compose(ops)


def main():
    seed = 42
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Current device:", torch.cuda.current_device())
        print("Device name:", torch.cuda.get_device_name(0))

    # ==================== HIPERPARÁMETROS MEJORADOS ====================
    batch_size = 8
    num_epochs = 100
    learning_rate = 5e-5
    weight_decay = 1e-3
    num_frames = 8

    # Augmentación
    enable_augmentation = True
    use_hflip = True
    use_color_jitter = True
    use_random_resized_crop = True
    use_gaussian_blur = False

    # Temporal + Anticipation
    use_temporal_augmentation = True
    temporal_max_jitter = 1
    use_toa_guided_sampling = True
    toa_center_strength = 0.3056937753654446
    anticipation_mode = True
    anticipation_offset = 5

    # Modelo GRU
    d_model = 256
    hidden_size = 96
    num_gru_layers = 1
    bidirectional = True          # ← Mejoría clave
    dropout = 0.45                # ← Más regularización

    # Early stopping
    patience = 6
    min_delta = 0.0015

    # ==================== RUN NAME ====================
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    frames_tag = f"{num_frames}f"
    run_name = (
        f"gru_improved_{timestamp}_{frames_tag}"
        f"_anticip{int(anticipation_mode)}_off{anticipation_offset}"
        f"_bidir{int(bidirectional)}_hs{hidden_size}"
        f"_do{dropout}_wd{weight_decay:.1e}"
    )
    print("Run name:", run_name)

    # ==================== WANDB ====================
    wandb.init(
        project="tfg-accident-prediction",
        name=run_name,
        config={
            "seed": seed,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "num_frames": num_frames,
            "model": "ResNet18 + BiGRU + Temporal Attention",
            "d_model": d_model,
            "hidden_size": hidden_size,
            "num_gru_layers": num_gru_layers,
            "bidirectional": bidirectional,
            "dropout": dropout,
            "patience": patience,
            "min_delta": min_delta,
            "enable_augmentation": enable_augmentation,
            "use_hflip": use_hflip,
            "use_color_jitter": use_color_jitter,
            "use_random_resized_crop": use_random_resized_crop,
            "use_gaussian_blur": use_gaussian_blur,
            "use_temporal_augmentation": use_temporal_augmentation,
            "temporal_max_jitter": temporal_max_jitter,
            "use_toa_guided_sampling": use_toa_guided_sampling,
            "toa_center_strength": toa_center_strength,
            "anticipation_mode": anticipation_mode,
            "anticipation_offset": anticipation_offset,
        }
    )

    # ==================== TRANSFORMS & DATASETS ====================
    train_transform = build_clip_transform(
        train=True,
        enable_augmentation=enable_augmentation,
        use_hflip=use_hflip,
        use_color_jitter=use_color_jitter,
        use_random_resized_crop=use_random_resized_crop,
        use_gaussian_blur=use_gaussian_blur,
    )
    val_transform = build_clip_transform(train=False)

    train_dataset = AccidentClipDataset(
        txt_path="/data-fast/data-server/vlopezmo/model/training/training_train.txt",
        rgb_root="/data-fast/data-server/vlopezmo/DADA2000",
        num_frames=num_frames,
        transform=train_transform,
        train=True,
        use_temporal_augmentation=use_temporal_augmentation,
        temporal_max_jitter=temporal_max_jitter,
        use_toa_guided_sampling=use_toa_guided_sampling,
        toa_center_strength=toa_center_strength,
        anticipation_mode=anticipation_mode,
        anticipation_offset=anticipation_offset,
    )

    val_dataset = AccidentClipDataset(
        txt_path="/data-fast/data-server/vlopezmo/model/training/training_val.txt",
        rgb_root="/data-fast/data-server/vlopezmo/DADA2000",
        num_frames=num_frames,
        transform=val_transform,
        train=False,
        use_temporal_augmentation=False,
        anticipation_mode=anticipation_mode,
        anticipation_offset=anticipation_offset,
    )

    print("Número de muestras de train:", len(train_dataset))
    print("Número de muestras de val:", len(val_dataset))

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True, persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True, persistent_workers=True
    )

    # ==================== MODELO ====================
    model = BaselineResNetGRUSimple(
        num_classes=2,
        d_model=d_model,
        hidden_size=hidden_size,
        num_gru_layers=num_gru_layers,
        bidirectional=bidirectional,
        dropout=dropout,
    ).to(device)

    # Congelar backbone las primeras 8 épocas
    for param in model.backbone.parameters():
        param.requires_grad = False
    print("Backbone congelado durante las primeras 8 épocas")

    # Loss con Label Smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=1
    )

    ckpt_dir = "/data-fast/data-server/vlopezmo/model/checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)
    best_val_ap = -1.0
    best_val_ap_for_stop = -1.0
    early_stop_counter = 0

    # ==================== TRAINING LOOP ====================
    for epoch in range(num_epochs):
        # Descongelar backbone a partir de la época 8
        if epoch == 8:
            for param in model.backbone.parameters():
                param.requires_grad = True
            print("Backbone descongelado - iniciando fine-tuning")

        model.train()
        running_loss = 0.0
        all_train_labels = []
        all_train_preds = []
        all_train_scores = []

        for batch_idx, (clips, labels) in enumerate(train_loader):
            clips = clips.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(clips)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            probs = torch.softmax(outputs, dim=1)[:, 1]

            all_train_labels.extend(labels.detach().cpu().numpy().tolist())
            all_train_preds.extend(preds.detach().cpu().numpy().tolist())
            all_train_scores.extend(probs.detach().cpu().numpy().tolist())

            if (batch_idx + 1) % 50 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] Batch [{batch_idx+1}/{len(train_loader)}] Loss: {loss.item():.4f}")

        train_loss = running_loss / len(train_loader)
        train_acc = accuracy_score(all_train_labels, all_train_preds)
        train_f1 = f1_score(all_train_labels, all_train_preds, zero_division=0)
        train_ap = average_precision_score(all_train_labels, all_train_scores)
        train_auc = roc_auc_score(all_train_labels, all_train_scores)

        val_loss, val_acc, val_f1, val_ap, val_auc = evaluate(model, val_loader, criterion, device)

        print(f"\nEpoch [{epoch+1}/{num_epochs}] completed")
        print(
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Train F1: {train_f1:.4f} | "
            f"Train AP: {train_ap:.4f} | Train AUC: {train_auc:.4f} || "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f} | "
            f"Val AP: {val_ap:.4f} | Val AUC: {val_auc:.4f}"
        )

        wandb.log({
            "epoch": epoch + 1,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_loss,
            "train_acc": train_acc,
            "train_f1": train_f1,
            "train_ap": train_ap,
            "train_roc_auc": train_auc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_f1": val_f1,
            "val_ap": val_ap,
            "val_roc_auc": val_auc,
        })

        old_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_ap)
        new_lr = optimizer.param_groups[0]["lr"]
        if new_lr != old_lr:
            print(f"Learning rate reducido: {old_lr:.2e} -> {new_lr:.2e}")

        # Guardar mejor modelo
        if val_ap > best_val_ap:
            best_val_ap = val_ap
            ckpt_path = os.path.join(ckpt_dir, f"{run_name}_best_val_ap.pt")
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_ap": val_ap,
                "val_acc": val_acc,
                "val_f1": val_f1,
                "val_auc": val_auc,
                "seed": seed,
            }, ckpt_path)
            print(f"Nuevo mejor modelo guardado en: {ckpt_path}")

        # Guardar último checkpoint
        last_ckpt_path = os.path.join(ckpt_dir, f"{run_name}_last.pt")
        torch.save({"epoch": epoch + 1, "model_state_dict": model.state_dict()}, last_ckpt_path)

        # Early stopping
        if val_ap > best_val_ap_for_stop + min_delta:
            best_val_ap_for_stop = val_ap
            early_stop_counter = 0
            print(f"Mejora en val_ap: {val_ap:.4f}")
        else:
            early_stop_counter += 1
            print(f"Sin mejora. EarlyStopping counter: {early_stop_counter}/{patience}")

        if early_stop_counter >= patience:
            print(f"Early stopping activado en epoch {epoch+1}")
            break

    wandb.finish()


if __name__ == "__main__":
    main()