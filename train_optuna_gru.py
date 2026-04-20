from torchvision.transforms import v2
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch
import torch.nn as nn
import os
import wandb
import random
import numpy as np
import math
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score, average_precision_score, roc_auc_score

from model.dataset import AccidentClipDataset
from model.mylibs.baseline_modelGRU import BaselineResNetGRU


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()

    running_loss = 0.0
    all_labels = []
    all_preds = []
    all_scores = []
    all_attn = []  # acumulamos pesos de atención para análisis

    for clips, labels in dataloader:
        clips = clips.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # El modelo ahora devuelve (logits, attn_weights)
        outputs, attn_weights = model(clips)
        loss = criterion(outputs, labels)

        running_loss += loss.item()

        preds = torch.argmax(outputs, dim=1)
        probs = torch.softmax(outputs, dim=1)[:, 1]

        all_labels.extend(labels.cpu().numpy().tolist())
        all_preds.extend(preds.cpu().numpy().tolist())
        all_scores.extend(probs.cpu().numpy().tolist())
        all_attn.append(attn_weights.cpu().numpy())

    val_loss = running_loss / len(dataloader)
    val_acc = accuracy_score(all_labels, all_preds)
    val_f1 = f1_score(all_labels, all_preds, zero_division=0)
    val_ap = average_precision_score(all_labels, all_scores)
    val_auc = roc_auc_score(all_labels, all_scores)

    # Media de pesos de atención por frame (para loggear a wandb)
    mean_attn = np.concatenate(all_attn, axis=0).mean(axis=0)  # [T]

    return val_loss, val_acc, val_f1, val_ap, val_auc, mean_attn


def build_clip_transform(
    train=False,
    image_size=(224, 224),
    enable_augmentation=False,
    use_hflip=False,
    use_color_jitter=False,
    use_random_resized_crop=False,
    use_gaussian_blur=False,
    use_random_erasing=False,
):
    ops = []

    if train and enable_augmentation and use_random_resized_crop:
        ops.append(
            v2.RandomResizedCrop(
                size=image_size,
                scale=(0.80, 1.0),
                ratio=(0.90, 1.10),
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
                v2.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.05,
                )
            )

        if use_gaussian_blur:
            ops.append(
                v2.RandomApply(
                    [v2.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))],
                    p=0.3,
                )
            )

    ops.extend([
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    if train and enable_augmentation and use_random_erasing:
        ops.append(
            v2.RandomErasing(p=0.15, scale=(0.02, 0.15), ratio=(0.3, 3.3))
        )

    return v2.Compose(ops)


# ── Cosine Annealing con Linear Warmup ────────────────────────────────────────
class CosineAnnealingWithWarmup(torch.optim.lr_scheduler._LRScheduler):
    """
    Linear warmup durante warmup_epochs, luego cosine decay hasta eta_min.
    Más estable que ReduceLROnPlateau para datasets pequeños.
    """
    def __init__(self, optimizer, warmup_epochs, total_epochs, eta_min=1e-6, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            alpha = (self.last_epoch + 1) / self.warmup_epochs
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            progress = (self.last_epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
            cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
            return [
                self.eta_min + (base_lr - self.eta_min) * cosine_factor
                for base_lr in self.base_lrs
            ]


def main():
    seed = 42
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("CUDA available:", torch.cuda.is_available())

    # ── Hiperparámetros ───────────────────────────────────────────────────────
    batch_size = 8
    num_epochs = 60
    weight_decay = 5e-4
    num_frames = 16

    # Backbone
    pretrained = True
    freeze_early = False
    freeze_all = True
    unfreeze_layer4 = False

    learning_rate = 3e-4
    warmup_epochs = 5
    eta_min = 1e-6

    # Augmentación espacial
    enable_augmentation = False
    use_hflip = True
    use_color_jitter = True
    use_random_resized_crop = True
    use_gaussian_blur = True
    use_random_erasing = True

    # Augmentación temporal
    use_temporal_augmentation = True
    temporal_max_jitter = 3
    toa_center_strength = 0.25

    # Modelo
    d_model = 128
    gru_hidden = 128
    gru_num_layers = 1
    dropout = 0.5
    bidirectional = True

    # MixUp temporal
    use_mixup = True
    mixup_alpha = 0.3
    mixup_prob = 0.5

    # Label smoothing (solo se aplica cuando NO hay MixUp en ese batch)
    label_smoothing = 0.1

    # Weighted sampler para desbalanceo
    use_weighted_sampler = True

    # Early stopping
    patience = 10
    min_delta = 0.001
    early_stop_counter = 0

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    frames_tag = "allf" if num_frames is None else f"{num_frames}f"
    run_name = (
        f"gru_v8_{timestamp}_{frames_tag}_seed{seed}"
        f"_freezeAll{int(freeze_all)}"
        f"_augTemp{int(use_temporal_augmentation)}"
        f"_aug{int(enable_augmentation)}"
        f"_cj{int(use_color_jitter)}"
        f"_flip{int(use_hflip)}"
        f"_rrc{int(use_random_resized_crop)}"
        f"_blur{int(use_gaussian_blur)}"
        f"_erase{int(use_random_erasing)}"
        f"_dm{d_model}"
        f"_gru{gru_num_layers}"
        f"_bi{int(bidirectional)}"
        f"_ls{int(label_smoothing * 10)}"
        f"_attnpool"
        f"_mixup{int(use_mixup)}a{int(mixup_alpha*10)}p{int(mixup_prob*10)}"
        f"_cosine_wu{warmup_epochs}"
        f"_wsamp{int(use_weighted_sampler)}"
    )
    print("Run name:", run_name)

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
            "model": "ResNet18 (pretrained, frozen) + Temporal GRU",
            "pooling": "attention",
            "d_model": d_model,
            "gru_num_layers": gru_num_layers,
            "bidirectional": bidirectional,
            "pretrained": pretrained,
            "freeze_early": freeze_early,
            "freeze_all": freeze_all,
            "dropout": dropout,
            "label_smoothing": label_smoothing,
            "patience": patience,
            "min_delta": min_delta,
            "stop_metric": "val_ap",
            "scheduler": "cosine_with_warmup",
            "warmup_epochs": warmup_epochs,
            "eta_min": eta_min,
            "use_mixup": use_mixup,
            "mixup_alpha": mixup_alpha,
            "mixup_prob": mixup_prob,
            "enable_augmentation": enable_augmentation,
            "use_hflip": use_hflip,
            "use_color_jitter": use_color_jitter,
            "use_random_resized_crop": use_random_resized_crop,
            "use_gaussian_blur": use_gaussian_blur,
            "use_random_erasing": use_random_erasing,
            "use_weighted_sampler": use_weighted_sampler,
            "temporal_max_jitter": temporal_max_jitter,
            "toa_center_strength": toa_center_strength,
        }
    )

    train_transform = build_clip_transform(
        train=True,
        image_size=(224, 224),
        enable_augmentation=enable_augmentation,
        use_hflip=use_hflip,
        use_color_jitter=use_color_jitter,
        use_random_resized_crop=use_random_resized_crop,
        use_gaussian_blur=use_gaussian_blur,
        use_random_erasing=use_random_erasing,
    )

    val_transform = build_clip_transform(
        train=False,
        image_size=(224, 224),
        enable_augmentation=False,
    )

    train_dataset = AccidentClipDataset(
        txt_path="/data-fast/data-server/vlopezmo/model/training/training_train.txt",
        rgb_root="/data-fast/data-server/vlopezmo/DADA2000",
        num_frames=num_frames,
        transform=train_transform,
        train=True,
        use_temporal_augmentation=use_temporal_augmentation,
        temporal_max_jitter=temporal_max_jitter,
        use_toa_guided_sampling=True,
        toa_center_strength=toa_center_strength,
    )

    val_dataset = AccidentClipDataset(
        txt_path="/data-fast/data-server/vlopezmo/model/training/training_val.txt",
        rgb_root="/data-fast/data-server/vlopezmo/DADA2000",
        num_frames=num_frames,
        transform=val_transform,
        train=False,
        use_temporal_augmentation=False,
    )

    print("Número de muestras de train:", len(train_dataset))
    print("Número de muestras de val:", len(val_dataset))

    # ── Diagnóstico de distribución de clases ─────────────────────────────────
    labels_train = [s["label"] for s in train_dataset.samples]
    labels_val = [s["label"] for s in val_dataset.samples]
    n_tr, n_va = len(labels_train), len(labels_val)
    pos_tr, pos_va = sum(labels_train), sum(labels_val)
    print(f"Train: {pos_tr} acc ({100*pos_tr/n_tr:.1f}%) | "
          f"{n_tr-pos_tr} no-acc ({100*(n_tr-pos_tr)/n_tr:.1f}%)")
    print(f"Val:   {pos_va} acc ({100*pos_va/n_va:.1f}%) | "
          f"{n_va-pos_va} no-acc ({100*(n_va-pos_va)/n_va:.1f}%)")

    # ── Weighted Sampler ──────────────────────────────────────────────────────
    train_sampler = None
    shuffle_train = True

    if use_weighted_sampler:
        class_counts = np.bincount(labels_train)
        class_weights_arr = 1.0 / class_counts.astype(np.float64)
        sample_weights = [class_weights_arr[l] for l in labels_train]
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        shuffle_train = False
        print(f"WeightedRandomSampler activado. Pesos por clase: {class_weights_arr}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        sampler=train_sampler,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
    )

    model = BaselineResNetGRU(
        num_classes=2,
        d_model=d_model,
        gru_hidden=gru_hidden,
        gru_layers=gru_num_layers,
        dropout=dropout,
        pretrained=pretrained,
        freeze_early=freeze_early,
        freeze_all=freeze_all,
        unfreeze_layer4=unfreeze_layer4,
        bidirectional=bidirectional,
    ).to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parámetros entrenables: {trainable_params:,} / {total_params:,} "
          f"({100*trainable_params/total_params:.1f}%)")

    # ── FIX: criterion_val DEFINIDO (antes faltaba y crasheaba en evaluate) ──
    criterion_val = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    scheduler = CosineAnnealingWithWarmup(
        optimizer,
        warmup_epochs=warmup_epochs,
        total_epochs=num_epochs,
        eta_min=eta_min,
    )

    ckpt_dir = "/data-fast/data-server/vlopezmo/model/checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)

    # ── FIX: una sola variable para tracking del mejor modelo y early stopping
    best_val_ap = -1.0

    for epoch in range(num_epochs):
        model.train()

        running_loss = 0.0
        all_train_labels = []
        all_train_preds = []
        all_train_scores = []

        for batch_idx, (clips, labels) in enumerate(train_loader):
            clips = clips.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # MixUp temporal
            apply_mixup = use_mixup and np.random.random() < mixup_prob
            if apply_mixup:
                lam = np.random.beta(mixup_alpha, mixup_alpha)
                lam = max(lam, 1.0 - lam)
                idx = torch.randperm(clips.size(0), device=device)

                clips_mixed = lam * clips + (1.0 - lam) * clips[idx]

                labels_one = nn.functional.one_hot(labels, 2).float()
                labels_mixed = lam * labels_one + (1.0 - lam) * labels_one[idx]
            else:
                clips_mixed = clips
                labels_mixed = labels

            optimizer.zero_grad()

            # FIX: desempaquetamos (logits, attn_weights); attn solo se usa en val
            outputs, _ = model(clips_mixed)

            # ── FIX: label smoothing SOLO cuando no hay MixUp ────────────────
            # Si aplicamos MixUp, las labels ya son soft — añadir label smoothing
            # encima suavizaría dos veces y diluiría la señal innecesariamente.
            if apply_mixup:
                loss = nn.functional.cross_entropy(outputs, labels_mixed)
            else:
                loss = nn.functional.cross_entropy(
                    outputs, labels, label_smoothing=label_smoothing
                )

            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            running_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            probs = torch.softmax(outputs, dim=1)[:, 1]

            all_train_labels.extend(labels.detach().cpu().numpy().tolist())
            all_train_preds.extend(preds.detach().cpu().numpy().tolist())
            all_train_scores.extend(probs.detach().cpu().numpy().tolist())

            if (batch_idx + 1) % 50 == 0:
                print(
                    f"Epoch [{epoch+1}/{num_epochs}] "
                    f"Batch [{batch_idx+1}/{len(train_loader)}] "
                    f"Loss: {loss.item():.4f}"
                )

        train_loss = running_loss / len(train_loader)
        train_acc = accuracy_score(all_train_labels, all_train_preds)
        train_f1 = f1_score(all_train_labels, all_train_preds, zero_division=0)
        train_ap = average_precision_score(all_train_labels, all_train_scores)
        train_auc = roc_auc_score(all_train_labels, all_train_scores)

        val_loss, val_acc, val_f1, val_ap, val_auc, mean_attn = evaluate(
            model, val_loader, criterion_val, device
        )

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        new_lr = optimizer.param_groups[0]["lr"]

        print(f"\nEpoch [{epoch+1}/{num_epochs}] completed  (lr: {current_lr:.2e} → {new_lr:.2e})")
        print(
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Train F1: {train_f1:.4f} | Train AP: {train_ap:.4f} | Train AUC: {train_auc:.4f} || "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
            f"Val F1: {val_f1:.4f} | Val AP: {val_ap:.4f} | Val AUC: {val_auc:.4f}"
        )

        # Log de la distribución media de atención temporal (qué frames mira el modelo)
        attn_table = wandb.Table(
            data=[[i, float(w)] for i, w in enumerate(mean_attn)],
            columns=["frame_idx", "attn_weight"],
        )

        wandb.log({
            "epoch": epoch + 1,
            "lr": current_lr,

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

            "val_attn_distribution": wandb.plot.bar(
                attn_table, "frame_idx", "attn_weight",
                title=f"Mean temporal attention (epoch {epoch+1})"
            ),
        })

        # ── FIX: tracking unificado de best_val_ap + early stopping ──────────
        improved = val_ap > best_val_ap + min_delta
        if improved:
            best_val_ap = val_ap
            early_stop_counter = 0
            print(f"Mejora en val_ap: {val_ap:.4f}")

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
                "train_loss": train_loss,
                "train_acc": train_acc,
                "train_f1": train_f1,
                "train_ap": train_ap,
                "train_auc": train_auc,
                "val_loss": val_loss,
                "num_frames": num_frames,
                "dropout": dropout,
                "label_smoothing": label_smoothing,
                "d_model": d_model,
                "gru_num_layers": gru_num_layers,
                "bidirectional": bidirectional,
                "pretrained": pretrained,
                "freeze_early": freeze_early,
                "freeze_all": freeze_all,
                "use_mixup": use_mixup,
                "mixup_alpha": mixup_alpha,
                "mixup_prob": mixup_prob,
                "enable_augmentation": enable_augmentation,
                "use_hflip": use_hflip,
                "use_color_jitter": use_color_jitter,
                "use_random_resized_crop": use_random_resized_crop,
                "use_gaussian_blur": use_gaussian_blur,
                "use_random_erasing": use_random_erasing,
                "use_weighted_sampler": use_weighted_sampler,
                "seed": seed,
            }, ckpt_path)
            print(f"Nuevo mejor modelo guardado en: {ckpt_path}")
        else:
            early_stop_counter += 1
            print(f"Sin mejora en val_ap. EarlyStopping counter: {early_stop_counter}/{patience}")

        # Guardar último checkpoint (siempre)
        last_ckpt_path = os.path.join(ckpt_dir, f"{run_name}_last.pt")
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_ap": val_ap,
            "val_acc": val_acc,
            "val_f1": val_f1,
            "val_auc": val_auc,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "train_f1": train_f1,
            "train_ap": train_ap,
            "train_auc": train_auc,
            "val_loss": val_loss,
            "num_frames": num_frames,
            "dropout": dropout,
            "label_smoothing": label_smoothing,
            "d_model": d_model,
            "gru_num_layers": gru_num_layers,
            "bidirectional": bidirectional,
            "pretrained": pretrained,
            "freeze_early": freeze_early,
            "freeze_all": freeze_all,
            "use_mixup": use_mixup,
            "mixup_alpha": mixup_alpha,
            "mixup_prob": mixup_prob,
            "enable_augmentation": enable_augmentation,
            "use_hflip": use_hflip,
            "use_color_jitter": use_color_jitter,
            "use_random_resized_crop": use_random_resized_crop,
            "use_gaussian_blur": use_gaussian_blur,
            "use_random_erasing": use_random_erasing,
            "use_weighted_sampler": use_weighted_sampler,
            "seed": seed,
        }, last_ckpt_path)

        if early_stop_counter >= patience:
            print(f"Early stopping activado en epoch {epoch+1}")
            break

    wandb.finish()


if __name__ == "__main__":
    main()