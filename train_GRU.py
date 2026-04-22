from torchvision.transforms import v2
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
import torch
import torch.nn as nn
import os
import wandb
import random
import numpy as np
import math
from datetime import datetime
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
)

from model.dataset import AccidentClipDataset
from model.mylibs.baseline_modelGRU import BaselineResNetGRU

def set_backbone_bn_eval(model):
    for m in model.backbone.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_labels_from_dataset(ds):
    if isinstance(ds, Subset):
        return [ds.dataset.samples[i]["label"] for i in ds.indices]
    return [s["label"] for s in ds.samples]


def get_samples_from_dataset(ds):
    if isinstance(ds, Subset):
        return [ds.dataset.samples[i] for i in ds.indices]
    return ds.samples


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()

    running_loss = 0.0
    all_labels = []
    all_preds = []
    all_scores = []
    all_attn = []

    for clips, labels in dataloader:
        clips = clips.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
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
    val_cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])

    mean_attn = np.concatenate(all_attn, axis=0).mean(axis=0)

    print(
        "Val probs:",
        f"min={np.min(all_scores):.4f}",
        f"max={np.max(all_scores):.4f}",
        f"mean={np.mean(all_scores):.4f}",
        f"std={np.std(all_scores):.4f}",
    )

    return val_loss, val_acc, val_f1, val_ap, val_auc, mean_attn, val_cm


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
                v2.ColorJitter(
                    brightness=0.10,
                    contrast=0.10,
                    saturation=0.10,
                    hue=0.02,
                )
            )

        if use_gaussian_blur:
            ops.append(
                v2.RandomApply(
                    [v2.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5))],
                    p=0.2,
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
            v2.RandomErasing(
                p=0.10,
                scale=(0.02, 0.10),
                ratio=(0.3, 3.3),
            )
        )

    return v2.Compose(ops)


class CosineAnnealingWithWarmup(torch.optim.lr_scheduler._LRScheduler):
    """
    Linear warmup + cosine decay hasta eta_min.
    Funciona con múltiples param_groups.
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

        progress = (self.last_epoch - self.warmup_epochs) / max(
            1, self.total_epochs - self.warmup_epochs
        )
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))

        return [
            self.eta_min + (base_lr - self.eta_min) * cosine_factor
            for base_lr in self.base_lrs
        ]


def main():
    seed = 42
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #torch.backends.cudnn.benchmark = True
    # FIX: deterministic=True para reproducibilidad estricta con seed fijo.
    # Con benchmark=True cuDNN elige algoritmos no deterministas según el workload.
    # Si priorizas velocidad sobre reproducibilidad, puedes quitar esta línea.
    #torch.backends.cudnn.deterministic = True
    print("CUDA available:", torch.cuda.is_available())

    # ── CONFIG PRINCIPAL ──────────────────────────────────────────────────────
    sanity_check = False

    batch_size = 32
    num_epochs = 30
    weight_decay = 1e-4
    # FIX: 32 frames captura mejor la dinámica temporal pre-accidente que 16.
    # Con backbone mayormente congelado el coste extra de la GRU es bajo.
    # Para revertir al ablation de 16f, comenta la línea siguiente y descomenta la de abajo.
    num_frames = 32
    # num_frames = 16  # ablation: menos contexto temporal, más rápido

    pretrained = True
    freeze_early = False

    # Fine-tuning parcial:
    # congelamos backbone completo y reactivamos solo layer4.
    freeze_all = True
    unfreeze_layer4 = True

    learning_rate_head = 1e-4
    learning_rate_layer4 = 3e-5

    warmup_epochs = 3
    eta_min = 1e-6

    # El .txt nuevo ya contiene ventanas pre-TOA:
    # label 1: distance 1-30
    # label 0: distance 61-120
    # distance 31-60 descartado
    anticipation_mode = False
    anticipation_offset = 1

    # Augmentación espacial moderada
    enable_augmentation = True
    use_hflip = True
    use_color_jitter = True
    use_random_resized_crop = False
    use_gaussian_blur = True
    use_random_erasing = False

    # Sin augmentación temporal para no alterar ventanas ya definidas
    use_temporal_augmentation = False
    temporal_max_jitter = 0
    use_toa_guided_sampling = False
    toa_center_strength = 0.0

    # Modelo
    d_model = 128
    gru_hidden = 128
    gru_num_layers = 1
    dropout = 0.3
    bidirectional = False

    # Regularización
    use_mixup = True
    mixup_alpha = 0.3
    # FIX: reducido de 0.3 a 0.15. Con clips de accidente, mezclas frecuentes entre
    # clases crean secuencias temporales fantasma que confunden la GRU.
    # Para ablation sin mixup pon use_mixup = False.
    mixup_prob = 0.15

    label_smoothing = 0.0
    use_weighted_sampler = False
    class_weights = None

    # Early stopping por AUC
    patience = 5
    min_delta = 0.001
    early_stop_counter = 0

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    frames_tag = "allf" if num_frames is None else f"{num_frames}f"

    run_name = (
        f"gru_v15_GAP_l4lastblock_{timestamp}_{frames_tag}_seed{seed}"
        f"_freezeAll{int(freeze_all)}_unfreezeL4{int(unfreeze_layer4)}"
        f"_lrHead{learning_rate_head:.0e}_lrL4{learning_rate_layer4:.0e}"
        f"_wd{weight_decay:.0e}"
        f"_aug{int(enable_augmentation)}"
        f"_dm{d_model}_gru{gru_num_layers}_bi{int(bidirectional)}"
        f"_do{int(dropout*100):02d}"
        f"_aucStop"
        f"_cosine_wu{warmup_epochs}"
    )

    print("Run name:", run_name)

    wandb.init(
        project="tfg-accident-prediction",
        name=run_name,
        config={
            "version": "v15_gap_full_layer4",
            "sanity_check": sanity_check,
            "seed": seed,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "weight_decay": weight_decay,
            "num_frames": num_frames,

            "task": "windowed_anticipation_gap",
            "positive_distance": "1-30",
            "discarded_distance": "31-60",
            "negative_distance": "61-120",

            "anticipation_mode": anticipation_mode,
            "anticipation_offset": anticipation_offset,

            "model": "ResNet18 layer4 last block finetune + Temporal GRU",
            "pooling": "attention",
            "d_model": d_model,
            "gru_hidden": gru_hidden,
            "gru_num_layers": gru_num_layers,
            "bidirectional": bidirectional,
            "dropout": dropout,

            "pretrained": pretrained,
            "freeze_all": freeze_all,
            "unfreeze_layer4": unfreeze_layer4,

            "lr_head": learning_rate_head,
            "lr_layer4": learning_rate_layer4,

            "label_smoothing": label_smoothing,
            "class_weights": class_weights,
            "use_weighted_sampler": use_weighted_sampler,

            "patience": patience,
            "min_delta": min_delta,
            "stop_metric": "val_auc",
            "checkpoint_metric": "val_auc",

            "scheduler": "cosine_with_warmup",
            "warmup_epochs": warmup_epochs,
            "eta_min": eta_min,

            "use_mixup": use_mixup,
            "enable_augmentation": enable_augmentation,
            "use_hflip": use_hflip,
            "use_color_jitter": use_color_jitter,
            "use_random_resized_crop": use_random_resized_crop,
            "use_gaussian_blur": use_gaussian_blur,
            "use_random_erasing": use_random_erasing,

            "use_temporal_augmentation": use_temporal_augmentation,
            "temporal_max_jitter": temporal_max_jitter,
            "use_toa_guided_sampling": use_toa_guided_sampling,
            "toa_center_strength": toa_center_strength,

            "start_only_baseline_auc": 0.7136,
            "start_only_baseline_ap": 0.6527,
        },
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
        use_toa_guided_sampling=use_toa_guided_sampling,
        toa_center_strength=toa_center_strength,
        anticipation_mode=anticipation_mode,
        anticipation_offset=anticipation_offset,
        drop_invalid_samples=True,
    )

    val_dataset = AccidentClipDataset(
        txt_path="/data-fast/data-server/vlopezmo/model/training/training_val.txt",
        rgb_root="/data-fast/data-server/vlopezmo/DADA2000",
        num_frames=num_frames,
        transform=val_transform,
        train=False,
        use_temporal_augmentation=False,
        temporal_max_jitter=0,
        use_toa_guided_sampling=False,
        toa_center_strength=0.0,
        anticipation_mode=anticipation_mode,
        anticipation_offset=anticipation_offset,
        drop_invalid_samples=True,
    )

    print("Número de muestras de train:", len(train_dataset))
    print("Número de muestras de val:", len(val_dataset))

    labels_train = get_labels_from_dataset(train_dataset)
    labels_val = get_labels_from_dataset(val_dataset)

    n_tr = len(labels_train)
    n_va = len(labels_val)
    pos_tr = sum(labels_train)
    pos_va = sum(labels_val)

    print(
        f"Train: {pos_tr} acc ({100 * pos_tr / n_tr:.1f}%) | "
        f"{n_tr - pos_tr} no-acc ({100 * (n_tr - pos_tr) / n_tr:.1f}%)"
    )
    print(
        f"Val:   {pos_va} acc ({100 * pos_va / n_va:.1f}%) | "
        f"{n_va - pos_va} no-acc ({100 * (n_va - pos_va) / n_va:.1f}%)"
    )

    train_samples_for_stats = get_samples_from_dataset(train_dataset)

    eff_lens_1 = [
        s["effective_end"] - s["start"] + 1
        for s in train_samples_for_stats
        if s["label"] == 1
    ]

    eff_lens_0 = [
        s["effective_end"] - s["start"] + 1
        for s in train_samples_for_stats
        if s["label"] == 0
    ]

    if eff_lens_1:
        print(
            f"Longitud efectiva label=1 (TRAIN): "
            f"min={min(eff_lens_1)}, "
            f"median={int(np.median(eff_lens_1))}, "
            f"max={max(eff_lens_1)}"
        )

    if eff_lens_0:
        print(
            f"Longitud efectiva label=0 (TRAIN): "
            f"min={min(eff_lens_0)}, "
            f"median={int(np.median(eff_lens_0))}, "
            f"max={max(eff_lens_0)}"
        )

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
    else:
        print("WeightedRandomSampler desactivado. Se usa shuffle=True.")

    num_workers = 6

    # FIX: pin_memory solo activo si hay CUDA — evita warning inofensivo en CPU.
    use_pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        persistent_workers=True,
        prefetch_factor=4,
        # FIX: drop_last=True evita que el último batch mini (< batch_size) produzca
        # una permutación mixup intra-clase desequilibrada o incluso de 1 solo elemento.
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        persistent_workers=True,
        prefetch_factor=4,
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

    model.train()
    set_backbone_bn_eval(model)

    bn_train = sum(
        1 for m in model.backbone.modules()
        if isinstance(m, nn.BatchNorm2d) and m.training
    )
    bn_total = sum(
        1 for m in model.backbone.modules()
        if isinstance(m, nn.BatchNorm2d)
    )

    print(f"[BN check] backbone BN en train: {bn_train}/{bn_total}  (esperado 0/{bn_total})")

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    print(
        f"Parámetros entrenables: {trainable_params:,} / {total_params:,} "
        f"({100 * trainable_params / total_params:.1f}%)"
    )

    class_counts = np.bincount(labels_train)
    cw = torch.tensor(
        [len(labels_train) / (2.0 * c) for c in class_counts],
        dtype=torch.float32,
        device=device,
    )
    print(f"Class weights para el loss: {cw.tolist()}")

    criterion_train = nn.CrossEntropyLoss(weight=cw)
    criterion_val = nn.CrossEntropyLoss()

    trainable_named_params = [
        (name, param) for name, param in model.named_parameters()
        if param.requires_grad
    ]

    print("Parámetros entrenables:")
    for name, param in trainable_named_params:
        print(f"  {name}: {param.numel():,}")

    head_params    = [p for n, p in trainable_named_params if "backbone" not in n]
    backbone_params = [p for n, p in trainable_named_params if "backbone" in n]

    param_groups = [{"params": head_params, "lr": learning_rate_head}]
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": learning_rate_layer4})

    optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)

    scheduler = CosineAnnealingWithWarmup(
        optimizer,
        warmup_epochs=warmup_epochs,
        total_epochs=num_epochs,
        eta_min=eta_min,
    )

    ckpt_dir = "/data-fast/data-server/vlopezmo/model/checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)

    best_val_auc = -1.0
    # FIX: guardamos también el mejor checkpoint por AP. Para anticipación de accidentes
    # AP pondera más los verdaderos positivos con pocas falsas alarmas, lo que es más
    # relevante operacionalmente que el AUC (ranking puro).
    best_val_ap = -1.0
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    for epoch in range(num_epochs):
        model.train()
        set_backbone_bn_eval(model)

        running_loss = 0.0
        all_train_labels = []
        all_train_preds = []
        all_train_scores = []

        for batch_idx, (clips, labels) in enumerate(train_loader):
            clips = clips.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            apply_mixup = use_mixup and np.random.random() < mixup_prob
 
            if apply_mixup:
                lam = float(np.random.beta(mixup_alpha, mixup_alpha))

                # FIX: mixup intra-clase — permutamos solo dentro de muestras del
                # mismo label. Mezclar positivo con negativo produce secuencias
                # temporales incoherentes que degradan la señal de la GRU.
                # Si alguna clase del batch tiene solo 1 muestra, degeneramos a
                # inter-clase como fallback (comportamiento anterior).
                idx = torch.zeros(clips.size(0), dtype=torch.long, device=device)
                for cls in labels.unique():
                    mask = (labels == cls).nonzero(as_tuple=True)[0]
                    if len(mask) > 1:
                        perm = mask[torch.randperm(len(mask), device=device)]
                        idx[mask] = perm
                    else:
                        # fallback: mapea a sí mismo (lam * x + (1-lam) * x = x)
                        idx[mask] = mask

                clips_mixed = lam * clips + (1.0 - lam) * clips[idx]
                labels_a = labels
                labels_b = labels[idx]
            else:
                clips_mixed = clips
                labels_a = labels
                labels_b = None
                lam = 1.0

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                outputs, _ = model(clips_mixed)

                if apply_mixup:
                    loss = (
                        lam * criterion_train(outputs, labels_a)
                        + (1.0 - lam) * criterion_train(outputs, labels_b)
                    )
                else:
                    loss = criterion_train(outputs, labels_a)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()

            if not apply_mixup:
                with torch.no_grad():
                    preds = torch.argmax(outputs, dim=1)
                    probs = torch.softmax(outputs, dim=1)[:, 1]
            
                    all_train_labels.extend(labels.detach().cpu().numpy().tolist())
                    all_train_preds.extend(preds.detach().cpu().numpy().tolist())
                    all_train_scores.extend(probs.detach().cpu().numpy().tolist())

            if (batch_idx + 1) % 50 == 0:
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}] "
                    f"Batch [{batch_idx + 1}/{len(train_loader)}] "
                    f"Loss: {loss.item():.4f}"
                )

        train_loss = running_loss / len(train_loader)
        if len(all_train_labels) > 0 and len(set(all_train_labels)) > 1:
            train_acc = accuracy_score(all_train_labels, all_train_preds)
            train_f1  = f1_score(all_train_labels, all_train_preds, zero_division=0)
            train_ap  = average_precision_score(all_train_labels, all_train_scores)
            train_auc = roc_auc_score(all_train_labels, all_train_scores)
            print(
                "Train probs:",
                f"min={np.min(all_train_scores):.4f}",
                f"max={np.max(all_train_scores):.4f}",
                f"mean={np.mean(all_train_scores):.4f}",
                f"std={np.std(all_train_scores):.4f}",
            )
        else:
            train_acc = train_f1 = train_ap = train_auc = float("nan")



        val_loss, val_acc, val_f1, val_ap, val_auc, mean_attn, val_cm = evaluate(
            model,
            val_loader,
            criterion_val,
            device,
        )

        current_lr_head = optimizer.param_groups[0]["lr"]
        current_lr_backbone = optimizer.param_groups[1]["lr"] if len(optimizer.param_groups) > 1 else 0.0

        scheduler.step()

        new_lr_head = optimizer.param_groups[0]["lr"]
        new_lr_backbone = optimizer.param_groups[1]["lr"] if len(optimizer.param_groups) > 1 else 0.0

        print(
            f"\nEpoch [{epoch + 1}/{num_epochs}] completed "
            f"(lr_head: {current_lr_head:.2e} → {new_lr_head:.2e}, "
            f"lr_backbone: {current_lr_backbone:.2e} → {new_lr_backbone:.2e})"
        )

        print(
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Train F1: {train_f1:.4f} | Train AP: {train_ap:.4f} | Train AUC: {train_auc:.4f} || "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
            f"Val F1: {val_f1:.4f} | Val AP: {val_ap:.4f} | Val AUC: {val_auc:.4f}"
        )

        print("Val confusion matrix [[TN, FP], [FN, TP]]:")
        print(val_cm)

        attn_table = wandb.Table(
            data=[[i, float(w)] for i, w in enumerate(mean_attn)],
            columns=["frame_idx", "attn_weight"],
        )

        wandb.log({
            "epoch": epoch + 1,
            "lr/head": current_lr_head,
            "lr/backbone": current_lr_backbone,

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

            "val_cm_tn": int(val_cm[0, 0]),
            "val_cm_fp": int(val_cm[0, 1]),
            "val_cm_fn": int(val_cm[1, 0]),
            "val_cm_tp": int(val_cm[1, 1]),

            "val_attn_distribution": wandb.plot.bar(
                attn_table,
                "frame_idx",
                "attn_weight",
                title=f"Mean temporal attention (epoch {epoch + 1})",
            ),
        }, step=epoch + 1)

        improved = val_auc > best_val_auc + min_delta

        if improved:
            best_val_auc = val_auc
            early_stop_counter = 0

            print(f"Mejora en val_auc: {val_auc:.4f}")

            ckpt_path = os.path.join(ckpt_dir, f"{run_name}_best_val_auc.pt")

            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),

                "val_auc": val_auc,
                "val_ap": val_ap,
                "val_acc": val_acc,
                "val_f1": val_f1,
                "val_loss": val_loss,
                "val_cm": val_cm,

                "train_auc": train_auc,
                "train_ap": train_ap,
                "train_acc": train_acc,
                "train_f1": train_f1,
                "train_loss": train_loss,

                "sanity_check": sanity_check,
                "anticipation_mode": anticipation_mode,
                "anticipation_offset": anticipation_offset,
                "num_frames": num_frames,
                "dropout": dropout,
                "d_model": d_model,
                "gru_hidden": gru_hidden,
                "gru_num_layers": gru_num_layers,
                "bidirectional": bidirectional,
                "freeze_all": freeze_all,
                "unfreeze_layer4": unfreeze_layer4,
                "label_smoothing": label_smoothing,
                "use_weighted_sampler": use_weighted_sampler,
                "seed": seed,
            }, ckpt_path)

            print(f"Nuevo mejor modelo guardado en: {ckpt_path}")

        else:
            early_stop_counter += 1
            print(
                f"Sin mejora en val_auc. "
                f"EarlyStopping counter: {early_stop_counter}/{patience}"
            )

        # FIX: checkpoint adicional por val_ap — más relevante para anticipación de
        # accidentes (premia alta precision con pocas falsas alarmas).
        # Early stopping sigue guiado por AUC; este checkpoint es solo para comparación.
        if val_ap > best_val_ap + min_delta:
            best_val_ap = val_ap
            print(f"Mejora en val_ap: {val_ap:.4f}")

            ckpt_ap_path = os.path.join(ckpt_dir, f"{run_name}_best_val_ap.pt")

            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),

                "val_auc": val_auc,
                "val_ap": val_ap,
                "val_acc": val_acc,
                "val_f1": val_f1,
                "val_loss": val_loss,
                "val_cm": val_cm,

                "train_auc": train_auc,
                "train_ap": train_ap,
                "train_acc": train_acc,
                "train_f1": train_f1,
                "train_loss": train_loss,

                "sanity_check": sanity_check,
                "anticipation_mode": anticipation_mode,
                "anticipation_offset": anticipation_offset,
                "num_frames": num_frames,
                "dropout": dropout,
                "d_model": d_model,
                "gru_hidden": gru_hidden,
                "gru_num_layers": gru_num_layers,
                "bidirectional": bidirectional,
                "freeze_all": freeze_all,
                "unfreeze_layer4": unfreeze_layer4,
                "label_smoothing": label_smoothing,
                "use_weighted_sampler": use_weighted_sampler,
                "seed": seed,
                "checkpoint_metric": "val_ap",
            }, ckpt_ap_path)

            print(f"Mejor modelo por AP guardado en: {ckpt_ap_path}")

        last_ckpt_path = os.path.join(ckpt_dir, f"{run_name}_last.pt")

        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),

            "val_auc": val_auc,
            "val_ap": val_ap,
            "val_acc": val_acc,
            "val_f1": val_f1,
            "val_loss": val_loss,
            "val_cm": val_cm,

            "train_auc": train_auc,
            "train_ap": train_ap,
            "train_acc": train_acc,
            "train_f1": train_f1,
            "train_loss": train_loss,

            "sanity_check": sanity_check,
            "anticipation_mode": anticipation_mode,
            "anticipation_offset": anticipation_offset,
            "seed": seed,
        }, last_ckpt_path)

        if early_stop_counter >= patience:
            print(f"Early stopping activado en epoch {epoch + 1}")
            break

    wandb.finish()


if __name__ == "__main__":
    main()