from pathlib import Path
import os
import random
import math
import json
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
from torchvision.transforms import v2
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    fbeta_score,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

try:
    import wandb
except Exception:
    wandb = None

from model.dataset import AccidentClipDataset, audit_dataset
from model.mylibs.baseline_modelGRU import BaselineResNetGRU


CONFIG = {
    # Paths
    "train_txt": "/data-fast/data-server/vlopezmo/model/training/training_train.txt",
    "val_txt": "/data-fast/data-server/vlopezmo/model/training/training_val.txt",
    "rgb_root": "/data-fast/data-server/vlopezmo/DADA2000",
    "output_dir": "checkpoints_anticipation_gru",

    # Task: anticipación real
    "task": "accident_anticipation_causal",
    "num_frames": 16,
    "image_size": [224, 224],
    "anticipation_mode": True,
    "anticipation_offset": 10,  # ajustar según FPS: 10 frames ≈ 1 s si FPS=10
    "strict_anticipation": True,
    "frame_index_offset": 1,

    # Data augmentation segura para conducción
    "enable_augmentation": True,
    "use_hflip": False,
    "use_mixup": False,
    "use_color_jitter": True,
    "use_random_resized_crop": True,
    "use_gaussian_blur": True,
    "use_random_erasing": False,
    "use_temporal_augmentation": True,
    "temporal_max_jitter": 1,
    "use_toa_guided_sampling": False,

    # Model
    "num_classes": 2,
    "d_model": 128,
    "gru_hidden": 128,
    "gru_layers": 1,
    "dropout": 0.30,
    "pretrained": True,
    "freeze_early": False,
    "freeze_all": True,
    "unfreeze_layer4": True,
    "unfreeze_mode": "full_layer4",
    "bidirectional": False,
    "pooling": "attention",

    # Training
    "seed": 42,
    "batch_size": 48,
    "num_workers": 6,
    "epochs": 40,
    "warmup_epochs": 3,
    "learning_rate_head": 1e-3,
    "learning_rate_backbone": 1e-5,
    "weight_decay": 1e-4,
    "grad_clip_norm": 5.0,
    "use_weighted_sampler": False,
    "drop_last": True,
    "early_stopping_patience": 10,
    "checkpoint_metric": "val_ap",

    # Logging
    "wandb_enabled": False,
    "wandb_project": "accident-anticipation-gru",
    "run_name": None,
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_samples_from_dataset(ds):
    if isinstance(ds, Subset):
        return [ds.dataset.samples[i] for i in ds.indices]
    return ds.samples


def get_labels_from_dataset(ds):
    return [s["label"] for s in get_samples_from_dataset(ds)]


def build_clip_transform(
    train: bool,
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
        ops.append(v2.RandomResizedCrop(size=image_size, scale=(0.90, 1.0), ratio=(0.95, 1.05), antialias=True))
    else:
        ops.append(v2.Resize(image_size, antialias=True))

    if train and enable_augmentation:
        if use_hflip:
            raise ValueError("use_hflip=True desactivado: no es seguro semánticamente para conducción.")
        if use_color_jitter:
            ops.append(v2.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.10, hue=0.02))
        if use_gaussian_blur:
            ops.append(v2.RandomApply([v2.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5))], p=0.2))

    ops.extend([
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    if train and enable_augmentation and use_random_erasing:
        ops.append(v2.RandomErasing(p=0.10, scale=(0.02, 0.10), ratio=(0.3, 3.3)))

    return v2.Compose(ops)


class CosineAnnealingWithWarmup(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs, total_epochs, eta_min=1e-6, last_epoch=-1):
        self.warmup_epochs = max(1, warmup_epochs)
        self.total_epochs = max(self.warmup_epochs + 1, total_epochs)
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            alpha = (self.last_epoch + 1) / self.warmup_epochs
            return [base_lr * alpha for base_lr in self.base_lrs]

        progress = (self.last_epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return [self.eta_min + (base_lr - self.eta_min) * cosine for base_lr in self.base_lrs]


def compute_threshold_metrics(labels, scores):
    labels = np.asarray(labels)
    scores = np.asarray(scores)

    out = {
        "best_f2": float("nan"),
        "best_f2_threshold": 0.5,
        "recall_at_fpr_10": float("nan"),
        "precision_at_recall_90": float("nan"),
    }

    if len(np.unique(labels)) < 2:
        return out

    precision, recall, pr_thresholds = precision_recall_curve(labels, scores)
    best_f2, best_thr = -1.0, 0.5
    for thr in pr_thresholds:
        preds = (scores >= thr).astype(int)
        f2 = fbeta_score(labels, preds, beta=2, zero_division=0)
        if f2 > best_f2:
            best_f2, best_thr = f2, float(thr)

    fpr, tpr, _ = roc_curve(labels, scores)
    valid = np.where(fpr <= 0.10)[0]
    recall_at_fpr_10 = float(np.max(tpr[valid])) if len(valid) else float("nan")

    valid_recall = np.where(recall >= 0.90)[0]
    precision_at_recall_90 = float(np.max(precision[valid_recall])) if len(valid_recall) else float("nan")

    out.update({
        "best_f2": float(best_f2),
        "best_f2_threshold": best_thr,
        "recall_at_fpr_10": recall_at_fpr_10,
        "precision_at_recall_90": precision_at_recall_90,
    })
    return out


@torch.no_grad()
def evaluate(model, dataloader, criterion, device, split_name="val"):
    model.eval()
    total_loss = 0.0
    all_labels, all_preds, all_scores, all_attn = [], [], [], []

    for clips, labels in dataloader:
        clips = clips.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            logits, attn = model(clips)
            loss = criterion(logits, labels)

        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = torch.argmax(logits, dim=1)

        total_loss += loss.item()
        all_labels.extend(labels.cpu().numpy().tolist())
        all_preds.extend(preds.cpu().numpy().tolist())
        all_scores.extend(probs.cpu().numpy().tolist())
        all_attn.append(attn.cpu().numpy())

    labels_arr = np.asarray(all_labels)
    preds_arr = np.asarray(all_preds)
    scores_arr = np.asarray(all_scores)

    metrics = {
        "loss": total_loss / max(1, len(dataloader)),
        "acc": accuracy_score(labels_arr, preds_arr),
        "f1": f1_score(labels_arr, preds_arr, zero_division=0),
        "cm": confusion_matrix(labels_arr, preds_arr, labels=[0, 1]),
        "labels": labels_arr,
        "preds": preds_arr,
        "scores": scores_arr,
    }

    if len(np.unique(labels_arr)) > 1:
        metrics["ap"] = average_precision_score(labels_arr, scores_arr)
        metrics["auc"] = roc_auc_score(labels_arr, scores_arr)
    else:
        metrics["ap"] = float("nan")
        metrics["auc"] = float("nan")

    metrics.update(compute_threshold_metrics(labels_arr, scores_arr))

    if all_attn:
        attn_np = np.concatenate(all_attn, axis=0)
        mean_attn = attn_np.mean(axis=0)
        metrics["attn_peak"] = int(mean_attn.argmax())
        metrics["attn_entropy"] = float(-np.sum(mean_attn * np.log(mean_attn + 1e-8)))
        metrics["mean_attn"] = mean_attn

    print(
        f"[{split_name}] loss={metrics['loss']:.4f} acc={metrics['acc']:.4f} "
        f"f1={metrics['f1']:.4f} ap={metrics['ap']:.4f} auc={metrics['auc']:.4f} "
        f"best_f2={metrics['best_f2']:.4f} thr={metrics['best_f2_threshold']:.3f} "
        f"recall@FPR10={metrics['recall_at_fpr_10']:.4f}"
    )
    print(f"[{split_name}] CM [[TN FP] [FN TP]]:\n{metrics['cm']}")

    return metrics


def build_optimizer(model, cfg):
    backbone_params = [p for n, p in model.named_parameters() if n.startswith("backbone.") and p.requires_grad]
    head_params = [p for n, p in model.named_parameters() if not n.startswith("backbone.") and p.requires_grad]

    param_groups = []
    if head_params:
        param_groups.append({"params": head_params, "lr": cfg["learning_rate_head"]})
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": cfg["learning_rate_backbone"]})

    return torch.optim.AdamW(param_groups, weight_decay=cfg["weight_decay"])


def save_checkpoint(path, model, optimizer, scheduler, epoch, cfg, metrics, threshold):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "config": cfg,
        "metrics": {k: v for k, v in metrics.items() if k not in {"labels", "preds", "scores", "mean_attn"}},
        "decision_threshold": threshold,
    }, path)


def main():
    cfg = CONFIG.copy()
    set_seed(cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    if cfg["wandb_enabled"]:
        if wandb is None:
            raise RuntimeError("wandb_enabled=True pero wandb no está instalado.")
        run_name = cfg["run_name"] or f"anticipation_gru_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        wandb.init(project=cfg["wandb_project"], name=run_name, config=cfg)

    train_transform = build_clip_transform(
        train=True,
        image_size=tuple(cfg["image_size"]),
        enable_augmentation=cfg["enable_augmentation"],
        use_hflip=cfg["use_hflip"],
        use_color_jitter=cfg["use_color_jitter"],
        use_random_resized_crop=cfg["use_random_resized_crop"],
        use_gaussian_blur=cfg["use_gaussian_blur"],
        use_random_erasing=cfg["use_random_erasing"],
    )
    val_transform = build_clip_transform(train=False, image_size=tuple(cfg["image_size"]))

    train_dataset = AccidentClipDataset(
        txt_path=cfg["train_txt"],
        rgb_root=cfg["rgb_root"],
        num_frames=cfg["num_frames"],
        transform=train_transform,
        train=True,
        use_temporal_augmentation=cfg["use_temporal_augmentation"],
        temporal_max_jitter=cfg["temporal_max_jitter"],
        use_toa_guided_sampling=cfg["use_toa_guided_sampling"],
        anticipation_mode=cfg["anticipation_mode"],
        anticipation_offset=cfg["anticipation_offset"],
        strict_anticipation=cfg["strict_anticipation"],
        frame_index_offset=cfg["frame_index_offset"],
    )
    val_dataset = AccidentClipDataset(
        txt_path=cfg["val_txt"],
        rgb_root=cfg["rgb_root"],
        num_frames=cfg["num_frames"],
        transform=val_transform,
        train=False,
        use_temporal_augmentation=False,
        use_toa_guided_sampling=False,
        anticipation_mode=cfg["anticipation_mode"],
        anticipation_offset=cfg["anticipation_offset"],
        strict_anticipation=cfg["strict_anticipation"],
        frame_index_offset=cfg["frame_index_offset"],
    )

    audit_dataset(train_dataset, val_dataset, cfg["num_frames"])

    labels_train = np.asarray(get_labels_from_dataset(train_dataset))
    class_counts = np.bincount(labels_train, minlength=2)
    if np.any(class_counts == 0):
        raise RuntimeError(f"Train sin ambas clases: class_counts={class_counts.tolist()}")

    class_weights = torch.tensor(
        [len(labels_train) / (2.0 * c) for c in class_counts],
        dtype=torch.float32,
        device=device,
    )
    print(f"Class counts: {class_counts.tolist()} | weights: {class_weights.detach().cpu().tolist()}")

    generator = torch.Generator()
    generator.manual_seed(cfg["seed"])

    sampler = None
    shuffle = True
    if cfg["use_weighted_sampler"]:
        sample_weights = np.array([class_weights[int(y)].detach().cpu().item() for y in labels_train])
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        shuffle = False

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        shuffle=shuffle,
        sampler=sampler,
        num_workers=cfg["num_workers"],
        pin_memory=pin_memory,
        drop_last=cfg["drop_last"],
        worker_init_fn=seed_worker,
        generator=generator,
        persistent_workers=cfg["num_workers"] > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=pin_memory,
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=generator,
        persistent_workers=cfg["num_workers"] > 0,
    )

    model = BaselineResNetGRU(
        num_classes=cfg["num_classes"],
        d_model=cfg["d_model"],
        gru_hidden=cfg["gru_hidden"],
        gru_layers=cfg["gru_layers"],
        dropout=cfg["dropout"],
        pretrained=cfg["pretrained"],
        freeze_early=cfg["freeze_early"],
        freeze_all=cfg["freeze_all"],
        unfreeze_layer4=cfg["unfreeze_layer4"],
        unfreeze_mode=cfg["unfreeze_mode"],
        bidirectional=cfg["bidirectional"],
        pooling=cfg["pooling"],
    ).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Parámetros entrenables: {trainable:,}/{total:,}")

    criterion_train = nn.CrossEntropyLoss(weight=class_weights)
    criterion_eval = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, cfg)
    scheduler = CosineAnnealingWithWarmup(
        optimizer,
        warmup_epochs=cfg["warmup_epochs"],
        total_epochs=cfg["epochs"],
        eta_min=1e-6,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    best_metric = -float("inf")
    best_epoch = -1
    epochs_without_improvement = 0

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        running_loss = 0.0
        train_labels, train_preds, train_scores = [], [], []

        for clips, labels in train_loader:
            clips = clips.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits, _ = model(clips)
                loss = criterion_train(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip_norm"])
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            probs = torch.softmax(logits.detach(), dim=1)[:, 1]
            preds = torch.argmax(logits.detach(), dim=1)
            train_labels.extend(labels.cpu().numpy().tolist())
            train_preds.extend(preds.cpu().numpy().tolist())
            train_scores.extend(probs.cpu().numpy().tolist())

        scheduler.step()

        train_loss = running_loss / max(1, len(train_loader))
        train_acc = accuracy_score(train_labels, train_preds)
        train_f1 = f1_score(train_labels, train_preds, zero_division=0)
        train_ap = average_precision_score(train_labels, train_scores) if len(np.unique(train_labels)) > 1 else float("nan")

        val_metrics = evaluate(model, val_loader, criterion_eval, device, split_name="val")
        current_metric = val_metrics[cfg["checkpoint_metric"].replace("val_", "")]

        lrs = [group["lr"] for group in optimizer.param_groups]
        print(
            f"Epoch {epoch:03d}/{cfg['epochs']} | train_loss={train_loss:.4f} "
            f"train_acc={train_acc:.4f} train_f1={train_f1:.4f} train_ap={train_ap:.4f} | "
            f"lr={lrs}"
        )

        if cfg["wandb_enabled"]:
            wandb.log({
                "epoch": epoch,
                "train/loss": train_loss,
                "train/acc": train_acc,
                "train/f1": train_f1,
                "train/ap": train_ap,
                "val/loss": val_metrics["loss"],
                "val/acc": val_metrics["acc"],
                "val/f1": val_metrics["f1"],
                "val/ap": val_metrics["ap"],
                "val/auc": val_metrics["auc"],
                "val/best_f2": val_metrics["best_f2"],
                "val/recall_at_fpr_10": val_metrics["recall_at_fpr_10"],
                "lr/head": lrs[0] if len(lrs) > 0 else 0.0,
                "lr/backbone": lrs[1] if len(lrs) > 1 else 0.0,
            })

        improved = not np.isnan(current_metric) and current_metric > best_metric
        if improved:
            best_metric = float(current_metric)
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(
                output_dir / "best_model.pt",
                model,
                optimizer,
                scheduler,
                epoch,
                cfg,
                val_metrics,
                threshold=val_metrics["best_f2_threshold"],
            )
            print(f"[BEST] epoch={epoch} {cfg['checkpoint_metric']}={best_metric:.4f}")
        else:
            epochs_without_improvement += 1

        save_checkpoint(
            output_dir / "last_model.pt",
            model,
            optimizer,
            scheduler,
            epoch,
            cfg,
            val_metrics,
            threshold=val_metrics["best_f2_threshold"],
        )

        if epochs_without_improvement >= cfg["early_stopping_patience"]:
            print(f"Early stopping en epoch {epoch}. Mejor epoch={best_epoch}, metric={best_metric:.4f}")
            break

    print(f"Entrenamiento terminado. Mejor epoch={best_epoch}, best_{cfg['checkpoint_metric']}={best_metric:.4f}")
    if cfg["wandb_enabled"]:
        wandb.finish()


if __name__ == "__main__":
    main()
