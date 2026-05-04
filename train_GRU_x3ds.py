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
    precision_score,
    recall_score,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
)

from model.dataset import AccidentClipDataset
from model.mylibs.x3d_gru import X3DGRU


def set_backbone_bn_eval(model):
    for m in model.backbone.modules():
        if isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d, nn.BatchNorm1d)):
            m.eval()


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_labels_from_dataset(ds):
    if isinstance(ds, Subset):
        return [ds.dataset.samples[i]["label"] for i in ds.indices]
    return [s["label"] for s in ds.samples]


def get_samples_from_dataset(ds):
    if isinstance(ds, Subset):
        return [ds.dataset.samples[i] for i in ds.indices]
    return ds.samples


def find_best_threshold(y_true, y_prob, metric="f1"):
    """
    Busca el mejor threshold usando P(label=1) en validación.

    metric="f1": maximiza F1.
    metric="precision_at_recall": maximiza precision con recall >= 0.80.
    metric="recall_at_precision": maximiza recall con precision >= 0.60.
    """
    thresholds = np.linspace(0.01, 0.99, 99)

    best = {
        "threshold": 0.5,
        "score": -1.0,
        "acc": 0.0,
        "f1": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "cm": None,
    }

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)

        if metric == "f1":
            score = f1
        elif metric == "precision_at_recall":
            score = precision if recall >= 0.80 else -1.0
        elif metric == "recall_at_precision":
            score = recall if precision >= 0.60 else -1.0
        else:
            raise ValueError(f"metric no soportada: {metric}")

        if score > best["score"]:
            best = {
                "threshold": float(t),
                "score": float(score),
                "acc": float(acc),
                "f1": float(f1),
                "precision": float(precision),
                "recall": float(recall),
                "cm": confusion_matrix(y_true, y_pred, labels=[0, 1]),
            }

    return best


@torch.no_grad()
def evaluate(model, dataloader, criterion, device, split_name="val"):
    model.eval()

    running_loss = 0.0
    all_labels = []
    all_preds = []
    all_scores = []
    all_attn = []

    for batch in dataloader:
        if len(batch) == 3:
            clips, labels, aux = batch
        else:
            clips, labels = batch
            aux = None
        clips = clips.to(device)
        labels = labels.to(device)

        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            model_out = model(clips)
            if len(model_out) == 3:
                outputs, attn_weights, _ = model_out
            else:
                outputs, attn_weights = model_out
            loss = criterion(outputs, labels)

        running_loss += loss.item()

        preds = torch.argmax(outputs, dim=1)
        probs = torch.softmax(outputs, dim=1)[:, 1]

        all_labels.extend(labels.cpu().numpy().tolist())
        all_preds.extend(preds.cpu().numpy().tolist())
        all_scores.extend(probs.cpu().numpy().tolist())
        all_attn.append(attn_weights.cpu().numpy())

    eval_loss = running_loss / len(dataloader)

    labels_arr = np.array(all_labels)
    preds_arr = np.array(all_preds)
    scores_arr = np.array(all_scores)

    eval_acc = accuracy_score(labels_arr, preds_arr)
    eval_f1 = f1_score(labels_arr, preds_arr, zero_division=0)
    eval_precision = precision_score(labels_arr, preds_arr, zero_division=0)
    eval_recall = recall_score(labels_arr, preds_arr, zero_division=0)

    best_thr = find_best_threshold(labels_arr, scores_arr, metric="f1")

    unique_labels = np.unique(labels_arr)
    if len(unique_labels) > 1:
        eval_ap = average_precision_score(labels_arr, scores_arr)
        eval_auc = roc_auc_score(labels_arr, scores_arr)
    else:
        eval_ap = float("nan")
        eval_auc = float("nan")
        print(f"[{split_name}] WARNING: solo hay una clase presente. AP/AUC = nan")

    eval_cm = confusion_matrix(labels_arr, preds_arr, labels=[0, 1])

    all_attn_np = np.concatenate(all_attn, axis=0)  # [N, T]
    mean_attn = all_attn_np.mean(axis=0)

    attn_max = float(mean_attn.max())
    attn_std = float(mean_attn.std())
    attn_peak = int(mean_attn.argmax())
    attn_entropy = float(-np.sum(mean_attn * np.log(mean_attn + 1e-8)))

    print(
        f"[{split_name} Attention] peak_subclip={attn_peak} | "
        f"max={attn_max:.4f} | std={attn_std:.4f} | entropy={attn_entropy:.4f} "
        f"(uniform={np.log(len(mean_attn)):.4f})"
    )

    print(
        f"{split_name.capitalize()} probs:",
        f"min={np.min(scores_arr):.4f}",
        f"max={np.max(scores_arr):.4f}",
        f"mean={np.mean(scores_arr):.4f}",
        f"std={np.std(scores_arr):.4f}",
    )

    n_pos = int(labels_arr.sum())
    n_neg = int(len(labels_arr) - n_pos)
    mean_score_pos = float(scores_arr[labels_arr == 1].mean()) if n_pos > 0 else float("nan")
    mean_score_neg = float(scores_arr[labels_arr == 0].mean()) if n_neg > 0 else float("nan")
    print(
        f"[{split_name} scores] mean_pos={mean_score_pos:.4f} | "
        f"mean_neg={mean_score_neg:.4f} | separation={mean_score_pos - mean_score_neg:.4f}"
    )

    print(
        f"[{split_name} threshold@0.5] "
        f"acc={eval_acc:.4f} | f1={eval_f1:.4f} | "
        f"precision={eval_precision:.4f} | recall={eval_recall:.4f}"
    )
    print(
        f"[{split_name} best_threshold_f1] "
        f"thr={best_thr['threshold']:.3f} | "
        f"acc={best_thr['acc']:.4f} | f1={best_thr['f1']:.4f} | "
        f"precision={best_thr['precision']:.4f} | recall={best_thr['recall']:.4f}"
    )
    print(f"[{split_name} best_threshold_f1] Confusion matrix [[TN, FP], [FN, TP]]:")
    print(best_thr["cm"])

    attn_entropy_pos = float("nan")
    attn_entropy_neg = float("nan")
    attn_peak_pos = -1
    attn_peak_neg = -1
    mean_attn_pos = None
    mean_attn_neg = None

    if np.any(labels_arr == 1):
        mean_attn_pos = all_attn_np[labels_arr == 1].mean(axis=0)
        attn_entropy_pos = float(-np.sum(mean_attn_pos * np.log(mean_attn_pos + 1e-8)))
        attn_peak_pos = int(mean_attn_pos.argmax())
        print(
            f"[{split_name} Attention POS] peak_subclip={attn_peak_pos} | "
            f"entropy={attn_entropy_pos:.4f}"
        )

    if np.any(labels_arr == 0):
        mean_attn_neg = all_attn_np[labels_arr == 0].mean(axis=0)
        attn_entropy_neg = float(-np.sum(mean_attn_neg * np.log(mean_attn_neg + 1e-8)))
        attn_peak_neg = int(mean_attn_neg.argmax())
        print(
            f"[{split_name} Attention NEG] peak_subclip={attn_peak_neg} | "
            f"entropy={attn_entropy_neg:.4f}"
        )

    return {
        "loss": eval_loss,
        "acc": eval_acc,
        "f1": eval_f1,
        "precision": eval_precision,
        "recall": eval_recall,
        "ap": eval_ap,
        "auc": eval_auc,
        "cm": eval_cm,
        "best_threshold": best_thr["threshold"],
        "best_threshold_acc": best_thr["acc"],
        "best_threshold_f1": best_thr["f1"],
        "best_threshold_precision": best_thr["precision"],
        "best_threshold_recall": best_thr["recall"],
        "best_threshold_cm": best_thr["cm"],
        "mean_attn": mean_attn,
        "attn_entropy": attn_entropy,
        "attn_entropy_pos": attn_entropy_pos,
        "attn_entropy_neg": attn_entropy_neg,
        "attn_peak": attn_peak,
        "attn_peak_pos": attn_peak_pos,
        "attn_peak_neg": attn_peak_neg,
        "mean_score_pos": mean_score_pos,
        "mean_score_neg": mean_score_neg,
        "score_separation": mean_score_pos - mean_score_neg,
        "labels": labels_arr,
        "preds": preds_arr,
        "scores": scores_arr,
    }


def build_clip_transform(
    train=False,
    image_size=(160, 160),
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
            mean=[0.45, 0.45, 0.45],
            std=[0.225, 0.225, 0.225],
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
def safe_aux_ce(criterion, logits, target):
    valid = target != -100
    if valid.any():
        return criterion(logits, target)
    return logits.sum() * 0.0

def main():
    seed = 44 #42,43,44
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("CUDA available:", torch.cuda.is_available())

    sanity_check = False

    batch_size = 6
    num_epochs = 30
    num_frames = 32
    image_size = (160, 160)

    pretrained = True
    x3d_model_name = "x3d_s"
    subclip_len = 13
    subclip_stride = 3

    freeze_backbone = True
    unfreeze_last_n_blocks = 1

    learning_rate_head = 1e-4
    learning_rate_backbone = 1e-6 #1e-5

    warmup_epochs = 3
    eta_min = 1e-6

    anticipation_mode = False
    anticipation_offset = 1

    enable_augmentation = True
    use_hflip = True
    use_color_jitter = True
    use_random_resized_crop = False
    use_gaussian_blur = True
    use_random_erasing = False

    use_temporal_augmentation = False
    temporal_max_jitter = 0
    use_toa_guided_sampling = False
    toa_center_strength = 0.0

    d_model = 128
    gru_hidden = 128
    gru_num_layers = 1
    dropout = 0.5
    bidirectional = True

    use_mixup = False
    mixup_alpha = 0.2
    mixup_prob = 0.3

    label_smoothing = 0.0
    use_weighted_sampler = False
    class_weights = None

    weight_decay = 1e-4 #5e-4
    run_val_sanity_check = True

    patience = 5
    min_delta = 0.001
    early_stop_counter = 0

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    frames_tag = "allf" if num_frames is None else f"{num_frames}f"

    run_name = (
        f"x3ds_gru_v1_{timestamp}_{frames_tag}_seed{seed}"
        f"_freezeBackbone{int(freeze_backbone)}_unfreezeLast{unfreeze_last_n_blocks}"
        f"_lrHead{learning_rate_head:.0e}_lrBackbone{learning_rate_backbone:.0e}"
        f"_wd{weight_decay:.0e}"
        f"_aug{int(enable_augmentation)}"
        f"_dm{d_model}_gru{gru_num_layers}_bi{int(bidirectional)}"
        f"_do{int(dropout*100):02d}"
        f"_aucStop"
        f"_cosine_wu{warmup_epochs}"
        f"_mix{int(use_mixup)}"
    )

    use_aux_annotations = True
    annotations_xlsx = "/data-fast/data-server/vlopezmo/DADA2000/dada_text_annotations.xlsx"

    lambda_type = 0.03
    lambda_weather = 0.0
    lambda_light = 0.0
    lambda_scene = 0.0
    lambda_linear = 0.0

    print("Run name:", run_name)

    wandb.init(
        project="TFG",
        name=run_name,
        config={
            "version": f"_aux{int(use_aux_annotations)}",
            "sanity_check": sanity_check,
            "seed": seed,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "weight_decay": weight_decay,
            "num_frames": num_frames,
            "task": "pre_toa_window_classification",
            "positive_distance": "1-30",
            "discarded_distance": "31-60",
            "negative_distance": "61-120",
            "anticipation_mode": anticipation_mode,
            "anticipation_offset": anticipation_offset,
            "model": "X3D-S backbone + Temporal GRU",
            "x3d_model_name": x3d_model_name,
            "subclip_len": subclip_len,
            "subclip_stride": subclip_stride,
            "image_size": image_size,
            "pooling": "attention",
            "d_model": d_model,
            "gru_hidden": gru_hidden,
            "gru_num_layers": gru_num_layers,
            "bidirectional": bidirectional,
            "dropout": dropout,
            "pretrained": pretrained,
            "freeze_backbone": freeze_backbone,
            "unfreeze_last_n_blocks": unfreeze_last_n_blocks,
            "lr_head": learning_rate_head,
            "lr_backbone": learning_rate_backbone,
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
            "mixup_alpha": mixup_alpha,
            "mixup_prob": mixup_prob,
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
            "threshold_tuning": "best_f1_on_validation",
            "start_only_baseline_auc": 0.7136,
            "start_only_baseline_ap": 0.6527,
            "use_aux_annotations": use_aux_annotations,
            "annotations_xlsx": annotations_xlsx,
            "lambda_type": lambda_type,
            "lambda_weather": lambda_weather,
            "lambda_light": lambda_light,
            "lambda_scene": lambda_scene,
            "lambda_linear": lambda_linear,
            "aux_strategy": "multitask_visual_heads",
        },
    )

    train_transform = build_clip_transform(
        train=True,
        image_size=(160, 160),
        enable_augmentation=enable_augmentation,
        use_hflip=use_hflip,
        use_color_jitter=use_color_jitter,
        use_random_resized_crop=use_random_resized_crop,
        use_gaussian_blur=use_gaussian_blur,
        use_random_erasing=use_random_erasing,
    )

    val_transform = build_clip_transform(
        train=False,
        image_size=(160, 160),
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
        annotations_xlsx=annotations_xlsx,
        use_aux_annotations=use_aux_annotations,
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
        annotations_xlsx=annotations_xlsx,
        use_aux_annotations=use_aux_annotations,
    )

    def count_aux_coverage(ds):
        count = 0
        for s in ds.samples:
            video_key = ds._normalize_video_id(s["video_id"])
            if video_key in ds.aux_by_video:
                count += 1
        return count

    if use_aux_annotations:
        train_aux_count = count_aux_coverage(train_dataset)
        val_aux_count = count_aux_coverage(val_dataset)

        print(f"Train samples con aux annotations: {train_aux_count}/{len(train_dataset)}")
        print(f"Val samples con aux annotations: {val_aux_count}/{len(val_dataset)}")

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

    num_workers = 8
    use_pin_memory = True
    use_persistent_workers = num_workers > 0

    loader_kwargs = dict(
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        persistent_workers=use_persistent_workers,
    )
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2
        loader_kwargs["worker_init_fn"] = seed_worker

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        sampler=train_sampler,
        drop_last=True,
        generator=g,
        **loader_kwargs,
    )

    train_eval_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        generator=g,
        **loader_kwargs,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        generator=g,
        **loader_kwargs,
    )

    model = X3DGRU(
        num_classes=2,
        model_name=x3d_model_name,
        pretrained=pretrained,
        subclip_len=subclip_len,
        subclip_stride=subclip_stride,
        d_model=d_model,
        gru_hidden=gru_hidden,
        gru_layers=gru_num_layers,
        dropout=dropout,
        bidirectional=bidirectional,
        freeze_backbone=freeze_backbone,
        unfreeze_last_n_blocks=unfreeze_last_n_blocks,
        use_aux_heads=use_aux_annotations,
        num_types=62,
        num_weather=4,
        num_light=2,
        num_scenes=5,
        num_linear=5,
    ).to(device)

    model.train()
    set_backbone_bn_eval(model)

    bn_train = sum(
        1 for m in model.backbone.modules()
        if isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d, nn.BatchNorm1d)) and m.training
    )
    bn_total = sum(
        1 for m in model.backbone.modules()
        if isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d, nn.BatchNorm1d))
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

    criterion_train = nn.CrossEntropyLoss(
        weight=cw,
        label_smoothing=label_smoothing,
    )
    criterion_eval = nn.CrossEntropyLoss()

    criterion_aux = nn.CrossEntropyLoss(ignore_index=-100)

    trainable_named_params = [
        (name, param) for name, param in model.named_parameters()
        if param.requires_grad
    ]

    print("Parámetros entrenables:")
    for name, param in trainable_named_params:
        print(f"  {name}: {param.numel():,}")

    head_params = [p for n, p in trainable_named_params if "backbone" not in n]
    backbone_params = [p for n, p in trainable_named_params if "backbone" in n]

    param_groups = [{"params": head_params, "lr": learning_rate_head}]
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": learning_rate_backbone})

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
    best_val_ap = -1.0
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    if run_val_sanity_check:
        print("\n[Sanity check] Evaluando modelo sin entrenar sobre val set...")
        sc = evaluate(model, val_loader, criterion_eval, device, split_name="val_sanity")
        print(
            f"[Sanity check] loss={sc['loss']:.4f} | acc={sc['acc']:.4f} | "
            f"f1={sc['f1']:.4f} | ap={sc['ap']:.4f} | auc={sc['auc']:.4f}"
        )
        print(f"[Sanity check] Confusion matrix @0.5:\n{sc['cm']}")
        print(
            f"[Sanity check] Best F1 threshold={sc['best_threshold']:.3f} | "
            f"f1={sc['best_threshold_f1']:.4f} | "
            f"precision={sc['best_threshold_precision']:.4f} | "
            f"recall={sc['best_threshold_recall']:.4f}"
        )
        print(f"[Sanity check] Confusion matrix best threshold:\n{sc['best_threshold_cm']}")
        if not np.isnan(sc["auc"]) and sc["auc"] > 0.60:
            print(
                f"[Sanity check] ADVERTENCIA: val_auc={sc['auc']:.4f} > 0.60 antes de entrenar. "
                "Posible data leak o problema en el split."
            )
        else:
            print("[Sanity check] OK — val_auc cerca de azar, split parece correcto.")
        print()

    for epoch in range(num_epochs):
        model.train()
        set_backbone_bn_eval(model)

        running_loss = 0.0
        num_mixup_batches = 0
        num_total_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            if len(batch) == 3:
                clips, labels, aux = batch
            else:
                clips, labels = batch
                aux = None
            
            clips = clips.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if aux is not None:
                aux = {k: v.to(device, non_blocking=True) for k, v in aux.items()}

            apply_mixup = use_mixup and np.random.random() < mixup_prob
            num_total_batches += 1

            if apply_mixup:
                num_mixup_batches += 1
                lam = float(np.random.beta(mixup_alpha, mixup_alpha))

                idx = torch.randperm(clips.size(0), device=device)

                clips_input = lam * clips + (1.0 - lam) * clips[idx]
                labels_a = labels
                labels_b = labels[idx]
            else:
                clips_input = clips
                labels_a = labels
                labels_b = None
                lam = 1.0

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                model_out = model(clips_input)

                if len(model_out) == 3:
                    outputs, _, aux_logits = model_out
                else:
                    outputs, _ = model_out
                    aux_logits = None

                if apply_mixup:
                    loss_acc = (
                        lam * criterion_train(outputs, labels_a)
                        + (1.0 - lam) * criterion_train(outputs, labels_b)
                    )
                else:
                    loss_acc = criterion_train(outputs, labels_a)

                loss = loss_acc

                # Loss auxiliar: NO la uso con mixup porque las etiquetas auxiliares no se mezclan bien.
                if (
                    use_aux_annotations
                    and aux is not None
                    and aux_logits is not None
                    and not apply_mixup
                ):
                    loss_aux_type = safe_aux_ce(criterion_aux, aux_logits["type"], aux["type"])
                    loss_aux_weather = safe_aux_ce(criterion_aux, aux_logits["weather"], aux["weather"])
                    loss_aux_light = safe_aux_ce(criterion_aux, aux_logits["light"], aux["light"])
                    loss_aux_scene = safe_aux_ce(criterion_aux, aux_logits["scene"], aux["scene"])
                    loss_aux_linear = safe_aux_ce(criterion_aux, aux_logits["linear"], aux["linear"])

                    loss_aux = (
                        lambda_type * loss_aux_type
                        + lambda_weather * loss_aux_weather
                        + lambda_light * loss_aux_light
                        + lambda_scene * loss_aux_scene
                        + lambda_linear * loss_aux_linear
                    )

                    loss = loss + loss_aux

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

            if (batch_idx + 1) % 50 == 0:
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}] "
                    f"Batch [{batch_idx + 1}/{len(train_loader)}] "
                    f"Loss: {loss.item():.4f} | Mixup: {apply_mixup}"
                )

        train_loss = running_loss / len(train_loader)

        train_metrics = evaluate(
            model,
            train_eval_loader,
            criterion_eval,
            device,
            split_name="train_eval",
        )

        val_metrics = evaluate(
            model,
            val_loader,
            criterion_eval,
            device,
            split_name="val",
        )

        current_lr_head = optimizer.param_groups[0]["lr"]
        current_lr_backbone = optimizer.param_groups[1]["lr"] if len(optimizer.param_groups) > 1 else 0.0

        scheduler.step()

        new_lr_head = optimizer.param_groups[0]["lr"]
        new_lr_backbone = optimizer.param_groups[1]["lr"] if len(optimizer.param_groups) > 1 else 0.0

        mixup_ratio = num_mixup_batches / max(1, num_total_batches)

        print(
            f"\nEpoch [{epoch + 1}/{num_epochs}] completed "
            f"(lr_head: {current_lr_head:.2e} → {new_lr_head:.2e}, "
            f"lr_backbone: {current_lr_backbone:.2e} → {new_lr_backbone:.2e})"
        )

        print(
            f"Train Loss(opt): {train_loss:.4f} | "
            f"Train Eval Acc@0.5: {train_metrics['acc']:.4f} | "
            f"Train Eval F1@0.5: {train_metrics['f1']:.4f} | "
            f"Train Eval AP: {train_metrics['ap']:.4f} | "
            f"Train Eval AUC: {train_metrics['auc']:.4f} || "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Acc@0.5: {val_metrics['acc']:.4f} | "
            f"Val F1@0.5: {val_metrics['f1']:.4f} | "
            f"Val AP: {val_metrics['ap']:.4f} | "
            f"Val AUC: {val_metrics['auc']:.4f}"
        )

        print(
            f"Val BestThrF1: thr={val_metrics['best_threshold']:.3f} | "
            f"acc={val_metrics['best_threshold_acc']:.4f} | "
            f"f1={val_metrics['best_threshold_f1']:.4f} | "
            f"precision={val_metrics['best_threshold_precision']:.4f} | "
            f"recall={val_metrics['best_threshold_recall']:.4f}"
        )

        print("Val confusion matrix @0.5 [[TN, FP], [FN, TP]]:")
        print(val_metrics["cm"])
        print("Val confusion matrix best F1 threshold [[TN, FP], [FN, TP]]:")
        print(val_metrics["best_threshold_cm"])

        attn_table = wandb.Table(
            data=[[i, float(w)] for i, w in enumerate(val_metrics["mean_attn"])],
            columns=["subclip_idx", "attn_weight"],
        )

        wandb.log({
            "epoch": epoch + 1,
            "lr/head": current_lr_head,
            "lr/backbone": current_lr_backbone,
            "train_loss_optim": train_loss,
            "train_mixup_ratio": mixup_ratio,

            "train_eval_loss": train_metrics["loss"],
            "train_eval_acc@0.5": train_metrics["acc"],
            "train_eval_f1@0.5": train_metrics["f1"],
            "train_eval_precision@0.5": train_metrics["precision"],
            "train_eval_recall@0.5": train_metrics["recall"],
            "train_eval_ap": train_metrics["ap"],
            "train_eval_roc_auc": train_metrics["auc"],
            "train_eval_score_separation": train_metrics["score_separation"],
            "train_eval_best_threshold": train_metrics["best_threshold"],
            "train_eval_best_f1": train_metrics["best_threshold_f1"],
            "train_eval_best_precision": train_metrics["best_threshold_precision"],
            "train_eval_best_recall": train_metrics["best_threshold_recall"],
            "train_eval_best_acc": train_metrics["best_threshold_acc"],

            "val_loss": val_metrics["loss"],
            "val_acc@0.5": val_metrics["acc"],
            "val_f1@0.5": val_metrics["f1"],
            "val_precision@0.5": val_metrics["precision"],
            "val_recall@0.5": val_metrics["recall"],
            "val_ap": val_metrics["ap"],
            "val_roc_auc": val_metrics["auc"],
            "val_score_separation": val_metrics["score_separation"],
            "val_best_threshold": val_metrics["best_threshold"],
            "val_best_f1": val_metrics["best_threshold_f1"],
            "val_best_precision": val_metrics["best_threshold_precision"],
            "val_best_recall": val_metrics["best_threshold_recall"],
            "val_best_acc": val_metrics["best_threshold_acc"],

            "val_cm@0.5_tn": int(val_metrics["cm"][0, 0]),
            "val_cm@0.5_fp": int(val_metrics["cm"][0, 1]),
            "val_cm@0.5_fn": int(val_metrics["cm"][1, 0]),
            "val_cm@0.5_tp": int(val_metrics["cm"][1, 1]),

            "val_best_cm_tn": int(val_metrics["best_threshold_cm"][0, 0]),
            "val_best_cm_fp": int(val_metrics["best_threshold_cm"][0, 1]),
            "val_best_cm_fn": int(val_metrics["best_threshold_cm"][1, 0]),
            "val_best_cm_tp": int(val_metrics["best_threshold_cm"][1, 1]),

            "val_attn_entropy": val_metrics["attn_entropy"],
            "val_attn_entropy_pos": val_metrics["attn_entropy_pos"],
            "val_attn_entropy_neg": val_metrics["attn_entropy_neg"],
            "val_attn_peak": val_metrics["attn_peak"],
            "val_attn_peak_pos": val_metrics["attn_peak_pos"],
            "val_attn_peak_neg": val_metrics["attn_peak_neg"],

            "val_attn_distribution": wandb.plot.bar(
                attn_table,
                "subclip_idx",
                "attn_weight",
                title=f"Mean subclip attention (epoch {epoch + 1})",
            ),
        }, step=epoch + 1)

        improved = (
            not np.isnan(val_metrics["auc"])
            and val_metrics["auc"] > best_val_auc + min_delta
        )

        if improved:
            best_val_auc = val_metrics["auc"]
            early_stop_counter = 0

            print(f"Mejora en val_auc: {val_metrics['auc']:.4f}")

            ckpt_path = os.path.join(ckpt_dir, f"{run_name}_best_val_auc.pt")

            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),

                "val_auc": val_metrics["auc"],
                "val_ap": val_metrics["ap"],
                "val_acc@0.5": val_metrics["acc"],
                "val_f1@0.5": val_metrics["f1"],
                "val_precision@0.5": val_metrics["precision"],
                "val_recall@0.5": val_metrics["recall"],
                "val_loss": val_metrics["loss"],
                "val_cm@0.5": val_metrics["cm"],
                "val_best_threshold": val_metrics["best_threshold"],
                "val_best_acc": val_metrics["best_threshold_acc"],
                "val_best_f1": val_metrics["best_threshold_f1"],
                "val_best_precision": val_metrics["best_threshold_precision"],
                "val_best_recall": val_metrics["best_threshold_recall"],
                "val_best_cm": val_metrics["best_threshold_cm"],

                "train_auc": train_metrics["auc"],
                "train_ap": train_metrics["ap"],
                "train_acc@0.5": train_metrics["acc"],
                "train_f1@0.5": train_metrics["f1"],
                "train_loss_eval": train_metrics["loss"],
                "train_loss_optim": train_loss,

                "sanity_check": sanity_check,
                "anticipation_mode": anticipation_mode,
                "anticipation_offset": anticipation_offset,
                "num_frames": num_frames,
                "dropout": dropout,
                "d_model": d_model,
                "gru_hidden": gru_hidden,
                "gru_num_layers": gru_num_layers,
                "bidirectional": bidirectional,
                "x3d_model_name": x3d_model_name,
                "subclip_len": subclip_len,
                "subclip_stride": subclip_stride,
                "image_size": image_size,
                "freeze_backbone": freeze_backbone,
                "unfreeze_last_n_blocks": unfreeze_last_n_blocks,
                "label_smoothing": label_smoothing,
                "use_weighted_sampler": use_weighted_sampler,
                "use_mixup": use_mixup,
                "mixup_alpha": mixup_alpha,
                "mixup_prob": mixup_prob,
                "seed": seed,
                "use_aux_annotations": use_aux_annotations,
                "use_aux_heads": use_aux_annotations,
                "num_types": 52,
                "num_weather": 4,
                "num_light": 2,
                "num_scenes": 5,
                "num_linear": 5,
                "lambda_type": lambda_type,
                "lambda_weather": lambda_weather,
                "lambda_light": lambda_light,
                "lambda_scene": lambda_scene,
                "lambda_linear": lambda_linear,
            }, ckpt_path)

            print(f"Nuevo mejor modelo guardado en: {ckpt_path}")

        else:
            early_stop_counter += 1
            print(
                f"Sin mejora en val_auc. "
                f"EarlyStopping counter: {early_stop_counter}/{patience}"
            )

        if (
            not np.isnan(val_metrics["ap"])
            and val_metrics["ap"] > best_val_ap + min_delta
        ):
            best_val_ap = val_metrics["ap"]
            print(f"Mejora en val_ap: {val_metrics['ap']:.4f}")

            ckpt_ap_path = os.path.join(ckpt_dir, f"{run_name}_best_val_ap.pt")

            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),

                "val_auc": val_metrics["auc"],
                "val_ap": val_metrics["ap"],
                "val_acc@0.5": val_metrics["acc"],
                "val_f1@0.5": val_metrics["f1"],
                "val_precision@0.5": val_metrics["precision"],
                "val_recall@0.5": val_metrics["recall"],
                "val_loss": val_metrics["loss"],
                "val_cm@0.5": val_metrics["cm"],
                "val_best_threshold": val_metrics["best_threshold"],
                "val_best_acc": val_metrics["best_threshold_acc"],
                "val_best_f1": val_metrics["best_threshold_f1"],
                "val_best_precision": val_metrics["best_threshold_precision"],
                "val_best_recall": val_metrics["best_threshold_recall"],
                "val_best_cm": val_metrics["best_threshold_cm"],

                "train_auc": train_metrics["auc"],
                "train_ap": train_metrics["ap"],
                "train_acc@0.5": train_metrics["acc"],
                "train_f1@0.5": train_metrics["f1"],
                "train_loss_eval": train_metrics["loss"],
                "train_loss_optim": train_loss,

                "sanity_check": sanity_check,
                "anticipation_mode": anticipation_mode,
                "anticipation_offset": anticipation_offset,
                "num_frames": num_frames,
                "dropout": dropout,
                "d_model": d_model,
                "gru_hidden": gru_hidden,
                "gru_num_layers": gru_num_layers,
                "bidirectional": bidirectional,
                "x3d_model_name": x3d_model_name,
                "subclip_len": subclip_len,
                "subclip_stride": subclip_stride,
                "image_size": image_size,
                "freeze_backbone": freeze_backbone,
                "unfreeze_last_n_blocks": unfreeze_last_n_blocks,
                "label_smoothing": label_smoothing,
                "use_weighted_sampler": use_weighted_sampler,
                "use_mixup": use_mixup,
                "mixup_alpha": mixup_alpha,
                "mixup_prob": mixup_prob,
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

            "val_auc": val_metrics["auc"],
            "val_ap": val_metrics["ap"],
            "val_acc@0.5": val_metrics["acc"],
            "val_f1@0.5": val_metrics["f1"],
            "val_loss": val_metrics["loss"],
            "val_cm@0.5": val_metrics["cm"],
            "val_best_threshold": val_metrics["best_threshold"],
            "val_best_f1": val_metrics["best_threshold_f1"],
            "val_best_precision": val_metrics["best_threshold_precision"],
            "val_best_recall": val_metrics["best_threshold_recall"],
            "val_best_cm": val_metrics["best_threshold_cm"],

            "train_auc": train_metrics["auc"],
            "train_ap": train_metrics["ap"],
            "train_acc@0.5": train_metrics["acc"],
            "train_f1@0.5": train_metrics["f1"],
            "train_loss_eval": train_metrics["loss"],
            "train_loss_optim": train_loss,

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
