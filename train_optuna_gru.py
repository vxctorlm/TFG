from pathlib import Path
from datetime import datetime
import os
import random
import json

import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from sklearn.metrics import accuracy_score, f1_score, average_precision_score, roc_auc_score

from model.dataset import AccidentClipDataset
from model.mylibs.baseline_modelGRU import BaselineResNetGRUSimple



PROJECT_NAME = "tfg-accident-prediction-optuna-phase1"

TRAIN_TXT = "/data-fast/data-server/vlopezmo/model/training/training_train.txt"<
VAL_TXT = "/data-fast/data-server/vlopezmo/model/training/training_val.txt"
RGB_ROOT = "/data-fast/data-server/vlopezmo/DADA2000"

CKPT_DIR = "/data-fast/data-server/vlopezmo/model/checkpoints_optuna_phase1"
STUDY_DB = "sqlite:///optuna_phase1.db"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_SEED = 42

# Fijos en fase 1
BATCH_SIZE = 4
NUM_EPOCHS = 25
PATIENCE = 5
MIN_DELTA = 0.0015

# augmentación visual fija en fase 1
ENABLE_AUGMENTATION = True
USE_HFLIP = True
USE_COLOR_JITTER = True
USE_RANDOM_RESIZED_CROP = True
USE_GAUSSIAN_BLUR = False

# fijos de tarea
USE_TEMPORAL_AUGMENTATION = True
ANTICIPATION_MODE = True
ANTICIPATION_OFFSET = 5
D_MODEL = 256
BIDIRECTIONAL = False
NUM_CLASSES = 2

# workers
NUM_WORKERS = 2
PIN_MEMORY = True


# =========================================================
# UTILIDADES
# =========================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


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
                v2.ColorJitter(
                    brightness=0.1,
                    contrast=0.1,
                    saturation=0.1,
                    hue=0.02,
                )
            )

        if use_gaussian_blur:
            ops.append(
                v2.RandomApply(
                    [v2.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5))],
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

    return v2.Compose(ops)


def compute_class_weights_from_dataset(dataset: AccidentClipDataset) -> torch.Tensor:
    labels = [sample["label"] for sample in dataset.samples]
    num_neg = sum(1 for x in labels if x == 0)
    num_pos = sum(1 for x in labels if x == 1)

    if num_neg == 0 or num_pos == 0:
        return torch.tensor([1.0, 1.0], dtype=torch.float32)

    total = num_neg + num_pos
    weights = torch.tensor(
        [
            total / (2.0 * num_neg),
            total / (2.0 * num_pos),
        ],
        dtype=torch.float32,
    )
    return weights


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


def build_dataloaders(
    num_frames,
    temporal_max_jitter,
    use_toa_guided_sampling,
    toa_center_strength,
    seed,
):
    train_transform = build_clip_transform(
        train=True,
        image_size=(224, 224),
        enable_augmentation=ENABLE_AUGMENTATION,
        use_hflip=USE_HFLIP,
        use_color_jitter=USE_COLOR_JITTER,
        use_random_resized_crop=USE_RANDOM_RESIZED_CROP,
        use_gaussian_blur=USE_GAUSSIAN_BLUR,
    )

    val_transform = build_clip_transform(
        train=False,
        image_size=(224, 224),
        enable_augmentation=False,
    )

    train_dataset = AccidentClipDataset(
        txt_path=TRAIN_TXT,
        rgb_root=RGB_ROOT,
        num_frames=num_frames,
        transform=train_transform,
        train=True,
        use_temporal_augmentation=USE_TEMPORAL_AUGMENTATION,
        temporal_max_jitter=temporal_max_jitter,
        use_toa_guided_sampling=use_toa_guided_sampling,
        toa_center_strength=toa_center_strength,
        anticipation_mode=ANTICIPATION_MODE,
        anticipation_offset=ANTICIPATION_OFFSET,
    )

    val_dataset = AccidentClipDataset(
        txt_path=VAL_TXT,
        rgb_root=RGB_ROOT,
        num_frames=num_frames,
        transform=val_transform,
        train=False,
        use_temporal_augmentation=False,
        use_toa_guided_sampling=False,
        toa_center_strength=0.0,
        anticipation_mode=ANTICIPATION_MODE,
        anticipation_offset=ANTICIPATION_OFFSET,
    )

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=(NUM_WORKERS > 0),
        worker_init_fn=seed_worker,
        generator=g,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=(NUM_WORKERS > 0),
        worker_init_fn=seed_worker,
    )

    return train_dataset, val_dataset, train_loader, val_loader


def suggest_hparams(trial: optuna.Trial):
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-4, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
    dropout = trial.suggest_float("dropout", 0.25, 0.50)
    num_frames = trial.suggest_categorical("num_frames", [8, 12, 16])
    hidden_size = trial.suggest_categorical("hidden_size", [64, 128, 256])
    num_gru_layers = trial.suggest_int("num_gru_layers", 1, 2)
    temporal_max_jitter = trial.suggest_int("temporal_max_jitter", 0, 3)

    use_toa_guided_sampling = trial.suggest_categorical(
        "use_toa_guided_sampling", [True, False]
    )

    if use_toa_guided_sampling:
        toa_center_strength = trial.suggest_float("toa_center_strength", 0.10, 0.50)
    else:
        toa_center_strength = 0.0

    return {
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "dropout": dropout,
        "num_frames": num_frames,
        "hidden_size": hidden_size,
        "num_gru_layers": num_gru_layers,
        "temporal_max_jitter": temporal_max_jitter,
        "use_toa_guided_sampling": use_toa_guided_sampling,
        "toa_center_strength": toa_center_strength,
    }


def objective(trial: optuna.Trial):
    trial_seed = BASE_SEED + trial.number
    set_seed(trial_seed)

    hparams = suggest_hparams(trial)

    train_dataset, val_dataset, train_loader, val_loader = build_dataloaders(
        num_frames=hparams["num_frames"],
        temporal_max_jitter=hparams["temporal_max_jitter"],
        use_toa_guided_sampling=hparams["use_toa_guided_sampling"],
        toa_center_strength=hparams["toa_center_strength"],
        seed=trial_seed,
    )

    class_weights = compute_class_weights_from_dataset(train_dataset).to(DEVICE)

    model = BaselineResNetGRUSimple(
        num_classes=NUM_CLASSES,
        d_model=D_MODEL,
        hidden_size=hparams["hidden_size"],
        num_gru_layers=hparams["num_gru_layers"],
        bidirectional=BIDIRECTIONAL,
        dropout=hparams["dropout"],
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=hparams["learning_rate"],
        weight_decay=hparams["weight_decay"],
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
    )

    best_val_ap = -1.0
    best_epoch = -1
    best_metrics = {}
    best_model_state = None

    best_val_ap_for_stop = -1.0
    early_stop_counter = 0

    for epoch in range(NUM_EPOCHS):
        model.train()

        running_loss = 0.0
        all_train_labels = []
        all_train_preds = []
        all_train_scores = []

        for clips, labels in train_loader:
            clips = clips.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

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

        train_loss = running_loss / len(train_loader)
        train_acc = accuracy_score(all_train_labels, all_train_preds)
        train_f1 = f1_score(all_train_labels, all_train_preds, zero_division=0)

        try:
            train_ap = average_precision_score(all_train_labels, all_train_scores)
        except ValueError:
            train_ap = float("nan")

        try:
            train_auc = roc_auc_score(all_train_labels, all_train_scores)
        except ValueError:
            train_auc = float("nan")

        val_loss, val_acc, val_f1, val_ap, val_auc = evaluate(
            model, val_loader, criterion, DEVICE
        )

        scheduler.step(val_ap)

        trial.set_user_attr(f"epoch_{epoch+1}_train_loss", float(train_loss))
        trial.set_user_attr(f"epoch_{epoch+1}_val_loss", float(val_loss))
        trial.set_user_attr(f"epoch_{epoch+1}_val_ap", float(val_ap))
        trial.set_user_attr(f"epoch_{epoch+1}_val_auc", float(val_auc))
        trial.set_user_attr(f"epoch_{epoch+1}_val_f1", float(val_f1))

        if val_ap > best_val_ap:
            best_val_ap = val_ap
            best_epoch = epoch + 1
            best_metrics = {
                "best_epoch": best_epoch,
                "train_loss": float(train_loss),
                "train_acc": float(train_acc),
                "train_f1": float(train_f1),
                "train_ap": float(train_ap),
                "train_auc": float(train_auc),
                "val_loss": float(val_loss),
                "val_acc": float(val_acc),
                "val_f1": float(val_f1),
                "val_ap": float(val_ap),
                "val_auc": float(val_auc),
            }
            best_model_state = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "hparams": hparams,
                "fixed_config": {
                    "batch_size": BATCH_SIZE,
                    "num_epochs": NUM_EPOCHS,
                    "patience": PATIENCE,
                    "min_delta": MIN_DELTA,
                    "enable_augmentation": ENABLE_AUGMENTATION,
                    "use_hflip": USE_HFLIP,
                    "use_color_jitter": USE_COLOR_JITTER,
                    "use_random_resized_crop": USE_RANDOM_RESIZED_CROP,
                    "use_gaussian_blur": USE_GAUSSIAN_BLUR,
                    "use_temporal_augmentation": USE_TEMPORAL_AUGMENTATION,
                    "anticipation_mode": ANTICIPATION_MODE,
                    "anticipation_offset": ANTICIPATION_OFFSET,
                    "d_model": D_MODEL,
                    "bidirectional": BIDIRECTIONAL,
                },
                "metrics": best_metrics,
                "seed": trial_seed,
                "class_weights": class_weights.detach().cpu(),
                "train_size": len(train_dataset),
                "val_size": len(val_dataset),
            }

        # early stopping
        if val_ap > best_val_ap_for_stop + MIN_DELTA:
            best_val_ap_for_stop = val_ap
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        # report a Optuna
        trial.report(best_val_ap, step=epoch)

        if trial.should_prune():
            raise optuna.TrialPruned()

        if early_stop_counter >= PATIENCE:
            break

    # guardar mejor checkpoint global de todo el estudio
    os.makedirs(CKPT_DIR, exist_ok=True)
    global_best_path = os.path.join(CKPT_DIR, "best_global_phase1.pt")

    current_best = None
    if os.path.exists(global_best_path):
        current_best = torch.load(global_best_path, map_location="cpu")

    previous_best_ap = -1.0
    if current_best is not None:
        previous_best_ap = current_best.get("metrics", {}).get("val_ap", -1.0)

    if best_model_state is not None and best_val_ap > previous_best_ap:
        torch.save(best_model_state, global_best_path)

    # guardar checkpoint del trial si fue decente
    trial_ckpt_path = os.path.join(CKPT_DIR, f"trial_{trial.number:04d}.pt")
    if best_model_state is not None:
        torch.save(best_model_state, trial_ckpt_path)

    # attrs legibles
    trial.set_user_attr("best_epoch", best_epoch)
    trial.set_user_attr("best_val_ap", float(best_val_ap))
    trial.set_user_attr("best_val_auc", float(best_metrics.get("val_auc", float("nan"))))
    trial.set_user_attr("best_val_f1", float(best_metrics.get("val_f1", float("nan"))))
    trial.set_user_attr("best_val_loss", float(best_metrics.get("val_loss", float("nan"))))
    trial.set_user_attr("train_size", len(train_dataset))
    trial.set_user_attr("val_size", len(val_dataset))
    trial.set_user_attr("trial_seed", trial_seed)

    return best_val_ap


def print_study_summary(study: optuna.Study):
    print("\n" + "=" * 80)
    print("ESTUDIO TERMINADO")
    print("=" * 80)
    print(f"Número de trials: {len(study.trials)}")
    print(f"Mejor trial: {study.best_trial.number}")
    print(f"Mejor val_ap: {study.best_value:.6f}")
    print("Mejores hiperparámetros:")
    for k, v in study.best_trial.params.items():
        print(f"  {k}: {v}")

    print("\nUser attrs del mejor trial:")
    for k, v in study.best_trial.user_attrs.items():
        if k.startswith("epoch_"):
            continue
        print(f"  {k}: {v}")

    print("=" * 80 + "\n")


def save_best_params_json(study: optuna.Study, out_path: str):
    payload = {
        "best_trial_number": study.best_trial.number,
        "best_value_val_ap": study.best_value,
        "best_params": study.best_trial.params,
        "best_user_attrs": {
            k: v for k, v in study.best_trial.user_attrs.items() if not k.startswith("epoch_")
        },
        "generated_at": datetime.now().isoformat(),
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def main():
    os.makedirs(CKPT_DIR, exist_ok=True)

    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Current device:", torch.cuda.current_device())
        print("Device name:", torch.cuda.get_device_name(0))
    else:
        print("Using CPU")

    sampler = TPESampler(seed=BASE_SEED, multivariate=True)
    pruner = MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=4,
        interval_steps=1,
    )

    study = optuna.create_study(
        study_name="accident_prediction_phase1",
        storage=STUDY_DB,
        load_if_exists=True,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
    )

    # ajusta n_trials a lo que quieras lanzar
    study.optimize(
        objective,
        n_trials=20,
        gc_after_trial=True,
        show_progress_bar=True,
    )

    print_study_summary(study)

    best_json_path = os.path.join(CKPT_DIR, "best_phase1_params.json")
    save_best_params_json(study, best_json_path)
    print(f"Mejores parámetros guardados en: {best_json_path}")


if __name__ == "__main__":
    main()