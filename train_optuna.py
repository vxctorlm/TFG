from torchvision.transforms import v2
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import os
import random
import numpy as np
import optuna
from sklearn.metrics import accuracy_score, f1_score, average_precision_score, roc_auc_score

from model.dataset import AccidentClipDataset
from model.mylibs.baseline_model import BaselineResNetTransformer


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


def objective(trial):
    seed = 42
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = trial.suggest_categorical("batch_size", [2, 4, 8])
    num_frames = trial.suggest_categorical("num_frames", [8, 16, 24])

    learning_rate = trial.suggest_float("learning_rate", 1e-5, 2e-4, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 5e-4, log=True)
    dropout = trial.suggest_float("dropout", 0.20, 0.45)

    use_temporal_augmentation = trial.suggest_categorical(
        "use_temporal_augmentation", [False, True]
    )
    use_toa_guided_sampling = trial.suggest_categorical(
        "use_toa_guided_sampling", [False, True]
    )

    temporal_max_jitter = trial.suggest_int("temporal_max_jitter", 0, 3)
    toa_center_strength = trial.suggest_float("toa_center_strength", 0.0, 0.7)

    patience = trial.suggest_int("patience", 5, 8)
    min_delta = trial.suggest_float("min_delta", 5e-4, 2e-3, log=True)

    scheduler_patience = trial.suggest_int("scheduler_patience", 2, 4)
    scheduler_factor = trial.suggest_categorical("scheduler_factor", [0.5, 0.7])

    num_epochs = 30
    image_size = (224, 224)
    num_transformer_layers = 1

    enable_augmentation = False
    use_hflip = False
    use_color_jitter = False
    use_random_resized_crop = False
    use_gaussian_blur = False

    if not use_temporal_augmentation:
        temporal_max_jitter = 0
        use_toa_guided_sampling = False
        toa_center_strength = 0.0

    if not use_toa_guided_sampling:
        toa_center_strength = 0.0

    train_transform = build_clip_transform(
        train=True,
        image_size=image_size,
        enable_augmentation=enable_augmentation,
        use_hflip=use_hflip,
        use_color_jitter=use_color_jitter,
        use_random_resized_crop=use_random_resized_crop,
        use_gaussian_blur=use_gaussian_blur,
    )

    val_transform = build_clip_transform(
        train=False,
        image_size=image_size,
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
    )

    val_dataset = AccidentClipDataset(
        txt_path="/data-fast/data-server/vlopezmo/model/training/training_val.txt",
        rgb_root="/data-fast/data-server/vlopezmo/DADA2000",
        num_frames=num_frames,
        transform=val_transform,
        train=False,
        use_temporal_augmentation=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
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

    model = BaselineResNetTransformer(
        num_classes=2,
        d_model=256,
        nhead=4,
        num_layers=num_transformer_layers,
        dim_feedforward=512,
        dropout=dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=scheduler_factor,
        patience=scheduler_patience,
    )

    ckpt_dir = "/data-fast/data-server/vlopezmo/model/checkpoints/optuna_full"
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"trial_{trial.number}_best.pt")

    best_val_ap = -1.0
    best_epoch = -1

    best_val_ap_for_stop = -1.0
    early_stop_counter = 0

    for epoch in range(num_epochs):
        model.train()

        running_loss = 0.0
        all_train_labels = []
        all_train_preds = []
        all_train_scores = []

        for clips, labels in train_loader:
            clips = clips.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(clips)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            probs = torch.softmax(outputs, dim=1)[:, 1]

            all_train_labels.extend(labels.detach().cpu().numpy().tolist())
            all_train_preds.extend(preds.detach().cpu().numpy().tolist())
            all_train_scores.extend(probs.detach().cpu().numpy().tolist())

        train_loss = running_loss / len(train_loader)

        try:
            train_acc = accuracy_score(all_train_labels, all_train_preds)
            train_f1 = f1_score(all_train_labels, all_train_preds, zero_division=0)
            train_ap = average_precision_score(all_train_labels, all_train_scores)
            train_auc = roc_auc_score(all_train_labels, all_train_scores)
        except ValueError:
            train_acc = float("nan")
            train_f1 = float("nan")
            train_ap = float("nan")
            train_auc = float("nan")

        val_loss, val_acc, val_f1, val_ap, val_auc = evaluate(
            model, val_loader, criterion, device
        )

        print(
            f"[Trial {trial.number}] Epoch {epoch+1}/{num_epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"train_f1={train_f1:.4f} train_ap={train_ap:.4f} train_auc={train_auc:.4f} || "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
            f"val_f1={val_f1:.4f} val_ap={val_ap:.4f} val_auc={val_auc:.4f}"
        )

        old_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_ap)
        new_lr = optimizer.param_groups[0]["lr"]

        if new_lr != old_lr:
            print(f"[Trial {trial.number}] LR reducido: {old_lr:.2e} -> {new_lr:.2e}")

        if val_ap > best_val_ap:
            best_val_ap = val_ap
            best_epoch = epoch + 1

            torch.save({
                "trial_number": trial.number,
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_ap": best_val_ap,
                "best_epoch": best_epoch,
                "params": trial.params,
            }, ckpt_path)

        # Optuna report/prune
        trial.report(best_val_ap, step=epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

        # Early stopping
        if val_ap > best_val_ap_for_stop + min_delta:
            best_val_ap_for_stop = val_ap
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if early_stop_counter >= patience:
            print(
                f"[Trial {trial.number}] Early stopping en epoch {epoch+1} "
                f"(best_val_ap={best_val_ap:.4f} en epoch {best_epoch})"
            )
            break

    trial.set_user_attr("best_epoch", best_epoch)
    trial.set_user_attr("best_val_ap", best_val_ap)
    trial.set_user_attr("checkpoint_path", ckpt_path)

    return best_val_ap


def main():
    study_name = "accident_prediction_optuna_full"
    storage = "sqlite:////data-fast/data-server/vlopezmo/model/optuna_accident_full.db"

    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=5,
        interval_steps=1,
    )

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        direction="maximize",
        pruner=pruner,
    )

    study.optimize(objective, n_trials=80)

    print("\n===== BEST TRIAL =====")
    print(f"Best value (best_val_ap): {study.best_value:.6f}")
    print("Best params:")
    for k, v in study.best_trial.params.items():
        print(f"  {k}: {v}")

    print("\nBest trial attrs:")
    for k, v in study.best_trial.user_attrs.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()