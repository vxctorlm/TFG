from torchvision.transforms import v2
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import os
import wandb
import random
import numpy as np
from datetime import datetime
from torch.amp import autocast, GradScaler
import torchmetrics
import optuna

from model.dataset import AccidentClipDataset
from model.mylibs.baseline_model import BaselineResNetTransformer


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


@torch.no_grad()
def evaluate(model, dataloader, criterion, device, metrics):
    model.eval()
    running_loss = 0.0

    for m in metrics.values():
        m.reset()

    for clips, labels in dataloader:
        clips = clips.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast(device_type='cuda', enabled=(device.type == 'cuda')):
            outputs = model(clips)
            loss = criterion(outputs, labels)

        running_loss += loss.item()

        preds = torch.argmax(outputs, dim=1)
        probs = torch.softmax(outputs, dim=1)[:, 1]

        for name, m in metrics.items():
            if name in ['ap', 'auc']:
                m.update(probs, labels)
            else:
                m.update(preds, labels)

    val_loss = running_loss / len(dataloader)
    res = {k: v.compute().item() for k, v in metrics.items()}
    return val_loss, res['acc'], res['f1'], res['ap'], res['auc']


def build_clip_transform(train=False, image_size=(224, 224), enable_augmentation=False):
    ops = []

    if train and enable_augmentation:
        ops.append(v2.RandomResizedCrop(size=image_size, scale=(0.90, 1.0), ratio=(0.95, 1.05), antialias=True))
        ops.append(v2.RandomHorizontalFlip(p=0.5))
        ops.append(v2.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02))
        ops.append(v2.RandomApply([v2.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5))], p=0.3))
    else:
        ops.append(v2.Resize(image_size, antialias=True))

    ops.extend([
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return v2.Compose(ops)


def objective(trial):
    seed = 42
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ====================== HIPERPARÁMETROS FIJOS ======================
    num_epochs = 100
    num_frames = 16
    enable_augmentation = True
    use_temporal_augmentation = True
    temporal_max_jitter = 2

    patience = 6
    min_delta = 0.0013

    # ====================== HIPERPARÁMETROS OPTUNA ======================
    learning_rate = trial.suggest_float("learning_rate", 4e-5, 1.5e-4, log=True)
    dropout = trial.suggest_float("dropout", 0.05, 0.35)
    num_transformer_layers = trial.suggest_int("num_transformer_layers", 1, 3)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 5e-4, log=True)
    toa_center_strength = trial.suggest_float("toa_center_strength", 0.35, 0.65)
    batch_size = trial.suggest_categorical("batch_size", [4, 8, 12])

    # ====================== WANDB ======================
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"optuna_trial_{trial.number}_{timestamp}"

    wandb.init(
        project="tfg-accident-prediction",
        name=run_name,
        reinit=True,
        config={
            "trial_number": trial.number,
            "seed": seed,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "dropout": dropout,
            "num_frames": num_frames,
            "num_transformer_layers": num_transformer_layers,
            "augmentation": enable_augmentation,
            "temporal_aug": use_temporal_augmentation,
            "toa_center_strength": toa_center_strength,
        }
    )

    # ====================== DATASETS ======================
    train_transform = build_clip_transform(train=True, enable_augmentation=enable_augmentation)
    val_transform = build_clip_transform(train=False)

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
    )

    print(f"\n[Trial {trial.number}]")
    print("Train samples:", len(train_dataset))
    print("Val samples:", len(val_dataset))
    print(f"Params: lr={learning_rate:.6f}, dropout={dropout:.3f}, "
          f"layers={num_transformer_layers}, wd={weight_decay:.6f}, "
          f"toa={toa_center_strength:.3f}, batch_size={batch_size}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    # ====================== MODELO ======================
    model = BaselineResNetTransformer(
        num_classes=2,
        d_model=256,
        nhead=4,
        num_layers=num_transformer_layers,
        dim_feedforward=512,
        dropout=dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    scaler = GradScaler(enabled=(device.type == "cuda"))

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3,
        threshold=min_delta
    )

    metrics = {
        'acc': torchmetrics.Accuracy(task="binary").to(device),
        'f1': torchmetrics.F1Score(task="binary").to(device),
        'ap': torchmetrics.AveragePrecision(task="binary").to(device),
        'auc': torchmetrics.AUROC(task="binary").to(device)
    }

    ckpt_dir = "/data-fast/data-server/vlopezmo/model/checkpoints/optuna"
    os.makedirs(ckpt_dir, exist_ok=True)

    best_val_ap = -1.0
    best_val_ap_for_stop = -1.0
    early_stop_counter = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for m in metrics.values():
            m.reset()

        for batch_idx, (clips, labels) in enumerate(train_loader):
            clips = clips.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type='cuda', enabled=(device.type == 'cuda')):
                outputs = model(clips)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            probs = torch.softmax(outputs, dim=1)[:, 1]

            for name, m in metrics.items():
                if name in ['ap', 'auc']:
                    m.update(probs, labels)
                else:
                    m.update(preds, labels)

            if (batch_idx + 1) % 50 == 0:
                print(
                    f"Trial [{trial.number}] Epoch [{epoch+1}/{num_epochs}] "
                    f"Batch [{batch_idx+1}/{len(train_loader)}] Loss: {loss.item():.4f}"
                )

        train_loss = running_loss / len(train_loader)
        res_train = {k: v.compute().item() for k, v in metrics.items()}

        val_loss, val_acc, val_f1, val_ap, val_auc = evaluate(
            model, val_loader, criterion, device, metrics
        )

        print(
            f"Trial [{trial.number}] Epoch [{epoch+1}/{num_epochs}] | "
            f"Train Loss: {train_loss:.4f} | Val AP: {val_ap:.4f} | "
            f"Val AUC: {val_auc:.4f} | Val F1: {val_f1:.4f}"
        )

        wandb.log({
            "trial_number": trial.number,
            "epoch": epoch + 1,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_loss,
            "train_acc": res_train["acc"],
            "train_f1": res_train["f1"],
            "train_ap": res_train["ap"],
            "train_auc": res_train["auc"],
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_f1": val_f1,
            "val_ap": val_ap,
            "val_roc_auc": val_auc,
        })

        scheduler.step(val_ap)

        if val_ap > best_val_ap:
            best_val_ap = val_ap
            torch.save({
                "trial_number": trial.number,
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "val_ap": val_ap,
                "val_f1": val_f1,
                "val_auc": val_auc,
                "params": trial.params,
            }, os.path.join(ckpt_dir, f"trial_{trial.number}_best_val_ap.pt"))
            print(f"→ Mejor modelo del trial guardado (val_ap = {val_ap:.4f})")

        # Reportar a Optuna
        trial.report(val_ap, epoch)

        # Pruning
        if trial.should_prune():
            wandb.finish()
            raise optuna.exceptions.TrialPruned()

        # Early stopping
        if val_ap > best_val_ap_for_stop + min_delta:
            best_val_ap_for_stop = val_ap
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"Early stopping activado en epoch {epoch+1}")
                break

    wandb.finish()
    return best_val_ap


def main():
    study = optuna.create_study(
        direction="maximize",
        study_name="accident_prediction_optuna",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    )

    study.optimize(objective, n_trials=20)

    print("\n====================== RESULTADOS OPTUNA ======================")
    print(f"Mejor trial: {study.best_trial.number}")
    print(f"Mejor val_ap: {study.best_value:.6f}")
    print("Mejores hiperparámetros:")
    for k, v in study.best_trial.params.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()