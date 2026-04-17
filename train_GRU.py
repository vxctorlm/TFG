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
    val_ap = average_precision_score(all_labels, all_scores)
    val_auc = roc_auc_score(all_labels, all_scores)

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


def main():
    seed = 42
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Current device:", torch.cuda.current_device())
        print("Device name:", torch.cuda.get_device_name(0))
    else:
        print("Using CPU")

    # ── Hiperparámetros ────────────────────────────────────────────────────────
    batch_size = 8
    num_epochs = 100
    weight_decay = 1e-4
    num_frames = 16

    # ── Backbone ───────────────────────────────────────────────────────────────
    # Configuración v5: backbone completamente congelado.
    # Las corridas con unfreeze_layer4 empeoraron val_ap (0.632 vs 0.637)
    # porque layer4 se especializa en los 2282 ejemplos de train incluso
    # con lr=2e-6. Cuello de botella es tamaño del dataset, no capacidad.
    pretrained = True
    freeze_early = False
    freeze_all = True
    unfreeze_layer4 = False

    learning_rate = 2e-5        # proj + GRU + classifier

    # ── Augmentación espacial ─────────────────────────────────────────────────
    enable_augmentation = True
    use_hflip = True
    use_color_jitter = True
    use_random_resized_crop = False
    use_gaussian_blur = False

    # ── Augmentación temporal ─────────────────────────────────────────────────
    use_temporal_augmentation = True
    temporal_max_jitter = 2
    # Bajado de 0.62 a 0.3: con 0.62 se introduce un sesgo en train
    # (frames centrados en el TOA) que no existe en val (muestreo uniforme).
    toa_center_strength = 0.3

    # ── Modelo ────────────────────────────────────────────────────────────────
    gru_num_layers = 1      # Vuelve a 1: con backbone congelado necesitamos más capacidad en GRU
    d_model = 256
    dropout = 0.35
    bidirectional = False

    # ── MixUp temporal ────────────────────────────────────────────────────────
    # Mezcla dos clips del mismo batch ponderadamente. Obliga al modelo
    # a generalizar mejor creando ejemplos sinteticos intermedios entre clases.
    # Util en datasets pequeños como este (2282 samples).
    use_mixup = True
    mixup_alpha = 0.2       # Beta(alpha, alpha): alpha bajo = mezclas suaves
    mixup_prob = 0.5        # Probabilidad de aplicar mixup a cada batch

    # ── Label smoothing ───────────────────────────────────────────────────────
    # Penaliza predicciones con prob > 0.9. Evita que val_loss se dispare
    # desde epoch 0 por logits sobreconfiados.
    label_smoothing = 0.1

    # ── Early stopping ────────────────────────────────────────────────────────
    patience = 8            # Subido a 5: con backbone congelado el aprendizaje es más lento
    min_delta = 0.001
    best_val_ap_for_stop = -1.0
    early_stop_counter = 0

    # ── Run name ──────────────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    frames_tag = "allf" if num_frames is None else f"{num_frames}f"
    run_name = (
        f"gru_v7_{timestamp}_{frames_tag}_seed{seed}"
        f"_freezeAll{int(freeze_all)}"
        f"_augTemp{int(use_temporal_augmentation)}"
        f"_aug{int(enable_augmentation)}"
        f"_cj{int(use_color_jitter)}"
        f"_flip{int(use_hflip)}"
        f"_dm{d_model}"
        f"_gru{gru_num_layers}"
        f"_bi{int(bidirectional)}"
        f"_ls{int(label_smoothing * 10)}"
        f"_meanpool"
        f"_mixup{int(use_mixup)}a{int(mixup_alpha*10)}p{int(mixup_prob*10)}"
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
            "pooling": "mean",
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
            "scheduler_metric": "val_ap",
            "use_mixup": use_mixup,
            "mixup_alpha": mixup_alpha,
            "mixup_prob": mixup_prob,
            "enable_augmentation": enable_augmentation,
            "use_hflip": use_hflip,
            "use_color_jitter": use_color_jitter,
            "use_random_resized_crop": use_random_resized_crop,
            "use_gaussian_blur": use_gaussian_blur,
        }
    )

    # ── Transforms ───────────────────────────────────────────────────────────
    train_transform = build_clip_transform(
        train=True,
        image_size=(224, 224),
        enable_augmentation=enable_augmentation,
        use_hflip=use_hflip,
        use_color_jitter=use_color_jitter,
        use_random_resized_crop=use_random_resized_crop,
        use_gaussian_blur=use_gaussian_blur,
    )

    val_transform = build_clip_transform(
        train=False,
        image_size=(224, 224),
        enable_augmentation=False,
    )

    # ── Datasets ──────────────────────────────────────────────────────────────
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
    labels_val   = [s["label"] for s in val_dataset.samples]
    n_tr, n_va   = len(labels_train), len(labels_val)
    pos_tr, pos_va = sum(labels_train), sum(labels_val)
    print(f"Train: {pos_tr} acc ({100*pos_tr/n_tr:.1f}%) | "
          f"{n_tr-pos_tr} no-acc ({100*(n_tr-pos_tr)/n_tr:.1f}%)")
    print(f"Val:   {pos_va} acc ({100*pos_va/n_va:.1f}%) | "
          f"{n_va-pos_va} no-acc ({100*(n_va-pos_va)/n_va:.1f}%)")
    diff = abs(pos_tr/n_tr - pos_va/n_va) * 100
    if diff > 10:
        print(f"Diferencia de distribucion entre splits: {diff:.1f}pp")
    else:
        print(f"Distribucion similar entre splits (diff={diff:.1f}pp)")

    # Distribucion 53/47 — no se necesitan pesos de clase.
    # Con distribucion casi balanceada los pesos introducen inestabilidad
    # sin aportar beneficio (causaron colapso de val_f1 a 0.20 en epoch 3).
    class_weights = None
    print("Class weights: desactivados (distribucion balanceada)")

    # ── DataLoaders ───────────────────────────────────────────────────────────
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

    # ── Modelo ────────────────────────────────────────────────────────────────
    model = BaselineResNetGRU(
        num_classes=2,
        d_model=d_model,
        num_layers=gru_num_layers,
        dropout=dropout,
        pretrained=pretrained,
        freeze_early=freeze_early,
        freeze_all=freeze_all,
        bidirectional=bidirectional,
    ).to(device)

    # FIX: dos criterios separados.
    # criterion_train: label_smoothing + pesos de clase para el loop de entrenamiento.
    # criterion_val:   sin label_smoothing para que val_loss sea comparable con
    #                  train_loss y no este artificialmente inflada.
    #                  Con label_smoothing incluso una prediccion perfecta tiene loss > 0.
    criterion_train = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    criterion_val   = nn.CrossEntropyLoss()

    # Solo optimizamos parametros entrenables (backbone congelado excluido)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3,
    )

    ckpt_dir = "/data-fast/data-server/vlopezmo/model/checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)

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

            # MixUp temporal: con probabilidad mixup_prob, mezcla este batch
            # con una permutacion de si mismo usando lambda ~ Beta(alpha, alpha).
            # Reemplaza el batch original por la version mezclada.
            apply_mixup = use_mixup and np.random.random() < mixup_prob
            if apply_mixup:
                lam = np.random.beta(mixup_alpha, mixup_alpha)
                # Garantizar que lam no sea trivial (evita no-ops)
                lam = max(lam, 1.0 - lam)
                idx = torch.randperm(clips.size(0), device=device)
                clips_mixed = lam * clips + (1.0 - lam) * clips[idx]
                labels_a, labels_b = labels, labels[idx]
            else:
                clips_mixed = clips

            optimizer.zero_grad()
            outputs = model(clips_mixed)

            if apply_mixup:
                # Loss mixta: combinacion convexa de los dos targets
                loss = lam * criterion_train(outputs, labels_a) \
                     + (1.0 - lam) * criterion_train(outputs, labels_b)
            else:
                loss = criterion_train(outputs, labels)

            loss.backward()

            # Gradient clipping: previene exploding gradients en RNNs
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            running_loss += loss.item()

            # Metricas de train: siempre sobre las labels originales (no mezcladas)
            # para que el reporte sea coherente con la evaluacion.
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

        val_loss, val_acc, val_f1, val_ap, val_auc = evaluate(
            model, val_loader, criterion_val, device
        )

        print(f"\nEpoch [{epoch+1}/{num_epochs}] completed")
        print(
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Train F1: {train_f1:.4f} | Train AP: {train_ap:.4f} | Train AUC: {train_auc:.4f} || "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
            f"Val F1: {val_f1:.4f} | Val AP: {val_ap:.4f} | Val AUC: {val_auc:.4f}"
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

        # Guardar mejor modelo según val_ap
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
                "seed": seed,
            }, ckpt_path)
            print(f"Nuevo mejor modelo guardado en: {ckpt_path}")

        # Guardar último checkpoint
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
            "seed": seed,
        }, last_ckpt_path)

        # Early stopping según val_ap
        if val_ap > best_val_ap_for_stop + min_delta:
            best_val_ap_for_stop = val_ap
            early_stop_counter = 0
            print(f"Mejora en val_ap: {val_ap:.4f}")
        else:
            early_stop_counter += 1
            print(f"Sin mejora en val_ap. EarlyStopping counter: {early_stop_counter}/{patience}")

        if early_stop_counter >= patience:
            print(f"Early stopping activado en epoch {epoch+1}")
            break

    wandb.finish()


if __name__ == "__main__":
    main()