import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from model.dataset import AccidentClipDataset
from model.mylibs.baseline_modelGRU import BaselineResNetGRU


def build_val_transform(image_size=(224, 224)):
    return v2.Compose([
        v2.Resize(image_size, antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()

    all_labels, all_preds, all_scores, all_attn = [], [], [], []

    for clips, labels in dataloader:
        clips = clips.to(device)
        labels = labels.to(device)

        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            outputs, attn_weights = model(clips)

        preds = torch.argmax(outputs, dim=1)
        probs = torch.softmax(outputs, dim=1)[:, 1]

        all_labels.extend(labels.cpu().numpy().tolist())
        all_preds.extend(preds.cpu().numpy().tolist())
        all_scores.extend(probs.cpu().numpy().tolist())
        all_attn.append(attn_weights.cpu().numpy())

    labels_arr = np.array(all_labels)
    preds_arr = np.array(all_preds)
    scores_arr = np.array(all_scores)
    attn_np = np.concatenate(all_attn, axis=0)  # [N, T]

    acc = accuracy_score(labels_arr, preds_arr)
    f1 = f1_score(labels_arr, preds_arr, zero_division=0)
    cm = confusion_matrix(labels_arr, preds_arr, labels=[0, 1])

    unique = np.unique(labels_arr)
    if len(unique) > 1:
        ap = average_precision_score(labels_arr, scores_arr)
        auc = roc_auc_score(labels_arr, scores_arr)
    else:
        ap = float("nan")
        auc = float("nan")
        print("WARNING: solo una clase en el split. AP/AUC = nan")

    mean_attn = attn_np.mean(axis=0)
    attn_entropy = float(-np.sum(mean_attn * np.log(mean_attn + 1e-8)))
    attn_peak = int(mean_attn.argmax())

    n_pos = int(labels_arr.sum())
    n_neg = len(labels_arr) - n_pos
    mean_score_pos = float(scores_arr[labels_arr == 1].mean()) if n_pos > 0 else float("nan")
    mean_score_neg = float(scores_arr[labels_arr == 0].mean()) if n_neg > 0 else float("nan")

    attn_entropy_pos, attn_peak_pos = float("nan"), -1
    attn_entropy_neg, attn_peak_neg = float("nan"), -1

    if n_pos > 0:
        m = attn_np[labels_arr == 1].mean(axis=0)
        attn_entropy_pos = float(-np.sum(m * np.log(m + 1e-8)))
        attn_peak_pos = int(m.argmax())

    if n_neg > 0:
        m = attn_np[labels_arr == 0].mean(axis=0)
        attn_entropy_neg = float(-np.sum(m * np.log(m + 1e-8)))
        attn_peak_neg = int(m.argmax())

    return {
        "acc": acc,
        "f1": f1,
        "ap": ap,
        "auc": auc,
        "cm": cm,
        "n_samples": len(labels_arr),
        "n_pos": n_pos,
        "n_neg": n_neg,
        "mean_score_pos": mean_score_pos,
        "mean_score_neg": mean_score_neg,
        "score_separation": mean_score_pos - mean_score_neg,
        "mean_attn": mean_attn,
        "attn_entropy": attn_entropy,
        "attn_peak": attn_peak,
        "attn_entropy_pos": attn_entropy_pos,
        "attn_peak_pos": attn_peak_pos,
        "attn_entropy_neg": attn_entropy_neg,
        "attn_peak_neg": attn_peak_neg,
        "labels": labels_arr,
        "preds": preds_arr,
        "scores": scores_arr,
    }


def print_results(m, split_name):
    tn, fp, fn, tp = m["cm"].ravel()
    print(f"\n{'='*60}")
    print(f"  Resultados — {split_name}")
    print(f"{'='*60}")
    print(f"  Muestras : {m['n_samples']}  (pos={m['n_pos']}, neg={m['n_neg']})")
    print(f"  Accuracy : {m['acc']:.4f}")
    print(f"  F1       : {m['f1']:.4f}")
    print(f"  AP       : {m['ap']:.4f}")
    print(f"  AUC      : {m['auc']:.4f}")
    print(f"  CM  [[TN={tn}, FP={fp}], [FN={fn}, TP={tp}]]")
    print(f"  Scores   : pos={m['mean_score_pos']:.4f} | neg={m['mean_score_neg']:.4f} "
          f"| sep={m['score_separation']:.4f}")
    print(f"  Attn     : peak={m['attn_peak']} | entropy={m['attn_entropy']:.4f}")
    print(f"             pos_peak={m['attn_peak_pos']} entropy={m['attn_entropy_pos']:.4f}")
    print(f"             neg_peak={m['attn_peak_neg']} entropy={m['attn_entropy_neg']:.4f}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Evalúa un checkpoint GRU sobre un split.")
    parser.add_argument("--ckpt", required=True, help="Ruta al .pt del checkpoint")
    parser.add_argument(
        "--txt",
        default="/data-fast/data-server/vlopezmo/model/training/training_val.txt",
        help="Fichero txt del split a evaluar",
    )
    parser.add_argument(
        "--rgb_root",
        default="/data-fast/data-server/vlopezmo/DADA2000",
    )
    parser.add_argument("--batch_size", type=int, default=48)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", default=None, help="cuda / cpu (auto si no se especifica)")
    parser.add_argument("--save_json", default=None, help="Ruta para guardar resultados en JSON")
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")
    print(f"Checkpoint: {args.ckpt}")

    # ── Config del modelo ─────────────────────────────────────────
    num_frames          = 16
    d_model             = 128
    gru_hidden          = 128
    gru_num_layers      = 1
    dropout             = 0.5
    bidirectional       = True
    freeze_all          = True
    unfreeze_layer4     = True
    unfreeze_mode       = "full_layer4"   # "last_block" | "full_layer4" | "layer3_layer4"
    anticipation_mode   = False
    anticipation_offset = 1
    # ──────────────────────────────────────────────────────────────

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    epoch = ckpt.get("epoch", "?")

    print(f"\nConfig (epoch {epoch}):")
    print(f"  num_frames={num_frames}, d_model={d_model}, gru_hidden={gru_hidden}, "
          f"gru_layers={gru_num_layers}, bidirectional={bidirectional}, dropout={dropout}")
    print(f"  freeze_all={freeze_all}, unfreeze_layer4={unfreeze_layer4}, "
          f"unfreeze_mode={unfreeze_mode}")
    print(f"  anticipation_mode={anticipation_mode}, anticipation_offset={anticipation_offset}")

    model = BaselineResNetGRU(
        num_classes=2,
        d_model=d_model,
        gru_hidden=gru_hidden,
        gru_layers=gru_num_layers,
        dropout=dropout,
        pretrained=False,
        freeze_early=False,
        freeze_all=freeze_all,
        unfreeze_layer4=unfreeze_layer4,
        unfreeze_mode=unfreeze_mode,
        bidirectional=bidirectional,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nParámetros totales: {total_params:,}")

    transform = build_val_transform()

    split_name = os.path.splitext(os.path.basename(args.txt))[0]

    dataset = AccidentClipDataset(
        txt_path=args.txt,
        rgb_root=args.rgb_root,
        num_frames=num_frames,
        transform=transform,
        train=False,
        use_temporal_augmentation=False,
        temporal_max_jitter=0,
        use_toa_guided_sampling=False,
        toa_center_strength=0.0,
        anticipation_mode=anticipation_mode,
        anticipation_offset=anticipation_offset,
        drop_invalid_samples=True,
    )

    print(f"\nMuestras en '{split_name}': {len(dataset)}")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=2 if args.num_workers > 0 else None,
    )

    print("Evaluando...\n")
    metrics = evaluate(model, loader, device)
    print_results(metrics, split_name)

    # Métricas del checkpoint en val (referencia del entrenamiento)
    if "val_auc" in ckpt:
        print("Métricas guardadas en el checkpoint (val durante entrenamiento):")
        print(f"  AUC={ckpt.get('val_auc', '?'):.4f} | "
              f"AP={ckpt.get('val_ap', '?'):.4f} | "
              f"Acc={ckpt.get('val_acc', '?'):.4f} | "
              f"F1={ckpt.get('val_f1', '?'):.4f}")

    if args.save_json:
        out = {
            "checkpoint": args.ckpt,
            "split": split_name,
            "epoch": epoch,
            "acc": metrics["acc"],
            "f1": metrics["f1"],
            "ap": float(metrics["ap"]) if not np.isnan(metrics["ap"]) else None,
            "auc": float(metrics["auc"]) if not np.isnan(metrics["auc"]) else None,
            "n_samples": metrics["n_samples"],
            "n_pos": metrics["n_pos"],
            "n_neg": metrics["n_neg"],
            "mean_score_pos": float(metrics["mean_score_pos"]),
            "mean_score_neg": float(metrics["mean_score_neg"]),
            "score_separation": float(metrics["score_separation"]),
            "attn_entropy": metrics["attn_entropy"],
            "attn_peak": metrics["attn_peak"],
            "cm": metrics["cm"].tolist(),
        }
        with open(args.save_json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Resultados guardados en: {args.save_json}")


if __name__ == "__main__":
    main()
