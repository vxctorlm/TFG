# diagnose_dataset.py
import sys
from collections import Counter, defaultdict

sys.path.insert(0, "/data-fast/data-server/vlopezmo/model")
from dataset import AccidentClipDataset

TRAIN_TXT = "/data-fast/data-server/vlopezmo/model/training/training_train.txt"
VAL_TXT   = "/data-fast/data-server/vlopezmo/model/training/training_val.txt"
RGB_ROOT  = "/data-fast/data-server/vlopezmo/DADA2000"

train_ds = AccidentClipDataset(
    txt_path=TRAIN_TXT, rgb_root=RGB_ROOT,
    num_frames=16, transform=None, train=False,
    drop_invalid_samples=True,
)
val_ds = AccidentClipDataset(
    txt_path=VAL_TXT, rgb_root=RGB_ROOT,
    num_frames=16, transform=None, train=False,
    drop_invalid_samples=True,
)

print(f"Clips train: {len(train_ds)}  | Clips val: {len(val_ds)}")

train_vids = [s["video_id"] for s in train_ds.samples]
val_vids   = [s["video_id"] for s in val_ds.samples]

train_set, val_set = set(train_vids), set(val_vids)
overlap = train_set & val_set

# === A) LEAKAGE ===
print(f"\n=== A) LEAKAGE (video_id en ambos splits) ===")
print(f"Vídeos únicos train: {len(train_set)}")
print(f"Vídeos únicos val:   {len(val_set)}")
print(f"SOLAPAMIENTO: {len(overlap)}   (debe ser 0)")
if overlap:
    # ¿cuántos clips de val provienen de vídeos que también están en train?
    leaked_val_clips = sum(1 for v in val_vids if v in overlap)
    print(f"Clips de val cuyo video_id ESTÁ en train: "
          f"{leaked_val_clips}/{len(val_vids)} "
          f"({100*leaked_val_clips/len(val_vids):.1f}%)")
    print(f"Ejemplos: {list(overlap)[:5]}")

# === B) CLIPS POR VÍDEO ===
print(f"\n=== B) Clips por vídeo ===")
tr_c = Counter(train_vids); va_c = Counter(val_vids)
def stats(c, name):
    vals = list(c.values())
    print(f"{name}: media={sum(vals)/len(vals):.2f}  "
          f"max={max(vals)}  "
          f"vid con >1 clip={sum(1 for v in vals if v>1)}/{len(vals)}")
stats(tr_c, "Train")
stats(va_c, "Val  ")

# === C) Mismo video_id con labels distintos ===
print(f"\n=== C) Vídeos con labels mixtos ===")
def mixed(ds_samples, name):
    by_vid = defaultdict(set)
    for s in ds_samples:
        by_vid[s["video_id"]].add(s["label"])
    mix = [v for v, ls in by_vid.items() if len(ls) > 1]
    print(f"{name}: {len(mix)} vídeos con clips de ambas clases")
    return mix
mixed(train_ds.samples, "Train")
mixed(val_ds.samples,   "Val  ")

# === D) Geometría de los clips (diluir señal) ===
print(f"\n=== D) Geometría temporal de los clips ===")
def clip_stats(samples, name):
    lens, toa_pos = [], []
    n_after_toa = 0  # positivos cuyo end > toa (accidente ocurre dentro de la ventana)
    for s in samples:
        L = s["effective_end"] - s["start"] + 1
        lens.append(L)
        if s["label"] == 1 and L > 0:
            # posición relativa del TOA dentro del clip [0, 1]
            rel = (s["toa"] - s["start"]) / max(1, s["end"] - s["start"])
            toa_pos.append(rel)
            if s["start"] <= s["toa"] <= s["effective_end"]:
                n_after_toa += 1
    import numpy as np
    lens = np.array(lens)
    print(f"{name}: long efectiva  min={lens.min()} med={int(np.median(lens))} "
          f"max={lens.max()} mean={lens.mean():.1f}")
    if toa_pos:
        toa_pos = np.array(toa_pos)
        print(f"   TOA relativo (positivos): "
              f"mean={toa_pos.mean():.2f} median={np.median(toa_pos):.2f}  "
              f"(0=start, 1=end)")
        n_pos = sum(1 for s in samples if s["label"]==1)
        print(f"   Positivos con TOA dentro de [start, effective_end]: "
              f"{n_after_toa}/{n_pos}")
clip_stats(train_ds.samples, "Train")
clip_stats(val_ds.samples,   "Val  ")