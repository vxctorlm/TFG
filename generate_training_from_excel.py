import pandas as pd
from pathlib import Path
from collections import defaultdict
import random


def main():
    excel_path = Path("/data-fast/data-server/vlopezmo/DADA2000/dada_text_annotations.xlsx")
    output_path = Path("/data-fast/data-server/vlopezmo/model/training/training_full.txt")
    dada_root = Path("/data-fast/data-server/vlopezmo/DADA2000")

    # Cuántas ventanas aleatorias samplear por vídeo sin accidente
    max_windows_per_no_acc_video = 2
    window_len = 64
    seed = 42
    rng = random.Random(seed)

    df = pd.read_excel(excel_path, sheet_name="Sheet1", header=0)

    # Construir video_id: índice dentro del tipo
    df = df.sort_values(["type", "video"]).reset_index(drop=True)
    df["vid_num"] = df.groupby("type").cumcount() + 1
    df["video_id"] = df["type"].astype(str) + "/" + df["vid_num"].apply(lambda x: f"{x:03d}")

    # Limpiar texto
    df["texts"] = df["texts"].fillna("").str.replace(r"\[CLS\]|\[SEP\]", "", regex=True).str.strip()

    acc_col = "whether an accident occurred (1/0)"
    toa_col = "accident frame"

    lines = []
    skipped = 0
    skipped_no_disk = 0

    # === 1. Vídeos CON accidente (label=1, igual que antes) ===
    valid = df[(df[acc_col] == 1) & (df[toa_col] > 0)].copy()
    print(f"Vídeos con accidente y toa>0: {len(valid)}")

    for _, row in valid.iterrows():
        video_id = row["video_id"]
        toa = int(row[toa_col])
        text = row["texts"] if row["texts"] else "accident"

        img_dir = dada_root / video_id / "images"
        if not img_dir.exists():
            skipped_no_disk += 1
            continue

        real_frames = len(list(img_dir.glob("*.png")))
        if real_frames == 0:
            skipped_no_disk += 1
            continue

        if toa > real_frames:
            skipped += 1
            continue

        # Igual que antes: una línea con start=1, end=real_frames, toa=toa
        # make_balanced_training_txt.py generará las ventanas sliding
        line = f"{video_id} 1 1 {real_frames} {toa},{text}"
        lines.append(line)

    print(f"  Vídeos con accidente en disco: {len(lines)}")
    print(f"  Skipped (no disco): {skipped_no_disk} | toa inválido: {skipped}")

    # === 2. Vídeos SIN accidente (label=0, negativos absolutos) ===
    no_acc = df[df[acc_col] == 0].copy()
    print(f"\nVídeos sin accidente en Excel: {len(no_acc)}")

    no_acc_added = 0
    no_acc_skipped = 0

    for _, row in no_acc.iterrows():
        video_id = row["video_id"]
        text = row["texts"] if row["texts"] else "no accident"

        img_dir = dada_root / video_id / "images"
        if not img_dir.exists():
            no_acc_skipped += 1
            continue

        real_frames = len(list(img_dir.glob("*.png")))
        if real_frames < window_len:
            no_acc_skipped += 1
            continue

        # Samplear hasta max_windows_per_no_acc_video ventanas aleatorias
        # toa=0 indica "sin accidente" — make_balanced_training_txt.py lo detectará
        max_start = real_frames - window_len  # último start válido
        n_windows = min(max_windows_per_no_acc_video, max(1, max_start // window_len))
        possible_starts = list(range(1, max_start + 1, window_len))
        selected_starts = rng.sample(possible_starts, min(n_windows, len(possible_starts)))

        for start in selected_starts:
            end = start + window_len - 1
            line = f"{video_id} 0 {start} {end} 0,{text}"
            lines.append(line)
            no_acc_added += 1

    print(f"  Vídeos sin accidente añadidos (ventanas): {no_acc_added}")
    print(f"  Vídeos sin accidente skipped: {no_acc_skipped}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")

    print(f"\nTotal líneas en training_full.txt: {len(lines)}")
    print(f"Guardado en: {output_path}")
    print("Ahora ejecuta make_balanced_training_txt.py")


if __name__ == "__main__":
    main()