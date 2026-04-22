"""
Genera training.txt desde dada_text_annotations.xlsx.

Formato de salida (igual que el training.txt original):
  video_id label start end toa,text

Solo incluye vídeos con accidente (whether an accident occurred == 1)
y con toa > 0.
"""
import pandas as pd
from pathlib import Path
from collections import defaultdict


def main():
    excel_path = Path("/data-fast/data-server/vlopezmo/DADA2000/dada_text_annotations.xlsx")
    output_path = Path("/data-fast/data-server/vlopezmo/model/training/training_full.txt")
    dada_root = Path("/data-fast/data-server/vlopezmo/DADA2000")

    df = pd.read_excel(excel_path, sheet_name="Sheet1", header=0)

    # Construir video_id: índice dentro del tipo (orden de aparición en Excel)
    df = df.sort_values(["type", "video"]).reset_index(drop=True)
    df["vid_num"] = df.groupby("type").cumcount() + 1
    df["video_id"] = df["type"].astype(str) + "/" + df["vid_num"].apply(lambda x: f"{x:03d}")

    # Limpiar texto
    df["texts"] = df["texts"].fillna("").str.replace(r"\[CLS\]|\[SEP\]", "", regex=True).str.strip()

    acc_col = "whether an accident occurred (1/0)"
    toa_col = "accident frame"

    # Solo vídeos con accidente y toa válido
    valid = df[(df[acc_col] == 1) & (df[toa_col] > 0)].copy()
    print(f"Vídeos con accidente y toa>0: {len(valid)}")

    # Verificar que existen en disco
    lines = []
    skipped = 0
    skipped_no_disk = 0

    for _, row in valid.iterrows():
        video_id = row["video_id"]
        toa = int(row[toa_col])
        text = row["texts"] if row["texts"] else "accident"
        total_frames = int(row["total frames"])

        img_dir = dada_root / video_id / "images"
        if not img_dir.exists():
            skipped_no_disk += 1
            continue

        # Usar frames reales del disco, no total_frames del Excel
        real_frames = len(list(img_dir.glob("*.png")))
        if real_frames == 0:
            skipped_no_disk += 1
            continue

        # Si toa > real_frames, el toa no es válido
        if toa > real_frames:
            skipped += 1
            continue

        line = f"{video_id} 1 1 {real_frames} {toa},{text}"
        lines.append(line)

    print(f"Vídeos encontrados en disco: {len(lines)}")
    print(f"Vídeos no encontrados en disco: {skipped_no_disk}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")

    print(f"\nGuardado en: {output_path}")
    print("Ahora ejecuta make_balanced_training_txt.py apuntando a training_full.txt")


if __name__ == "__main__":
    main()