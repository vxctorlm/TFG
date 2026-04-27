from pathlib import Path
from collections import defaultdict
import random


def main():
    input_txt = Path("/data-fast/data-server/vlopezmo/model/training/training_balanced.txt")
    output_train_txt = Path("/data-fast/data-server/vlopezmo/model/training/training_train.txt")
    output_val_txt = Path("/data-fast/data-server/vlopezmo/model/training/training_val.txt")
    val_ratio = 0.2
    seed = 42

    # -----------------------------
    # Leer líneas y agrupar por vídeo
    # -----------------------------
    video_to_lines = defaultdict(list)

    with open(input_txt, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue

            # formato: video_id label start end toa,text
            left, _ = line.split(",", 1)
            parts = left.split()
            video_id = parts[0]

            video_to_lines[video_id].append(line)

    # -----------------------------
    # Lista de vídeos únicos
    # -----------------------------
    video_ids = list(video_to_lines.keys())
    random.Random(seed).shuffle(video_ids)

    num_videos = len(video_ids)
    num_val_videos = max(1, int(num_videos * val_ratio))

    val_video_ids = set(video_ids[:num_val_videos])
    train_video_ids = set(video_ids[num_val_videos:])

    # -----------------------------
    # Construir líneas de train y val
    # -----------------------------
    train_lines = []
    val_lines = []

    for video_id, lines in video_to_lines.items():
        if video_id in val_video_ids:
            val_lines.extend(lines)
        else:
            train_lines.extend(lines)

    # -----------------------------
    # Guardar archivos
    # -----------------------------
    with open(output_train_txt, "w", encoding="utf-8") as f:
        for line in train_lines:
            f.write(line + "\n")

    with open(output_val_txt, "w", encoding="utf-8") as f:
        for line in val_lines:
            f.write(line + "\n")

    # -----------------------------
    # Resumen
    # -----------------------------
    print("Split completado.")
    print(f"Archivo original: {input_txt}")
    print(f"Vídeos totales: {num_videos}")
    print(f"Vídeos train: {len(train_video_ids)}")
    print(f"Vídeos val: {len(val_video_ids)}")
    print(f"Muestras train: {len(train_lines)}")
    print(f"Muestras val: {len(val_lines)}")
    print(f"Guardado train en: {output_train_txt}")
    print(f"Guardado val en: {output_val_txt}")


if __name__ == "__main__":
    main()