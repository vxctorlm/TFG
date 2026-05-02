from pathlib import Path
from collections import defaultdict
import random
import statistics


def parse_line(line):
    line = line.rstrip("\n")
    left, text = line.split(",", 1)
    parts = left.split()
    return {
        "video_id": parts[0],
        "label": int(parts[1]),
        "start": int(parts[2]),
        "end": int(parts[3]),
        "toa": int(parts[4]),
        "text": text.strip(),
    }


def make_line(video_id, label, start, end, toa, text):
    return f"{video_id} {label} {start} {end} {toa},{text}"

def generate_windows_for_video(
    video_id, toa, text,
    window_len=64, stride=16,
    positive_horizon=20, gray_zone=80,
    negative_max_distance=180,
    max_neg_per_video=2, max_pos_per_video=2,
    seed=42,
):
    rng = random.Random(seed + hash(video_id) % 10_000_000)

    candidates_neg = []
    candidates_pos = []

    max_end = toa - 1
    if max_end < window_len:
        return []

    max_possible_distance = toa - window_len
    if max_possible_distance <= positive_horizon + 5:
        return []

    negative_min_distance = positive_horizon + gray_zone + 1

    # CAMBIO: si el vídeo es corto, reducir gray_zone para poder generar negativos
    # Mínimo gap real posible: toa - window_len (distancia máxima alcanzable)
    max_possible_distance = toa - window_len  # con start=1, end=window_len
    if max_possible_distance < negative_min_distance:
        # Reducir gray_zone al mínimo que permita al menos 1 negativo
        # Aseguramos que haya al menos 10 frames de margen entre zona gris y positivos
        effective_gray_zone = max(10, max_possible_distance - positive_horizon - 1)
        negative_min_distance = positive_horizon + effective_gray_zone + 1
    
    start = 1
    while True:
        end = start + window_len - 1
        if end > max_end:
            break

        distance = toa - end

        if 1 <= distance <= positive_horizon:
            candidates_pos.append(make_line(video_id, 1, start, end, toa, text))
        elif negative_min_distance <= distance <= negative_max_distance:
            candidates_neg.append(make_line(video_id, 0, start, end, toa, text))

        start += stride

    rng.shuffle(candidates_neg)
    rng.shuffle(candidates_pos)

    selected = candidates_neg[:max_neg_per_video] + candidates_pos[:max_pos_per_video]
    return selected

def describe_starts(lines):
    starts_pos, starts_neg = [], []
    for line in lines:
        s = parse_line(line)
        (starts_pos if s["label"] == 1 else starts_neg).append(s["start"])

    def describe(values, name):
        if not values:
            print(f"  {name}: sin muestras")
            return
        print(f"  {name}: n={len(values)}, min={min(values)}, max={max(values)}, "
              f"media={sum(values)/len(values):.1f}, mediana={statistics.median(values):.1f}")

    print("Distribución de start:")
    describe(starts_pos, "Positivos")
    describe(starts_neg, "Negativos")


def describe_distances(lines):
    dist_pos, dist_neg = [], []
    for line in lines:
        s = parse_line(line)
        d = s["toa"] - s["end"]
        (dist_pos if s["label"] == 1 else dist_neg).append(d)

    def describe(values, name):
        if not values:
            print(f"  {name}: sin muestras")
            return
        print(f"  {name}: n={len(values)}, min={min(values)}, max={max(values)}, "
              f"media={sum(values)/len(values):.1f}, mediana={statistics.median(values):.1f}")

    print("Distribución de distance = toa - end:")
    describe(dist_pos, "Positivos")
    describe(dist_neg, "Negativos (incluyendo absolutos con toa=0 → distance negativa)")


def describe_video_label_distribution(lines):
    vid_labels = defaultdict(set)
    for line in lines:
        s = parse_line(line)
        vid_labels[s["video_id"]].add(s["label"])

    only_pos = sum(1 for ls in vid_labels.values() if ls == {1})
    only_neg = sum(1 for ls in vid_labels.values() if ls == {0})
    both     = sum(1 for ls in vid_labels.values() if ls == {0, 1})
    print(f"Vídeos solo positivos:    {only_pos}")
    print(f"Vídeos solo negativos:    {only_neg}  ← negativos absolutos aquí")
    print(f"Vídeos con ambas clases:  {both}")


def main():
    input_txt  = Path("/data-fast/data-server/vlopezmo/model/training/training_full.txt")
    output_txt = Path("/data-fast/data-server/vlopezmo/model/training/training_balanced.txt")

    window_len            = 64
    stride                = 16
    positive_horizon      = 20
    gray_zone             = 80
    negative_max_distance = 180
    max_neg_per_video     = 2
    max_pos_per_video     = 2
    seed                  = 42

    video_to_samples = defaultdict(list)   # vídeos CON accidente (toa > 0)
    absolute_negatives = []                # vídeos SIN accidente (toa == 0)

    with open(input_txt, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            sample = parse_line(line)

            if sample["label"] == 1 and sample["toa"] > 0:
                # Vídeo con accidente → sliding window en generate_windows_for_video
                video_to_samples[sample["video_id"]].append(sample)

            elif sample["label"] == 0 and sample["toa"] == 0:
                # Negativo absoluto (vídeo sin accidente) → pasar directamente
                absolute_negatives.append(
                    make_line(sample["video_id"], 0,
                              sample["start"], sample["end"], 0, sample["text"])
                )

    new_lines = []
    skipped_videos = 0

    for video_id, samples in video_to_samples.items():
        sample = samples[0]
        lines = generate_windows_for_video(
            video_id=video_id,
            toa=sample["toa"],
            text=sample["text"],
            window_len=window_len,
            stride=stride,
            positive_horizon=positive_horizon,
            gray_zone=gray_zone,
            negative_max_distance=negative_max_distance,
            max_neg_per_video=max_neg_per_video,
            max_pos_per_video=max_pos_per_video,
            seed=seed,
        )
        if not lines:
            skipped_videos += 1
            continue

        # Si el vídeo no generó ningún negativo sliding,
        # usar ventanas del Excel original (start=1) como negativos de rescate
        has_neg = any(parse_line(l)["label"] == 0 for l in lines)
        if not has_neg:
            # El vídeo en training_full.txt tiene start=1, end=real_frames
            # Cogemos una ventana muy alejada del TOA: los primeros 64 frames
            toa = sample["toa"]
            if toa > window_len * 2:  # solo si hay margen suficiente
                rescue_end = window_len  # frame 64
                rescue_distance = toa - rescue_end
                # Solo añadir si está fuera de la gray zone
                if rescue_distance > positive_horizon + gray_zone:
                    lines.append(make_line(
                        video_id, 0, 1, window_len, toa, sample["text"]
                    ))

        new_lines.extend(lines)

    # Añadir negativos absolutos
    new_lines.extend(absolute_negatives)
    random.Random(seed).shuffle(new_lines)

    output_txt.parent.mkdir(parents=True, exist_ok=True)
    with open(output_txt, "w", encoding="utf-8") as f:
        for line in new_lines:
            f.write(line + "\n")

    num_pos = sum(1 for l in new_lines if parse_line(l)["label"] == 1)
    num_neg = len(new_lines) - num_pos

    print("Nuevo training generado.")
    print(f"Input:  {input_txt}")
    print(f"Output: {output_txt}\n")
    print(f"Vídeos con accidente leídos:         {len(video_to_samples)}")
    print(f"Vídeos saltados (sin ventana válida): {skipped_videos}")
    print(f"Negativos absolutos añadidos:         {len(absolute_negatives)}\n")
    print(f"Total muestras: {len(new_lines)}")
    print(f"Positivos: {num_pos}")
    print(f"Negativos: {num_neg}  "
          f"(sliding: {num_neg - len(absolute_negatives)} | absolutos: {len(absolute_negatives)})\n")
    print("Configuración:")
    print(f"  window_len={window_len}, stride={stride}")
    print(f"  positive_horizon={positive_horizon}, gray_zone={gray_zone}, "
          f"negative_max_distance={negative_max_distance}")
    print(f"  max_pos_per_video={max_pos_per_video}, max_neg_per_video={max_neg_per_video}\n")

    describe_starts(new_lines)
    print()
    describe_distances(new_lines)
    print()
    describe_video_label_distribution(new_lines)

    # --- DIAGNÓSTICO TOA vídeos solo-positivos ---
    import numpy as np
    print(f"\nTOA de vídeos solo-positivos tras fix:")
    vid_labels_diag2 = defaultdict(set)
    for line in new_lines:
        s = parse_line(line)
        vid_labels_diag2[s["video_id"]].add(s["label"])
    only_pos_after = sum(1 for ls in vid_labels_diag2.values() if ls == {1})
    print(f"Vídeos solo positivos: {only_pos_after}  (antes: 710)")


if __name__ == "__main__":
    main()