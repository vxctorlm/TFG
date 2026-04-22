from pathlib import Path
from collections import defaultdict
import random
import statistics


def parse_line(line):
    """
    Formato esperado:
    video_id label start end toa,text
    """
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
    video_id,
    toa,
    text,
    window_len=64,
    stride=16,
    positive_horizon=30,
    gray_zone=30,
    negative_max_distance=120,
    max_neg_per_video=2,
    max_pos_per_video=2,
    seed=42,
):
    """
    Genera ventanas antes del accidente.

    Definición:
      - label = 1 si distance = toa - end está entre 1 y positive_horizon.
      - se descartan ventanas con distance entre positive_horizon+1 y
        positive_horizon+gray_zone.
      - label = 0 si distance está entre positive_horizon+gray_zone+1
        y negative_max_distance.

    Ejemplo:
      positive_horizon = 30
      gray_zone = 30
      negative_max_distance = 120

      positivo:   distance 1-30
      descartado: distance 31-60
      negativo:   distance 61-120

    Esto evita que positivos y negativos sean ventanas casi idénticas.
    """

    rng = random.Random(seed + hash(video_id) % 10_000_000)

    candidates_neg = []
    candidates_pos = []

    max_end = toa - 1

    if max_end < window_len:
        return []

    negative_min_distance = positive_horizon + gray_zone + 1

    start = 1

    while True:
        end = start + window_len - 1

        if end > max_end:
            break

        distance = toa - end

        if 1 <= distance <= positive_horizon:
            candidates_pos.append(
                make_line(video_id, 1, start, end, toa, text)
            )

        elif negative_min_distance <= distance <= negative_max_distance:
            candidates_neg.append(
                make_line(video_id, 0, start, end, toa, text)
            )

        start += stride

    rng.shuffle(candidates_neg)
    rng.shuffle(candidates_pos)

    selected_neg = candidates_neg[:max_neg_per_video]
    selected_pos = candidates_pos[:max_pos_per_video]

    return selected_neg + selected_pos


def describe_starts(lines):
    starts_pos = []
    starts_neg = []

    for line in lines:
        sample = parse_line(line)
        if sample["label"] == 1:
            starts_pos.append(sample["start"])
        else:
            starts_neg.append(sample["start"])

    def describe(values):
        if not values:
            return "sin muestras"

        return (
            f"n={len(values)}, "
            f"min={min(values)}, "
            f"max={max(values)}, "
            f"media={sum(values) / len(values):.2f}, "
            f"mediana={statistics.median(values):.2f}"
        )

    print("Distribución de start:")
    print(f"  Positivos: {describe(starts_pos)}")
    print(f"  Negativos: {describe(starts_neg)}")


def describe_distances(lines):
    distances_pos = []
    distances_neg = []

    for line in lines:
        sample = parse_line(line)
        distance = sample["toa"] - sample["end"]

        if sample["label"] == 1:
            distances_pos.append(distance)
        else:
            distances_neg.append(distance)

    def describe(values):
        if not values:
            return "sin muestras"

        return (
            f"n={len(values)}, "
            f"min={min(values)}, "
            f"max={max(values)}, "
            f"media={sum(values) / len(values):.2f}, "
            f"mediana={statistics.median(values):.2f}"
        )

    print("Distribución de distance = toa - end:")
    print(f"  Positivos: {describe(distances_pos)}")
    print(f"  Negativos: {describe(distances_neg)}")


def main():
    input_txt = Path("/data-fast/data-server/vlopezmo/model/training/training_full.txt")
    output_txt = Path("/data-fast/data-server/vlopezmo/model/training/training_balanced.txt")

    # Parámetros principales
    window_len = 64
    stride = 16

    # label 1: accidente en los próximos 30 frames
    positive_horizon = 30

    # distance 31-60 se descarta como zona gris
    gray_zone = 30

    # label 0: accidente entre 61 y 120 frames después del clip
    negative_max_distance = 120

    # Máximo por clase y vídeo
    max_neg_per_video = 2
    max_pos_per_video = 2

    seed = 42

    video_to_samples = defaultdict(list)

    with open(input_txt, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            sample = parse_line(line)

            # Ignoramos negativos antiguos, especialmente los de toa=-1/start=1.
            if sample["label"] == 1 and sample["toa"] > 0:
                video_to_samples[sample["video_id"]].append(sample)

    new_lines = []
    skipped_videos = 0

    for video_id, samples in video_to_samples.items():
        # Normalmente hay una anotación positiva por vídeo.
        # Si hay varias, cogemos la primera para evitar duplicados.
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

        new_lines.extend(lines)

    random.Random(seed).shuffle(new_lines)

    output_txt.parent.mkdir(parents=True, exist_ok=True)

    with open(output_txt, "w", encoding="utf-8") as f:
        for line in new_lines:
            f.write(line + "\n")

    num_pos = 0
    num_neg = 0

    for line in new_lines:
        sample = parse_line(line)
        if sample["label"] == 1:
            num_pos += 1
        else:
            num_neg += 1

    print("Nuevo training generado.")
    print(f"Input:  {input_txt}")
    print(f"Output: {output_txt}")
    print()
    print(f"Vídeos positivos leídos: {len(video_to_samples)}")
    print(f"Vídeos saltados por no tener ventana válida: {skipped_videos}")
    print()
    print(f"Total muestras: {len(new_lines)}")
    print(f"Positivos: {num_pos}")
    print(f"Negativos: {num_neg}")
    print()

    print("Configuración:")
    print(f"  window_len = {window_len}")
    print(f"  stride = {stride}")
    print(f"  positive_horizon = {positive_horizon}")
    print(f"  gray_zone = {gray_zone}")
    print(f"  negative_max_distance = {negative_max_distance}")
    print(f"  max_pos_per_video = {max_pos_per_video}")
    print(f"  max_neg_per_video = {max_neg_per_video}")
    print()
    print(f"  positivos:   distance 1-{positive_horizon}")
    print(f"  descartados: distance {positive_horizon + 1}-{positive_horizon + gray_zone}")
    print(
        f"  negativos:   distance "
        f"{positive_horizon + gray_zone + 1}-{negative_max_distance}"
    )
    print()

    describe_starts(new_lines)
    print()
    describe_distances(new_lines)


if __name__ == "__main__":
    main()