from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import tv_tensors


class AccidentClipDataset(Dataset):
    """
    Dataset para anticipación de accidentes.

    Regla central para positivos:
        el último frame visible debe ser estrictamente anterior al TOA.

    Si anticipation_mode=True:
        effective_end = min(end, toa - anticipation_offset) para label=1

    Esto evita que el modelo vea el accidente o frames posteriores.
    """

    def __init__(
        self,
        txt_path,
        rgb_root,
        num_frames: Optional[int] = 16,
        transform=None,
        train: bool = False,
        use_temporal_augmentation: bool = False,
        temporal_max_jitter: int = 1,
        use_toa_guided_sampling: bool = False,
        toa_center_strength: float = 0.0,
        anticipation_mode: bool = True,
        anticipation_offset: int = 10,
        drop_invalid_samples: bool = True,
        strict_anticipation: bool = True,
        frame_index_offset: int = 1,
    ):
        self.txt_path = Path(txt_path)
        self.rgb_root = Path(rgb_root)
        self.num_frames = num_frames
        self.transform = transform
        self.train = train

        self.use_temporal_augmentation = use_temporal_augmentation
        self.temporal_max_jitter = int(temporal_max_jitter)
        self.use_toa_guided_sampling = use_toa_guided_sampling
        self.toa_center_strength = float(toa_center_strength)

        self.anticipation_mode = anticipation_mode
        self.anticipation_offset = int(anticipation_offset)
        self.drop_invalid_samples = drop_invalid_samples
        self.strict_anticipation = strict_anticipation
        self.frame_index_offset = int(frame_index_offset)

        if self.anticipation_offset < 1:
            raise ValueError("anticipation_offset debe ser >= 1 para evitar ver el TOA.")
        if self.use_toa_guided_sampling:
            raise ValueError(
                "use_toa_guided_sampling=True no es recomendable para anticipación real: "
                "usa información futura del TOA para muestrear. Déjalo en False."
            )

        self.samples = self._load_samples()

    def _parse_line(self, line: str) -> Dict:
        line = line.strip()
        if not line:
            raise ValueError("Línea vacía")

        if "," in line:
            left, text = line.split(",", 1)
        else:
            left, text = line, ""

        parts = left.split()
        if len(parts) < 5:
            raise ValueError(f"Formato inválido en {self.txt_path}: {line}")

        return {
            "video_id": parts[0],
            "label": int(parts[1]),
            "start": int(parts[2]),
            "end": int(parts[3]),
            "toa": int(parts[4]),
            "text": text.strip(),
        }

    def _compute_effective_end(self, sample: Dict) -> int:
        if self.anticipation_mode and sample["label"] == 1:
            return min(sample["end"], sample["toa"] - self.anticipation_offset)
        return sample["end"]

    def _load_samples(self) -> List[Dict]:
        if not self.txt_path.exists():
            raise FileNotFoundError(f"No existe el fichero de anotaciones: {self.txt_path}")

        samples = []
        discarded_counts = {0: 0, 1: 0}
        temporal_leaks = []

        with open(self.txt_path, "r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                sample = self._parse_line(line)
                sample["line_number"] = line_number
                sample["effective_end"] = self._compute_effective_end(sample)
                sample["visible_len"] = sample["effective_end"] - sample["start"] + 1
                sample["toa_margin"] = sample["toa"] - sample["effective_end"]

                if self.drop_invalid_samples and sample["effective_end"] < sample["start"]:
                    discarded_counts[sample["label"]] += 1
                    continue

                if (
                    self.strict_anticipation
                    and self.anticipation_mode
                    and sample["label"] == 1
                    and sample["effective_end"] >= sample["toa"]
                ):
                    temporal_leaks.append(sample)
                    if self.drop_invalid_samples:
                        discarded_counts[sample["label"]] += 1
                        continue

                samples.append(sample)

        total_discarded = sum(discarded_counts.values())
        if total_discarded > 0:
            print(
                f"[AccidentClipDataset] descartadas={total_discarded} "
                f"(label=0: {discarded_counts[0]}, label=1: {discarded_counts[1]})"
            )

        if temporal_leaks and self.strict_anticipation and not self.drop_invalid_samples:
            examples = temporal_leaks[:5]
            msg = "\n".join(
                f"video={s['video_id']} start={s['start']} end={s['end']} "
                f"toa={s['toa']} effective_end={s['effective_end']}"
                for s in examples
            )
            raise ValueError(f"Fuga temporal en positivos:\n{msg}")

        if len(samples) == 0:
            raise ValueError(f"Dataset vacío después de filtrar: {self.txt_path}")

        return samples

    def __len__(self):
        return len(self.samples)

    def _sample_frame_indices(self, start: int, end: int) -> np.ndarray:
        total_len = end - start + 1
        if total_len <= 0:
            raise ValueError(f"Intervalo inválido: start={start}, end={end}")

        if self.num_frames is None:
            indices = np.arange(start, end + 1)
        else:
            indices = np.linspace(start, end, self.num_frames)

        if self.train and self.use_temporal_augmentation and self.num_frames is not None:
            if self.temporal_max_jitter > 0:
                jitter = np.random.randint(
                    -self.temporal_max_jitter,
                    self.temporal_max_jitter + 1,
                    size=self.num_frames,
                )
                indices = indices + jitter

            indices = np.clip(indices, start, end)
            indices = np.sort(indices)

            # Frame dropout ligero: realista y no usa información futura.
            if self.num_frames > 1 and np.random.random() < 0.10:
                drop_idx = np.random.randint(1, self.num_frames)
                indices[drop_idx] = indices[drop_idx - 1]

        return np.round(indices).astype(int)

    def _frame_path(self, video_id: str, frame_idx: int) -> Path:
        # frame_index_offset=1 asume anotaciones 0-indexed y ficheros 0001.png.
        filename = f"{frame_idx + self.frame_index_offset:04d}.png"
        return self.rgb_root / video_id / "images" / filename

    @staticmethod
    def _pil_to_uint8_chw(img: Image.Image) -> torch.Tensor:
        arr = np.array(img, dtype=np.uint8)
        return torch.from_numpy(arr).permute(2, 0, 1).contiguous()

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        frame_indices = self._sample_frame_indices(sample["start"], sample["effective_end"])

        frames = []
        for frame_idx in frame_indices:
            img_path = self._frame_path(sample["video_id"], int(frame_idx))
            if not img_path.exists():
                raise FileNotFoundError(f"No existe el frame: {img_path}")
            with Image.open(img_path) as img:
                img = img.convert("RGB")
                frames.append(self._pil_to_uint8_chw(img))

        clip = tv_tensors.Video(torch.stack(frames, dim=0))  # [T, C, H, W]
        if self.transform is not None:
            clip = self.transform(clip)

        label = torch.tensor(sample["label"], dtype=torch.long)
        return torch.as_tensor(clip), label



def audit_dataset(train_dataset: AccidentClipDataset, val_dataset: AccidentClipDataset, num_frames: int) -> None:
    """Auditoría dura para evitar experimentos inválidos de anticipación."""
    print("\n========== DATASET AUDIT ==========")

    train_videos = {s["video_id"] for s in train_dataset.samples}
    val_videos = {s["video_id"] for s in val_dataset.samples}
    overlap = train_videos & val_videos

    print(f"Train videos únicos: {len(train_videos)}")
    print(f"Val videos únicos:   {len(val_videos)}")
    print(f"Overlap train/val:   {len(overlap)}")

    if overlap:
        examples = list(sorted(overlap))[:20]
        raise RuntimeError(
            "Fuga por video_id entre train y val. "
            f"Ejemplos: {examples}. Genera splits por vídeo antes de entrenar."
        )

    for name, dataset in [("TRAIN", train_dataset), ("VAL", val_dataset)]:
        print(f"\n--- {name} ---")
        samples = dataset.samples

        for label in [0, 1]:
            ss = [s for s in samples if s["label"] == label]
            if not ss:
                raise RuntimeError(f"{name}: no hay muestras de label={label}")

            lengths = np.array([s["visible_len"] for s in ss])
            margins = np.array([s["toa_margin"] for s in ss])
            short = int(np.sum(lengths < num_frames))

            print(f"label={label}: n={len(ss)}")
            print(
                f"  visible_len: min={lengths.min()} median={np.median(lengths):.1f} max={lengths.max()}"
            )
            print(
                f"  toa-effective_end: min={margins.min()} median={np.median(margins):.1f} max={margins.max()}"
            )
            print(f"  clips < {num_frames} frames: {short}/{len(ss)} ({100*short/len(ss):.1f}%)")

        positive_leaks = [
            s for s in samples
            if s["label"] == 1 and s["effective_end"] >= s["toa"]
        ]
        if positive_leaks:
            examples = positive_leaks[:5]
            msg = "\n".join(
                f"video={s['video_id']} start={s['start']} end={s['end']} "
                f"toa={s['toa']} effective_end={s['effective_end']}"
                for s in examples
            )
            raise RuntimeError(f"Fuga temporal: positivos ven el accidente.\n{msg}")

    print("[OK] Sin overlap por vídeo y sin fuga temporal en positivos.")
    print("========== END DATASET AUDIT ==========" )
