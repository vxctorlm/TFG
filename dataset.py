from pathlib import Path
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import tv_tensors
import pandas as pd

def safe_int_minus_one(x):
    if pd.isna(x):
        return -100
    return int(x) - 1

class AccidentClipDataset(Dataset):
    def __init__(
        self,
        txt_path,
        rgb_root,
        num_frames=None,
        transform=None,
        train=False,
        use_temporal_augmentation=False,
        temporal_max_jitter=2,
        use_toa_guided_sampling=True,
        toa_center_strength=0.5,
        anticipation_mode=False,
        anticipation_offset=1,
        drop_invalid_samples=True,
        annotations_xlsx=None,
        use_aux_annotations=False,
    ):
        self.txt_path = Path(txt_path)
        self.rgb_root = Path(rgb_root)
        self.num_frames = num_frames
        self.transform = transform

        self.train = train
        self.use_temporal_augmentation = use_temporal_augmentation
        self.temporal_max_jitter = temporal_max_jitter
        self.use_toa_guided_sampling = use_toa_guided_sampling
        self.toa_center_strength = toa_center_strength

        self.anticipation_mode = anticipation_mode
        self.anticipation_offset = anticipation_offset
        self.drop_invalid_samples = drop_invalid_samples

        self.annotations_xlsx = annotations_xlsx
        self.use_aux_annotations = use_aux_annotations
        self.aux_by_video = self._load_aux_annotations() if use_aux_annotations else {}

        self.samples = self._load_samples()

    def _load_samples(self):
        samples = []
        # FIX: contamos descartados por label para detectar pérdidas silenciosas de positivos
        discarded_counts = {0: 0, 1: 0}

        with open(self.txt_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                left, text = line.split(",", 1)
                parts = left.split()

                sample = {
                    "video_id": parts[0],
                    "label": int(parts[1]),
                    "start": int(parts[2]),
                    "end": int(parts[3]),
                    "toa": int(parts[4]),
                    "text": text.strip(),
                }

                # Si esta en anticipación, redefine el final efectivo
                if self.anticipation_mode and sample["label"] == 1:
                    effective_end = min(sample["end"], sample["toa"] - self.anticipation_offset)
                else:
                    effective_end = sample["end"]

                sample["effective_end"] = effective_end

                # Filtrar muestras inválidas: ventana invertida
                if self.drop_invalid_samples and effective_end < sample["start"]:
                    discarded_counts[sample["label"]] += 1
                    continue

                # Garantiza end < TOA para muestras pre-TOA
                if sample["toa"] > 0 and sample["end"] >= sample["toa"]:
                    raise ValueError(
                        f"Muestra inválida en {self.txt_path.name}: "
                        f"video_id={sample['video_id']}, label={sample['label']}, "
                        f"start={sample['start']}, end={sample['end']}, toa={sample['toa']}. "
                        f"Se esperaba end < TOA (régimen estricto de anticipación)."
                    )

                if self.anticipation_mode and sample["toa"] > 0:
                    if effective_end >= sample["toa"] - self.anticipation_offset + 1:
                        raise ValueError(
                            f"effective_end={effective_end} no respeta anticipation_offset="
                            f"{self.anticipation_offset} para video_id={sample['video_id']} "
                            f"(toa={sample['toa']})."
                        )

                samples.append(sample)

        total_discarded = sum(discarded_counts.values())
        if total_discarded > 0:
            print(
                f"[AccidentClipDataset] Muestras descartadas (effective_end < start): "
                f"{total_discarded} total "
                f"(label=0: {discarded_counts[0]}, label=1: {discarded_counts[1]}) "
                f"← revisar si label=1 > 0 con anticipation_mode=True"
            )

        return samples

    def _normalize_video_id(self, video_id):
        """
        Convierte IDs tipo '001', '000001', '1', '1.mp4' a int cuando sea posible.
        """
        video_id = str(video_id)
        video_id = Path(video_id).stem

        digits = "".join(ch for ch in video_id if ch.isdigit())
        if digits == "":
            return video_id

        return int(digits)


    def _load_aux_annotations(self):
        if self.annotations_xlsx is None:
            return {}

        df = pd.read_excel(self.annotations_xlsx, sheet_name="Sheet1")

        col_weather = "weather(sunny,rainy,snowy,foggy)1-4"
        col_light = "light(day,night)1-2"
        col_scene = "scenes(highway,tunnel,mountain,urban,rural)1-5"
        col_linear = "linear(arterials,curve,intersection,T-junction,ramp) 1-5"

        aux_by_video = {}

        type_values = sorted(df["type"].dropna().astype(int).unique())
        self.type_to_idx = {v: i for i, v in enumerate(type_values)}
        print(f"[Aux type mapping] {len(self.type_to_idx)} tipos únicos: {type_values}")

        for _, row in df.iterrows():
            video_key = self._normalize_video_id(row["video"])

            aux_by_video[video_key] = {
                "weather": safe_int_minus_one(row[col_weather]),
                "light": safe_int_minus_one(row[col_light]),
                "scene": safe_int_minus_one(row[col_scene]),
                "linear": safe_int_minus_one(row[col_linear]),
                "type": self.type_to_idx[int(row["type"])],
            }

        print(f"[Aux annotations] Cargadas {len(aux_by_video)} anotaciones desde {self.annotations_xlsx}")
        return aux_by_video

    def __len__(self):
        return len(self.samples)

    def _sample_frame_indices(self, start, end, toa):
        total_frames = np.arange(start, end + 1)
        total_len = len(total_frames)

        if total_len <= 0:
            raise ValueError(f"Intervalo inválido: start={start}, end={end}")

        # Si hay menos frames disponibles que num_frames, samplear CON REPETICIÓN
        # para garantizar siempre la misma longitud de salida (necesario para batch).
        # FIX: usar np.round antes de astype(int) para evitar sesgo hacia el inicio
        # del clip que produce .astype(int) (trunca en lugar de redondear).
        if self.num_frames is not None and total_len < self.num_frames:
            base_indices = np.linspace(start, end, self.num_frames)
            return np.round(base_indices).astype(int)

        if self.num_frames is None:
            return total_frames

        base_indices = np.linspace(start, end, self.num_frames)

        # Val/test: determinista, sin augmentation
        if not (self.train and self.use_temporal_augmentation):
            return np.round(base_indices).astype(int)

        indices = base_indices.copy()

        # 1. Jitter temporal: añade variabilidad entre epochs
        if self.temporal_max_jitter > 0:
            jitter = np.random.randint(
                -self.temporal_max_jitter,
                self.temporal_max_jitter + 1,
                size=self.num_frames,
            )
            indices = indices + jitter

        # 2. TOA-guided shift: guía suavemente el centro hacia el momento del accidente.
        if self.use_toa_guided_sampling:
            current_center = indices.mean()
            toa_clamped = min(max(toa, start), end)
            desired_shift = toa_clamped - current_center
            shift = self.toa_center_strength * desired_shift
            indices = indices + shift

        indices = np.clip(indices, start, end)
        indices = np.sort(indices)
        indices = indices.astype(int)

        # Eliminar duplicados. tras clip+sort pueden aparecer frames repetidos
        indices = np.unique(indices)
        if len(indices) < self.num_frames:
            indices = np.round(np.interp(
                np.linspace(0, len(indices) - 1, self.num_frames),
                np.arange(len(indices)),
                indices,
            )).astype(int)

        # 3. Temporal reversal (p=0.08): invierte el orden del clip.
        """
        if np.random.random() < 0.08:
            indices = indices[::-1].copy()
        """

        # 4. Frame dropout (p=0.10): reemplaza un frame aleatorio por el anterior.
        if np.random.random() < 0.10:
            drop_idx = np.random.randint(1, self.num_frames)
            indices[drop_idx] = indices[drop_idx - 1]

        return indices

    def _frame_path(self, video_id, frame_idx):
        filename = f"{frame_idx + 1:04d}.png"
        return self.rgb_root / video_id / "images" / filename

    def _pil_to_uint8_chw(self, img):
        arr = np.array(img, dtype=np.uint8)  # [H, W, C]
        tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # [C, H, W]
        return tensor

    def __getitem__(self, idx):
        sample = self.samples[idx]

        frame_indices = self._sample_frame_indices(
            sample["start"],
            sample["effective_end"],
            sample["toa"],
        )

        frames = []

        for frame_idx in frame_indices:
            img_path = self._frame_path(sample["video_id"], frame_idx)

            if not img_path.exists():
                raise FileNotFoundError(f"No existe el frame: {img_path}")

            with Image.open(img_path) as img:
                img = img.convert("RGB")
                img = self._pil_to_uint8_chw(img)

            frames.append(img)

        clip = torch.stack(frames, dim=0)  # [T, C, H, W]
        clip = tv_tensors.Video(clip)

        if self.transform is not None:
            clip = self.transform(clip)

        clip = torch.as_tensor(clip)
        label = torch.tensor(sample["label"], dtype=torch.long)

        if self.use_aux_annotations:
            video_key = self._normalize_video_id(sample["video_id"])

            aux = self.aux_by_video.get(
                video_key,
                {
                    "weather": -100,
                    "light": -100,
                    "scene": -100,
                    "linear": -100,
                    "type": -100,
                },
            )

            aux = {
                k: torch.tensor(v, dtype=torch.long)
                for k, v in aux.items()
            }

            return clip, label, aux

        return clip, label

    