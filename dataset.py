from pathlib import Path
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import tv_tensors


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

        self.samples = self._load_samples()

    def _load_samples(self):
        samples = []

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

                # Si estamos en anticipación, redefinimos el final efectivo
                if self.anticipation_mode and sample["label"] == 1:
                    effective_end = min(sample["end"], sample["toa"] - self.anticipation_offset)
                else:
                    effective_end = sample["end"]

                sample["effective_end"] = effective_end

                # Filtrar muestras inválidas
                if self.drop_invalid_samples and effective_end < sample["start"]:
                    continue

                samples.append(sample)

        return samples

    def __len__(self):
        return len(self.samples)

    def _sample_frame_indices(self, start, end, toa):
        total_frames = np.arange(start, end + 1)
        total_len = len(total_frames)

        if total_len <= 0:
            raise ValueError(f"Intervalo inválido: start={start}, end={end}")

        if self.num_frames is None or self.num_frames >= total_len:
            return total_frames

        # Muestreo base uniforme
        base_indices = np.linspace(start, end, self.num_frames)

        # Val/test: determinista, sin augmentation
        if not (self.train and self.use_temporal_augmentation):
            return base_indices.astype(int)

        indices = base_indices.copy()

        # FIX: orden corregido — primero jitter, luego TOA shift.
        # Antes el orden era el inverso: el jitter deshacía parcialmente
        # el guiado hacia el TOA. Ahora el TOA shift es la última palabra.

        # 1. Jitter temporal: añade variabilidad entre epochs
        if self.temporal_max_jitter > 0:
            jitter = np.random.randint(
                -self.temporal_max_jitter,
                self.temporal_max_jitter + 1,
                size=self.num_frames,
            )
            indices = indices + jitter

        # 2. TOA-guided shift: guía suavemente el centro hacia el momento del accidente.
        # Se aplica después del jitter para que sea la operación dominante.
        if self.use_toa_guided_sampling:
            current_center = indices.mean()
            toa_clamped = min(max(toa, start), end)
            desired_shift = toa_clamped - current_center
            shift = self.toa_center_strength * desired_shift
            indices = indices + shift

        indices = np.clip(indices, start, end)
        indices = np.sort(indices)
        indices = indices.astype(int)

        # FIX: eliminar duplicados — tras clip+sort pueden aparecer frames repetidos
        # (varios índices apuntando al mismo frame), especialmente en los bordes del clip.
        # Interpolamos para mantener siempre exactamente num_frames frames.
        indices = np.unique(indices)
        if len(indices) < self.num_frames:
            indices = np.round(np.interp(
                np.linspace(0, len(indices) - 1, self.num_frames),
                np.arange(len(indices)),
                indices,
            )).astype(int)

        # 3. Temporal reversal (p=0.08): invierte el orden del clip.
        # Obliga al modelo a no depender únicamente del orden temporal.
        # Probabilidad baja para no introducir demasiado ruido semántico
        # (un accidente al revés es poco natural).
        if np.random.random() < 0.08:
            indices = indices[::-1].copy()

        # 4. Frame dropout (p=0.10): reemplaza un frame aleatorio por el anterior.
        # Simula frames perdidos o corrompidos, habituales en dashcams.
        # Se excluye el frame 0 para no perder el contexto inicial.
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

        return clip, label