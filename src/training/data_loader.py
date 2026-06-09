"""PyTorch Dataset and DataLoader implementations for fitness action training.

Three dataset types matching the three model architectures:
  - PhaseDataset: returns (keypoints_window, phase_label)
  - ErrorDataset: returns (keypoints_window, error_labels, phase_id)
  - QualityDataset: returns (user_sequence, reference_sequence, score)
"""

import glob
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# ─── Data augmentation utilities ─────────────────────────────────

def _add_gaussian_noise(x: np.ndarray, std: float = 0.01) -> np.ndarray:
    noise = np.random.randn(*x.shape).astype(np.float32) * std
    return x + noise


def _time_warp(x: np.ndarray, max_warp: float = 0.1) -> np.ndarray:
    """Randomly stretch/compress the time axis."""
    T = x.shape[0]
    warp_factor = 1.0 + np.random.uniform(-max_warp, max_warp)
    new_T = max(1, int(T * warp_factor))
    indices = np.linspace(0, T - 1, new_T)
    warped = np.zeros_like(x)
    for i, idx in enumerate(indices):
        lo = int(np.floor(idx))
        hi = min(lo + 1, T - 1)
        frac = idx - lo
        warped[i] = x[lo] * (1 - frac) + x[hi] * frac
    return warped


def _random_dropout(x: np.ndarray, p: float = 0.05) -> np.ndarray:
    """Randomly zero out keypoints (simulate occlusion)."""
    mask = np.random.random(x.shape) > p
    return x * mask.astype(np.float32)


def _scale_jitter(x: np.ndarray, scale_range: tuple = (0.9, 1.1)) -> np.ndarray:
    scale = np.random.uniform(*scale_range)
    return x * scale


# ─── Phase Classification Dataset ────────────────────────────────

class PhaseDataset(Dataset):
    """Dataset for action phase classification (PhaseClassifier / TCN).

    Returns:
        x: (window_size, input_dim) keypoint window
        y: phase label index (0..num_phases-1)
    """

    def __init__(
        self,
        data_dir: str,
        window_size: int = 30,
        stride: int = 5,
        input_dim: int = 99,
        num_phases: int = 5,
        augment: bool = False,
        file_list: list[str] | None = None,
    ):
        self.window_size = window_size
        self.stride = stride
        self.input_dim = input_dim
        self.num_phases = num_phases
        self.augment = augment

        self._samples: list[tuple[np.ndarray, int]] = []
        self._load(data_dir, file_list)

    def _load(self, data_dir: str, file_list: list[str] | None):
        if file_list is None:
            file_list = glob.glob(os.path.join(data_dir, "**/*.npy"), recursive=True)

        for f in file_list:
            try:
                data = np.load(f, allow_pickle=True).item()
                landmarks = data["landmarks"]       # (T, 33, 4)
                phases = data.get("phases")          # (T,) phase per frame, or None
                label = data.get("label", "unknown")

                features = landmarks[:, :, :3].reshape(landmarks.shape[0], -1)
                features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

                T = features.shape[0]
                for start in range(0, T - self.window_size + 1, self.stride):
                    window = features[start:start + self.window_size].astype(np.float32)
                    # Aggregate phase label from the window
                    if phases is not None and len(phases) > start + self.window_size:
                        phase_window = phases[start:start + self.window_size]
                        p = int(np.argmax(np.bincount(phase_window)))  # majority vote
                    else:
                        p = 0  # default if no phase info
                    self._samples.append((window, p))
            except Exception as e:
                # Skip corrupt files silently in batch loading
                pass

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x, y = self._samples[idx]
        x = torch.from_numpy(x.copy())  # (T, D)

        if self.augment:
            x_np = x.numpy()
            if random.random() < 0.5:
                x_np = _add_gaussian_noise(x_np, 0.01)
            if random.random() < 0.3:
                x_np = _scale_jitter(x_np)
            if random.random() < 0.05:
                x_np = _random_dropout(x_np, 0.05)
            x = torch.from_numpy(x_np)

        return x, torch.tensor(y, dtype=torch.long)


# ─── Error Detection Dataset ─────────────────────────────────────

class ErrorDataset(Dataset):
    """Dataset for error detection (ErrorDetectorModel / LSTM+Attention).

    Returns:
        x: (window_size, input_dim) keypoint window
        error_labels: (num_error_types,) binary multi-label vector
        phase_id: (1,) phase label
    """

    def __init__(
        self,
        data_dir: str,
        window_size: int = 30,
        stride: int = 5,
        input_dim: int = 99,
        num_error_types: int = 10,
        augment: bool = False,
        file_list: list[str] | None = None,
    ):
        self.window_size = window_size
        self.stride = stride
        self.input_dim = input_dim
        self.num_error_types = num_error_types
        self.augment = augment

        self._samples: list[tuple[np.ndarray, np.ndarray, int]] = []
        self._load(data_dir, file_list)

    def _load(self, data_dir: str, file_list: list[str] | None):
        if file_list is None:
            file_list = glob.glob(os.path.join(data_dir, "**/*.npy"), recursive=True)

        for f in file_list:
            try:
                data = np.load(f, allow_pickle=True).item()
                landmarks = data["landmarks"]
                error_labels = data.get("error_labels", [])
                phase = data.get("phase", 0)

                features = landmarks[:, :, :3].reshape(landmarks.shape[0], -1)
                features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

                # Convert error label names/IDs to multi-hot vector
                error_vec = np.zeros(self.num_error_types, dtype=np.float32)
                if isinstance(error_labels, list):
                    for e in error_labels:
                        if isinstance(e, int) and 0 <= e < self.num_error_types:
                            error_vec[e] = 1.0
                        elif isinstance(e, str):
                            h = abs(hash(e)) % self.num_error_types
                            error_vec[h] = 1.0

                T = features.shape[0]
                for start in range(0, T - self.window_size + 1, self.stride):
                    window = features[start:start + self.window_size].astype(np.float32)
                    self._samples.append((window, error_vec, int(phase)))
            except Exception:
                pass

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, err, phase = self._samples[idx]
        x = torch.from_numpy(x.copy())
        if self.augment and random.random() < 0.5:
            x_np = x.numpy()
            x_np = _add_gaussian_noise(x_np, 0.01)
            x = torch.from_numpy(x_np)
        return x, torch.from_numpy(err), torch.tensor(phase, dtype=torch.long)


# ─── Quality Scoring Dataset ─────────────────────────────────────

class QualityDataset(Dataset):
    """Dataset for quality scoring (QualityScorerModel / Siamese LSTM).

    Returns:
        user_seq: (window_size, input_dim)
        ref_seq: (window_size, input_dim)
        score: (1,) float [0, 100]
    """

    def __init__(
        self,
        data_dir: str,
        reference_dir: str = "",
        window_size: int = 30,
        input_dim: int = 99,
        augment: bool = False,
        file_list: list[str] | None = None,
    ):
        self.window_size = window_size
        self.input_dim = input_dim
        self.augment = augment

        self._references: dict[str, np.ndarray] = {}
        if reference_dir:
            self._load_references(reference_dir)

        self._samples: list[tuple[np.ndarray, np.ndarray, float]] = []
        self._load(data_dir, file_list)

    def _load_references(self, ref_dir: str):
        """Load reference templates keyed by movement name."""
        for f in glob.glob(os.path.join(ref_dir, "*.npy")):
            try:
                data = np.load(f, allow_pickle=True)
                if isinstance(data, np.ndarray) and data.ndim == 2:
                    name = os.path.splitext(os.path.basename(f))[0]
                    self._references[name] = data.astype(np.float32)
            except Exception:
                pass

    def _load(self, data_dir: str, file_list: list[str] | None):
        if file_list is None:
            file_list = glob.glob(os.path.join(data_dir, "**/*.npy"), recursive=True)

        for f in file_list:
            try:
                data = np.load(f, allow_pickle=True).item()
                landmarks = data["landmarks"]
                label = data.get("label", "unknown")
                score = data.get("score", 50.0)

                features = landmarks[:, :, :3].reshape(landmarks.shape[0], -1)
                features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
                features = features.astype(np.float32)

                # Get reference for this movement
                ref = self._references.get(label, features)  # fallback to self
                if ref.ndim == 1:
                    ref = ref.reshape(-1, self.input_dim)

                # Align lengths
                min_len = min(features.shape[0], ref.shape[0], self.window_size)
                if min_len < self.window_size:
                    continue  # skip short sequences

                # Random crop to window_size
                user_start = random.randint(0, features.shape[0] - self.window_size)
                ref_start = random.randint(0, ref.shape[0] - self.window_size)
                user_win = features[user_start:user_start + self.window_size]
                ref_win = ref[ref_start:ref_start + self.window_size]

                self._samples.append((user_win, ref_win, float(score)))
            except Exception:
                pass

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        user, ref, score = self._samples[idx]
        user_t = torch.from_numpy(user.copy())
        ref_t = torch.from_numpy(ref.copy())
        if self.augment and random.random() < 0.5:
            user_np = user_t.numpy()
            user_np = _add_gaussian_noise(user_np, 0.01)
            user_t = torch.from_numpy(user_np)
        return user_t, ref_t, torch.tensor(score, dtype=torch.float32)


# ─── IMU Dataset (for 1D-CNN + LSTM model from research proposal) ──

class IMUDataset(Dataset):
    """Dataset for IMU-based action classification (research proposal Method 3).

    Input: (window_size, 6) — accel_xyz + gyro_xyz
    Output: movement class label

    This uses virtual IMU data generated from landmark sequences until
    physical IMU hardware is available.
    """

    def __init__(
        self,
        data_dir: str,
        window_size: int = 128,
        stride: int = 16,
        augment: bool = False,
        file_list: list[str] | None = None,
    ):
        self.window_size = window_size
        self.stride = stride
        self.augment = augment

        self._samples: list[tuple[np.ndarray, int]] = []
        self._load(data_dir, file_list)

    def _load(self, data_dir: str, file_list: list[str] | None):
        if file_list is None:
            file_list = glob.glob(os.path.join(data_dir, "**/*.npy"), recursive=True)

        for f in file_list:
            try:
                data = np.load(f, allow_pickle=True).item()
                imu_data = data.get("imu_data")
                label = data.get("label_idx", data.get("label", 0))

                if imu_data is None:
                    continue  # skip files without pre-computed IMU data

                # imu_data shape: (T, 6) — accel(3) + gyro(3)
                imu = np.nan_to_num(imu_data, nan=0.0).astype(np.float32)
                T = imu.shape[0]
                for start in range(0, T - self.window_size + 1, self.stride):
                    window = imu[start:start + self.window_size]
                    self._samples.append((window, int(label) if isinstance(label, (int, float)) else 0))
            except Exception:
                pass

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x, y = self._samples[idx]
        x = torch.from_numpy(x.copy())
        if self.augment and random.random() < 0.5:
            x_np = x.numpy()
            x_np = _add_gaussian_noise(x_np, 0.005)
            if random.random() < 0.3:
                x_np = _time_warp(x_np, 0.05)
            x = torch.from_numpy(x_np)
        return x, torch.tensor(y, dtype=torch.long)


# ─── DataLoader factory ──────────────────────────────────────────

def create_dataloaders(
    data_dir: str,
    dataset_type: str = "phase",
    batch_size: int = 64,
    num_workers: int = 4,
    train_split: float = 0.7,
    val_split: float = 0.15,
    window_size: int = 30,
    stride: int = 5,
    augment: bool = True,
    file_list: list[str] | None = None,
    reference_dir: str = "",
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test DataLoaders for a given dataset type.

    Args:
        data_dir: path to .npy data files
        dataset_type: "phase" | "error" | "quality" | "imu"
        batch_size: batch size
        num_workers: DataLoader workers
        train_split: fraction for training
        val_split: fraction for validation
        window_size: sliding window size
        stride: window stride
        augment: enable data augmentation (train split only)
        file_list: explicit file list (overrides data_dir glob)
        reference_dir: for quality dataset, reference templates dir

    Returns:
        (train_loader, val_loader, test_loader)
    """
    if file_list is None:
        file_list = sorted(glob.glob(os.path.join(data_dir, "**/*.npy"), recursive=True))

    random.shuffle(file_list)
    n = len(file_list)
    train_end = int(n * train_split)
    val_end = train_end + int(n * val_split)

    train_files = file_list[:train_end]
    val_files = file_list[train_end:val_end]
    test_files = file_list[val_end:]

    ds_cls = {
        "phase": PhaseDataset,
        "error": ErrorDataset,
        "quality": QualityDataset,
        "imu": IMUDataset,
    }.get(dataset_type)

    if ds_cls is None:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    common_kwargs = dict(
        window_size=window_size, stride=stride, file_list=None
    )

    if dataset_type == "quality":
        common_kwargs["reference_dir"] = reference_dir

    train_ds = ds_cls(data_dir=data_dir, augment=augment, file_list=train_files, **common_kwargs)  # type: ignore[arg-type]
    val_ds = ds_cls(data_dir=data_dir, augment=False, file_list=val_files, **common_kwargs)  # type: ignore[arg-type]
    test_ds = ds_cls(data_dir=data_dir, augment=False, file_list=test_files, **common_kwargs)  # type: ignore[arg-type]

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    print(f"[DataLoader] {dataset_type} | train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")
    return train_loader, val_loader, test_loader
