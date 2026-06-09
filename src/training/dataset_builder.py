"""训练数据集构建器 — 将骨骼序列切分为固定长度窗口的PyTorch数据集."""

import os
import glob
import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, DataLoader


@dataclass
class DatasetConfig:
    """数据集构建配置."""
    window_size: int = 30         # 滑动窗口帧数
    stride: int = 5               # 窗口步长
    input_dim: int = 99           # 33 × 3 (只取xyz)
    num_phases: int = 5           # 阶段数
    train_split: float = 0.7
    val_split: float = 0.15
    augment: bool = True
    noise_std: float = 0.01


class SkeletonDataset(Dataset):
    """骨骼序列PyTorch数据集.

    将变长的骨骼序列切分为固定长度窗口。
    """

    def __init__(
        self,
        data_dir: str,
        config: DatasetConfig,
        phase: str = "train",    # train / val / test
        label_map: dict = None,
        file_list: list = None,
    ):
        """
        Args:
            data_dir: 包含 .npy 文件的目录
            config: 数据集配置
            phase: train/val/test
            label_map: 动作名称→数字标签映射
            file_list: 显式文件列表（优先级高于 data_dir glob）
        """
        self.config = config
        self.phase = phase

        # 加载所有 .npy 文件
        self._samples = []  # [(window, label, error_labels)]
        self._load_data(data_dir, label_map, file_list)

        # 数据增强
        self.augment = config.augment and phase == "train"

    def _load_data(self, data_dir: str, label_map: dict = None, file_list: list = None):
        """加载并切分数据."""
        if file_list is not None:
            npy_files = file_list
        else:
            npy_files = glob.glob(os.path.join(data_dir, "**/*.npy"), recursive=True)

        for f in npy_files:
            try:
                data = np.load(f, allow_pickle=True).item()
                landmarks = data["landmarks"]       # (T, 33, 4)
                label = data.get("label", "unknown")
                error_labels = data.get("error_labels", [])

                # 提取 xyz，展平为 (T, 99)
                features = landmarks[:, :, :3].reshape(landmarks.shape[0], -1)

                # 填充 NaN
                features = np.nan_to_num(features, nan=0.0)

                # 滑动窗口切分
                T = features.shape[0]
                for start in range(0, T - self.config.window_size + 1, self.config.stride):
                    window = features[start:start + self.config.window_size]
                    label_idx = label_map.get(label, 0) if label_map else 0
                    self._samples.append((window, label_idx, error_labels))

            except Exception as e:
                print(f"[DatasetBuilder] 加载失败 {f}: {e}")

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, idx):
        features, label, error_labels = self._samples[idx]
        x = torch.FloatTensor(features)  # (window_size, 99)

        # 数据增强
        if self.augment:
            x = self._augment(x)

        return x, label

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        """数据增强."""
        # 高斯噪声
        if self.config.noise_std > 0:
            noise = torch.randn_like(x) * self.config.noise_std
            x = x + noise

        # 随机缩放
        if np.random.random() < 0.3:
            scale = np.random.uniform(0.9, 1.1)
            x = x * scale

        # 随机时间反转（只对部分对称动作）
        if np.random.random() < 0.1:
            x = torch.flip(x, dims=[0])

        return x


class DatasetBuilder:
    """数据集构建器 — 负责整理和划分数据集."""

    def __init__(self, config: DatasetConfig = None):
        self.config = config or DatasetConfig()
        self.label_map: dict = {}
        self._reverse_label_map: dict = {}

    def build_label_map(self, data_dir: str):
        """从数据目录自动构建标签映射."""
        # 扫描子目录作为类别
        subdirs = [
            d for d in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, d))
        ]
        subdirs = sorted(subdirs)
        self.label_map = {name: i for i, name in enumerate(subdirs)}
        self._reverse_label_map = {i: name for name, i in self.label_map.items()}
        return self.label_map

    def create_dataloaders(
        self,
        data_dir: str,
        batch_size: int = 64,
        num_workers: int = 4,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """创建训练/验证/测试 DataLoader."""
        if not self.label_map:
            self.build_label_map(data_dir)

        # 收集所有样本并划分
        npy_files = glob.glob(os.path.join(data_dir, "**/*.npy"), recursive=True)
        np.random.shuffle(npy_files)

        n = len(npy_files)
        train_end = int(n * self.config.train_split)
        val_end = train_end + int(n * self.config.val_split)

        train_files = npy_files[:train_end]
        val_files = npy_files[train_end:val_end]
        test_files = npy_files[val_end:]

        # 为每个split创建独立的Dataset
        # 注意: SkeletonDataset 需要调整以接收文件列表
        # 这里简化处理，使用临时目录方式
        train_ds = self._build_from_files(train_files)
        val_ds = self._build_from_files(val_files)
        test_ds = self._build_from_files(test_files)

        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        )
        test_loader = DataLoader(
            test_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        )

        print(f"[DatasetBuilder] 训练:{len(train_ds)} 验证:{len(val_ds)} 测试:{len(test_ds)}")
        return train_loader, val_loader, test_loader

    def _build_from_files(self, file_list: list) -> Dataset:
        """从文件列表构建数据集."""
        return SkeletonDataset(
            "",
            self.config,
            phase="train",
            label_map=self.label_map,
            file_list=file_list,
        )

    def save(self, path: str):
        """保存标签映射."""
        np.save(path, {"label_map": self.label_map})

    def load(self, path: str):
        """加载标签映射."""
        data = np.load(path, allow_pickle=True).item()
        self.label_map = data["label_map"]
        self._reverse_label_map = {i: n for n, i in self.label_map.items()}
