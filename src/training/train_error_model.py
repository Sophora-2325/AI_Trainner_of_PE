"""动作分析模型训练脚本.

训练三个模型:
1. PhaseClassifier — 动作阶段分类器 (TCN)
2. ErrorDetectorModel — 错误检测模型 (LSTM + Attention)
3. QualityScorerModel — 动作质量评分模型 (Siamese LSTM)

在 WSL2 或 Windows 中运行:
    python -m src.training.train_error_model --data dataset/processed/train/
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Optional
import logging

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# 模型定义
# ═══════════════════════════════════════════════════════════════

class TemporalBlock(nn.Module):
    """TCN 的基本残差块."""
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.2):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.dropout = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv2.weight)
        if self.downsample is not None:
            nn.init.kaiming_normal_(self.downsample.weight)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.dropout(out)
        out = self.conv2(out)
        res = x if self.downsample is None else self.downsample(x)
        # 裁剪到相同长度
        out = out[:, :, :res.size(2)]
        return F.relu(out + res)


class PhaseClassifier(nn.Module):
    """动作阶段分类器 — TCN架构.

    Input:  (B, window_size, input_dim)  →  permute → (B, input_dim, window_size)
    Output: (B, num_phases)
    """

    def __init__(
        self,
        input_dim: int = 99,
        num_phases: int = 5,
        hidden_dim: int = 128,
        num_layers: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        layers = []
        for i in range(num_layers):
            dilation = 2 ** i
            layers.append(
                TemporalBlock(hidden_dim, hidden_dim, kernel_size, dilation, dropout)
            )
        self.tcn = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(hidden_dim, num_phases)

    def forward(self, x):
        # x: (B, T, D) → (B, D, T)
        x = self.input_proj(x).transpose(1, 2)
        x = self.tcn(x)
        x = self.pool(x).squeeze(-1)
        return self.classifier(x)


class ErrorDetectorModel(nn.Module):
    """错误检测模型 — LSTM + Multi-Head Attention.

    Input:  (B, window_size, input_dim) + (B, num_phases) phase嵌入
    Output: (B, num_error_types) 多标签分类
    """

    def __init__(
        self,
        input_dim: int = 99,
        num_error_types: int = 10,
        num_phases: int = 5,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.phase_embed = nn.Embedding(num_phases, 32)

        self.lstm = nn.LSTM(
            input_dim + 32, hidden_dim, num_layers,
            batch_first=True, bidirectional=True, dropout=dropout,
        )

        self.attention = nn.MultiheadAttention(
            hidden_dim * 2, num_heads, dropout=dropout, batch_first=True,
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_error_types),
        )

    def forward(self, x, phase_ids):
        # x: (B, T, D), phase_ids: (B,)
        B, T, D = x.shape
        phase_emb = self.phase_embed(phase_ids).unsqueeze(1).expand(-1, T, -1)
        x = torch.cat([x, phase_emb], dim=-1)

        lstm_out, _ = self.lstm(x)       # (B, T, 2*H)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        pooled = attn_out.mean(dim=1)     # (B, 2*H)

        return self.classifier(pooled)


class QualityScorerModel(nn.Module):
    """质量评分模型 — Siamese LSTM.

    将用户序列和标准序列编码后计算余弦相似度作为评分。
    """

    def __init__(
        self,
        input_dim: int = 99,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
    ):
        super().__init__()
        self.encoder = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, bidirectional=True,
        )
        self.projection = nn.Linear(hidden_dim * 2, embedding_dim)

    def encode(self, x):
        # x: (B, T, D) → embedding: (B, embedding_dim)
        out, (h_n, _) = self.encoder(x)
        # 使用最后层的 hidden state
        h = torch.cat([h_n[-2], h_n[-1]], dim=-1)  # (B, 2*H)
        return self.projection(h)

    def forward(self, user_seq, ref_seq):
        user_emb = F.normalize(self.encode(user_seq), dim=-1)
        ref_emb = F.normalize(self.encode(ref_seq), dim=-1)
        # 余弦相似度 → [0, 100]
        sim = F.cosine_similarity(user_emb, ref_emb, dim=-1)
        return (sim + 1) * 50.0  # 映射到 [0, 100]


# ═══════════════════════════════════════════════════════════════
# 训练函数
# ═══════════════════════════════════════════════════════════════

def train_phase_classifier(
    model: PhaseClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 100,
    lr: float = 1e-3,
    device: str = "cuda",
    save_path: str = "models/phase_classifier.pt",
):
    """训练动作阶段分类器."""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # 验证
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                pred = logits.argmax(dim=-1)
                correct += (pred == y).sum().item()
                total += y.size(0)

        val_acc = correct / total if total > 0 else 0
        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)

        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d} | Loss: {train_loss/len(train_loader):.4f} | Val Acc: {val_acc:.4f}")

    print(f"训练完成 | 最佳验证准确率: {best_val_acc:.4f}")
    return model


def train_error_detector(
    model: ErrorDetectorModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 100,
    lr: float = 1e-3,
    device: str = "cuda",
    save_path: str = "models/error_detector.pt",
):
    """训练错误检测模型（多标签分类）."""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    criterion = nn.BCEWithLogitsLoss()

    best_val_f1 = 0.0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            x, error_labels, phase_ids = [b.to(device) for b in batch]
            optimizer.zero_grad()
            logits = model(x, phase_ids)
            loss = criterion(logits, error_labels.float())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d} | Loss: {train_loss/len(train_loader):.4f}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"训练完成 | 模型保存至: {save_path}")
    return model


def train_quality_scorer(
    model: QualityScorerModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 100,
    lr: float = 1e-3,
    device: str = "cuda",
    save_path: str = "models/quality_scorer.pt",
):
    """训练动作质量评分模型."""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for user_seq, ref_seq, scores in train_loader:
            user_seq = user_seq.to(device)
            ref_seq = ref_seq.to(device)
            scores = scores.to(device)

            optimizer.zero_grad()
            pred = model(user_seq, ref_seq)
            loss = criterion(pred, scores)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d} | Loss: {train_loss/len(train_loader):.4f}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"训练完成 | 模型保存至: {save_path}")
    return model


# ═══════════════════════════════════════════════════════════════
# CLI入口
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练健身动作分析模型")
    parser.add_argument("--model", type=str, default="phase_classifier",
                        choices=["phase_classifier", "error_detector", "quality_scorer"])
    parser.add_argument("--data", type=str, required=True, help="训练数据目录")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save", type=str, default="models/")
    args = parser.parse_args()

    # 数据加载器（需根据实际数据格式调整）
    # train_loader, val_loader, test_loader = ...

    # 训练
    if args.model == "phase_classifier":
        model = PhaseClassifier(num_phases=5)
        # train_phase_classifier(model, train_loader, val_loader, ...)
    elif args.model == "error_detector":
        model = ErrorDetectorModel(num_error_types=10)
        # train_error_detector(model, train_loader, val_loader, ...)
    elif args.model == "quality_scorer":
        model = QualityScorerModel()
        # train_quality_scorer(model, train_loader, val_loader, ...)

    print(f"模型训练脚本就绪。请确保训练数据已放置在 {args.data}")
