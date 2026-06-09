"""评分历史记录与折线图 — 第6周功能.
记录每次练习的评分历史，绘制折线图 (matplotlib).

运行方式:
  python scripts/score_history.py                    # 显示历史折线图
  python scripts/score_history.py --record score.json  # 记录一次评分
  python scripts/score_history.py --export report.png  # 导出图表为图片
"""

import json
import os
import sys
import argparse
from datetime import datetime
from collections import defaultdict
from typing import Optional


HISTORY_FILE = "score_history.json"


def load_history(filepath: str = HISTORY_FILE) -> list[dict]:
    """加载评分历史."""
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_history(history: list[dict], filepath: str = HISTORY_FILE):
    """保存评分历史."""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def record_score(movement: str, score: int, suggestions: list[str] = None,
                 filepath: str = HISTORY_FILE):
    """记录一次练习评分.

    Args:
        movement: 动作名称
        score: 评分 0-10
        suggestions: 改进建议列表
        filepath: 历史文件路径
    """
    history = load_history(filepath)

    record = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "timestamp": datetime.now().timestamp(),
        "movement": movement,
        "score": score,
        "suggestions": suggestions or [],
    }
    history.append(record)
    save_history(history, filepath)

    print(f"[评分记录] {movement}: {score}/10 分 | 历史共 {len(history)} 条")
    return record


def plot_history(history: list[dict] = None, output_path: str = None,
                 filepath: str = HISTORY_FILE):
    """绘制评分历史折线图.

    Args:
        history: 历史数据 (None 则从文件加载)
        output_path: 输出图片路径 (None 则显示窗口)
        filepath: 历史文件路径
    """
    try:
        import matplotlib
        matplotlib.use("TkAgg" if output_path is None else "Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[错误] matplotlib 未安装, 请运行: pip install matplotlib")
        return None

    if history is None:
        history = load_history(filepath)

    if not history:
        print("[评分历史] 暂无记录")
        # 仍显示空图表以提示用户
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.text(0.5, 0.5, "暂无练习记录\n请先进行一次动作评分",
                ha="center", va="center", fontsize=14, color="gray",
                transform=ax.transAxes)
        ax.set_title("评分历史 (空)")
        if output_path:
            plt.savefig(output_path, dpi=100, bbox_inches="tight")
            print(f"[图表] 已保存: {output_path}")
        else:
            plt.show()
        return None

    # 按动作分组
    by_movement = defaultdict(list)
    for r in history:
        by_movement[r["movement"]].append(r)

    # 颜色映射
    colors = {"squat": "#4dc9f6", "pushup": "#ff6b6b", "deadlift": "#ffd93d"}
    default_colors = ["#4ecdc4", "#ff6b6b", "#ffd93d", "#a29bfe", "#fd79a8"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8),
                                     gridspec_kw={"height_ratios": [2, 1]})
    fig.suptitle("AI 健身教练 — 评分历史", fontsize=15, fontweight="bold")

    # ─── 上子图: 各动作评分折线 ──────────────────────
    for ci, (movement, records) in enumerate(by_movement.items()):
        records.sort(key=lambda r: r["timestamp"])
        x = list(range(1, len(records) + 1))
        y = [r["score"] for r in records]
        color = colors.get(movement, default_colors[ci % len(default_colors)])
        ax1.plot(x, y, "o-", color=color, linewidth=2, markersize=6,
                 label=f"{_movement_name(movement)} (平均 {sum(y)/len(y):.1f})")

    ax1.axhline(y=5, color="red", linestyle="--", alpha=0.4, linewidth=1,
                label="及格线 (5分)")
    ax1.axhline(y=8, color="green", linestyle="--", alpha=0.4, linewidth=1,
                label="优秀线 (8分)")
    ax1.set_ylabel("评分 (0-10)")
    ax1.set_ylim(0, 10.5)
    ax1.legend(loc="best", fontsize=8)
    ax1.grid(True, alpha=0.3)

    # ─── 下子图: 练习次数柱状图 ──────────────────────
    movements = list(by_movement.keys())
    counts = [len(by_movement[m]) for m in movements]
    bar_colors = [colors.get(m, default_colors[i % len(default_colors)])
                  for i, m in enumerate(movements)]
    bars = ax2.bar(range(len(movements)), counts, color=bar_colors)
    ax2.set_xticks(range(len(movements)))
    ax2.set_xticklabels([_movement_name(m) for m in movements], fontsize=9)
    ax2.set_ylabel("练习次数")
    ax2.set_title("各动作练习次数统计")
    for bar, count in zip(bars, counts):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 str(count), ha="center", fontsize=9)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=120, bbox_inches="tight")
        print(f"[图表] 已导出: {output_path}")
    else:
        plt.show()

    return fig


def _movement_name(m: str) -> str:
    names = {"squat": "深蹲", "pushup": "俯卧撑", "deadlift": "硬拉",
             "pullup": "引体向上", "plank": "平板支撑"}
    return names.get(m, m)


# ─── CLI ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="评分历史记录与图表")
    parser.add_argument("--record", "-r", type=str, default=None,
                        help="记录一次评分 (JSON 文件路径, 来自 compare_with_lm.py 输出)")
    parser.add_argument("--movement", "-m", type=str, default="squat",
                        help="动作名称 (--record 时使用)")
    parser.add_argument("--export", "-e", type=str, default=None,
                        help="导出图表为图片 (默认显示窗口)")
    parser.add_argument("--history-file", default=HISTORY_FILE,
                        help=f"历史数据文件 (默认: {HISTORY_FILE})")
    args = parser.parse_args()

    if args.record:
        # 从 compare_with_lm.py 的输出 JSON 中读取分数
        with open(args.record, "r", encoding="utf-8") as f:
            data = json.load(f)
        score = data.get("score", 0)
        suggestions = data.get("suggestions", [])
        record_score(args.movement, score, suggestions, args.history_file)

    # 始终显示/导出图表
    history = load_history(args.history_file)
    plot_history(history, args.export, args.history_file)


if __name__ == "__main__":
    main()
