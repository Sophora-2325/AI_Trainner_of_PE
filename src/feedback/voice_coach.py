"""Voice coach — priority-queued TTS announcements for fitness training.

Wraps TTSEngine with:
  - Priority queue: EMERGENCY > WARNING > HINT > count > encouragement
  - Stage transition announcements (SETUP→DESCENT: "下蹲")
  - Rep counting (every 5 reps summary)
  - Anti-overlap throttling

Research proposal Method 4: "一级为提示级，APP发出语音提示'注意角度'；
二级为报警级，节点振动马达同时APP发出'动作错误，请调整'"
"""

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional

from src.feedback.tts_engine import TTSEngine


class MessagePriority(IntEnum):
    """Priority levels — higher value = higher priority."""
    ENCOURAGEMENT = 0
    REP_COUNT = 10
    STAGE_TRANSITION = 20
    HINT = 30       # InterventionLevel.HINT
    WARNING = 40    # InterventionLevel.WARNING
    EMERGENCY = 50  # InterventionLevel.EMERGENCY


@dataclass(order=True)
class VoiceMessage:
    """A queued voice message with priority."""
    priority: int
    text: str = field(compare=False)
    timestamp: float = field(default_factory=time.time, compare=False)


# ─── Stage → Chinese command mapping ─────────────────────────────

STAGE_NAMES = {
    "SETUP": "准备",
    "DESCENT": "下蹲",
    "BOTTOM": "底部",
    "ASCENT": "起身",
    "LOCKOUT": "锁定",
    "REST": "休息",
    "PULL": "拉起",
    "LOWER": "下放",
    "DEAD_HANG": "悬垂",
    "PULL_UP": "上拉",
    "TOP": "顶部",
    "HOLD": "保持",
}

CORRECTION_PHRASES = {
    "knee_valgus": "膝盖向外打开，对准脚尖方向",
    "heel_lift": "重心放在全脚掌，脚跟踩实地面",
    "back_rounding": "收紧核心，胸部挺起，保持脊柱中立",
    "insufficient_depth": "下蹲至大腿与地面平行或更低",
    "knee_over_toe": "臀部向后坐，重心放在脚后跟",
    "torso_too_upright": "适度前倾躯干",
    "asymmetry": "检查双脚对称，均匀分配体重",
    "hips_rising_first": "同步伸展髋膝，不要先抬臀部",
    "sagging_hips": "收紧核心和臀部，身体呈一条直线",
    "elbow_flare": "手肘贴近身体，约45度角",
}

ENCOURAGEMENT_PHRASES = [
    "动作标准，继续保持",
    "做得很好",
    "节奏控制得很好",
    "保持这个状态",
    "加油，再来一个",
]


class VoiceCoach:
    """Priority-queued voice coach for real-time fitness feedback.

    Usage:
        vc = VoiceCoach()
        vc.announce_stage("DESCENT")            # "下蹲"
        vc.announce_rep(5, 85)                  # "第5次，85分，良好"
        vc.announce_correction("knee_valgus")   # "膝盖向外打开"
        vc.run()  # starts background processing thread
    """

    def __init__(
        self,
        tts_engine: Optional[TTSEngine] = None,
        enabled: bool = True,
        rep_summary_interval: int = 5,
    ):
        self.tts = tts_engine or TTSEngine()
        self.enabled = enabled
        self.rep_summary_interval = rep_summary_interval

        self._queue: list[VoiceMessage] = []
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Debounce: minimum seconds between announcements
        self._last_spoke = 0.0
        self._min_interval = 1.5  # seconds

        # Recent messages dedup
        self._recent: deque = deque(maxlen=5)

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._process_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False

    # ─── Public API ───────────────────────────────────────────

    def announce_stage(self, stage: str):
        """Announce phase transition."""
        name = STAGE_NAMES.get(stage, stage)
        self._enqueue(MessagePriority.STAGE_TRANSITION, name)

    def announce_rep(self, rep_count: int, score: float = 0):
        """Announce rep count (only every N reps)."""
        if rep_count % self.rep_summary_interval != 0:
            return
        if score >= 90:
            level = "优秀"
        elif score >= 75:
            level = "良好"
        elif score >= 60:
            level = "需要注意"
        else:
            level = "需要改进"
        self._enqueue(MessagePriority.REP_COUNT,
                      f"第{rep_count}次，{score:.0f}分，{level}")

    def announce_correction(self, error_id: str):
        """Announce correction for a detected error."""
        phrase = CORRECTION_PHRASES.get(error_id, "注意动作姿势")
        self._enqueue(MessagePriority.WARNING, phrase)

    def announce_hint(self, message: str):
        """Announce a low-priority hint."""
        self._enqueue(MessagePriority.HINT, message)

    def announce_warning(self, message: str):
        """Announce a warning (e.g., safety threshold exceeded)."""
        self._enqueue(MessagePriority.WARNING, message)

    def announce_emergency(self, message: str):
        """Immediate emergency announcement — clears queue."""
        with self._lock:
            self._queue.clear()
        self._enqueue(MessagePriority.EMERGENCY, message)

    def announce_encouragement(self):
        """Random encouragement phrase."""
        import random
        self._enqueue(MessagePriority.ENCOURAGEMENT,
                      random.choice(ENCOURAGEMENT_PHRASES))

    # ─── Internal ─────────────────────────────────────────────

    def _enqueue(self, priority: MessagePriority, text: str):
        if not self.enabled or not text.strip():
            return

        # Dedup: skip if same text spoken recently
        if text in self._recent:
            return

        with self._lock:
            # Lower priority messages are dropped when a higher priority
            # message is queued (not already speaking)
            msg = VoiceMessage(priority=int(priority), text=text)
            # Remove same-priority duplicates
            self._queue = [m for m in self._queue if m.text != text]
            self._queue.append(msg)
            # Sort descending by priority
            self._queue.sort(key=lambda m: m.priority, reverse=True)

    def _process_loop(self):
        """Background thread: drain queue and speak."""
        while self._running:
            msg = None
            with self._lock:
                if self._queue:
                    # Skip if TTS is still speaking and message is not emergency
                    if self.tts.is_speaking:
                        top = self._queue[0]
                        if top.priority < MessagePriority.EMERGENCY:
                            time.sleep(0.1)
                            continue
                    msg = self._queue.pop(0)

            if msg is not None:
                now = time.time()
                if now - self._last_spoke >= self._min_interval or msg.priority >= MessagePriority.EMERGENCY:
                    self._recent.append(msg.text)
                    self.tts.speak(msg.text, block=True)
                    self._last_spoke = time.time()
            else:
                time.sleep(0.1)

    def set_enabled(self, enabled: bool):
        self.enabled = enabled
