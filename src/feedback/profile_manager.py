"""User profile manager — adaptive safety thresholds from training history.

Records per-user training data and recommends personalized safety thresholds
based on individual ROM statistics (mean ± 2σ of recent N sessions).

Research proposal Method 4: "用户可根据自身情况在APP中自定义各关节的安全阈值，
系统根据历史数据动态统计用户的日常活动范围，推荐个性化阈值"
"""

import json
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class SessionRecord:
    """One training session summary."""
    timestamp: float = 0.0
    movement: str = ""
    rep_count: int = 0
    avg_score: float = 0.0
    joint_rom: dict[str, dict[str, float]] = field(default_factory=dict)
    # joint_rom: {joint_name: {min, max, mean, std}}


@dataclass
class UserProfile:
    """Persistent user profile."""
    user_id: str = "default"
    created_at: float = 0.0
    sessions: list[SessionRecord] = field(default_factory=list)
    custom_thresholds: dict[str, dict[str, float]] = field(default_factory=dict)


class ProfileManager:
    """Manage user profiles and adaptive threshold recommendations.

    Usage:
        pm = ProfileManager("profiles/user_001.json")
        pm.record_session(session)
        thresholds = pm.recommend_thresholds("squat")
    """

    def __init__(self, profile_path: str = "profiles/default.json"):
        self.profile_path = profile_path
        self.profile: UserProfile = self._load()

    def _load(self) -> UserProfile:
        if os.path.exists(self.profile_path):
            try:
                with open(self.profile_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                sessions = [SessionRecord(**s) for s in data.get("sessions", [])]
                return UserProfile(
                    user_id=data.get("user_id", "default"),
                    created_at=data.get("created_at", time.time()),
                    sessions=sessions,
                    custom_thresholds=data.get("custom_thresholds", {}),
                )
            except Exception:
                pass
        return UserProfile(user_id="default", created_at=time.time())

    def save(self):
        os.makedirs(os.path.dirname(self.profile_path), exist_ok=True)
        with open(self.profile_path, "w", encoding="utf-8") as f:
            json.dump({
                "user_id": self.profile.user_id,
                "created_at": self.profile.created_at,
                "sessions": [{
                    "timestamp": s.timestamp,
                    "movement": s.movement,
                    "rep_count": s.rep_count,
                    "avg_score": s.avg_score,
                    "joint_rom": s.joint_rom,
                } for s in self.profile.sessions],
                "custom_thresholds": self.profile.custom_thresholds,
            }, f, ensure_ascii=False, indent=2)

    def record_session(self, session: SessionRecord):
        """Append a training session to history."""
        self.profile.sessions.append(session)
        # Keep only last 50 sessions
        if len(self.profile.sessions) > 50:
            self.profile.sessions = self.profile.sessions[-50:]
        self.save()

    def recommend_thresholds(
        self,
        movement: str,
        num_recent: int = 10,
        safety_factor: float = 0.8,
    ) -> dict[str, dict[str, float]]:
        """Recommend personalized safety thresholds for a movement.

        Computes per-joint ROM from recent N sessions of this movement,
        then applies safety_factor:
          - beginner (factor=0.8): conservative, narrow range
          - advanced (factor=0.95): close to natural ROM

        Args:
            movement: movement name (e.g. "squat")
            num_recent: look back N sessions
            safety_factor: 0.0~1.0, smaller = more conservative

        Returns:
            {joint_name: {min, max, recommended_min, recommended_max}}
        """
        relevant = [
            s for s in self.profile.sessions
            if s.movement == movement
        ][-num_recent:]

        if not relevant:
            return {}

        # Collect ROM data per joint
        joint_data: dict[str, list[float]] = defaultdict(list)
        for session in relevant:
            for joint_name, stats in session.joint_rom.items():
                joint_data[joint_name].extend([stats.get("min", 0), stats.get("max", 0)])

        thresholds = {}
        for joint_name, values in joint_data.items():
            if len(values) < 4:
                continue
            arr = np.array(values)
            mean = float(np.mean(arr))
            std = float(np.std(arr))

            # Safe range: mean ± 2σ, widened by safety_factor
            range_half = 2.0 * std / safety_factor
            thresholds[joint_name] = {
                "observed_mean": round(mean, 1),
                "observed_std": round(std, 1),
                "recommended_min": round(mean - range_half, 1),
                "recommended_max": round(mean + range_half, 1),
            }

        return thresholds

    def set_custom_threshold(self, movement: str, joint: str, min_val: float, max_val: float):
        """Manually override a threshold."""
        key = f"{movement}/{joint}"
        self.profile.custom_thresholds[key] = {"min": min_val, "max": max_val}
        self.save()

    def get_threshold(self, movement: str, joint: str) -> Optional[dict]:
        """Get threshold for a specific joint, custom overrides recommended."""
        custom = self.profile.custom_thresholds.get(f"{movement}/{joint}")
        if custom:
            return custom
        rec = self.recommend_thresholds(movement)
        return rec.get(joint)

    def get_summary(self) -> dict:
        """Get user summary statistics."""
        if not self.profile.sessions:
            return {"total_sessions": 0}
        scores = [s.avg_score for s in self.profile.sessions if s.avg_score > 0]
        reps = [s.rep_count for s in self.profile.sessions]
        movements = set(s.movement for s in self.profile.sessions)
        return {
            "total_sessions": len(self.profile.sessions),
            "avg_score": round(np.mean(scores), 1) if scores else 0,
            "total_reps": sum(reps),
            "movements_trained": list(movements),
            "last_session": self.profile.sessions[-1].timestamp if self.profile.sessions else 0,
        }
