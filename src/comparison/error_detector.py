"""错误模式检测器 — 基于规则引擎识别常见健身动作错误."""

from dataclasses import dataclass
from typing import Optional
import numpy as np
from src.pose.tracker import Phase


@dataclass
class MovementError:
    """动作错误."""
    id: str
    name: str
    severity: str             # high / medium / low
    advice: str
    detail: str = ""          # 额外描述
    current_value: float = 0.0
    threshold: float = 0.0


class ErrorDetector:
    """基于预定义规则的动作错误检测器.

    混合策略:
    1. 预定义阈值规则 (config/movements.yaml) — 覆盖已知错误
    2. 训练的分类模型 — 覆盖复杂模式（后续阶段集成）
    """

    def __init__(self, movement_rules: dict):
        """
        Args:
            movement_rules: 从 movements.yaml 加载的错误规则
        """
        self.rules = movement_rules

    def detect(
        self,
        joint_angles: dict,
        landmarks: Optional[np.ndarray],
        phase: Phase,
        movement: str,
    ) -> list[MovementError]:
        """检测当前帧的动作错误.

        Args:
            joint_angles: 关节角度字典
            landmarks: (33, 4) 关键点
            phase: 当前动作阶段
            movement: 动作名称

        Returns:
            检测到的错误列表
        """
        errors = []

        # 通过规则引擎检测
        rule_errors = self._check_rules(joint_angles, landmarks, phase, movement)
        errors.extend(rule_errors)

        # 通过启发式方法检测（补充规则未覆盖的错误）
        heuristic_errors = self._check_heuristics(joint_angles, landmarks, phase, movement)
        errors.extend(heuristic_errors)

        return errors

    def _check_rules(
        self,
        angles: dict,
        landmarks: Optional[np.ndarray],
        phase: Phase,
        movement: str,
    ) -> list[MovementError]:
        """执行 YAML 中定义的规则."""
        errors = []
        movement_rules = self.rules.get(movement, {}).get("errors", [])

        for rule in movement_rules:
            try:
                condition = rule["condition"]
                if self._eval_condition(condition, angles, phase):
                    errors.append(MovementError(
                        id=rule["id"],
                        name=rule["name"],
                        severity=rule.get("severity", "medium"),
                        advice=rule.get("advice", ""),
                    ))
            except Exception:
                continue

        return errors

    def _eval_condition(
        self,
        condition: str,
        angles: dict,
        phase: Phase,
    ) -> bool:
        """安全地评估条件表达式.

        支持的表达式格式:
        - "knee_valgus_angle > 12"
        - "knee_angle > 100 and phase == BOTTOM"
        - "abs(hip_flexion_r - hip_flexion_l) > symmetry_threshold"
        """
        # 构建安全的局部变量
        local_vars = {**angles, "phase": phase}

        # 特殊变量映射
        for v in angles:
            # 用下划线连接的名字也注册为变量
            safe_key = v.replace(" ", "_")
            if safe_key != v:
                local_vars[safe_key] = angles[v]

        # 替换 and / or 为 Python 运算符
        expr = condition.replace(" and ", " and ").replace(" or ", " or ")

        try:
            # 安全评估（仅数学运算和比较）
            return bool(eval(expr, {"__builtins__": {}}, local_vars))
        except Exception:
            return False

    def _check_heuristics(
        self,
        angles: dict,
        landmarks: Optional[np.ndarray],
        phase: Phase,
        movement: str,
    ) -> list[MovementError]:
        """启发式错误检测（补充规则引擎）."""
        errors = []

        if landmarks is None:
            return errors

        pts = landmarks[:, :3]

        if movement == "squat":
            errors.extend(self._heuristic_squat(angles, pts, phase))
        elif movement == "deadlift":
            errors.extend(self._heuristic_deadlift(angles, pts, phase))
        elif movement == "pushup":
            errors.extend(self._heuristic_pushup(angles, pts, phase))
        elif movement == "pullup":
            errors.extend(self._heuristic_pullup(angles, pts, phase))
        elif movement == "plank":
            errors.extend(self._heuristic_plank(angles, pts, phase))

        return errors

    # ─── 各动作的启发式检测 ──────────────────────────────────

    def _heuristic_squat(
        self, angles: dict, pts: np.ndarray, phase: Phase
    ) -> list[MovementError]:
        """深蹲专项错误检测."""
        errors = []

        # 膝内扣检测（膝-踝连线偏移）
        knee_valgus_r = angles.get("knee_valgus_angle_r", 0)
        knee_valgus_l = angles.get("knee_valgus_angle_l", 0)
        if abs(knee_valgus_r) > 12:
            errors.append(MovementError(
                id="knee_valgus_r", name="右膝内扣",
                severity="high", advice="右膝向外打开，对准右脚尖方向",
                current_value=knee_valgus_r, threshold=12,
            ))
        if abs(knee_valgus_l) > 12:
            errors.append(MovementError(
                id="knee_valgus_l", name="左膝内扣",
                severity="high", advice="左膝向外打开，对准左脚尖方向",
                current_value=knee_valgus_l, threshold=12,
            ))

        # 膝过脚尖检测
        knee_toe_offset = self._calc_knee_toe_offset(pts)
        if knee_toe_offset > 0.08 and phase in (Phase.DESCENT, Phase.BOTTOM):
            errors.append(MovementError(
                id="knee_over_toe", name="膝过脚尖",
                severity="low", advice="臀部向后坐，重心放在脚后跟",
                current_value=knee_toe_offset, threshold=0.08,
            ))

        # 脚跟抬起检测
        heel_height = self._calc_heel_height(pts)
        if heel_height > 0.03 and phase in (Phase.DESCENT, Phase.BOTTOM):
            errors.append(MovementError(
                id="heel_lift", name="脚跟抬起",
                severity="medium", advice="重心放在全脚掌，脚跟踩实地面",
                current_value=heel_height, threshold=0.03,
            ))

        # 深度不足检测
        knee_angle = angles.get("knee_angle_r", 180)
        if knee_angle > 100 and phase == Phase.BOTTOM:
            errors.append(MovementError(
                id="insufficient_depth", name="深度不足",
                severity="medium", advice="继续下蹲至大腿与地面平行",
                current_value=knee_angle, threshold=100,
            ))

        # 左右不对称
        knee_sym = angles.get("knee_symmetry", 0)
        if knee_sym > 10:
            errors.append(MovementError(
                id="asymmetry", name="左右不对称",
                severity="medium", advice="均匀分配体重，保持双侧对称",
                current_value=knee_sym, threshold=10,
            ))

        return errors

    def _heuristic_deadlift(
        self, angles: dict, pts: np.ndarray, phase: Phase
    ) -> list[MovementError]:
        """硬拉专项错误检测."""
        errors = []

        lumbar = angles.get("lumbar_extension", 0)
        if lumbar < -10:
            errors.append(MovementError(
                id="back_rounding", name="弓背",
                severity="high", advice="挺胸收腹，保持背部平直",
                current_value=lumbar, threshold=-10,
            ))

        if abs(lumbar) > 5 and phase == Phase.SETUP:
            errors.append(MovementError(
                id="poor_setup", name="起始姿势不良",
                severity="medium", advice="调整起始姿势，收紧背部和核心",
                current_value=lumbar, threshold=5,
            ))

        return errors

    def _heuristic_pushup(
        self, angles: dict, pts: np.ndarray, phase: Phase
    ) -> list[MovementError]:
        """俯卧撑专项错误检测."""
        errors = []

        # 塌腰检测
        hip_drop = self._calc_hip_drop(pts)
        if hip_drop > 0.05:
            errors.append(MovementError(
                id="sagging_hips", name="塌腰",
                severity="high", advice="收紧核心和臀部，身体呈一条直线",
                current_value=hip_drop, threshold=0.05,
            ))

        # 肘外展检测
        elbow_flare_r = self._calc_elbow_flare(pts, side="right")
        elbow_flare_l = self._calc_elbow_flare(pts, side="left")
        avg_flare = (elbow_flare_r + elbow_flare_l) / 2
        if avg_flare > 60:
            errors.append(MovementError(
                id="elbow_flare", name="肘部外展过大",
                severity="medium", advice="手肘贴近身体，保持约45度角",
                current_value=avg_flare, threshold=60,
            ))

        return errors

    def _heuristic_pullup(
        self, angles: dict, pts: np.ndarray, phase: Phase
    ) -> list[MovementError]:
        """引体向上专项."""
        errors = []

        elbow_r = angles.get("elbow_angle_r", 180)
        if elbow_r > 45 and phase == Phase.TOP:
            errors.append(MovementError(
                id="half_rep", name="幅度不够",
                severity="high", advice="拉至下巴过杠",
                current_value=elbow_r, threshold=45,
            ))

        elbow_sym = abs(
            angles.get("elbow_angle_r", 0) - angles.get("elbow_angle_l", 0)
        )
        if elbow_sym > 15:
            errors.append(MovementError(
                id="uneven_pull", name="左右不平衡",
                severity="medium", advice="保持两侧均匀发力",
                current_value=elbow_sym, threshold=15,
            ))

        return errors

    def _heuristic_plank(
        self, angles: dict, pts: np.ndarray, phase: Phase
    ) -> list[MovementError]:
        """平板支撑专项."""
        errors = []

        hip_drop = self._calc_hip_drop(pts)
        if hip_drop > 0.04:
            errors.append(MovementError(
                id="hip_sag", name="髋部下塌",
                severity="high", advice="收紧臀部，抬高髋部至与肩同高",
                current_value=hip_drop, threshold=0.04,
            ))

        return errors

    # ─── 辅助计算 ────────────────────────────────────────────

    @staticmethod
    def _calc_knee_toe_offset(pts: np.ndarray) -> float:
        """计算膝盖相对于脚尖的前伸距离（归一化）."""
        knee_avg = (pts[25, :2] + pts[26, :2]) / 2.0   # 膝中点 (x, z)
        toe_avg = (pts[31, :2] + pts[32, :2]) / 2.0    # 趾中点
        body_scale = np.linalg.norm(pts[24, :3] - pts[26, :3])  # 大腿长度

        if body_scale < 1e-9:
            return 0.0
        forward_offset = knee_avg[0] - toe_avg[0]  # x方向差异
        return float(forward_offset / body_scale)

    @staticmethod
    def _calc_heel_height(pts: np.ndarray) -> float:
        """估算脚跟离地高度（归一化）."""
        ankle_y = (pts[27, 1] + pts[28, 1]) / 2.0
        toe_y = (pts[31, 1] + pts[32, 1]) / 2.0
        heel_y = (pts[29, 1] + pts[30, 1]) / 2.0
        body_scale = np.linalg.norm(pts[24, :3] - pts[26, :3])

        if body_scale < 1e-9:
            return 0.0
        # 脚跟高于脚尖时表示脚跟抬起
        return float(max(0, heel_y - toe_y) / body_scale)

    @staticmethod
    def _calc_hip_drop(pts: np.ndarray) -> float:
        """计算髋部下塌程度（俯卧撑/平板）."""
        shoulder_y = (pts[11, 1] + pts[12, 1]) / 2.0
        hip_y = (pts[23, 1] + pts[24, 1]) / 2.0
        ankle_y = (pts[27, 1] + pts[28, 1]) / 2.0
        body_length = np.linalg.norm(
            (pts[11, :2] + pts[12, :2]) / 2.0 -
            (pts[27, :2] + pts[28, :2]) / 2.0
        )

        if body_length < 1e-9:
            return 0.0
        # 髋部相对于肩-踝连线的下坠量
        expected_hip_y = (shoulder_y + ankle_y) / 2.0
        return float((expected_hip_y - hip_y) / body_length)

    @staticmethod
    def _calc_elbow_flare(pts: np.ndarray, side: str = "right") -> float:
        """计算肘部外展角度."""
        if side == "right":
            shoulder, elbow, wrist = 12, 14, 16
        else:
            shoulder, elbow, wrist = 11, 13, 15

        upper_arm = pts[elbow, :2] - pts[shoulder, :2]
        torso_ref = np.array([0, -1])  # 躯干参考方向

        norm_arm = np.linalg.norm(upper_arm)
        if norm_arm < 1e-9:
            return 0.0

        cos_angle = np.dot(upper_arm / norm_arm, torso_ref)
        angle = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
        return float(angle)
