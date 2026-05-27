"""AI 健身教练 — 主程序

完整流程：
  摄像头 → MediaPipe Pose → OpenSim IK (via socket) →
  标准动作对比 → 评分 → 错误检测 → Qwen2-7B 建议 → TTS 播报

使用方式:
  python app.py                          # 默认深蹲模式（摄像头）
  python app.py --movement deadlift      # 硬拉模式
  python app.py --no-opensim             # 关闭OpenSim连接（纯规则模式）
  python app.py --no-llm                 # 关闭LLM（仅规则建议）
  python app.py --video video.mp4        # 处理视频文件（headless模式）
  python app.py --video video.mp4 -o out.mp4  # 指定输出路径
"""

import sys
import os
import time
import argparse
import signal
import yaml
import cv2
import numpy as np
from collections import Counter, deque

# 添加项目根目录
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.pose.estimator import PoseEstimator
from src.pose.preprocessor import LandmarkSmoother, normalize_pose, mediapipe_to_opensim_markers
from src.pose.tracker import MovementPhaseTracker, Phase
from src.bridge.socket_client import OpenSimClient, IKResponse
from src.comparison.movement_library import MovementLibrary, MovementTemplate
from src.comparison.scorer import MovementScorer, RepScore
from src.comparison.error_detector import ErrorDetector
from src.feedback.llm_advisor import LLMAdvisor, AdviceContext
from src.feedback.tts_engine import TTSEngine
from src.ui.overlay import OverlayRenderer
from src.ui.dashboard import DashboardData, ConsoleDashboard


class FitnessCoach:
    """AI 健身教练主控制器."""

    def __init__(self, config: dict):
        self.config = config

        # ─── 模块初始化 ─────────────────────────────────
        print("[Coach] 初始化模块...")

        # 姿态估计
        pose_cfg = config.get("pose", {})
        self.estimator = PoseEstimator(
            model_complexity=pose_cfg.get("model_complexity", 2),
            min_detection_confidence=pose_cfg.get("min_detection_confidence", 0.5),
            min_tracking_confidence=pose_cfg.get("min_tracking_confidence", 0.5),
        )
        self.smoother = LandmarkSmoother(
            window_size=pose_cfg.get("smooth_window", 5)
        )

        # 动作库 + 阶段追踪
        self.library = MovementLibrary(data_dir=config.get("movement_data_dir", "movement_data"))
        self.library.load_config(config.get("movement_config", "config/movements.yaml"))
        self.phase_tracker = MovementPhaseTracker()

        # 评分 + 错误检测
        scorer_cfg = config.get("scoring", {})
        self.scorer = MovementScorer(
            joint_deviation_weight=scorer_cfg.get("joint_deviation_weight", 0.50),
            symmetry_weight=scorer_cfg.get("symmetry_weight", 0.20),
            stability_weight=scorer_cfg.get("stability_weight", 0.15),
            tempo_weight=scorer_cfg.get("tempo_weight", 0.15),
        )
        self.error_detector = ErrorDetector(self.library._config)

        # OpenSim 桥接
        bridge_cfg = config.get("bridge", {})
        self.opensim_client = OpenSimClient(
            host=bridge_cfg.get("host", "localhost"),
            request_port=bridge_cfg.get("request_port", 5000),
            result_port=bridge_cfg.get("result_port", 5001),
            timeout=bridge_cfg.get("timeout_ms", 50) / 1000.0,
        )

        # LLM
        llm_cfg = config.get("llm", {})
        self.use_llm = config.get("use_llm", True)
        if self.use_llm:
            self.advisor = LLMAdvisor(
                model_path=llm_cfg.get("model_path", "Qwen/Qwen2-7B-Instruct"),
                device=llm_cfg.get("device", "cuda"),
                use_4bit=llm_cfg.get("use_4bit_quantization", True),
            )
        else:
            self.advisor = None

        # TTS
        tts_cfg = config.get("tts", {})
        self.tts = TTSEngine(
            engine=tts_cfg.get("engine", "edge-tts"),
            voice=tts_cfg.get("voice", "zh-CN-XiaoxiaoNeural"),
        )

        # UI
        ui_cfg = config.get("ui", {})
        self.overlay = OverlayRenderer(
            show_skeleton=ui_cfg.get("show_skeleton_overlay", True),
            show_score=ui_cfg.get("show_score_panel", True),
            show_phase=ui_cfg.get("show_phase_indicator", True),
        )
        self.dashboard = DashboardData()
        self.console = ConsoleDashboard()

        # ─── 状态变量 ─────────────────────────────────
        self.movement = ""
        self.template: MovementTemplate = None
        self.use_opensim = config.get("use_opensim", True)
        self.running = False
        self.paused = False

        # 动作计数
        self.rep_count = 0
        self._rep_phases: list = []      # 记录一次动作经历的阶段
        self._last_rep_end_frame = 0
        self._frame_idx = 0

        # 反馈节流：每N帧或阶段切换时生成建议
        self._last_advice_frame = -60   # 至少间隔2秒
        self._last_advice = ""

        print("[Coach] 初始化完成")

    def start(self, movement: str = "squat", video_path: str = None):
        """开始训练会话.

        Args:
            movement: 训练动作名称
            video_path: 输入视频路径。提供时进入headless模式，结果写入输出视频文件
        """
        self.movement = movement
        self._video_mode = video_path is not None

        print(f"\n{'='*50}")
        print(f"  AI 健身教练 — {self.library._config.get(movement, {}).get('name', movement)}")
        if self._video_mode:
            print(f"  输入视频: {video_path}")
            print(f"  模式: headless（结果写入输出视频）")
        else:
            print(f"  按 'q' 退出 | 'p' 暂停 | 'r' 重置次数")
        print(f"{'='*50}\n")

        # 加载动作模板
        self.template = self.library.load_template(movement)
        if self.template is None:
            print(f"[Coach] 无法加载动作模板: {movement}")
            return

        # 更新阶段追踪器
        self.phase_tracker.movement = movement

        # 连接 OpenSim (可选)
        if self.use_opensim:
            ok = self.opensim_client.connect()
            if ok:
                print("[Coach] OpenSim 连接已建立 (WSL2)")
            else:
                print("[Coach] OpenSim 不可用，使用本地几何IK")

        # 加载 LLM (可选)
        if self.use_llm and self.advisor is not None:
            self.advisor.load()

        # 打开视频输入（摄像头或文件）
        if self._video_mode:
            self.cap = cv2.VideoCapture(video_path)
            if not self.cap.isOpened():
                print(f"[Coach] 无法打开视频: {video_path}")
                return
            total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_fps = self.cap.get(cv2.CAP_PROP_FPS)
            print(f"[Coach] 视频: {total_frames} 帧, {video_fps:.1f} fps, "
                  f"时长 {total_frames/video_fps:.1f}s" if video_fps > 0 else f"[Coach] 视频: {total_frames} 帧")

            # 初始化输出视频
            output_path = getattr(self, '_output_path', None)
            if output_path is None:
                base = os.path.splitext(os.path.basename(video_path))[0]
                output_path = os.path.join(
                    os.path.dirname(video_path) or ".",
                    f"{base}_analyzed.mp4"
                )
            frame_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(output_path, fourcc, video_fps, (frame_w, frame_h))
            print(f"[Coach] 输出视频: {output_path}")
        else:
            camera_cfg = self.config.get("camera", {})
            self.cap = cv2.VideoCapture(camera_cfg.get("device_id", 0))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_cfg.get("width", 1280))
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_cfg.get("height", 720))
            self.cap.set(cv2.CAP_PROP_FPS, camera_cfg.get("fps", 30))
            self._writer = None

        self.running = True
        self._main_loop()

    def stop(self):
        """停止训练."""
        self.running = False
        if hasattr(self, "cap") and self.cap is not None:
            self.cap.release()
        if hasattr(self, "_writer") and self._writer is not None:
            self._writer.release()
            print("[Coach] 输出视频已保存")
        if not getattr(self, "_video_mode", False):
            cv2.destroyAllWindows()
        self.estimator.close()
        self.opensim_client.disconnect()
        print("\n[Coach] 训练结束")

    def _main_loop(self):
        """主循环."""
        fps_counter = deque(maxlen=30)
        last_time = time.time()

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                if self._video_mode:
                    break  # 视频播放完毕
                continue

            if not self._video_mode:
                frame = cv2.flip(frame, 1)  # 摄像头模式：水平翻转（镜像）
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            self._frame_idx += 1

            # ─── 1. 姿态估计 ────────────────────────
            result = self.estimator.detect(frame_rgb, timestamp=time.time())

            if not result.detected:
                # 未检测到人体
                cv2.putText(frame, "未检测到人体", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                self._write_or_show(frame)
                if not self._video_mode and cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            # 平滑滤波
            landmarks = self.smoother.update(result.world_landmarks)

            # ─── 2. 逆运动学求解 ──────────────────────
            joint_angles = self._solve_ik(landmarks)

            # ─── 3. 动作阶段检测 ──────────────────────
            phase = self.phase_tracker.update(joint_angles, landmarks)

            # 检测动作完成（深蹲：SETUP→DESCENT→BOTTOM→ASCENT→LOCKOUT）
            self._detect_rep_completion(phase)

            # ─── 4. 标准动作对比 ──────────────────────
            phase_idx = self._phase_to_template_index(phase)
            template_angles = self.template.get_frame_at(phase_idx)

            # 对齐用户关节名称与模板
            user_vec = self._angles_to_vector(joint_angles, self.template.joint_names)

            # ─── 5. 评分 ──────────────────────────────
            errors = self.error_detector.detect(
                joint_angles, landmarks, phase, self.movement
            )
            frame_score = self.scorer.score_frame(
                joint_angles, template_angles,
                self.template.joint_names, landmarks, phase, errors,
            )
            self.dashboard.update(frame_score.total, joint_angles, phase)

            # ─── 6. LLM 建议生成 (周期性) ──────────────
            advice = self._maybe_generate_advice(frame_score, errors, phase)

            # ─── 7. UI 渲染 ───────────────────────────
            output = self.overlay.render(
                frame,
                landmarks_2d=result.landmarks_2d,
                score=frame_score,
                phase=phase,
                movement_name=self.library._config.get(self.movement, {}).get("name", self.movement),
                rep_count=self.rep_count,
                advice=advice or self._last_advice,
            )
            if advice:
                self._last_advice = advice

            # 控制台输出
            self.console.update(frame_score.total, phase, errors, self.rep_count)

            # FPS 计算
            now = time.time()
            fps_counter.append(1.0 / max(now - last_time, 0.001))
            last_time = now
            fps = np.mean(fps_counter) if fps_counter else 0
            cv2.putText(output, f"FPS: {fps:.0f}", (10, output.shape[0] - 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            self._write_or_show(output)

            if not self._video_mode:
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("p"):
                    self.paused = not self.paused
                    print(f"[Coach] {'暂停' if self.paused else '继续'}")
                elif key == ord("r"):
                    self._reset_rep()
                    print("[Coach] 次数已重置")

        self.stop()

    # ─── 内部方法 ──────────────────────────────────────────

    def _write_or_show(self, frame: np.ndarray):
        """写入视频文件（headless模式）或显示窗口（摄像头模式）."""
        if self._video_mode and self._writer is not None:
            self._writer.write(frame)
        elif not self._video_mode:
            cv2.imshow("AI Fitness Coach", frame)

    def _solve_ik(self, landmarks: np.ndarray) -> dict:
        """求解逆运动学."""
        if self.use_opensim and self.opensim_client._connected:
            resp = self.opensim_client.solve_ik(landmarks, self.movement)
            if resp.success:
                return resp.joint_angles

        # 回退：本地几何IK
        from src.bridge.socket_server import GeometricIKSolver
        return GeometricIKSolver.solve(landmarks)

    def _detect_rep_completion(self, phase: Phase):
        """检测一次完整动作的完成."""
        self._rep_phases.append(phase)

        # 保持最近 20 个阶段记录
        if len(self._rep_phases) > 20:
            self._rep_phases = self._rep_phases[-20:]

        # 检测完整周期：SETUP→...→LOCKOUT
        if (phase == Phase.LOCKOUT and
            Phase.BOTTOM in self._rep_phases[-10:] and
            self._frame_idx - self._last_rep_end_frame > 30):  # 至少30帧间隔

            self.rep_count += 1
            self._last_rep_end_frame = self._frame_idx

            # 汇总一次动作的评分
            rep_score = self.scorer.score_rep(self.movement)
            print(f"\n[Rep #{self.rep_count}] 评分: {rep_score.total:.0f}/100 | "
                  f"关节:{rep_score.joint_deviation:.0f} 对称:{rep_score.symmetry:.0f} "
                  f"稳定:{rep_score.stability:.0f} 节奏:{rep_score.tempo:.0f}")

            if rep_score.error_summary:
                print(f"  错误: {rep_score.error_summary}")

            # 生成完整动作反馈
            self._generate_rep_feedback(rep_score)

            # 重置，准备下一次计数
            self.scorer.reset()
            self._rep_phases.clear()

    def _maybe_generate_advice(
        self,
        score,
        errors: list,
        phase: Phase,
    ) -> str:
        """周期性生成实时建议（非LLM，仅规则建议）."""
        # LLM 只在动作完成时调用（性能考虑）
        # 实时建议使用规则
        if errors and self._frame_idx - self._last_advice_frame > 60:
            self._last_advice_frame = self._frame_idx
            # 取最严重错误
            sev = {"high": 0, "medium": 1, "low": 2}
            worst = sorted(errors, key=lambda e: sev.get(e.severity, 3))[0]
            return worst.advice
        return ""

    def _generate_rep_feedback(self, rep_score: RepScore):
        """动作完成后生成LLM反馈."""
        errors_for_llm = []
        for eid, count in rep_score.error_summary.items():
            # 从 error_detector 规则中查找错误详情
            rules = self.library.get_error_rules(self.movement)
            for r in rules:
                if r["id"] == eid:
                    from src.comparison.error_detector import MovementError
                    errors_for_llm.append(MovementError(
                        id=eid, name=r.get("name", eid),
                        severity=r.get("severity", "medium"),
                        advice=r.get("advice", ""),
                    ))
                    break

        context = AdviceContext(
            movement=self.movement,
            movement_name=self.library._config.get(self.movement, {}).get("name", self.movement),
            phase="complete",
            score=rep_score.total,
            errors=errors_for_llm,
            rep_count=self.rep_count,
            duration=rep_score.duration_seconds,
        )

        # LLM 反馈
        result = {"level": "良好", "score": f"{rep_score.total:.0f}/100",
                   "advice": "继续保持！", "errors": []}
        if self.use_llm and self.advisor is not None:
            result = self.advisor.generate_rep_feedback(context)
        else:
            # 规则模式
            if errors_for_llm:
                result["advice"] = "；".join(e.advice for e in errors_for_llm[:2])
            if rep_score.total < 60:
                result["level"] = "需要改进"
            elif rep_score.total >= 90:
                result["level"] = "优秀"

        # TTS 播报
        tts_text = f"{result['level']}，{result['score']}分。{result['advice']}"
        self.tts.speak(tts_text)

    def _reset_rep(self):
        """重置动作计数."""
        self.rep_count = 0
        self.scorer.reset()
        self.dashboard.reset()
        self._rep_phases.clear()

    @staticmethod
    def _phase_to_template_index(phase: Phase) -> int:
        """将 Phase 映射到模板帧索引."""
        mapping = {
            Phase.SETUP:   10,
            Phase.DESCENT: 30,
            Phase.BOTTOM:  45,
            Phase.ASCENT:  60,
            Phase.LOCKOUT: 80,
            Phase.REST:    10,
        }
        return mapping.get(phase, 45)

    @staticmethod
    def _angles_to_vector(angles: dict, joint_names: list) -> np.ndarray:
        """将关节角度字典转为向量."""
        result = []
        for name in joint_names:
            result.append(angles.get(name, 0.0))
        return np.array(result)


# ═══════════════════════════════════════════════════════════════
# CLI入口
# ═══════════════════════════════════════════════════════════════

def load_config(config_path: str = "config/settings.yaml") -> dict:
    """加载YAML配置."""
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


def main():
    parser = argparse.ArgumentParser(description="AI 健身教练")
    parser.add_argument("--movement", "-m", type=str, default="squat",
                        choices=["squat", "deadlift", "pushup", "pullup", "plank"],
                        help="训练动作 (默认: squat)")
    parser.add_argument("--config", "-c", type=str, default="config/settings.yaml",
                        help="配置文件路径")
    parser.add_argument("--no-opensim", action="store_true",
                        help="禁用 OpenSim IK（使用纯几何方法）")
    parser.add_argument("--no-llm", action="store_true",
                        help="禁用 LLM 建议（使用规则模式）")
    parser.add_argument("--video", "-v", type=str, default=None,
                        help="输入视频路径（headless模式，不弹窗，结果写入文件）")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="输出视频路径（默认自动生成: {input}_analyzed.mp4）")
    parser.add_argument("--list-movements", action="store_true",
                        help="列出所有支持的动作")
    args = parser.parse_args()

    config = load_config(args.config)
    config["use_opensim"] = not args.no_opensim
    config["use_llm"] = not args.no_llm

    if args.list_movements:
        lib = MovementLibrary()
        lib.load_config(config.get("movement_config", "config/movements.yaml"))
        print("支持的动作:")
        for m in lib.list_movements():
            print(f"  - {m}: {lib._config[m].get('name', m)}")
        return

    coach = FitnessCoach(config)
    signal.signal(signal.SIGINT, lambda s, f: coach.stop())

    try:
        coach._output_path = args.output  # 在 start() 之前设置
        coach.start(movement=args.movement, video_path=args.video)
    except Exception as e:
        print(f"[Coach] 致命错误: {e}")
        import traceback
        traceback.print_exc()
        coach.stop()


if __name__ == "__main__":
    main()
