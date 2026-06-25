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
import json
import argparse
import signal
import yaml
import cv2
import numpy as np
from collections import Counter, deque
from typing import Optional

# 添加项目根目录
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.pose.estimator import PoseEstimator
from src.pose.preprocessor import LandmarkSmoother, normalize_pose, mediapipe_to_opensim_markers
from src.pose.tracker import MovementPhaseTracker, Phase
from src.bridge.socket_client import OpenSimClient, IKResponse
from src.bridge.ws_server import TwinWebSocketServer
from src.comparison.movement_library import MovementLibrary, MovementTemplate
from src.comparison.scorer import MovementScorer, RepScore
from src.comparison.error_detector import ErrorDetector
from src.feedback.llm_advisor import LLMAdvisor, AdviceContext
from src.feedback.tts_engine import TTSEngine
from src.feedback.voice_coach import VoiceCoach
from src.feedback.intervention_engine import InterventionEngine, InterventionLevel, InterventionResult
from src.feedback.profile_manager import ProfileManager, SessionRecord
from src.ui.overlay import OverlayRenderer
from src.ui.dashboard import DashboardData, ConsoleDashboard
from src.webots.op2_client import OP2RealtimeClient


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
                model_path=llm_cfg.get("model_path", "qwen2:7b"),
                device=llm_cfg.get("device", "cuda"),
                use_4bit=llm_cfg.get("use_4bit_quantization", True),
                use_ollama=llm_cfg.get("use_ollama", True),
                ollama_host=llm_cfg.get("ollama_host", "http://localhost:11434"),
            )
        else:
            self.advisor = None

        # TTS
        tts_cfg = config.get("tts", {})
        self.tts = TTSEngine(
            engine=tts_cfg.get("engine", "edge-tts"),
            voice=tts_cfg.get("voice", "zh-CN-XiaoxiaoNeural"),
        )

        # Voice coach (priority-queued announcements)
        voice_cfg = config.get("voice", {})
        self.voice_coach = VoiceCoach(
            tts_engine=self.tts,
            enabled=voice_cfg.get("enabled", True),
            rep_summary_interval=voice_cfg.get("rep_summary_interval", 5),
        )

        # Intervention engine (3-level graded intervention)
        self.intervention_engine: Optional[InterventionEngine] = None

        # Profile manager (adaptive thresholds)
        profile_cfg = config.get("profile", {})
        self.profile_manager = ProfileManager(
            profile_path=profile_cfg.get("path", "profiles/default.json")
        )

        # Webots OP2 real-time client
        webots_cfg = config.get("webots", {})
        self.webots_client: Optional[OP2RealtimeClient] = None
        if config.get("webots_op2_enabled", False):
            self.webots_client = OP2RealtimeClient(
                host=webots_cfg.get("op2_host", "localhost"),
                port=webots_cfg.get("op2_port", 10020),
            )

        # WebSocket 服务器 (第5周: 3D孪生对比)
        ws_cfg = config.get("websocket", {})
        self.ws_server: Optional[TwinWebSocketServer] = None
        if config.get("ws_enabled", True):
            self.ws_server = TwinWebSocketServer(
                host=ws_cfg.get("host", "localhost"),
                port=ws_cfg.get("port", 8765),
            )
            self.ws_server.on_movement_change = self._on_frontend_movement_change

        # UI
        ui_cfg = config.get("ui", {})
        self.overlay = OverlayRenderer(
            show_skeleton=ui_cfg.get("show_skeleton_overlay", True),
            show_score=ui_cfg.get("show_score_panel", True),
            show_phase=ui_cfg.get("show_phase_indicator", True),
        )
        self.dashboard = DashboardData()
        self.console = ConsoleDashboard()
        self.display_mode = config.get("display_mode", "opencv")  # opencv | web | headless

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
        self._last_phase = None          # track phase transitions for voice announcements

        # 反馈节流：每N帧或阶段切换时生成建议
        self._last_advice_frame = -60   # 至少间隔2秒
        self._last_advice = ""

        # 评分历史 (第6周)
        self._score_history: list[dict] = []

        print("[Coach] 初始化完成")

    def _on_frontend_movement_change(self, movement: str):
        """前端动作下拉框切换回调."""
        print(f"[Coach] 前端切换动作: {movement}")

        # 加载新模板
        self.template = self.library.load_template(movement)
        if self.template is None:
            print(f"[Coach] 无法加载动作模板: {movement}")
            return

        self.movement = movement
        self.phase_tracker.movement = movement

        # 更新干预引擎
        movement_cfg = self.library._config.get(movement, {})
        self.intervention_engine = InterventionEngine(
            movement=movement, rom_config=movement_cfg
        )

        # 发送新模板到前端
        if self.ws_server and self.ws_server.connected:
            self._send_template_to_frontend()

        print(f"[Coach] 已切换到: {self.library._config.get(movement, {}).get('name', movement)}")

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
            self.display_mode = "headless"
        elif self.display_mode == "web":
            print("[Coach] Web 控制台模式 — 请在浏览器打开 http://localhost:8080")
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

        # 初始化干预引擎
        movement_cfg = self.library._config.get(movement, {})
        self.intervention_engine = InterventionEngine(
            movement=movement,
            rom_config=movement_cfg,
        )

        # 启动语音播报
        self.voice_coach.start()

        # 连接 OpenSim (可选)
        if self.use_opensim:
            ok = self.opensim_client.connect()
            if ok:
                print("[Coach] OpenSim 连接已建立 (WSL2)")
            else:
                print("[Coach] OpenSim 不可用，使用本地几何IK")

        # 连接 Webots OP2 (可选)
        if self.webots_client is not None:
            ok = self.webots_client.connect()
            if ok:
                print("[Coach] Webots OP2 实时镜像已连接")
            else:
                print("[Coach] Webots OP2 不可用（请先启动 Webots 仿真）")

        # 加载 LLM (可选)
        if self.use_llm and self.advisor is not None:
            self.advisor.load()

        # 启动 WebSocket 服务器 (第5周: 3D孪生对比)
        if self.ws_server is not None:
            self.ws_server.start()
            # 发送模板数据到前端
            self._send_template_to_frontend()

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
            from src.utils.camera import open_camera
            camera_cfg = self.config.get("camera", {})
            self.cap, opened_id = open_camera(
                device_id=camera_cfg.get("device_id", 0),
                width=camera_cfg.get("width", 1280),
                height=camera_cfg.get("height", 720),
                fps=camera_cfg.get("fps", 30),
            )
            if self.cap is None:
                print("[Coach] 无法打开摄像头，请检查：")
                print("  1) 摄像头是否被其他软件占用（Teams/Zoom/相机）")
                print("  2) Windows 设置 → 隐私 → 摄像头 是否允许桌面应用")
                print("  3) 尝试: python web_coach.py --video test_squat.mp4")
                if self.display_mode == "web" and self.ws_server is not None:
                    self.ws_server.send_stats({
                        "score": 0, "phase": "ERROR",
                        "movement_name": "摄像头不可用",
                        "rep_count": 0, "advice": "无法读取摄像头，请关闭占用摄像头的程序后重启",
                        "fps": 0,
                    })
                return
            else:
                camera_cfg["device_id"] = opened_id
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
        if self.webots_client is not None:
            self.webots_client.disconnect()
        if self.ws_server is not None:
            self.ws_server.stop()
        self.voice_coach.stop()
        # Record session to profile
        session = SessionRecord(
            timestamp=time.time(),
            movement=self.movement,
            rep_count=self.rep_count,
            avg_score=float(np.mean(self.dashboard.scores)) if self.dashboard.scores else 0.0,
        )
        self.profile_manager.record_session(session)

        # 保存评分历史 (第6周)
        self._save_score_history()

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
                if self.cap is None or not self.cap.isOpened():
                    if self.display_mode == "web" and self.ws_server is not None:
                        self.ws_server.send_stats({
                            "score": 0, "phase": "ERROR",
                            "movement_name": "摄像头不可用",
                            "rep_count": 0,
                            "advice": "无法读取摄像头帧，请检查设备或改用 --video test_squat.mp4",
                            "fps": 0,
                        })
                    time.sleep(0.5)
                    continue
                if self._frame_idx == 0:
                    self._frame_idx += 1
                    print(f"[Coach] 无法读取摄像头 (device_id={self.config.get('camera', {}).get('device_id', 0)})，请检查设备ID")
                time.sleep(0.05)
                continue

            if not self._video_mode:
                frame = cv2.flip(frame, 1)  # 摄像头模式：水平翻转（镜像）
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            self._frame_idx += 1

            # ─── 1. 姿态估计 ────────────────────────
            result = self.estimator.detect(frame_rgb, timestamp=time.time())

            if not result.detected:
                from src.ui.text_renderer import draw_text
                draw_text(frame, "未检测到人体", (50, 30), (0, 0, 255), font_size=28)
                self._write_or_show(frame)
                if not self._video_mode and cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            # 平滑滤波
            landmarks = self.smoother.update(result.world_landmarks)

            # 发送到 3D 孪生对比页面 (第5周)
            if self.ws_server is not None and self.ws_server.connected:
                self.ws_server.send_pose(result.world_landmarks, self._frame_idx)

            # ─── 2. 逆运动学求解 ──────────────────────
            joint_angles = self._solve_ik(landmarks)

            # Webots OP2 实时镜像发送
            if self._frame_idx == 5:
                print(f"[Coach] webots_client={self.webots_client} is_connected={self.webots_client.is_connected if self.webots_client else 'N/A'}")
            if self.webots_client is not None and self.webots_client.is_connected:
                self.webots_client.send_joint_angles(joint_angles)

            # ─── 3. 动作阶段检测 ──────────────────────
            phase = self.phase_tracker.update(joint_angles, landmarks)

            # 阶段切换语音播报
            if phase != self._last_phase and self._last_phase is not None:
                self.voice_coach.announce_stage(phase.name)
            self._last_phase = phase

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

            # ─── 6. 干预评估 ──────────────────────────
            intervention: Optional[InterventionResult] = None
            if self.intervention_engine is not None:
                intervention = self.intervention_engine.evaluate(
                    joint_angles, phase, landmarks, timestamp=time.time()
                )
                if intervention.level == InterventionLevel.HINT:
                    self.voice_coach.announce_hint(intervention.message)
                elif intervention.level == InterventionLevel.WARNING:
                    self.voice_coach.announce_warning(intervention.message)
                elif intervention.level == InterventionLevel.EMERGENCY:
                    self.voice_coach.announce_emergency(intervention.message)

            # ─── 7. LLM 建议生成 (周期性) ──────────────
            advice = self._maybe_generate_advice(frame_score, errors, phase)

            # ─── 8. UI 渲染 ───────────────────────────
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

            self._write_or_show(output, frame_score=frame_score, phase=phase,
                                advice=advice or self._last_advice, fps=fps)

            if not self._video_mode:
                if self.display_mode == "opencv":
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                    elif key == ord("p"):
                        self.paused = not self.paused
                        print(f"[Coach] {'暂停' if self.paused else '继续'}")
                    elif key == ord("r"):
                        self._reset_rep()
                        print("[Coach] 次数已重置")
                else:
                    time.sleep(0.001)

        self.stop()

    # ─── 内部方法 ──────────────────────────────────────────

    def _write_or_show(self, frame: np.ndarray, frame_score=None, phase=None, advice: str = "", fps: float = 0):
        """写入视频 / Web 推送 / OpenCV 窗口显示."""
        if self.display_mode == "web" and self.ws_server is not None:
            ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if ok:
                self.ws_server.send_frame(buf.tobytes())
            if frame_score is not None:
                movement_name = self.library._config.get(self.movement, {}).get("name", self.movement)
                self.ws_server.send_stats({
                    "score": round(float(frame_score.total), 1),
                    "joint": round(float(frame_score.joint_deviation), 1),
                    "symmetry": round(float(frame_score.symmetry), 1),
                    "stability": round(float(frame_score.stability), 1),
                    "tempo": round(float(frame_score.tempo), 1),
                    "phase": phase.name if phase else "",
                    "movement": self.movement,
                    "movement_name": movement_name,
                    "rep_count": self.rep_count,
                    "advice": advice or "",
                    "fps": round(fps, 1),
                })
            return

        if self._video_mode and self._writer is not None:
            self._writer.write(frame)
        elif self.display_mode == "opencv" and not self._video_mode:
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

            # Voice announcement for rep count
            self.voice_coach.announce_rep(self.rep_count, rep_score.total)

            # 生成完整动作反馈
            self._generate_rep_feedback(rep_score)

            # 记录评分历史 (第6周)
            self._record_score_to_history(rep_score.total)

            # 为最严重的错误播报纠正提示
            if rep_score.error_summary:
                sev = {"high": 0, "medium": 1, "low": 2}
                worst = sorted(
                    rep_score.error_summary.keys(),
                    key=lambda eid: sev.get(
                        next((r.get("severity", "low") for r in self.library.get_error_rules(self.movement) if r["id"] == eid), "low"),
                        3
                    )
                )
                if worst:
                    self.voice_coach.announce_correction(worst[0])

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

    # ─── 第5-6周辅助方法 ──────────────────────────────────────

    def _send_template_to_frontend(self):
        """将当前动作模板发送到 3D 孪生页面."""
        if self.ws_server is None or not self.ws_server.connected:
            return
        if self.template is None:
            return

        # 从模板的 joint_angle_sequence 反算关键点 或 从 JSON 加载原始关键点
        try:
            from src.utils.paths import resource_path
            json_path = resource_path("templates", f"template_{self.movement}.json")
        except ImportError:
            json_path = os.path.join("templates", f"template_{self.movement}.json")
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                template_seq = json.load(f)
            self.ws_server.send_template(template_seq)
            print(f"[Coach] 已发送模板到前端: {self.movement}")
        else:
            print(f"[Coach] 模板 JSON 不存在: {json_path}")

    def _record_score_to_history(self, score: float):
        """记录一次评分到内存历史."""
        self._score_history.append({
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "timestamp": time.time(),
            "movement": self.movement,
            "score": round(score, 1),
        })
        # 自动绘制并保存图表 (第6周)
        if len(self._score_history) % 5 == 0:
            self._plot_score_chart()

    def _plot_score_chart(self):
        """绘制评分历史折线图 (matplotlib)."""
        try:
            from scripts.score_history import plot_history
            output = f"charts/score_{self.movement}_{len(self._score_history)}.png"
            os.makedirs("charts", exist_ok=True)
            plot_history(self._score_history, output_path=output)
        except ImportError:
            pass  # matplotlib 未安装时静默跳过

    def _save_score_history(self):
        """保存评分历史到文件."""
        if not self._score_history:
            return
        try:
            from scripts.score_history import save_history, load_history
            existing = load_history()
            existing.extend(self._score_history)
            save_history(existing)
            print(f"[Coach] 评分历史已保存 ({len(self._score_history)} 条新记录)")
        except Exception:
            pass

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
    from src.utils.paths import resource_path, setup_runtime
    setup_runtime()
    if not os.path.isabs(config_path) and not os.path.exists(config_path):
        bundled = resource_path(config_path)
        if os.path.exists(bundled):
            config_path = bundled
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        mov_cfg = cfg.get("movement_config", "config/movements.yaml")
        if not os.path.isabs(mov_cfg) and not os.path.exists(mov_cfg):
            bundled_mov = resource_path(mov_cfg)
            if os.path.exists(bundled_mov):
                cfg["movement_config"] = bundled_mov
        return cfg
    return {}


def main():
    parser = argparse.ArgumentParser(description="AI 健身教练")
    parser.add_argument("--movement", "-m", type=str, default="squat",
                        choices=["squat", "deadlift", "pushup", "pullup", "plank", "shooting"],
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
    parser.add_argument("--web-ui", action="store_true",
                        help="使用 Web 控制台显示（浏览器界面，无需 OpenCV 窗口）")
    parser.add_argument("--list-movements", action="store_true",
                        help="列出所有支持的动作")
    parser.add_argument("--webots", action="store_true",
                        help="启用 Webots OP2 实时镜像模式")
    args = parser.parse_args()

    config = load_config(args.config)
    config["use_opensim"] = not args.no_opensim
    config["use_llm"] = not args.no_llm
    config["webots_op2_enabled"] = args.webots
    if args.web_ui:
        config["display_mode"] = "web"
    if args.video:
        config["display_mode"] = "headless"

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
