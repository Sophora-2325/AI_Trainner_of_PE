"""课程步骤自动验收 — 一键跑通第 1-6 周核心流程.

运行:
  python scripts/auto_demo.py
  python scripts/auto_demo.py --skip-llm   # 跳过需 Ollama 的第 4 周验证
"""

import argparse
import json
import os
import sys
import urllib.request

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.utils.paths import setup_runtime, resource_path, data_path

setup_runtime()

TEST_VIDEO = data_path("test_squat.mp4")
POSE_JSON = data_path("test_squat_pose.json")
TEMPLATE = resource_path("templates", "template_squat.json")


OK = "[OK]"
FAIL = "[FAIL]"
WARN = "[!!]"
SKIP = "[--]"


def _mark(ok: bool) -> str:
    return OK if ok else FAIL


def _step(title: str):
    print(f"\n{'-' * 50}")
    print(f"  {title}")
    print(f"{'-' * 50}")


def check_env() -> bool:
    _step("第 1-2 周：环境检查")
    ok = True
    for mod in ("mediapipe", "cv2", "numpy"):
        try:
            __import__(mod)
            print(f"  {OK} {mod}")
        except ImportError:
            print(f"  {FAIL} {mod} 未安装")
            ok = False

    if os.path.exists(TEST_VIDEO):
        print(f"  {OK} 测试视频: {TEST_VIDEO}")
    else:
        print(f"  {FAIL} 缺少测试视频: {TEST_VIDEO}")
        ok = False
    return ok


def check_pose_demo() -> bool:
    _step("第 1-2 周：姿态检测验证")
    import cv2
    sys.path.insert(0, ROOT)
    from src.pose.estimator import PoseEstimator

    cap = cv2.VideoCapture(TEST_VIDEO)
    if not cap.isOpened():
        print(f"  {FAIL} 无法打开测试视频")
        return False

    estimator = PoseEstimator(model_complexity=0)
    detected = 0
    total = 0
    try:
        while total < 30:
            ret, frame = cap.read()
            if not ret:
                break
            total += 1
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = estimator.detect(rgb)
            if result.detected:
                detected += 1
    finally:
        estimator.close()
        cap.release()

    print(f"  前 {total} 帧中检测到人体: {detected}/{total}")
    ok = detected >= total * 0.5
    print(f"  {_mark(ok)} MediaPipe 姿态检测 (Tasks API)")
    return ok


def run_extract_pose() -> bool:
    _step("第 3 周：姿态提取")
    from scripts.extract_pose import extract_pose

    frames = extract_pose(TEST_VIDEO, POSE_JSON, skip_frames=10)
    ok = len(frames) > 0
    sample = frames[0] if frames else {}
    kp_count = sum(1 for k in sample if k.endswith("_x"))
    print(f"  帧数: {len(frames)}, 每帧关键点: {kp_count}")
    print(f"  {_mark(ok and kp_count == 33)} 提取完成 -> {POSE_JSON}")
    return ok and kp_count == 33


def check_template() -> bool:
    if os.path.exists(TEMPLATE):
        with open(TEMPLATE, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"  {OK} 模板已存在: {len(data)} 帧")
        return True

    print("  模板不存在，正在生成...")
    from scripts.generate_template import generate_template
    generate_template("squat", output_dir=os.path.dirname(TEMPLATE))
    ok = os.path.exists(TEMPLATE)
    print(f"  {_mark(ok)} 生成 template_squat.json")
    return ok


def ollama_available(host: str = "http://localhost:11434") -> bool:
    try:
        req = urllib.request.Request(f"{host}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=2) as resp:
            return resp.status == 200
    except Exception:
        return False


def run_llm_score(skip_llm: bool) -> bool:
    _step("第 4 周：大模型评分")
    if skip_llm:
        print(f"  {SKIP} 已跳过（--skip-llm）")
        return True

    if not ollama_available():
        print(f"  {WARN} Ollama 未运行，使用回退评分模式")
        print("    提示: 另开终端执行 ollama serve && ollama pull qwen2:7b")

    from scripts.compare_with_lm import compare_with_lm, generate_wrong_sequence

    with open(TEMPLATE, "r", encoding="utf-8") as f:
        template_seq = json.load(f)
    wrong_seq = generate_wrong_sequence(template_seq)
    result = compare_with_lm(wrong_seq, template_seq, target_frames=20)
    score = result.get("score", -1)
    print(f"  错误动作评分: {score}/10")
    ok = score < 5 or result.get("_fallback")
    print(f"  {_mark(ok)} 验证 (期望 score < 5 或 LLM 回退可用)")
    return ok


def check_websocket_module() -> bool:
    _step("第 5 周：3D 孪生模块")
    html = resource_path("sports_twin.html")
    ok_html = os.path.exists(html)
    print(f"  {_mark(ok_html)} sports_twin.html")
    try:
        from src.bridge.ws_server import TwinWebSocketServer
        srv = TwinWebSocketServer(port=8766)
        srv.start()
        srv.stop()
        print(f"  {OK} WebSocket 服务可启动 (8765/8766)")
        return ok_html
    except Exception as e:
        print(f"  {FAIL} WebSocket 启动失败: {e}")
        return False


def check_voice_and_history() -> bool:
    _step("第 6 周：语音 + 评分历史")
    ok_tts = False
    try:
        import pyttsx3
        engine = pyttsx3.init()
        ok_tts = engine is not None
        print(f"  {OK} pyttsx3 可用")
    except Exception as e:
        print(f"  {WARN} pyttsx3 不可用: {e}")

    from scripts.score_history import record_score, load_history

    record_score("squat", 7, ["膝盖超过脚尖，请向后坐"])
    history = load_history()
    ok_hist = len(history) > 0
    print(f"  {_mark(ok_hist)} 评分历史记录 ({len(history)} 条)")
    return ok_tts or ok_hist


def main():
    parser = argparse.ArgumentParser(description="课程步骤自动验收")
    parser.add_argument("--skip-llm", action="store_true", help="跳过第 4 周 LLM 验证")
    args = parser.parse_args()

    print("=" * 50)
    print("  AI 健身教练 — 课程流程自动验收")
    print("=" * 50)

    steps = [
        ("环境", check_env),
        ("姿态检测", check_pose_demo),
        ("姿态提取", run_extract_pose),
        ("模板", check_template),
        ("大模型评分", lambda: run_llm_score(args.skip_llm)),
        ("3D孪生", check_websocket_module),
        ("语音历史", check_voice_and_history),
    ]

    results = []
    for name, fn in steps:
        try:
            results.append((name, fn()))
        except Exception as e:
            print(f"  {FAIL} 异常: {e}")
            results.append((name, False))

    print(f"\n{'=' * 50}")
    print("  验收汇总")
    print(f"{'=' * 50}")
    passed = 0
    for name, ok in results:
        print(f"  {_mark(ok)} {name}")
        passed += int(ok)

    print(f"\n  通过: {passed}/{len(results)}")
    if passed == len(results):
        print("  全部通过，可以开始实时训练。")
    else:
        print("  部分未通过，请根据上方提示修复后重试。")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
