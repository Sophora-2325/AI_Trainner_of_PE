"""大模型动作比对评分 — 第4周核心功能.

compare_with_lm(current_seq, template_seq):
  1. 将两段骨骼点序列采样到相同长度 (各20帧)
  2. 构造 Prompt 发送给大模型
  3. 解析返回的 JSON: {"score": 0-10整数, "suggestions": ["建议1","建议2","建议3"]}

运行方式:
  python scripts/compare_with_lm.py \
      --user user_pose.json \
      --template templates/template_squat.json

验证方式:
  python scripts/compare_with_lm.py --test  # 自动生成错误动作并验证 score < 5

依赖:
  大模型通过 Ollama API 调用 (ollama_host 默认 http://localhost:11434)
  需先启动: ollama serve 并拉取模型: ollama pull qwen2:7b
"""

import json
import os
import sys
import argparse
import math
import copy
import random
import urllib.request
from typing import Optional


# ─── 核心关节索引 (MediaPipe 33点中用于动作分析的主要关节) ───

MAJOR_JOINT_INDICES = {
    11: "left_shoulder", 12: "right_shoulder",
    13: "left_elbow",    14: "right_elbow",
    15: "left_wrist",    16: "right_wrist",
    23: "left_hip",      24: "right_hip",
    25: "left_knee",     26: "right_knee",
    27: "left_ankle",    28: "right_ankle",
    29: "left_heel",     30: "right_heel",
    31: "left_foot_index", 32: "right_foot_index",
}

ALL_LANDMARK_NAMES = [
    "nose",
    "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear",
    "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_pinky", "right_pinky",
    "left_index", "right_index",
    "left_thumb", "right_thumb",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]


def _sample_sequence(seq: list[dict], target_frames: int = 20) -> list[dict]:
    """将关键点序列采样到指定帧数 (线性插值).

    Args:
        seq: 原始序列, [{frame, joint_x, joint_y, joint_z, ...}, ...]
        target_frames: 目标帧数

    Returns:
        采样后的序列
    """
    n = len(seq)
    if n < 2:
        return seq * target_frames if seq else []

    sampled = []
    for i in range(target_frames):
        src_idx = (i / (target_frames - 1)) * (n - 1) if target_frames > 1 else 0.0
        idx_lo = int(math.floor(src_idx))
        idx_hi = min(idx_lo + 1, n - 1)
        frac = src_idx - idx_lo

        frame_data = {"frame": i}
        for key in seq[0]:
            if key == "frame":
                continue
            v_lo = seq[idx_lo].get(key, 0.0)
            v_hi = seq[idx_hi].get(key, 0.0)
            frame_data[key] = round(v_lo + (v_hi - v_lo) * frac, 6)

        sampled.append(frame_data)

    return sampled


def _serialize_frame(frame: dict, compact: bool = True) -> str:
    """将一帧关键点序列化为可读文本.

    Args:
        frame: {frame, nose_x, nose_y, nose_z, ...}
        compact: True=只输出主要关节, False=输出全部33点

    Returns:
        序列化字符串
    """
    if compact:
        parts = []
        for idx, name in MAJOR_JOINT_INDICES.items():
            x = frame.get(f"{name}_x", 0)
            y = frame.get(f"{name}_y", 0)
            z = frame.get(f"{name}_z", 0)
            parts.append(f"{name}({x:.3f},{y:.3f},{z:.3f})")
        return f"帧{frame['frame']}: {' '.join(parts)}"
    else:
        parts = [f"帧{frame['frame']}:"]
        for name in ALL_LANDMARK_NAMES:
            x = frame.get(f"{name}_x", 0)
            y = frame.get(f"{name}_y", 0)
            z = frame.get(f"{name}_z", 0)
            parts.append(f"{name}({x:.3f},{y:.3f},{z:.3f})")
        return " ".join(parts)


def _build_prompt(current_seq: list[dict], template_seq: list[dict]) -> str:
    """构造大模型评分的 Prompt.

    Args:
        current_seq: 用户动作序列 (20帧)
        template_seq: 标准模板序列 (20帧)

    Returns:
        完整的 Prompt 字符串
    """
    user_lines = [_serialize_frame(f, compact=True) for f in current_seq]
    template_lines = [_serialize_frame(f, compact=True) for f in template_seq]

    prompt = f"""对比以下两段骨骼点序列(每帧33点)。第一段是用户动作，第二段是标准模板。
输出JSON:{{"score":0-10整数,"suggestions":["建议1","建议2","建议3"]}}

[用户动作 — 20帧]
{chr(10).join(user_lines)}

[标准模板 — 20帧]
{chr(10).join(template_lines)}"""

    return prompt


def _call_ollama(prompt: str, model: str = "qwen2:7b",
                 host: str = "http://localhost:11434",
                 timeout: int = 60) -> Optional[dict]:
    """调用 Ollama API 生成评分结果.

    Args:
        prompt: 完整 prompt
        model: Ollama 模型名
        host: Ollama 服务地址
        timeout: 请求超时(秒)

    Returns:
        解析后的 JSON dict 或 None
    """
    body = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,     # 低温确保输出稳定
            "num_predict": 150,     # 足够的 token 用于返回 JSON
        },
    }).encode("utf-8")

    try:
        req = urllib.request.Request(
            f"{host}/api/generate",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())

        response_text = data.get("response", "").strip()
        print(f"[compare_with_lm] LLM 原始响应:\n{response_text}\n")

        # 尝试从响应中提取 JSON
        return _parse_json_response(response_text)

    except urllib.error.URLError as e:
        print(f"[compare_with_lm] Ollama 连接失败 ({host}): {e}")
        print("[compare_with_lm] 请确认: 1) ollama serve 已启动  2) 已执行 ollama pull qwen2:7b")
        return None
    except Exception as e:
        print(f"[compare_with_lm] 调用失败: {e}")
        return None


def _parse_json_response(text: str) -> Optional[dict]:
    """从 LLM 响应文本中提取 JSON.

    容错策略:
      1. 尝试直接解析整段文本
      2. 尝试匹配 {...} 模式
      3. 尝试修复常见问题 (如未闭合引号)
    """
    # 策略1: 直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 策略2: 查找 JSON 块
    import re
    # 匹配 { ... } 块 (非贪婪)
    match = re.search(r'\{[^{}]*"score"\s*:\s*\d+[^{}]*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # 策略3: 尝试提取 score 和 suggestions
    score_match = re.search(r'"score"\s*:\s*(\d+)', text)
    suggestions_match = re.findall(r'"建议\d+"\s*:\s*"([^"]*)"', text)

    if score_match:
        result = {"score": int(score_match.group(1))}
        if suggestions_match:
            result["suggestions"] = suggestions_match
        else:
            result["suggestions"] = ["请检查动作姿势"]
        return result

    print("[compare_with_lm] 无法从响应中解析 JSON")
    return None


def compare_with_lm(
    current_seq: list[dict],
    template_seq: list[dict],
    model: str = "qwen2:7b",
    host: str = "http://localhost:11434",
    target_frames: int = 20,
) -> dict:
    """对比用户动作与标准模板，通过大模型评分.

    Args:
        current_seq: 用户动作关键点序列 [{frame, nose_x, nose_y, ...}, ...]
        template_seq: 标准模板关键点序列
        model: Ollama 模型名
        host: Ollama 服务地址
        target_frames: 采样目标帧数 (默认20)

    Returns:
        {"score": int, "suggestions": [str, str, str]} 或
        {"error": str} 失败时

    验证标准: 明显错误动作的 score 应 < 5
    """
    print(f"[compare_with_lm] 输入: 用户={len(current_seq)}帧, 模板={len(template_seq)}帧")

    # 1. 采样到相同长度
    user_sampled = _sample_sequence(current_seq, target_frames)
    tmpl_sampled = _sample_sequence(template_seq, target_frames)
    print(f"[compare_with_lm] 采样后: 各 {target_frames} 帧")

    # 2. 构造 Prompt
    prompt = _build_prompt(user_sampled, tmpl_sampled)
    print(f"[compare_with_lm] Prompt 长度: {len(prompt)} 字符")

    # 3. 调用大模型
    result = _call_ollama(prompt, model=model, host=host)

    if result is None:
        # 回退: 基于规则计算一个近似分数
        return _fallback_compare(user_sampled, tmpl_sampled)

    # 4. 规范化结果
    score = result.get("score", 5)
    suggestions = result.get("suggestions", ["请检查动作姿势"])

    # 确保 score 在 0-10 范围
    score = max(0, min(10, int(score)))

    # 确保有3条建议
    while len(suggestions) < 3:
        suggestions.append("请继续保持" if score >= 7 else "请注意动作规范")

    result["score"] = score
    result["suggestions"] = suggestions[:3]

    return result


def _fallback_compare(user_seq: list[dict], template_seq: list[dict]) -> dict:
    """大模型不可用时的回退评分 (基于关键关节平均偏差).

    偏差映射:
      0.00-0.02 → score 9-10
      0.02-0.05 → score 7-8
      0.05-0.10 → score 4-6
      >0.10     → score 1-3
    """
    total_dev = 0.0
    count = 0

    for uf, tf in zip(user_seq, template_seq):
        for idx, name in MAJOR_JOINT_INDICES.items():
            for axis in ["x", "y", "z"]:
                key = f"{name}_{axis}"
                total_dev += abs(uf.get(key, 0.0) - tf.get(key, 0.0))
                count += 1

    avg_dev = total_dev / max(count, 1)

    if avg_dev < 0.02:
        score = 9
        adv = "动作非常标准"
    elif avg_dev < 0.05:
        score = 7
        adv = "动作基本标准，注意细节控制"
    elif avg_dev < 0.10:
        score = 5
        adv = "存在明显偏差，请对照标准动作调整"
    elif avg_dev < 0.20:
        score = 3
        adv = "动作偏差较大，建议放慢速度重新练习"
    else:
        score = 1
        adv = "动作严重偏差，请停止并重新学习标准动作"

    print(f"[compare_with_lm] 回退评分: avg_dev={avg_dev:.4f} → score={score}")

    return {
        "score": score,
        "suggestions": [adv, "建议对照镜子或录像自我检查", "建议从较轻负荷开始练习"],
        "_fallback": True,
        "_avg_deviation": round(avg_dev, 4),
    }


def generate_wrong_sequence(template_seq: list[dict]) -> list[dict]:
    """根据标准模板生成一个"明显错误"的动作序列 (用于验证).

    错误类型:
      - 膝内扣 (膝盖X坐标偏内)
      - 深度不足 (膝角不够小)
      - 躯干过于直立

    Args:
        template_seq: 标准模板序列

    Returns:
        故意引入错误的动作序列
    """
    wrong_seq = copy.deepcopy(template_seq)

    for f, frame in enumerate(wrong_seq):
        phase_progress = f / max(len(wrong_seq) - 1, 1)

        # 在下降/底部阶段引入膝内扣 (膝盖X坐标向内侧偏移)
        if 0.2 < phase_progress < 0.6:
            # 右膝内扣: X坐标减小 (向中线移动)
            frame["right_knee_x"] = round(frame.get("right_knee_x", 0) - 0.12, 6)
            # 左膝内扣
            frame["left_knee_x"] = round(frame.get("left_knee_x", 0) + 0.12, 6)

        # 深度不足: 底部阶段膝角不够小 → 臀部Y坐标不够低
        if 0.4 < phase_progress < 0.55:
            for joint in ["left_knee", "right_knee", "left_hip", "right_hip"]:
                frame[f"{joint}_y"] = round(frame.get(f"{joint}_y", 0) + 0.20, 6)

        # 躯干过于直立: 肩部X偏移减小
        if 0.2 < phase_progress < 0.7:
            frame["left_shoulder_x"] = round(frame.get("left_shoulder_x", 0) * 0.7, 6)
            frame["right_shoulder_x"] = round(frame.get("right_shoulder_x", 0) * 0.7, 6)

    return wrong_seq


# ─── CLI ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="大模型动作比对评分")
    parser.add_argument("--user", "-u", default=None,
                        help="用户动作 JSON 文件路径")
    parser.add_argument("--template", "-t", default="templates/template_squat.json",
                        help="标准模板 JSON 文件路径")
    parser.add_argument("--model", default="qwen2:7b",
                        help="Ollama 模型名 (默认: qwen2:7b)")
    parser.add_argument("--host", default="http://localhost:11434",
                        help="Ollama 服务地址")
    parser.add_argument("--frames", type=int, default=20,
                        help="采样目标帧数 (默认: 20)")
    parser.add_argument("--test", action="store_true",
                        help="运行验证: 生成错误动作并确保 score < 5")
    parser.add_argument("--output", "-o", default=None,
                        help="输出结果 JSON 文件")
    args = parser.parse_args()

    # ─── 验证模式 ────────────────────────────────────────
    if args.test:
        print("=" * 60)
        print("[验证] 大模型评分 — 错误动作应得分 < 5")
        print("=" * 60)

        # 加载模板
        tmpl_path = args.template
        if not os.path.exists(tmpl_path):
            print(f"[验证] 模板文件不存在: {tmpl_path}")
            print("[验证] 请先运行: python scripts/generate_template.py")
            sys.exit(1)

        with open(tmpl_path, "r", encoding="utf-8") as f:
            template_seq = json.load(f)

        # 生成错误动作
        wrong_seq = generate_wrong_sequence(template_seq)

        # 调用大模型比对
        result = compare_with_lm(
            wrong_seq, template_seq,
            model=args.model, host=args.host,
            target_frames=args.frames,
        )

        score = result.get("score", -1)
        suggestions = result.get("suggestions", [])

        print(f"\n{'='*60}")
        print(f"[验证结果]")
        print(f"  评分: {score}/10")
        print(f"  建议:")
        for i, s in enumerate(suggestions, 1):
            print(f"    {i}. {s}")

        if score < 5:
            print(f"\n  ✓ 验证通过: 错误动作得分 {score} < 5")
        else:
            print(f"\n  ✗ 验证失败: 错误动作得分 {score} >= 5, 不符合预期")
            print(f"    可能原因: 1) LLM 未正确理解任务  2) 错误动作不够明显")
            if result.get("_fallback"):
                print(f"    注意: 当前使用回退模式 (LLM不可用), 请启动 Ollama 后重试")

        return

    # ─── 正常评分模式 ────────────────────────────────────
    if args.user is None:
        print("请指定用户动作 JSON 文件: --user <path>")
        print("或运行验证模式: --test")
        sys.exit(1)

    if not os.path.exists(args.user):
        print(f"用户动作文件不存在: {args.user}")
        sys.exit(1)

    if not os.path.exists(args.template):
        print(f"模板文件不存在: {args.template}")
        print("请先运行: python scripts/generate_template.py")
        sys.exit(1)

    with open(args.user, "r", encoding="utf-8") as f:
        current_seq = json.load(f)
    with open(args.template, "r", encoding="utf-8") as f:
        template_seq = json.load(f)

    result = compare_with_lm(
        current_seq, template_seq,
        model=args.model, host=args.host,
        target_frames=args.frames,
    )

    print(f"\n{'='*60}")
    print(f"[评分结果]")
    print(f"  综合评分: {result['score']}/10")
    print(f"  改进建议:")
    for i, s in enumerate(result.get("suggestions", []), 1):
        print(f"    {i}. {s}")
    print(f"{'='*60}")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"结果已保存: {args.output}")


if __name__ == "__main__":
    main()
