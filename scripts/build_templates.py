"""重建标准动作模板.

默认: 生物力学 FK 生成「理想标准动作」(一个完整周期, 90帧)
可选: --from-video 从参考视频提取 (需自行提供标准动作录像)

用法:
  python scripts/build_templates.py                 # 全部 FK 标准模板
  python scripts/build_templates.py -m squat        # 仅深蹲
  python scripts/build_templates.py -m squat --from-video ref.mp4
"""

import argparse
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.utils.paths import resource_path, data_path

MOVEMENTS = ["squat", "deadlift", "pushup", "pullup", "plank", "shooting"]
TEMPLATE_FRAMES = 90


def build_from_fk(movement: str) -> str:
    from scripts.generate_template import generate_template
    out_dir = os.path.dirname(resource_path("templates", "template_squat.json"))
    return generate_template(movement, output_dir=out_dir)


def build_from_video(movement: str, video: str, skip: int = 2) -> str:
    from scripts.extract_pose import extract_pose
    from scripts.template_utils import refine_for_template

    out_path = resource_path("templates", f"template_{movement}.json")
    if not os.path.isabs(video):
        video = data_path(video)
    if not os.path.exists(video):
        raise FileNotFoundError(f"参考视频不存在: {video}")

    raw_path = out_path + ".raw.json"
    extract_pose(video, output_path=raw_path, skip_frames=skip, coord_space="world")
    with open(raw_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    refined = refine_for_template(raw, target_frames=TEMPLATE_FRAMES)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(refined, f, ensure_ascii=False, indent=2)
    try:
        os.remove(raw_path)
    except OSError:
        pass

    print(f"[build_templates] 视频模板: {len(raw)} 帧 → 精炼 {len(refined)} 帧")
    return out_path


def build_one(movement: str, from_video: str = None, skip: int = 2) -> str:
    if from_video:
        print(f"[build_templates] {movement}: 从参考视频提取并精炼")
        return build_from_video(movement, from_video, skip)

    if movement == "squat":
        print(f"[build_templates] {movement}: 实测关键帧标准深蹲 ({TEMPLATE_FRAMES} 帧)")
    else:
        print(f"[build_templates] {movement}: FK 生物力学标准动作 ({TEMPLATE_FRAMES} 帧)")
    return build_from_fk(movement)


def main():
    parser = argparse.ArgumentParser(description="重建标准动作模板")
    parser.add_argument("-m", "--movement", choices=MOVEMENTS, default=None)
    parser.add_argument("--from-video", metavar="VIDEO",
                        help="从标准动作参考视频提取 (不提供则用 FK 理想模板)")
    parser.add_argument("--skip", type=int, default=2, help="视频下采样 (仅 --from-video)")
    args = parser.parse_args()

    targets = [args.movement] if args.movement else MOVEMENTS
    print("=" * 50)
    print("  重建标准模板")
    print("  默认: 实测关键帧/FK | 可选: --from-video 参考录像")
    print("=" * 50)

    if args.from_video and not args.movement:
        parser.error("--from-video 需配合 -m 指定动作，例如: -m squat --from-video ref.mp4")

    for m in targets:
        vid = args.from_video if m == args.movement else None
        path = build_one(m, from_video=vid, skip=args.skip)
        print(f"  -> {path}\n")

    print("完成。请刷新浏览器 (Ctrl+F5)。")


if __name__ == "__main__":
    main()
