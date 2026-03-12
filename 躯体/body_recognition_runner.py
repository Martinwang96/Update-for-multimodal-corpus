#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Body recognition stage runner.

Pipeline:
1) Video -> MMPoseInferencer -> raw predictions JSON
2) raw predictions JSON -> best-instance filtering -> *-tonly.json

CLI output:
- Writes structured summary JSON (if --summary-json provided)
- Prints human-readable logs
- Prints machine-readable progress lines: PROGRESS_JSON:{...}
- Exit code 0 on success, non-zero on failure
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List


PROGRESS_PREFIX = "PROGRESS_JSON:"


def filter_best_instance_per_frame(data: List[Dict[str, Any]], score_thresh: float) -> List[Dict[str, Any]]:
    filtered: List[Dict[str, Any]] = []
    for idx, frame in enumerate(data):
        instances = frame.get("instances", []) if isinstance(frame, dict) else []
        frame_id = frame.get("frame_id", idx) if isinstance(frame, dict) else idx

        if not instances:
            filtered.append({"frame_id": frame_id, "instances": []})
            continue

        counts: List[int] = []
        for inst in instances:
            scores = inst.get("keypoint_scores", []) if isinstance(inst, dict) else []
            try:
                cnt = sum(1 for s in scores if float(s) > score_thresh)
            except Exception:
                cnt = 0
            counts.append(cnt)

        if not counts:
            filtered.append({"frame_id": frame_id, "instances": []})
            continue

        best_idx = max(range(len(counts)), key=lambda i: counts[i])
        best_inst = instances[best_idx]
        filtered.append({"frame_id": frame_id, "instances": [best_inst]})

    return filtered


def _resolve_device(device: str) -> str:
    if device and device.lower() != "auto":
        return device
    try:
        import torch

        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _find_raw_json(predictions_dir: Path, video_stem: str) -> Path:
    if not predictions_dir.exists():
        raise FileNotFoundError(f"predictions directory not found: {predictions_dir}")

    all_json = sorted(predictions_dir.rglob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not all_json:
        raise FileNotFoundError(f"no json found under predictions directory: {predictions_dir}")

    candidates = [
        p
        for p in all_json
        if p.stem.startswith(video_stem) and not p.stem.endswith("-tonly") and "tonly" not in p.stem
    ]
    if candidates:
        return candidates[0]

    non_tonly = [p for p in all_json if "tonly" not in p.stem]
    if non_tonly:
        return non_tonly[0]

    return all_json[0]


def _get_video_total_frames(video_path: Path) -> int:
    try:
        import cv2
    except Exception:
        return 0

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    except Exception:
        total = 0
    finally:
        cap.release()
    return max(0, total)


def _emit_progress(frames_done: int, frames_total: int):
    payload = {
        "stage": "recognition",
        "frames_done": int(max(0, frames_done)),
        "frames_total": int(max(0, frames_total)),
    }
    print(PROGRESS_PREFIX + json.dumps(payload, ensure_ascii=False), flush=True)


def run_body_recognition(
    video_path: str,
    output_dir: str,
    pose2d: str = "rtmo",
    batch_size: int = 16,
    device: str = "auto",
    save_vis: bool = False,
    score_thresh: float = 0.6,
) -> Dict[str, Any]:
    vp = Path(video_path).expanduser().resolve()
    if not vp.is_file():
        raise FileNotFoundError(f"video not found: {vp}")

    out_root = Path(output_dir).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    predictions_dir = out_root / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)

    vis_dir = out_root / "vis"
    if save_vis:
        vis_dir.mkdir(parents=True, exist_ok=True)

    try:
        from mmpose.apis import MMPoseInferencer
    except Exception as e:
        raise RuntimeError(
            "failed to import mmpose (and dependencies like mmcv/mmengine). "
            "Please use a Python environment with mmpose installed."
        ) from e

    runtime_device = _resolve_device(device)
    inferencer = MMPoseInferencer(pose2d=pose2d, device=runtime_device)

    kwargs = {
        "inputs": str(vp),
        "batch_size": int(batch_size),
        "out_dir": str(out_root),
        "show": False,
    }
    if save_vis:
        kwargs["vis_out_dir"] = str(vis_dir)

    total_frames = _get_video_total_frames(vp)
    if total_frames > 0:
        _emit_progress(0, total_frames)

    results_generator = inferencer(**kwargs)

    frame_count = 0
    for i, _ in enumerate(results_generator):
        frame_count = i + 1
        if frame_count == 1 or frame_count % 10 == 0:
            _emit_progress(frame_count, total_frames)

    final_total = total_frames if total_frames > 0 else frame_count
    _emit_progress(frame_count, final_total)

    raw_json_path = _find_raw_json(predictions_dir, vp.stem)

    with raw_json_path.open("r", encoding="utf-8") as f:
        raw_data = json.load(f)
    if not isinstance(raw_data, list):
        raise ValueError(f"unexpected raw json schema, expected list frames: {raw_json_path}")

    filtered_data = filter_best_instance_per_frame(raw_data, float(score_thresh))

    tonly_path = raw_json_path.with_name(f"{raw_json_path.stem}-tonly.json")
    with tonly_path.open("w", encoding="utf-8") as f:
        json.dump(filtered_data, f, ensure_ascii=False, indent=2)

    return {
        "video_path": str(vp),
        "output_dir": str(out_root),
        "predictions_dir": str(predictions_dir),
        "vis_dir": str(vis_dir),
        "raw_json_path": str(raw_json_path),
        "tonly_json_path": str(tonly_path),
        "pose2d": pose2d,
        "batch_size": int(batch_size),
        "device_requested": device,
        "device_used": runtime_device,
        "save_vis": bool(save_vis),
        "score_thresh": float(score_thresh),
        "frame_count": int(frame_count),
        "inference_finished": True,
        "inference_message": "推理已结束，JSON文件已生成。",
    }


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run body recognition and generate raw/tonly JSON.")
    p.add_argument("--video-path", required=True, help="Input video path")
    p.add_argument("--output-dir", required=True, help="Recognition output root directory")
    p.add_argument("--pose2d", default="rtmo", help="MMPose pose2d alias (default: rtmo)")
    p.add_argument("--batch-size", type=int, default=16, help="Inference batch size (default: 16)")
    p.add_argument("--device", default="auto", help="Device string, e.g. auto/cuda:0/cpu")
    p.add_argument("--save-vis", action="store_true", help="Whether to save visualization video")
    p.add_argument("--score-thresh", type=float, default=0.6, help="Filtering score threshold (default: 0.6)")
    p.add_argument("--summary-json", default="", help="Optional path to write structured summary json")
    return p


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    try:
        result = run_body_recognition(
            video_path=args.video_path,
            output_dir=args.output_dir,
            pose2d=args.pose2d,
            batch_size=args.batch_size,
            device=args.device,
            save_vis=args.save_vis,
            score_thresh=args.score_thresh,
        )

        if args.summary_json:
            summary_path = Path(args.summary_json).expanduser().resolve()
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

        print("[body-recognition] success")
        print(f"raw_json_path: {result['raw_json_path']}")
        print(f"tonly_json_path: {result['tonly_json_path']}")
        print("[body-recognition] inference finished: 推理已结束，JSON文件已生成。")
        if args.summary_json:
            print(f"summary_json: {Path(args.summary_json).expanduser().resolve()}")
        return 0

    except Exception as e:
        print("[body-recognition] failed", file=sys.stderr)
        print(f"error: {type(e).__name__}: {e}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
