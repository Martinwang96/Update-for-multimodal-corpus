#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import importlib.util
import json
import sys
import threading
import time
import traceback
import uuid
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from flask import Flask, abort, jsonify, render_template, request, send_file, url_for

BASE = Path(__file__).resolve().parent
COMBINED_FILE = BASE / (Path(__file__).name.replace("-web.py", "-2d.py"))
ORDER = ["tilt", "shrug", "displacement", "rotation"]
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}

JOBS: Dict[str, Dict[str, Any]] = {}
JOB_LOCK = threading.Lock()
MAX_JOB_LOG_LINES = 200


def load_combined_module():
    spec = importlib.util.spec_from_file_location("combined2d", str(COMBINED_FILE))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {COMBINED_FILE}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


COMBINED = load_combined_module()
APP = Flask(
    __name__,
    template_folder=str(BASE / "templates"),
    static_folder=str(BASE / "static"),
)


def _to_float(v: str, default: float) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _to_int(v: str, default: int) -> int:
    try:
        return int(v)
    except Exception:
        return default


def _opt_float(v: str) -> Optional[float]:
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None
    try:
        return float(s)
    except Exception:
        return None


def _is_odd_ge_3(v: int) -> bool:
    return v >= 3 and (v % 2 == 1)


def _require_poly_lt_window(name: str, poly: int, win: int):
    if not (1 <= poly < win):
        raise ValueError(f"{name} 参数错误：需要满足 1 <= poly < window。当前 poly={poly}, window={win}")


def _require_odd_window(name: str, win: int):
    if not _is_odd_ge_3(win):
        raise ValueError(f"{name} 参数错误：window 需为奇数且 >= 3。当前 window={win}")


def _resolve_video_path(path_str: str) -> Path:
    p = Path(path_str).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"视频文件不存在: {p}")
    if p.suffix.lower() not in VIDEO_EXTENSIONS:
        allowed = ", ".join(sorted(VIDEO_EXTENSIONS))
        raise ValueError(f"视频格式不支持: {p.suffix}。支持: {allowed}")
    return p


def _validate_runtime_inputs(config: Dict[str, Any]):
    if float(config.get("fps", 0)) <= 0:
        raise ValueError("FPS 必须大于 0。")

    conf = float(config.get("min_confidence", -1))
    if not (0.0 <= conf <= 1.0):
        raise ValueError("最小关键点置信度需在 0~1。")

    rec = config.get("recognition", {})
    if int(rec.get("batch_size", 0)) <= 0:
        raise ValueError("body-recognition 的 batch_size 必须 > 0。")
    if float(rec.get("score_thresh", -1)) < 0:
        raise ValueError("body-recognition 的 score_thresh 不能小于 0。")

    params = config.get("params", {})
    selected = {m for m in config.get("modules", ORDER) if m in ORDER}

    if "tilt" in selected:
        tilt = params.get("tilt", {})
        _require_odd_window("倾斜", int(tilt.get("smooth_window", 0)))
        _require_poly_lt_window("倾斜", int(tilt.get("smooth_poly", 0)), int(tilt.get("smooth_window", 0)))

    if "shrug" in selected:
        shrug = params.get("shrug", {})
        _require_odd_window("耸肩", int(shrug.get("smooth_window", 0)))
        _require_poly_lt_window("耸肩", int(shrug.get("smooth_poly", 0)), int(shrug.get("smooth_window", 0)))

    if "displacement" in selected:
        displacement = params.get("displacement", {})
        _require_odd_window("位移-平滑", int(displacement.get("smooth_window", 0)))
        _require_poly_lt_window(
            "位移-平滑",
            int(displacement.get("smooth_poly", 0)),
            int(displacement.get("smooth_window", 0)),
        )
        _require_odd_window("位移-运动学", int(displacement.get("kinematics_window", 0)))
        _require_poly_lt_window(
            "位移-运动学",
            int(displacement.get("kinematics_poly", 0)),
            int(displacement.get("kinematics_window", 0)),
        )

    if "rotation" in selected:
        rotation = params.get("rotation", {})
        _require_odd_window("转动", int(rotation.get("smooth_window", 0)))
        _require_poly_lt_window("转动", int(rotation.get("smooth_poly", 0)), int(rotation.get("smooth_window", 0)))


def build_form_defaults() -> Dict[str, Any]:
    dp = COMBINED.build_default_params()
    return {
        "input_path": "",
        "output_root": str(BASE / "combined_2d_runs"),
        "rec_python": sys.executable,
        "rec_pose2d": "rtmo",
        "rec_batch_size": "16",
        "rec_device": "auto",
        "rec_score_thresh": "0.6",
        "rec_json_source": "tonly",
        "rec_save_vis": False,
        "fps": "30.0",
        "min_confidence": "0.3",
        "save_plots": False,
        "parallel": True,
        "modules": ORDER.copy(),
        "tilt_smooth_window": str(dp["tilt"]["smooth_window"]),
        "tilt_smooth_poly": str(dp["tilt"]["smooth_poly"]),
        "tilt_outlier_diff_threshold": str(dp["tilt"]["outlier_diff_threshold"]),
        "tilt_threshold_override": "",
        "tilt_min_duration_sec": str(dp["tilt"]["min_duration_sec"]),
        "tilt_max_duration_sec": str(dp["tilt"]["max_duration_sec"]),
        "tilt_merge_gap_sec": str(dp["tilt"]["merge_gap_sec"]),
        "shrug_smooth_window": str(dp["shrug"]["smooth_window"]),
        "shrug_smooth_poly": str(dp["shrug"]["smooth_poly"]),
        "shrug_baseline_window_sec": str(dp["shrug"]["baseline_window_sec"]),
        "shrug_threshold_override": "",
        "shrug_min_duration_sec": str(dp["shrug"]["min_duration_sec"]),
        "shrug_max_duration_sec": str(dp["shrug"]["max_duration_sec"]),
        "shrug_merge_gap_sec": str(dp["shrug"]["merge_gap_sec"]),
        "displacement_smooth_window": str(dp["displacement"]["smooth_window"]),
        "displacement_smooth_poly": str(dp["displacement"]["smooth_poly"]),
        "displacement_baseline_window_sec": str(dp["displacement"]["baseline_window_sec"]),
        "displacement_kinematics_window": str(dp["displacement"]["kinematics_window"]),
        "displacement_kinematics_poly": str(dp["displacement"]["kinematics_poly"]),
        "displacement_threshold_pa_override": "",
        "displacement_threshold_move_override": "",
        "displacement_min_duration_sec": str(dp["displacement"]["min_duration_sec"]),
        "displacement_max_duration_sec": str(dp["displacement"]["max_duration_sec"]),
        "displacement_merge_gap_sec": str(dp["displacement"]["merge_gap_sec"]),
        "rotation_left_shoulder_idx": str(dp["rotation"]["left_shoulder_idx"]),
        "rotation_right_shoulder_idx": str(dp["rotation"]["right_shoulder_idx"]),
        "rotation_smooth_window": str(dp["rotation"]["smooth_window"]),
        "rotation_smooth_poly": str(dp["rotation"]["smooth_poly"]),
        "rotation_threshold_override": "",
        "rotation_min_duration_sec": str(dp["rotation"]["min_duration_sec"]),
        "rotation_max_duration_sec": str(dp["rotation"]["max_duration_sec"]),
        "rotation_merge_gap_sec": str(dp["rotation"]["merge_gap_sec"]),
    }


def _build_config_from_form() -> Dict[str, Any]:
    input_path_raw = str(request.form.get("input_path", "")).strip()
    if not input_path_raw:
        raise ValueError("请输入视频路径。")
    input_path = _resolve_video_path(input_path_raw)

    output_root = Path(request.form.get("output_root", str(BASE / "combined_2d_runs"))).expanduser().resolve()
    run_root = output_root / input_path.stem

    modules = request.form.getlist("modules") or ORDER.copy()
    modules = [m for m in ORDER if m in modules]
    if not modules:
        modules = ORDER.copy()

    config = {
        "entry_mode": "video",
        "video_path": str(input_path),
        "run_root": str(run_root),
        "export_layer_csv": True,
        "recognition": {
            "python": request.form.get("rec_python", sys.executable),
            "pose2d": request.form.get("rec_pose2d", "rtmo"),
            "batch_size": _to_int(request.form.get("rec_batch_size", "16"), 16),
            "device": request.form.get("rec_device", "auto"),
            "save_vis": bool(request.form.get("rec_save_vis")),
            "score_thresh": _to_float(request.form.get("rec_score_thresh", "0.6"), 0.6),
            "json_source": request.form.get("rec_json_source", "tonly"),
        },
        "fps": _to_float(request.form.get("fps", "30.0"), 30.0),
        "min_confidence": _to_float(request.form.get("min_confidence", "0.3"), 0.3),
        "save_plots": bool(request.form.get("save_plots")),
        "parallel": bool(request.form.get("parallel")),
        "modules": modules,
        "params": {
            "tilt": {
                "smooth_window": _to_int(request.form.get("tilt_smooth_window", "11"), 11),
                "smooth_poly": _to_int(request.form.get("tilt_smooth_poly", "3"), 3),
                "outlier_diff_threshold": _to_float(
                    request.form.get("tilt_outlier_diff_threshold", "25"),
                    25.0,
                ),
                "threshold_override": _opt_float(request.form.get("tilt_threshold_override", "")),
                "min_duration_sec": _to_float(request.form.get("tilt_min_duration_sec", "0.1"), 0.1),
                "max_duration_sec": _to_float(request.form.get("tilt_max_duration_sec", "5.0"), 5.0),
                "merge_gap_sec": _to_float(request.form.get("tilt_merge_gap_sec", "0.3"), 0.3),
            },
            "shrug": {
                "smooth_window": _to_int(request.form.get("shrug_smooth_window", "7"), 7),
                "smooth_poly": _to_int(request.form.get("shrug_smooth_poly", "2"), 2),
                "baseline_window_sec": _to_float(
                    request.form.get("shrug_baseline_window_sec", "1.5"),
                    1.5,
                ),
                "threshold_override": _opt_float(request.form.get("shrug_threshold_override", "")),
                "min_duration_sec": _to_float(request.form.get("shrug_min_duration_sec", "0.15"), 0.15),
                "max_duration_sec": _to_float(request.form.get("shrug_max_duration_sec", "2.0"), 2.0),
                "merge_gap_sec": _to_float(request.form.get("shrug_merge_gap_sec", "0.3"), 0.3),
            },
            "displacement": {
                "smooth_window": _to_int(request.form.get("displacement_smooth_window", "7"), 7),
                "smooth_poly": _to_int(request.form.get("displacement_smooth_poly", "2"), 2),
                "baseline_window_sec": _to_float(
                    request.form.get("displacement_baseline_window_sec", "2.0"),
                    2.0,
                ),
                "kinematics_window": _to_int(
                    request.form.get("displacement_kinematics_window", "9"),
                    9,
                ),
                "kinematics_poly": _to_int(request.form.get("displacement_kinematics_poly", "2"), 2),
                "threshold_pa_override": _opt_float(
                    request.form.get("displacement_threshold_pa_override", ""),
                ),
                "threshold_move_override": _opt_float(
                    request.form.get("displacement_threshold_move_override", ""),
                ),
                "min_duration_sec": _to_float(
                    request.form.get("displacement_min_duration_sec", "0.2"),
                    0.2,
                ),
                "max_duration_sec": _to_float(
                    request.form.get("displacement_max_duration_sec", "10.0"),
                    10.0,
                ),
                "merge_gap_sec": _to_float(
                    request.form.get("displacement_merge_gap_sec", "0.4"),
                    0.4,
                ),
            },
            "rotation": {
                "left_shoulder_idx": _to_int(request.form.get("rotation_left_shoulder_idx", "5"), 5),
                "right_shoulder_idx": _to_int(request.form.get("rotation_right_shoulder_idx", "6"), 6),
                "smooth_window": _to_int(request.form.get("rotation_smooth_window", "11"), 11),
                "smooth_poly": _to_int(request.form.get("rotation_smooth_poly", "3"), 3),
                "threshold_override": _opt_float(request.form.get("rotation_threshold_override", "")),
                "min_duration_sec": _to_float(request.form.get("rotation_min_duration_sec", "0.1"), 0.1),
                "max_duration_sec": _to_float(request.form.get("rotation_max_duration_sec", "2.0"), 2.0),
                "merge_gap_sec": _to_float(request.form.get("rotation_merge_gap_sec", "0.3"), 0.3),
            },
        },
    }
    _validate_runtime_inputs(config)
    return config


def _trim_job_logs(job: Dict[str, Any]):
    logs = job.get("logs", [])
    if len(logs) > MAX_JOB_LOG_LINES:
        job["logs"] = logs[-MAX_JOB_LOG_LINES:]


def _append_log(job: Dict[str, Any], message: str):
    msg = str(message).strip()
    if not msg:
        return
    logs = job.setdefault("logs", [])
    if not logs or logs[-1] != msg:
        logs.append(msg)
    _trim_job_logs(job)


def _clamp_progress(v: Any) -> float:
    try:
        x = float(v)
    except Exception:
        return 0.0
    return max(0.0, min(100.0, x))


def _update_job_progress(job_id: str, event: Dict[str, Any]):
    with JOB_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return

        stage = str(event.get("stage", job.get("stage", "prepare")))
        message = str(event.get("message", "")).strip()
        progress = _clamp_progress(event.get("progress", job.get("progress", 0.0)))
        eta = event.get("eta_sec", None)
        try:
            eta_val = None if eta is None else max(0.0, float(eta))
        except Exception:
            eta_val = job.get("eta_sec", None)

        job["status"] = "running"
        job["stage"] = stage
        job["progress"] = progress
        job["eta_sec"] = eta_val
        job["last_update"] = time.time()
        if message:
            job["message"] = message
            _append_log(job, message)


def _job_public_view(job: Dict[str, Any]) -> Dict[str, Any]:
    now = time.time()
    started = job.get("started_at")
    finished = job.get("finished_at")

    elapsed = 0.0
    if isinstance(started, (int, float)):
        if isinstance(finished, (int, float)):
            elapsed = max(0.0, finished - started)
        else:
            elapsed = max(0.0, now - started)

    return {
        "job_id": job.get("job_id"),
        "status": job.get("status"),
        "stage": job.get("stage"),
        "progress": float(job.get("progress", 0.0)),
        "eta_sec": job.get("eta_sec"),
        "elapsed_sec": elapsed,
        "message": job.get("message", ""),
        "error": job.get("error", ""),
        "summary": job.get("summary"),
        "created_at": job.get("created_at"),
        "started_at": started,
        "finished_at": finished,
        "logs": job.get("logs", [])[-30:],
    }


def _run_job(job_id: str, config: Dict[str, Any]):
    with JOB_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return
        job["status"] = "running"
        job["stage"] = "prepare"
        job["progress"] = 1.0
        job["eta_sec"] = None
        job["message"] = "准备阶段：任务已启动"
        job["started_at"] = time.time()
        job["last_update"] = time.time()
        _append_log(job, job["message"])

    try:
        summary = COMBINED.run_pipeline(config, progress_callback=lambda e: _update_job_progress(job_id, e))
        with JOB_LOCK:
            job = JOBS.get(job_id)
            if not job:
                return
            job["status"] = "completed"
            job["stage"] = "done"
            job["progress"] = 100.0
            job["eta_sec"] = 0.0
            job["message"] = "流程完成"
            job["summary"] = summary
            job["finished_at"] = time.time()
            job["last_update"] = time.time()
            _append_log(job, "流程完成")
    except Exception as e:
        err = f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}"
        with JOB_LOCK:
            job = JOBS.get(job_id)
            if not job:
                return
            job["status"] = "error"
            job["stage"] = "error"
            job["eta_sec"] = None
            job["message"] = "流程失败"
            job["error"] = err
            job["finished_at"] = time.time()
            job["last_update"] = time.time()
            _append_log(job, f"流程失败：{type(e).__name__}: {e}")


@APP.get("/preview-video")
def preview_video():
    raw_path = str(request.args.get("path", "")).strip()
    if not raw_path:
        abort(400, "missing path")

    try:
        video_path = _resolve_video_path(raw_path)
    except FileNotFoundError:
        abort(404, "video file not found")
    except Exception as e:
        abort(400, str(e))

    return send_file(str(video_path), conditional=True)


@APP.get("/api/preview-cover")
def preview_cover():
    raw_path = str(request.args.get("path", "")).strip()
    if not raw_path:
        abort(400, "缺少视频路径")

    try:
        video_path = _resolve_video_path(raw_path)
    except FileNotFoundError:
        abort(404, "视频文件不存在")
    except Exception as e:
        abort(400, str(e))

    try:
        import cv2
    except Exception as e:
        abort(500, f"无法导入 OpenCV：{type(e).__name__}: {e}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        abort(500, "无法打开视频文件")

    try:
        ok, frame = cap.read()
        if not ok or frame is None:
            abort(500, "无法读取视频首帧")
        enc_ok, buf = cv2.imencode(".jpg", frame)
        if not enc_ok:
            abort(500, "视频首帧编码失败")
        return send_file(BytesIO(buf.tobytes()), mimetype="image/jpeg", max_age=0)
    finally:
        cap.release()


@APP.post("/api/jobs")
def create_job():
    try:
        config = _build_config_from_form()
    except Exception as e:
        return jsonify({"ok": False, "error": f"{type(e).__name__}: {e}"}), 400

    job_id = uuid.uuid4().hex[:12]
    job = {
        "job_id": job_id,
        "status": "queued",
        "stage": "prepare",
        "progress": 0.0,
        "eta_sec": None,
        "message": "任务已创建",
        "error": "",
        "summary": None,
        "created_at": time.time(),
        "started_at": None,
        "finished_at": None,
        "last_update": time.time(),
        "logs": [],
        "run_root": config.get("run_root", ""),
    }

    with JOB_LOCK:
        JOBS[job_id] = job

    t = threading.Thread(target=_run_job, args=(job_id, config), daemon=True)
    t.start()

    return jsonify({
        "ok": True,
        "job_id": job_id,
        "run_root": config.get("run_root", ""),
        "preview_cover_url": url_for("preview_cover", path=config.get("video_path", "")),
    })


@APP.get("/api/jobs/<job_id>")
def get_job(job_id: str):
    with JOB_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return jsonify({"ok": False, "error": "job not found"}), 404
        data = _job_public_view(job)
    data["ok"] = True
    return jsonify(data)


@APP.get("/")
def index():
    return render_template(
        "综合处理-web.html",
        form=build_form_defaults(),
        order=ORDER,
    )


if __name__ == "__main__":
    print("Open: http://127.0.0.1:7860")
    APP.run(host="127.0.0.1", port=7860, debug=False)
