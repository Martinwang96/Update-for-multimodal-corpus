#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import concurrent.futures as cf
import csv
import importlib.util
import json
import math
import os
import subprocess
import sys
import traceback
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

os.environ.setdefault("MPLBACKEND", "Agg")
BASE = Path(__file__).resolve().parent

SPECS = {
    "tilt": {"file": "\u503e\u659c-2d.py", "alias": "m_tilt", "label": "Tilt(2D)"},
    "shrug": {"file": "\u8038\u80a9-2d.py", "alias": "m_shrug", "label": "Shrug(2D rel-lift)"},
    "displacement": {"file": "\u4f4d\u79fb-2d.py", "alias": "m_disp", "label": "Displacement(2D X deviation)"},
    "rotation": {"file": "\u8f6c\u52a8-2d.py", "alias": "m_rot", "label": "Rotation(2D orientation)"},
}
ORDER = ["tilt", "shrug", "displacement", "rotation"]
CACHE: Dict[str, Any] = {}
LAYER_CSV_FILENAMES = {
    "tilt": "tilt_layer.csv",
    "shrug": "shrug_layer.csv",
    "displacement": "displacement_layer.csv",
    "rotation": "rotation_layer.csv",
}
LAYER_CSV_COLUMNS = [
    "module",
    "event_name",
    "event_name_field",
    "start_frame",
    "end_frame",
    "start_time_sec",
    "end_time_sec",
    "duration_sec",
    "start_time_str",
    "end_time_str",
    "peak_value",
    "peak_metric",
    "threshold",
    "units",
    "status",
    "error",
    "source_csv",
]
LAYER_PEAK_FIELD = {
    "tilt": "max_angle_0_90",
    "shrug": "peak_shrug_distance",
    "displacement": "peak_deviation_distance_x",
    "rotation": "angle_sum_abs",
}

# ---------- module load ----------
def load_mod(key: str):
    if key in CACHE:
        return CACHE[key]
    p = BASE / SPECS[key]["file"]
    if not p.is_file():
        raise FileNotFoundError(f"module file not found: {p}")
    spec = importlib.util.spec_from_file_location(SPECS[key]["alias"], str(p))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module: {p}")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    CACHE[key] = m
    return m


# ---------- helpers ----------
def save_fig(fig, path: Path, dpi: int = 140) -> Optional[str]:
    if fig is None:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    try:
        import matplotlib.pyplot as plt

        plt.close(fig)
    except Exception:
        pass
    return str(path)


def nanmax0(arr: Any) -> float:
    a = np.asarray(arr)
    if a.size == 0 or np.all(np.isnan(a)):
        return 0.0
    return float(np.nanmax(a))


def _safe_progress_callback(progress_callback: Optional[Callable[[Dict[str, Any]], None]], event: Dict[str, Any]):
    if progress_callback is None:
        return
    try:
        progress_callback(event)
    except Exception:
        pass


def _emit_progress(
    progress_callback: Optional[Callable[[Dict[str, Any]], None]],
    stage: str,
    progress: float,
    message: str,
    eta_sec: Optional[float] = None,
    extra: Optional[Dict[str, Any]] = None,
):
    p = max(0.0, min(100.0, float(progress)))
    payload: Dict[str, Any] = {
        "stage": stage,
        "progress": p,
        "message": message,
    }
    if eta_sec is not None and math.isfinite(float(eta_sec)):
        payload["eta_sec"] = max(0.0, float(eta_sec))
    if extra:
        payload.update(extra)
    _safe_progress_callback(progress_callback, payload)


def ask(prompt: str, default: Optional[str] = None) -> str:
    suffix = f" [{default}]" if default is not None else ""
    v = input(f"{prompt}{suffix}: ").strip()
    if v:
        return v
    return default if default is not None else ""


def ask_bool(prompt: str, default: bool = True) -> bool:
    hint = "Y/n" if default else "y/N"
    v = input(f"{prompt} ({hint}): ").strip().lower()
    if not v:
        return default
    return v in {"y", "yes", "1", "true"}


def ask_int(prompt: str, default: int, validator=None) -> int:
    while True:
        raw = ask(prompt, str(default))
        try:
            val = int(raw)
            if validator and not validator(val):
                raise ValueError("invalid")
            return val
        except ValueError:
            print("Invalid integer.")


def ask_float(prompt: str, default: float, validator=None) -> float:
    while True:
        raw = ask(prompt, str(default))
        try:
            val = float(raw)
            if validator and not validator(val):
                raise ValueError("invalid")
            return val
        except ValueError:
            print("Invalid number.")


def ask_opt_float(prompt: str) -> Optional[float]:
    raw = input(f"{prompt} [blank=auto/recommended]: ").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        print("Invalid number, fallback to auto/recommended.")
        return None


def ask_choice(prompt: str, options: Dict[str, str], default_key: str) -> str:
    print(prompt)
    for k, v in options.items():
        print(f"  {k}. {v}")
    while True:
        raw = ask("Choose", default_key).strip()
        if raw in options:
            return raw
        print("Invalid choice.")


def is_odd_ge_3(x: int) -> bool:
    return x >= 3 and (x % 2 == 1)


# ---------- parameter collection ----------
def choose_modules() -> List[str]:
    print("\nModules:")
    for i, key in enumerate(ORDER, start=1):
        print(f"  {i}. {SPECS[key]['label']}")
    raw = ask("Select module indexes (comma-separated, blank=all)", "")
    if not raw:
        return ORDER.copy()

    chosen: List[str] = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if tok.isdigit():
            idx = int(tok) - 1
            if 0 <= idx < len(ORDER):
                chosen.append(ORDER[idx])
        elif tok in ORDER:
            chosen.append(tok)

    chosen = [k for k in ORDER if k in chosen]
    return chosen or ORDER.copy()


def collect_tilt_params() -> Dict[str, Any]:
    m = load_mod("tilt")
    print("\n[Tilt] parameters")
    return {
        "smooth_window": ask_int("Smooth window (odd >=3)", int(m.DEFAULT_SMOOTH_WINDOW), is_odd_ge_3),
        "smooth_poly": ask_int("Smooth poly (1 <= poly < window)", int(m.DEFAULT_SMOOTH_POLY), lambda x: x >= 1),
        "outlier_diff_threshold": ask_float("Outlier angle jump threshold (deg/frame, 0=off)", float(m.DEFAULT_OUTLIER_DIFF_THRESHOLD), lambda x: x >= 0),
        "threshold_override": ask_opt_float("Final tilt threshold (deg)"),
        "min_duration_sec": ask_float("Min duration (sec)", float(m.DEFAULT_MIN_DURATION_SEC), lambda x: x >= 0),
        "max_duration_sec": ask_float("Max duration (sec, 0=unlimited)", float(m.DEFAULT_MAX_DURATION_SEC), lambda x: x >= 0),
        "merge_gap_sec": ask_float("Merge gap (sec)", float(m.DEFAULT_MERGE_GAP_SEC), lambda x: x >= 0),
    }


def collect_shrug_params() -> Dict[str, Any]:
    m = load_mod("shrug")
    print("\n[Shrug] parameters")
    return {
        "smooth_window": ask_int("Y smooth window (odd >=3)", int(m.DEFAULT_SMOOTH_Y_WINDOW), is_odd_ge_3),
        "smooth_poly": ask_int("Y smooth poly (1 <= poly < window)", int(m.DEFAULT_SMOOTH_Y_POLY), lambda x: x >= 1),
        "baseline_window_sec": ask_float("Baseline window (sec)", float(m.DEFAULT_BASELINE_WINDOW_SEC), lambda x: x > 0),
        "threshold_override": ask_opt_float("Final shrug threshold (px)"),
        "min_duration_sec": ask_float("Min duration (sec)", float(m.DEFAULT_MIN_DURATION_SEC), lambda x: x >= 0),
        "max_duration_sec": ask_float("Max duration (sec, 0=unlimited)", float(m.DEFAULT_MAX_DURATION_SEC), lambda x: x >= 0),
        "merge_gap_sec": ask_float("Merge gap (sec)", float(m.DEFAULT_MERGE_GAP_SEC), lambda x: x >= 0),
    }


def collect_displacement_params() -> Dict[str, Any]:
    m = load_mod("displacement")
    print("\n[Displacement] parameters")
    return {
        "smooth_window": ask_int("Position smooth window (odd >=3)", int(m.DEFAULT_SMOOTH_POS_WINDOW), is_odd_ge_3),
        "smooth_poly": ask_int("Position smooth poly (1 <= poly < window)", int(m.DEFAULT_SMOOTH_POS_POLY), lambda x: x >= 1),
        "baseline_window_sec": ask_float("Baseline window (sec)", float(m.DEFAULT_BASELINE_WINDOW_SEC), lambda x: x > 0),
        "kinematics_window": ask_int("Kinematics window (odd >=3)", int(m.DEFAULT_KINEMATICS_WINDOW), is_odd_ge_3),
        "kinematics_poly": ask_int("Kinematics poly (1 <= poly < window)", int(m.DEFAULT_KINEMATICS_POLY), lambda x: x >= 1),
        "threshold_pa_override": ask_opt_float("Final PA threshold (px)"),
        "threshold_move_override": ask_opt_float("Final Movement threshold (px)"),
        "min_duration_sec": ask_float("Min duration (sec)", float(m.DEFAULT_MIN_DURATION_SEC), lambda x: x >= 0),
        "max_duration_sec": ask_float("Max duration (sec, 0=unlimited)", float(m.DEFAULT_MAX_DURATION_SEC), lambda x: x >= 0),
        "merge_gap_sec": ask_float("Merge gap (sec)", float(m.DEFAULT_MERGE_GAP_SEC), lambda x: x >= 0),
    }


def collect_rotation_params() -> Dict[str, Any]:
    m = load_mod("rotation")
    print("\n[Rotation] parameters")
    return {
        "left_shoulder_idx": ask_int("Left shoulder keypoint index", 5, lambda x: x >= 0),
        "right_shoulder_idx": ask_int("Right shoulder keypoint index", 6, lambda x: x >= 0),
        "smooth_window": ask_int("Smooth window (odd >=3)", int(m.DEFAULT_SMOOTH_WINDOW), is_odd_ge_3),
        "smooth_poly": ask_int("Smooth poly (1 <= poly < window)", int(m.DEFAULT_SMOOTH_POLY), lambda x: x >= 1),
        "threshold_override": ask_opt_float("Final rotation threshold (deg/frame)"),
        "min_duration_sec": ask_float("Min duration (sec)", float(m.DEFAULT_MIN_DURATION_SEC_EVENT), lambda x: x >= 0),
        "max_duration_sec": ask_float("Max duration (sec, 0=unlimited)", float(m.DEFAULT_MAX_DURATION_SEC_EVENT), lambda x: x >= 0),
        "merge_gap_sec": ask_float("Merge gap (sec)", float(m.DEFAULT_MERGE_GAP_SEC_EVENT), lambda x: x >= 0),
    }



# ---------- body-recognition stage ----------
def run_body_recognition_stage(
    recognition_python: str,
    video_path: Path,
    recognition_root: Path,
    pose2d: str,
    batch_size: int,
    device: str,
    save_vis: bool,
    score_thresh: float,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    recognition_root.mkdir(parents=True, exist_ok=True)
    summary_path = recognition_root / "recognition_summary.json"

    runner_path = BASE / "body_recognition_runner.py"
    if not runner_path.is_file():
        raise FileNotFoundError(f"body_recognition_runner.py not found: {runner_path}")

    cmd = [
        recognition_python,
        str(runner_path),
        "--video-path",
        str(video_path),
        "--output-dir",
        str(recognition_root),
        "--pose2d",
        pose2d,
        "--batch-size",
        str(batch_size),
        "--device",
        device,
        "--score-thresh",
        str(score_thresh),
        "--summary-json",
        str(summary_path),
    ]
    if save_vis:
        cmd.append("--save-vis")

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
    except FileNotFoundError as e:
        raise RuntimeError(f"recognition python not found: {recognition_python}") from e

    stdout_lines: List[str] = []
    progress_prefix = "PROGRESS_JSON:"

    assert proc.stdout is not None
    while True:
        line = proc.stdout.readline()
        if line == "" and proc.poll() is not None:
            break
        if not line:
            continue

        line = line.rstrip("\r\n")
        stdout_lines.append(line)

        if line.startswith(progress_prefix):
            raw = line[len(progress_prefix) :].strip()
            try:
                payload = json.loads(raw)
                if isinstance(payload, dict):
                    _safe_progress_callback(progress_callback, payload)
            except Exception:
                pass

    return_code = proc.wait()
    stdout_text = "\n".join(stdout_lines)

    if return_code != 0:
        raise RuntimeError(
            "body-recognition stage failed.\n"
            f"cmd: {' '.join(cmd)}\n"
            f"stdout:\n{stdout_text}\n"
            "stderr:\n"
        )

    if not summary_path.is_file():
        raise RuntimeError(
            "body-recognition stage finished with zero code but summary json is missing.\n"
            f"stdout:\n{stdout_text}\n"
            "stderr:\n"
        )

    result = json.loads(summary_path.read_text(encoding="utf-8"))
    result["runner_stdout"] = stdout_text
    result["runner_stderr"] = ""
    return result


# ---------- output folder ----------
def module_output_dir(key: str, root: Path, stem: str) -> Path:
    m = load_mod(key)
    if key == "tilt":
        name = m.ANALYSIS_FOLDER_TEMPLATE.format(stem)
    elif key == "shrug":
        name = m.ANALYSIS_FOLDER_TEMPLATE.format(stem)
    elif key == "displacement":
        name = f"{stem}_deviationX_analysis_{m.TRACKING_POINT_MODE}"
    else:
        name = m.ANALYSIS_FOLDER_TEMPLATE_ORIENTATION.format(stem)
    return root / name



def _stringify_threshold(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return str(value)


def _extract_event_name(row: Dict[str, Any]) -> tuple[str, str]:
    for field in ("event_type", "event_label"):
        val = str(row.get(field, "")).strip()
        if val:
            return val, field
    return "", ""


def _extract_units(row: Dict[str, Any]) -> str:
    keys = [k for k in row.keys() if k.startswith("units") and str(row.get(k, "")).strip()]
    if not keys:
        return ""
    if len(keys) == 1:
        return str(row.get(keys[0], "")).strip()
    parts = [f"{k}={str(row.get(k, '')).strip()}" for k in keys]
    return "; ".join([p for p in parts if p and not p.endswith("=")])


def _read_csv_rows(csv_path: Path) -> List[Dict[str, Any]]:
    errors: List[str] = []
    for enc in ("utf-8-sig", "utf-8"):
        try:
            with csv_path.open("r", encoding=enc, newline="") as f:
                reader = csv.DictReader(f)
                return [dict(r) for r in reader]
        except Exception as e:
            errors.append(f"{enc}: {type(e).__name__}: {e}")
    raise RuntimeError("failed to read source csv; " + " | ".join(errors))


def _status_row(module_key: str, status: str, result: Optional[Dict[str, Any]] = None, error: str = "") -> Dict[str, str]:
    row = {k: "" for k in LAYER_CSV_COLUMNS}
    row["module"] = module_key
    row["status"] = status
    row["error"] = error
    if result:
        row["threshold"] = _stringify_threshold(result.get("threshold"))
        row["source_csv"] = str(result.get("csv") or "")
    return row


def _build_layer_event_rows(module_key: str, result: Dict[str, Any]) -> List[Dict[str, str]]:
    csv_path_raw = result.get("csv")
    if not csv_path_raw:
        return []

    csv_path = Path(str(csv_path_raw)).expanduser().resolve()
    if not csv_path.is_file():
        return []

    raw_rows = _read_csv_rows(csv_path)
    if not raw_rows:
        return []

    peak_field = LAYER_PEAK_FIELD[module_key]
    threshold_str = _stringify_threshold(result.get("threshold"))

    rows: List[Dict[str, str]] = []
    for src in raw_rows:
        event_name, event_name_field = _extract_event_name(src)
        row = {k: "" for k in LAYER_CSV_COLUMNS}
        row["module"] = module_key
        row["event_name"] = event_name
        row["event_name_field"] = event_name_field
        row["start_frame"] = str(src.get("start_frame", ""))
        row["end_frame"] = str(src.get("end_frame", ""))
        row["start_time_sec"] = str(src.get("start_time_sec", ""))
        row["end_time_sec"] = str(src.get("end_time_sec", ""))
        row["duration_sec"] = str(src.get("duration_sec", ""))
        row["start_time_str"] = str(src.get("start_time_str", ""))
        row["end_time_str"] = str(src.get("end_time_str", ""))
        row["peak_metric"] = peak_field
        row["peak_value"] = str(src.get(peak_field, ""))
        row["threshold"] = threshold_str
        row["units"] = _extract_units(src)
        row["status"] = "ok"
        row["error"] = ""
        row["source_csv"] = str(csv_path)
        rows.append(row)
    return rows


def export_layer_csvs(results: List[Dict[str, Any]], selected_modules: List[str], layer_csv_dir: Path) -> Dict[str, Any]:
    layer_csv_dir.mkdir(parents=True, exist_ok=True)
    selected_set = {m for m in selected_modules if m in ORDER}
    result_map = {r.get("module"): r for r in results if isinstance(r, dict) and r.get("module") in ORDER}

    files: Dict[str, str] = {}
    status_map: Dict[str, str] = {}

    for key in ORDER:
        out_path = layer_csv_dir / LAYER_CSV_FILENAMES[key]
        result = result_map.get(key)

        if key not in selected_set:
            rows = [_status_row(key, "skipped", result)]
            status = "skipped"
        elif result is None:
            rows = [_status_row(key, "error", None, "missing module result")]
            status = "error"
        elif result.get("status") != "ok":
            rows = [_status_row(key, "error", result, str(result.get("error", "module failed")))]
            status = "error"
        else:
            try:
                event_rows = _build_layer_event_rows(key, result)
            except Exception as e:
                event_rows = []
                rows = [_status_row(key, "error", result, f"layer csv parse failed: {type(e).__name__}: {e}")]
                status = "error"
            else:
                if event_rows:
                    rows = event_rows
                    status = "ok"
                else:
                    rows = [_status_row(key, "no_event", result)]
                    status = "no_event"

        with out_path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=LAYER_CSV_COLUMNS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

        files[key] = str(out_path)
        status_map[key] = status

    return {
        "layer_csv_dir": str(layer_csv_dir),
        "layer_csv_files": files,
        "layer_csv_status": status_map,
    }

# ---------- module runners (logic preserved by calling original functions) ----------
def run_tilt(task: Dict[str, Any]) -> Dict[str, Any]:
    m = load_mod("tilt")
    jp = Path(task["json"])
    od = Path(task["out"])
    fps = float(task["fps"])
    conf = float(task["conf"])
    plots = bool(task["plots"])
    p = task["params"]

    sw = int(p["smooth_window"])
    sp = int(p["smooth_poly"])
    if not (1 <= sp < sw):
        sp = max(1, min(sp, sw - 1))

    coords, _, total_frames, valid_mask = m.load_required_keypoints_2d(jp, m.DEFAULT_REQUIRED_KP_INDICES, conf)
    min_req = max(10, sw)
    if total_frames == 0 or int(np.sum(valid_mask)) < min_req:
        raise ValueError(f"tilt valid frames insufficient: {int(np.sum(valid_mask))} < {min_req}")

    ls, rs, lh, rh = m.interpolate_keypoint_positions(coords)
    ms = (ls + rs) / 2.0
    mh = (lh + rh) / 2.0
    ms[np.any(np.isnan(ls), axis=1) | np.any(np.isnan(rs), axis=1)] = np.nan
    mh[np.any(np.isnan(lh), axis=1) | np.any(np.isnan(rh), axis=1)] = np.nan

    raw = m.calculate_torso_tilt_angle_2d(mh, ms)
    if np.all(np.isnan(raw)):
        raise ValueError("tilt raw angle is all NaN")

    corr = raw.copy()
    outlier_thr = float(p["outlier_diff_threshold"])
    if outlier_thr > 0 and len(corr) > 1:
        s = pd.Series(corr)
        fill_lim = max(5, int(fps * 0.2)) if fps > 0 else 5
        filled = s.interpolate(method="linear", limit_direction="both", limit=fill_lim).ffill().bfill().to_numpy()
        if not np.all(np.isnan(filled)):
            d = np.diff(filled, prepend=np.nan)
            idx = np.where(np.abs(d) > outlier_thr)[0]
            if len(idx) > 0:
                s2 = pd.Series(corr)
                s2.iloc[np.unique(idx)] = np.nan
                corr = s2.interpolate(method="linear", limit_direction="both", limit=10).to_numpy()

    sm = m.smooth_angles(corr, sw, sp)
    if np.all(np.isnan(sm)):
        sm = corr
    vel = m.compute_angular_velocity(sm, fps)

    rec, ana = m.recommend_tilt_threshold_2d(corr, default_thresh=float(m.DEFAULT_TILT_THRESHOLD))
    final_thr = float(p["threshold_override"]) if p["threshold_override"] is not None else float(rec)
    if not math.isfinite(final_thr) or final_thr <= 0:
        final_thr = float(rec)

    threshold_plot = None
    if plots and ana:
        fig = m.plot_tilt_threshold_analysis_2d(corr, ana, target_name=f"{m.TARGET_NAME} - {jp.stem}")
        threshold_plot = save_fig(fig, od / f"{jp.stem}_tilt_threshold_analysis.png", 110)

    min_df = max(1, int(float(p["min_duration_sec"]) * fps)) if fps > 0 else 1
    max_sec = float(p["max_duration_sec"])
    max_df = int(max_sec * fps) if (max_sec > 0 and fps > 0) else None
    gap_df = int(float(p["merge_gap_sec"]) * fps) if fps > 0 else 0

    events = m.detect_tilt_events_2d(sm, final_thr, min_df, max_df, gap_df, fps)
    csv = m.export_tilt_events_to_csv(events, od / m.OUTPUT_CSV_TEMPLATE.format(jp.stem, final_thr))

    analysis_plot = None
    if plots:
        fig = m.plot_tilt_analysis_2d(
            angles_0_180_raw=raw,
            smoothed_angles_0_180=sm,
            angular_velocities=vel,
            events=events,
            threshold_0_90=final_thr,
            fps=fps,
            title_prefix=f"{jp.stem}\n",
            target_name=m.TARGET_NAME,
        )
        analysis_plot = save_fig(fig, od / f"{jp.stem}_tilt_analysis_final.png", 150)

    rel = np.minimum(sm, 180.0 - sm)
    return {
        "module": "tilt",
        "label": SPECS["tilt"]["label"],
        "status": "ok",
        "event_count": len(events),
        "threshold": final_thr,
        "max_value": nanmax0(rel),
        "csv": str(csv) if csv else None,
        "threshold_plot": threshold_plot,
        "analysis_plot": analysis_plot,
        "output_dir": str(od),
    }


def run_shrug(task: Dict[str, Any]) -> Dict[str, Any]:
    m = load_mod("shrug")
    jp = Path(task["json"])
    od = Path(task["out"])
    fps = float(task["fps"])
    conf = float(task["conf"])
    plots = bool(task["plots"])
    p = task["params"]

    sw = int(p["smooth_window"])
    sp = int(p["smooth_poly"])
    if not (1 <= sp < sw):
        sp = max(1, min(sp, sw - 1))

    ys, _, flags, total, unit = m.load_shoulder_hip_y_coordinates_2d(jp, m.DEFAULT_REQUIRED_KP_INDICES, conf)
    lsr, rsr, lhr, rhr = ys
    vls, vrs, vlh, vrh = flags

    min_req = max(10, sw, int(float(p["baseline_window_sec"]) * fps * 1.1) if fps > 0 else 10)
    n_sh = int(np.sum(vls & vrs))
    n_hip = int(np.sum(vlh & vrh))
    if total == 0 or n_sh < min_req or n_hip < min_req:
        raise ValueError(f"shrug valid frames insufficient: shoulder={n_sh}, hip={n_hip}, need={min_req}")

    lsi = m.interpolate_single_coordinate(lsr, vls)
    rsi = m.interpolate_single_coordinate(rsr, vrs)
    lhi = m.interpolate_single_coordinate(lhr, vlh)
    rhi = m.interpolate_single_coordinate(rhr, vrh)

    lss = m.smooth_single_coordinate(lsi, sw, sp)
    rss = m.smooth_single_coordinate(rsi, sw, sp)
    lhs = m.smooth_single_coordinate(lhi, sw, sp)
    rhs = m.smooth_single_coordinate(rhi, sw, sp)

    rel_y, base_y, dist = m.calculate_relative_y_and_shrug_distance(
        lss, rss, lhs, rhs, fps, float(p["baseline_window_sec"])
    )
    if np.all(np.isnan(dist)):
        raise ValueError("shrug distance signal all NaN")

    rec, ana = m.recommend_shrug_distance_threshold(dist, coord_units=unit, method="gmm")
    final_thr = float(p["threshold_override"]) if p["threshold_override"] is not None else float(rec)
    if not math.isfinite(final_thr) or final_thr <= 0:
        final_thr = float(rec)

    threshold_plot = None
    if plots and ana:
        fig = m.plot_shrug_distance_threshold_analysis(dist, ana, "shoulder relative lift", unit)
        threshold_plot = save_fig(fig, od / f"{jp.stem}_shrug_dist_threshold_analysis.png", 110)

    min_df = max(1, int(float(p["min_duration_sec"]) * fps)) if fps > 0 else 1
    max_sec = float(p["max_duration_sec"])
    max_df = int(max_sec * fps) if (max_sec > 0 and fps > 0) else None
    gap_df = int(float(p["merge_gap_sec"]) * fps) if fps > 0 else 0

    det = m.detect_events_from_signal_single_threshold(dist, final_thr, min_df, max_df, gap_df, fps)
    events = m.analyze_shrug_events_distance(det, dist, rel_y, fps=fps, coord_units=unit)
    csv = m.export_shrug_events_to_csv(events, od / m.OUTPUT_CSV_TEMPLATE.format(jp.stem, final_thr))

    analysis_plot = None
    if plots:
        fig = m.plot_shrug_analysis_distance(
            shrug_distance_signal=dist,
            relative_y_pos=rel_y,
            baseline_relative_y=base_y,
            events=events,
            threshold_shrug=final_thr,
            fps=fps,
            coord_units=unit,
            title_prefix=f"{jp.stem}\n",
        )
        analysis_plot = save_fig(fig, od / f"{jp.stem}_shrug_analysis_final.png", 150)

    return {
        "module": "shrug",
        "label": SPECS["shrug"]["label"],
        "status": "ok",
        "event_count": len(events),
        "threshold": final_thr,
        "max_value": nanmax0(dist),
        "csv": str(csv) if csv else None,
        "threshold_plot": threshold_plot,
        "analysis_plot": analysis_plot,
        "output_dir": str(od),
    }


def run_displacement(task: Dict[str, Any]) -> Dict[str, Any]:
    m = load_mod("displacement")
    jp = Path(task["json"])
    od = Path(task["out"])
    fps = float(task["fps"])
    conf = float(task["conf"])
    plots = bool(task["plots"])
    p = task["params"]

    sw = int(p["smooth_window"])
    sp = int(p["smooth_poly"])
    if not (1 <= sp < sw):
        sp = max(1, min(sp, sw - 1))

    kw = int(p["kinematics_window"])
    kp = int(p["kinematics_poly"])
    if not (1 <= kp < kw):
        kp = max(1, min(kp, kw - 1))

    pos, valid, _, unit = m.load_and_calculate_tracked_point_2d(jp, m.CENTROID_KP_INDICES, m.LOAD_MODE, conf)
    min_req = max(10, sw, int(float(p["baseline_window_sec"]) * fps * 1.1) if fps > 0 else 10, kw)
    if int(np.sum(valid)) < (min_req + 1):
        raise ValueError(f"displacement valid frames insufficient: {int(np.sum(valid))} < {min_req + 1}")

    inter = m.interpolate_invalid_frames_2d(pos, valid)
    sm2d = m.smooth_positions_2d(inter, sw, sp)
    x = sm2d[:, 0]

    base_x, dev_x = m.calculate_baseline_and_deviation_x(x, fps, float(p["baseline_window_sec"]))
    if np.all(np.isnan(dev_x)):
        raise ValueError("displacement deviation signal all NaN")

    vx, ax = m.compute_kinematics_x(x, fps, kw, kp)
    frame_dx = m.compute_frame_displacements_x(x)

    rec_pa, rec_mv, ana = m.recommend_deviation_thresholds_x(dev_x, coord_units=unit, method="gmm")

    pa = float(p["threshold_pa_override"]) if p["threshold_pa_override"] is not None else float(rec_pa)
    mv = float(p["threshold_move_override"]) if p["threshold_move_override"] is not None else float(rec_mv)
    if pa < 0:
        pa = max(0.0, float(rec_pa))
    if mv <= pa:
        mv = max(pa * 1.1, float(rec_mv), pa + 1e-6)

    threshold_plot = None
    if plots and ana:
        fig = m.plot_deviation_threshold_analysis_x(dev_x, ana, coord_units=unit)
        threshold_plot = save_fig(fig, od / f"{jp.stem}_deviationX_threshold_analysis.png", 110)

    min_df = max(1, int(float(p["min_duration_sec"]) * fps)) if fps > 0 else 1
    max_sec = float(p["max_duration_sec"])
    max_df = int(max_sec * fps) if (max_sec > 0 and fps > 0) else None
    gap_df = int(float(p["merge_gap_sec"]) * fps) if fps > 0 else 0

    cand = m.detect_events_deviation_multi_threshold_x(dev_x, pa, mv, min_df, max_df, gap_df)
    events = m.analyze_and_classify_events_deviation_x(cand, dev_x, mv, frame_dx, x, vx, ax, fps, unit)

    csv_name = f"{jp.stem}_events_{m.TRACKING_POINT_MODE}_devX_pa{pa:.1f}_mv{mv:.1f}.csv"
    csv = m.export_events_to_csv_x(events, od / csv_name)

    analysis_plot = None
    if plots:
        fig = m.plot_deviation_analysis_x(
            deviation_distance_x=dev_x,
            positions_x=x,
            baseline_x=base_x,
            events=events,
            threshold_pose_adj_dev_x=pa,
            threshold_movement_dev_x=mv,
            fps=fps,
            coord_units=unit,
            title_prefix=f"{jp.stem}\n",
        )
        analysis_plot = save_fig(fig, od / f"{jp.stem}_deviationX_analysis_final.png", 150)

    return {
        "module": "displacement",
        "label": SPECS["displacement"]["label"],
        "status": "ok",
        "event_count": len(events),
        "event_detail": {
            "movement": sum(1 for e in events if e.get("event_type") == "Movement"),
            "pose_adjustment": sum(1 for e in events if e.get("event_type") == "Pose Adjustment"),
        },
        "threshold": {"pose_adjustment": pa, "movement": mv},
        "max_value": nanmax0(dev_x),
        "csv": str(csv) if csv else None,
        "threshold_plot": threshold_plot,
        "analysis_plot": analysis_plot,
        "output_dir": str(od),
    }


def run_rotation(task: Dict[str, Any]) -> Dict[str, Any]:
    m = load_mod("rotation")
    jp = Path(task["json"])
    od = Path(task["out"])
    fps = float(task["fps"])
    conf = float(task["conf"])
    plots = bool(task["plots"])
    p = task["params"]

    sw = int(p["smooth_window"])
    sp = int(p["smooth_poly"])
    if not (1 <= sp < sw):
        sp = max(1, min(sp, sw - 1))

    ang, len_chg, total = m.compute_body_orientation_2d(
        jp,
        int(p["left_shoulder_idx"]),
        int(p["right_shoulder_idx"]),
        min_confidence=conf,
        smooth=True,
        smooth_window=sw,
        smooth_poly=sp,
    )
    if total == 0:
        raise ValueError("rotation total frames is zero")
    if len(ang) == 0:
        raise ValueError("rotation orientation change is empty")

    rec, ana = m.recommend_rotation_threshold(ang, default_thresh=float(m.DEFAULT_ROTATION_THRESHOLD))
    final_thr = float(p["threshold_override"]) if p["threshold_override"] is not None else float(rec)
    if not math.isfinite(final_thr) or final_thr <= 0:
        final_thr = float(rec)

    threshold_plot = None
    if plots and ana:
        fig = m.plot_threshold_analysis(ang, ana, target_name_plot=m.TARGET_NAME_ORIENTATION)
        threshold_plot = save_fig(fig, od / f"{jp.stem}_orientation_threshold_analysis.png", 110)

    min_df = max(1, int(float(p["min_duration_sec"]) * fps)) if fps > 0 else 1
    max_sec = float(p["max_duration_sec"])
    max_df = int(max_sec * fps) if (max_sec > 0 and fps > 0) else (total or 99999)
    gap_df = int(float(p["merge_gap_sec"]) * fps) if fps > 0 else 0

    events = m.detect_orientation_change_events(ang, final_thr, min_df, max_df, gap_df, fps)
    csv = m.export_events_to_csv(events, od / m.OUTPUT_CSV_TEMPLATE_ORIENTATION.format(jp.stem, final_thr))

    analysis_plot = None
    if plots:
        fig = m.plot_orientation_analysis_2d(
            orientation_changes=ang,
            length_changes=len_chg,
            fps_param=fps,
            events_list=events,
            threshold_val=final_thr,
            title_prefix=f"{jp.stem}\n",
        )
        analysis_plot = save_fig(fig, od / f"{jp.stem}_orientation_analysis_final.png", 130)

    return {
        "module": "rotation",
        "label": SPECS["rotation"]["label"],
        "status": "ok",
        "event_count": len(events),
        "threshold": final_thr,
        "max_value": nanmax0(np.abs(np.asarray(ang))),
        "csv": str(csv) if csv else None,
        "threshold_plot": threshold_plot,
        "analysis_plot": analysis_plot,
        "output_dir": str(od),
    }


RUNNERS = {
    "tilt": run_tilt,
    "shrug": run_shrug,
    "displacement": run_displacement,
    "rotation": run_rotation,
}


def exec_task(task: Dict[str, Any]) -> Dict[str, Any]:
    os.environ.setdefault("MPLBACKEND", "Agg")
    k = task["module"]
    t0 = time.perf_counter()
    try:
        result = RUNNERS[k](task)
        result["elapsed_sec"] = float(time.perf_counter() - t0)
        return result
    except Exception as e:
        return {
            "module": k,
            "label": SPECS[k]["label"],
            "status": "error",
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(),
            "elapsed_sec": float(time.perf_counter() - t0),
        }
    finally:
        try:
            import matplotlib.pyplot as plt

            plt.close("all")
        except Exception:
            pass


def build_default_params() -> Dict[str, Dict[str, Any]]:
    tilt = load_mod("tilt")
    shrug = load_mod("shrug")
    disp = load_mod("displacement")
    rot = load_mod("rotation")
    return {
        "tilt": {
            "smooth_window": int(tilt.DEFAULT_SMOOTH_WINDOW),
            "smooth_poly": int(tilt.DEFAULT_SMOOTH_POLY),
            "outlier_diff_threshold": float(tilt.DEFAULT_OUTLIER_DIFF_THRESHOLD),
            "threshold_override": None,
            "min_duration_sec": float(tilt.DEFAULT_MIN_DURATION_SEC),
            "max_duration_sec": float(tilt.DEFAULT_MAX_DURATION_SEC),
            "merge_gap_sec": float(tilt.DEFAULT_MERGE_GAP_SEC),
        },
        "shrug": {
            "smooth_window": int(shrug.DEFAULT_SMOOTH_Y_WINDOW),
            "smooth_poly": int(shrug.DEFAULT_SMOOTH_Y_POLY),
            "baseline_window_sec": float(shrug.DEFAULT_BASELINE_WINDOW_SEC),
            "threshold_override": None,
            "min_duration_sec": float(shrug.DEFAULT_MIN_DURATION_SEC),
            "max_duration_sec": float(shrug.DEFAULT_MAX_DURATION_SEC),
            "merge_gap_sec": float(shrug.DEFAULT_MERGE_GAP_SEC),
        },
        "displacement": {
            "smooth_window": int(disp.DEFAULT_SMOOTH_POS_WINDOW),
            "smooth_poly": int(disp.DEFAULT_SMOOTH_POS_POLY),
            "baseline_window_sec": float(disp.DEFAULT_BASELINE_WINDOW_SEC),
            "kinematics_window": int(disp.DEFAULT_KINEMATICS_WINDOW),
            "kinematics_poly": int(disp.DEFAULT_KINEMATICS_POLY),
            "threshold_pa_override": None,
            "threshold_move_override": None,
            "min_duration_sec": float(disp.DEFAULT_MIN_DURATION_SEC),
            "max_duration_sec": float(disp.DEFAULT_MAX_DURATION_SEC),
            "merge_gap_sec": float(disp.DEFAULT_MERGE_GAP_SEC),
        },
        "rotation": {
            "left_shoulder_idx": 5,
            "right_shoulder_idx": 6,
            "smooth_window": int(rot.DEFAULT_SMOOTH_WINDOW),
            "smooth_poly": int(rot.DEFAULT_SMOOTH_POLY),
            "threshold_override": None,
            "min_duration_sec": float(rot.DEFAULT_MIN_DURATION_SEC_EVENT),
            "max_duration_sec": float(rot.DEFAULT_MAX_DURATION_SEC_EVENT),
            "merge_gap_sec": float(rot.DEFAULT_MERGE_GAP_SEC_EVENT),
        },
    }


def run_pipeline(
    config: Dict[str, Any],
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    pipeline_start = time.perf_counter()
    _emit_progress(progress_callback, "prepare", 2.0, "准备阶段：参数校验与任务初始化")

    entry_mode = config.get("entry_mode", "json")
    run_root = Path(config["run_root"]).expanduser().resolve()
    run_root.mkdir(parents=True, exist_ok=True)

    recognition_info: Optional[Dict[str, Any]] = None
    selected_json_mode = "json_direct"
    recognition_elapsed = 0.0

    if entry_mode == "video":
        video_path = Path(config["video_path"]).expanduser().resolve()
        if not video_path.is_file():
            raise FileNotFoundError(f"video not found: {video_path}")

        rec_cfg = config.get("recognition", {})
        rec_start = time.perf_counter()
        _emit_progress(progress_callback, "recognition", 8.0, "识别阶段：正在启动 body-recognition")

        def _on_rec_progress(event: Dict[str, Any]):
            rec_elapsed = max(0.0, time.perf_counter() - rec_start)
            done = int(event.get("frames_done", 0) or 0)
            total = int(event.get("frames_total", 0) or 0)

            ratio = 0.0
            eta = None
            msg = "识别阶段进行中"
            if total > 0:
                ratio = min(1.0, done / float(total))
                msg = f"识别阶段：已处理 {done}/{total} 帧"
                if done > 0 and rec_elapsed > 0:
                    remaining = max(0, total - done)
                    eta = rec_elapsed / float(done) * float(remaining)
            elif done > 0:
                msg = f"识别阶段：已处理 {done} 帧"

            overall = 8.0 + 52.0 * ratio
            _emit_progress(
                progress_callback,
                "recognition",
                overall,
                msg,
                eta_sec=eta,
                extra={"frames_done": done, "frames_total": total},
            )

        recognition_info = run_body_recognition_stage(
            recognition_python=str(rec_cfg.get("python", sys.executable)),
            video_path=video_path,
            recognition_root=run_root / "recognition",
            pose2d=str(rec_cfg.get("pose2d", "rtmo")),
            batch_size=int(rec_cfg.get("batch_size", 16)),
            device=str(rec_cfg.get("device", "auto")),
            save_vis=bool(rec_cfg.get("save_vis", False)),
            score_thresh=float(rec_cfg.get("score_thresh", 0.6)),
            progress_callback=_on_rec_progress,
        )
        recognition_elapsed = float(time.perf_counter() - rec_start)

        _emit_progress(
            progress_callback,
            "recognition",
            60.0,
            "识别阶段完成：推理已结束，JSON文件已生成。",
            eta_sec=0.0,
        )

        src = str(rec_cfg.get("json_source", "tonly")).lower()
        if src == "raw":
            json_path = Path(recognition_info["raw_json_path"]).resolve()
            selected_json_mode = "raw"
        else:
            json_path = Path(recognition_info["tonly_json_path"]).resolve()
            selected_json_mode = "tonly"

        input_stem_for_output = video_path.stem
    else:
        json_path = Path(config["json_path"]).expanduser().resolve()
        if not json_path.is_file():
            raise FileNotFoundError(f"json not found: {json_path}")
        input_stem_for_output = json_path.stem
        _emit_progress(progress_callback, "recognition", 60.0, "识别阶段跳过：直接使用JSON输入", eta_sec=0.0)

    analysis_root = run_root / "analysis"
    analysis_root.mkdir(parents=True, exist_ok=True)

    fps = float(config.get("fps", 30.0))
    min_conf = float(config.get("min_confidence", 0.3))
    save_plots = bool(config.get("save_plots", True))
    run_parallel = bool(config.get("parallel", True))

    modules = config.get("modules", ORDER.copy())
    modules = [m for m in ORDER if m in modules]
    if not modules:
        modules = ORDER.copy()

    default_params = build_default_params()
    incoming_params = config.get("params", {})
    params: Dict[str, Dict[str, Any]] = {}
    for m in modules:
        merged = dict(default_params[m])
        merged.update(incoming_params.get(m, {}))
        params[m] = merged

    tasks = []
    for key in modules:
        out_dir = module_output_dir(key, analysis_root, input_stem_for_output)
        out_dir.mkdir(parents=True, exist_ok=True)
        tasks.append(
            {
                "module": key,
                "json": str(json_path),
                "out": str(out_dir),
                "fps": fps,
                "conf": min_conf,
                "plots": save_plots,
                "params": params[key],
            }
        )

    _emit_progress(progress_callback, "processing", 62.0, "数据处理阶段：正在执行四模块分析")
    processing_start = time.perf_counter()

    def _processing_progress(completed: int, total: int):
        elapsed = max(0.0, time.perf_counter() - processing_start)
        ratio = (completed / float(total)) if total > 0 else 1.0
        overall = 62.0 + 34.0 * min(1.0, ratio)
        eta = None
        if completed > 0 and total > 0 and completed < total:
            eta = elapsed / float(completed) * float(total - completed)
        _emit_progress(
            progress_callback,
            "processing",
            overall,
            f"数据处理阶段：已完成 {completed}/{total} 个模块",
            eta_sec=eta,
            extra={"modules_done": completed, "modules_total": total},
        )

    results: List[Dict[str, Any]] = []
    parallel_requested = bool(run_parallel and len(tasks) > 1)
    parallel_used = False
    parallel_fallback_error = ""
    completed = 0

    if parallel_requested:
        try:
            max_workers = min(len(tasks), max(1, os.cpu_count() or 1))
            with cf.ProcessPoolExecutor(max_workers=max_workers) as ex:
                future_map = {ex.submit(exec_task, t): t["module"] for t in tasks}
                for fut in cf.as_completed(future_map):
                    results.append(fut.result())
                    completed += 1
                    _processing_progress(completed, len(tasks))
            parallel_used = True
        except Exception as e:
            parallel_fallback_error = f"{type(e).__name__}: {e}"
            results = []
            completed = 0
            for t in tasks:
                results.append(exec_task(t))
                completed += 1
                _processing_progress(completed, len(tasks))
    else:
        for t in tasks:
            results.append(exec_task(t))
            completed += 1
            _processing_progress(completed, len(tasks))

    results.sort(key=lambda x: ORDER.index(x["module"]))
    processing_elapsed = float(time.perf_counter() - processing_start)

    export_layer_csv = bool(config.get("export_layer_csv", True))
    layer_csv_info = {
        "layer_csv_dir": "",
        "layer_csv_files": {},
        "layer_csv_status": {},
    }

    csv_export_elapsed = 0.0
    if export_layer_csv:
        csv_start = time.perf_counter()
        layer_csv_info = export_layer_csvs(results, modules, analysis_root / "layer_csv")
        csv_export_elapsed = float(time.perf_counter() - csv_start)

    total_elapsed = float(time.perf_counter() - pipeline_start)
    module_elapsed_sec = {
        r.get("module", f"m_{idx}"): float(r.get("elapsed_sec", 0.0))
        for idx, r in enumerate(results)
        if isinstance(r, dict)
    }

    _emit_progress(progress_callback, "processing", 98.0, "数据处理阶段完成：正在整理汇总与CSV", eta_sec=0.0)

    summary = {
        "entry_mode": entry_mode,
        "selected_json_mode": selected_json_mode,
        "json_input_path": str(json_path),
        "run_root": str(run_root),
        "analysis_root": str(analysis_root),
        "recognition": recognition_info,
        "fps": fps,
        "min_confidence": min_conf,
        "parallel": run_parallel,
        "parallel_requested": parallel_requested,
        "parallel_used": parallel_used,
        "parallel_fallback_error": parallel_fallback_error,
        "selected_modules": modules,
        "params": params,
        "results": results,
        "layer_csv_dir": layer_csv_info.get("layer_csv_dir", ""),
        "layer_csv_files": layer_csv_info.get("layer_csv_files", {}),
        "layer_csv_status": layer_csv_info.get("layer_csv_status", {}),
        "timing": {
            "total_elapsed_sec": total_elapsed,
            "recognition_elapsed_sec": recognition_elapsed,
            "processing_elapsed_sec": processing_elapsed,
            "csv_export_elapsed_sec": csv_export_elapsed,
            "module_elapsed_sec": module_elapsed_sec,
        },
    }
    summary_path = run_root / "combined_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary["summary_json"] = str(summary_path)

    _emit_progress(progress_callback, "done", 100.0, "全部流程完成", eta_sec=0.0)
    return summary


def main():
    print("Unified 2D Interface: body-recognition + tilt/shrug/displacement/rotation")
    print("Core logic of 4 modules is preserved. Orchestration only.")
    print("-" * 74)

    entry = ask_choice(
        "Input mode:",
        {
            "1": "Video input (run body-recognition first)",
            "2": "JSON input (skip body-recognition)",
        },
        "1",
    )

    config: Dict[str, Any] = {}

    if entry == "1":
        video_path = Path(ask("Input video path")).expanduser().resolve()
        if not video_path.is_file():
            print(f"Video not found: {video_path}")
            sys.exit(1)

        root_default = video_path.parent / "combined_2d_runs"
        root_dir = Path(ask("Output root", str(root_default))).expanduser().resolve()
        run_root = root_dir / video_path.stem

        recognition_cfg = {
            "python": ask("Recognition python path", sys.executable),
            "pose2d": ask("Recognition pose2d alias", "rtmo"),
            "batch_size": ask_int("Recognition batch size", 16, lambda x: x > 0),
            "device": ask("Recognition device (auto/cuda:0/cpu)", "auto"),
            "save_vis": ask_bool("Save recognition visualization video", False),
            "score_thresh": ask_float("Best-instance filter score threshold", 0.6, lambda x: x >= 0),
        }
        src = ask_choice(
            "JSON source for 4 modules:",
            {"1": "raw predictions json", "2": "filtered tonly json"},
            "2",
        )
        recognition_cfg["json_source"] = "raw" if src == "1" else "tonly"

        config.update({
            "entry_mode": "video",
            "video_path": str(video_path),
            "run_root": str(run_root),
            "recognition": recognition_cfg,
        })
    else:
        json_path = Path(ask("Input JSON path")).expanduser().resolve()
        if not json_path.is_file():
            print(f"JSON not found: {json_path}")
            sys.exit(1)

        root_default = json_path.parent / "combined_2d_runs"
        root_dir = Path(ask("Output root", str(root_default))).expanduser().resolve()
        run_root = root_dir / json_path.stem

        config.update({
            "entry_mode": "json",
            "json_path": str(json_path),
            "run_root": str(run_root),
        })

    config["fps"] = ask_float("FPS", 30.0, lambda x: x > 0)
    config["min_confidence"] = ask_float("Min keypoint confidence", 0.3, lambda x: 0.0 <= x <= 1.0)
    config["save_plots"] = ask_bool("Save plots", True)
    config["parallel"] = ask_bool("Run selected modules in parallel", True)

    modules = choose_modules()
    config["modules"] = modules

    params: Dict[str, Dict[str, Any]] = {}
    for key in modules:
        if key == "tilt":
            params[key] = collect_tilt_params()
        elif key == "shrug":
            params[key] = collect_shrug_params()
        elif key == "displacement":
            params[key] = collect_displacement_params()
        else:
            params[key] = collect_rotation_params()
    config["params"] = params

    print("\nRunning pipeline...")
    try:
        summary = run_pipeline(config)
    except Exception as e:
        print(f"Pipeline failed: {type(e).__name__}: {e}")
        print(traceback.format_exc())
        sys.exit(2)

    print("\n=== Summary ===")
    for r in summary.get("results", []):
        if r.get("status") == "ok":
            print(f"- {r['label']}: OK, events={r.get('event_count', 0)}, output={r.get('output_dir')}")
        else:
            print(f"- {r['label']}: ERROR, {r.get('error')}")
    print(f"Summary JSON: {summary.get('summary_json')}")

    if any(r.get("status") != "ok" for r in summary.get("results", [])):
        print("\nSome modules failed. See traceback in summary JSON.")
        sys.exit(3)

    print("\nAll modules finished.")


if __name__ == "__main__":
    main()




