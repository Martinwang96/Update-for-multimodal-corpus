'''
Author: Martinwang96 -git
Date: 2025-04-16 18:29:10
Contact: martingwang01@163.com
LONG LIVE McDonald's
Copyright (c) 2025 by Martin Wang in Language of Sciences, Shanghai International Studies University, All Rights Reserved. 
'''

# -*- coding: utf-8 -*-
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.ticker import FuncFormatter # 用于格式化时间轴
from scipy.signal import savgol_filter
import pandas as pd
from sklearn.mixture import GaussianMixture
import csv
import warnings
import sys # 用于 sys.exit
import math # 用于角度计算中的 math.degrees

# ==================================
#  == 配置区域 (作为交互输入的默认值) ==
# ==================================

# 1. 检测目标 (仅用于标签和文件名)
TARGET_NAME = "身体倾斜 (2D)"
OUTPUT_CSV_TEMPLATE = '{}_events_tilt_2d_thresh{:.1f}.csv'
ANALYSIS_FOLDER_TEMPLATE = '{}_tilt_2d_analysis'

# 2. 定义关键点索引 (COCO 17-keypoint format is common for MMPose)
# 这些通常是固定的，但如果需要也可以做成交互输入
DEFAULT_LEFT_SHOULDER_IDX = 5
DEFAULT_RIGHT_SHOULDER_IDX = 6
DEFAULT_LEFT_HIP_IDX = 11
DEFAULT_RIGHT_HIP_IDX = 12
# 将默认值组合成元组
DEFAULT_REQUIRED_KP_INDICES = (DEFAULT_LEFT_SHOULDER_IDX, DEFAULT_RIGHT_SHOULDER_IDX, DEFAULT_LEFT_HIP_IDX, DEFAULT_RIGHT_HIP_IDX)

# 3. 坐标系假设 (2D Image Plane)
# 垂直轴 (向下) 方向向量 - 基本固定
VERTICAL_AXIS_2D = np.array([0.0, 1.0])

# 4. 默认计算参数 (用户可以覆盖)
DEFAULT_FPS = 30.0
DEFAULT_MIN_KEYPOINT_CONFIDENCE = 0.3
DEFAULT_SMOOTH_WINDOW = 11
DEFAULT_SMOOTH_POLY = 3
DEFAULT_OUTLIER_DIFF_THRESHOLD = 25.0 # 0 表示禁用

# 默认事件检测参数 (用户可以覆盖)
DEFAULT_TILT_THRESHOLD = 15.0
DEFAULT_MIN_DURATION_SEC = 0.1
DEFAULT_MAX_DURATION_SEC = 5.0 # 0 表示无限制
DEFAULT_MERGE_GAP_SEC = 0.3

# Matplotlib settings
rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'sans-serif'] # Fallback fonts
rcParams['axes.unicode_minus'] = False
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')
warnings.filterwarnings("ignore", message="Graph is not weighted")
warnings.filterwarnings("ignore", category=RuntimeWarning) # 忽略一些计算中的 RuntimeWarning

# ==================================
#  == 辅助函数 (通用) ==
# ==================================
def frame_to_time_str(frame_number, fps):
    """将帧号转换为 MM:SS.ms 格式字符串"""
    if not isinstance(fps, (int, float)) or fps <= 0: return f"Frame {frame_number} (无效 FPS)"
    total_seconds = frame_number / fps
    minutes = int(total_seconds // 60)
    seconds = total_seconds % 60
    return f"{minutes:02d}:{seconds:06.3f}"

def time_formatter(x, pos, fps):
    """Matplotlib FuncFormatter for time axis"""
    return frame_to_time_str(x, fps)

# ==================================
#  == 核心函数 (数据加载、处理 - 完整定义) ==
# ==================================
def load_required_keypoints_2d(json_path, kp_indices, min_confidence):
    """
    从 JSON 加载指定的 2D 关键点坐标和置信度。
    (函数体与之前提供的一致，包含错误处理和格式检查)
    """
    json_path = Path(json_path)
    if not json_path.exists(): raise FileNotFoundError(f"文件未找到: {json_path}")
    num_required_kps = len(kp_indices)
    if num_required_kps == 0: raise ValueError("kp_indices 不能为空。")

    try:
        with json_path.open('r', encoding='utf-8') as f: data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"无法解析 JSON 文件: {json_path} - {e}")
    except Exception as e:
        raise IOError(f"读取文件时出错: {json_path} - {e}")

    total_frames = len(data)
    if total_frames == 0:
        warnings.warn(f"JSON 文件 '{json_path.name}' 为空或格式不正确。")
        coords_list = [np.full((0, 2), np.nan) for _ in range(num_required_kps)]
        scores_list = [np.full(0, np.nan) for _ in range(num_required_kps)]
        valid_frame_mask = np.zeros(0, dtype=bool)
        return coords_list, scores_list, 0, valid_frame_mask

    coords_list = [np.full((total_frames, 2), np.nan) for _ in range(num_required_kps)]
    scores_list = [np.full(total_frames, np.nan) for _ in range(num_required_kps)]
    valid_frame_mask = np.zeros(total_frames, dtype=bool)
    processed_frames = 0

    for frame_idx, frame_info in enumerate(data):
        if not isinstance(frame_info, dict) or 'instances' not in frame_info: continue
        instances = frame_info.get("instances", [])
        if not isinstance(instances, list) or not instances: continue
        inst = instances[0]
        if not isinstance(inst, dict): continue

        keypoints_raw = inst.get("keypoints")
        scores_raw = inst.get("keypoint_scores")
        if keypoints_raw is None or scores_raw is None: continue

        try:
            keypoints = np.array(keypoints_raw)
            scores = np.array(scores_raw)
        except Exception: continue

        if keypoints.size == 0 or scores.size == 0: continue

        if keypoints.ndim == 1 and keypoints.size % 2 == 0 and keypoints.shape[0] > 0 :
            try: keypoints = keypoints.reshape(-1, 2)
            except ValueError: continue
        elif keypoints.ndim != 2 or keypoints.shape[1] != 2: continue

        num_kps_available = keypoints.shape[0]
        if num_kps_available == 0: continue

        valid_frame_mask[frame_idx] = True
        processed_frames += 1

        for list_idx, kp_idx in enumerate(kp_indices):
            if 0 <= kp_idx < num_kps_available and 0 <= kp_idx < len(scores):
                if scores[kp_idx] >= min_confidence:
                    coords_list[list_idx][frame_idx] = keypoints[kp_idx, :2]
                    scores_list[list_idx][frame_idx] = scores[kp_idx]

    if processed_frames == 0:
        warnings.warn(f"文件 '{json_path.name}': 未找到包含有效实例数据的帧。")

    return coords_list, scores_list, total_frames, valid_frame_mask

def interpolate_keypoint_positions(coords_list, limit_gap_frames=15):
    """
    对关键点坐标序列进行线性插值以填充 NaN 值。
    (函数体与之前提供的一致)
    """
    if not coords_list: return []
    interpolated_coords_list = []
    num_frames = coords_list[0].shape[0]

    for i, coords in enumerate(coords_list):
        if coords.shape[0] != num_frames:
            raise ValueError(f"插值错误：关键点 {i} 的帧数 ({coords.shape[0]}) 与第一个 ({num_frames}) 不匹配。")

        df = pd.DataFrame(coords, columns=['x', 'y'])
        df_interpolated = df.interpolate(method='linear', axis=0, limit=limit_gap_frames,
                                         limit_direction='both', limit_area=None)
        interpolated_coords_list.append(df_interpolated.to_numpy())
    return interpolated_coords_list

def calculate_torso_tilt_angle_2d(mid_hip_pos, mid_shoulder_pos, vert_axis=VERTICAL_AXIS_2D):
    """
    计算 2D 躯干与指定垂直轴的夹角 (0-180 度)。
    (函数体与之前提供的一致，包含 NaN 和零向量处理)
    """
    mid_hip_pos = np.asarray(mid_hip_pos)
    mid_shoulder_pos = np.asarray(mid_shoulder_pos)
    if mid_hip_pos.shape != mid_shoulder_pos.shape or mid_hip_pos.ndim != 2 or mid_hip_pos.shape[1] != 2:
        raise ValueError("输入的中点坐标形状不正确，应为 (F, 2)。")

    torso_vectors = mid_shoulder_pos - mid_hip_pos
    invalid_input_mask = np.any(np.isnan(mid_hip_pos), axis=1) | np.any(np.isnan(mid_shoulder_pos), axis=1)
    norms = np.linalg.norm(torso_vectors, axis=1)
    is_zero_norm = np.isclose(norms, 0, atol=1e-6)
    safe_norms = np.where(is_zero_norm, 1.0, norms)

    unit_torso_vectors = np.full_like(torso_vectors, np.nan)
    valid_norm_mask = ~is_zero_norm
    if np.any(valid_norm_mask): # Check if there are any valid vectors before division
       # Ensure broadcasting is correct if safe_norms becomes 1D
       unit_torso_vectors[valid_norm_mask] = torso_vectors[valid_norm_mask] / safe_norms[valid_norm_mask, np.newaxis]


    dot_product = np.full(torso_vectors.shape[0], np.nan)
    # Check for valid unit vectors before dot product
    valid_unit_vectors_mask = ~np.any(np.isnan(unit_torso_vectors), axis=1)
    if np.any(valid_unit_vectors_mask):
      dot_product[valid_unit_vectors_mask] = np.einsum('ij,j->i', unit_torso_vectors[valid_unit_vectors_mask], vert_axis)

    cos_theta = np.clip(dot_product, -1.0, 1.0)
    angles_rad = np.arccos(cos_theta)
    angles_deg = np.degrees(angles_rad)
    final_invalid_mask = invalid_input_mask | is_zero_norm
    angles_deg[final_invalid_mask] = np.nan
    return angles_deg

def smooth_angles(angles, window, poly):
    """
    使用 Savitzky-Golay 滤波器平滑角度序列。
    (函数体与之前提供的一致，包含窗口/阶数检查和 NaN 处理)
    """
    angles_arr = np.asarray(angles)
    num_frames = len(angles_arr)
    valid_mask = ~np.isnan(angles_arr)
    num_valid = np.sum(valid_mask)

    try:
        eff_window = max(3, int(window))
        if eff_window % 2 == 0: eff_window += 1
        eff_poly = min(int(poly), eff_window - 1); eff_poly = max(1, eff_poly)
    except (TypeError, ValueError):
        warnings.warn("平滑窗口或阶数无效，使用默认值。")
        eff_window = max(3, int(DEFAULT_SMOOTH_WINDOW) | 1)
        eff_poly = min(int(DEFAULT_SMOOTH_POLY), eff_window - 1); eff_poly = max(1, eff_poly)

    if num_valid < eff_window or num_frames < eff_window:
        return angles_arr

    angles_series = pd.Series(angles_arr)
    angles_filled = angles_series.interpolate(method='linear', limit_direction='both', limit_area='inside', limit=int(eff_window*2))
    angles_filled = angles_filled.fillna(method='ffill').fillna(method='bfill')
    angles_to_smooth = angles_filled.to_numpy()

    if np.any(np.isnan(angles_to_smooth)):
         warnings.warn("平滑前角度序列仍含 NaN，返回原始数据。")
         return angles_arr

    try:
        smoothed = savgol_filter(angles_to_smooth, eff_window, eff_poly)
        smoothed[~valid_mask] = np.nan
        return smoothed
    except Exception as e:
        warnings.warn(f"角度平滑失败: {e}。返回原始(填充后)数据。")
        return angles_to_smooth

def compute_angular_velocity(angles_deg, fps):
    """
    计算角度序列的瞬时角速度 (度/秒)。
    (函数体与之前提供的一致，使用 np.gradient)
    """
    angles_arr = np.asarray(angles_deg)
    if not isinstance(fps, (int, float)) or fps <= 0:
        # warnings.warn(f"无效的 FPS 值 ({fps})，无法计算角速度。")
        return np.full_like(angles_arr, np.nan)
    if angles_arr.size < 2: return np.full_like(angles_arr, np.nan)

    original_nan_mask = np.isnan(angles_arr)
    angles_series = pd.Series(angles_arr)
    limit_frames = max(5, int(fps * 0.5)) if fps > 0 else 5
    angles_filled = angles_series.interpolate(method='linear', limit_direction='both', limit_area='inside', limit=limit_frames)
    angles_filled = angles_filled.fillna(method='ffill').fillna(method='bfill')
    angles_ready = angles_filled.to_numpy()

    if np.all(np.isnan(angles_ready)): return np.full_like(angles_arr, np.nan)

    dt = 1.0 / fps if fps > 0 else 1.0 # Avoid division by zero if fps is somehow 0
    angular_velocities_dps = np.gradient(angles_ready, dt)
    angular_velocities_dps[original_nan_mask] = np.nan
    abs_angular_velocity_dps = np.abs(angular_velocities_dps)
    return abs_angular_velocity_dps

def recommend_tilt_threshold_2d(tilt_angles_0_180, default_thresh, method='gmm', min_thresh_val=5.0):
    """
    推荐倾斜阈值 (作用于 0-90 度范围)。
    (函数体与之前提供的一致，包含 GMM 和百分位法)
    """
    analysis_data = {}
    tilt_angles_np = np.asarray(tilt_angles_0_180)
    if tilt_angles_np.size == 0 or np.all(np.isnan(tilt_angles_np)):
         warnings.warn("输入角度为空或全为 NaN，无法推荐阈值。")
         return default_thresh, {'fallback': {'threshold_definite': default_thresh}}

    angles_relative_to_vertical = np.minimum(tilt_angles_np, 180.0 - tilt_angles_np)
    angles_relative_to_vertical[np.isnan(tilt_angles_np)] = np.nan
    valid_abs_angles = angles_relative_to_vertical[~np.isnan(angles_relative_to_vertical)]

    if len(valid_abs_angles) < 10:
        warnings.warn(f"有效角度数据点过少 ({len(valid_abs_angles)} < 10)，使用默认阈值 {default_thresh:.1f}°。")
        return default_thresh, {'fallback': {'threshold_definite': default_thresh}}

    non_zero_valid = valid_abs_angles[valid_abs_angles > 1.0]
    if len(non_zero_valid) < 20:
        warnings.warn(f"有效非零角度(>1°)数据不足 (<20)，回退到百分位法。")
        if len(non_zero_valid) < 2:
             warnings.warn(f"有效非零角度(>1°)数据极少 (<2)，使用默认阈值 {default_thresh:.1f}°。")
             return default_thresh, {'fallback': {'threshold_definite': default_thresh}}
        method = 'percentile'

    rec = default_thresh
    final_method = 'default'

    if method == 'gmm':
        try:
            X = non_zero_valid.reshape(-1, 1)
            gmm = GaussianMixture(n_components=2, random_state=42, covariance_type='spherical',
                                  n_init=10, reg_covar=1e-5, init_params='k-means++')
            gmm.fit(X)
            if not gmm.converged_:
                 warnings.warn("GMM 未收敛，回退百分位法。")
                 method = 'percentile'
            else:
                 means = gmm.means_.flatten()
                 weights = gmm.weights_.flatten()
                 if gmm.covariance_type == 'spherical':
                     variances = np.array(gmm.covariances_)
                     stds = np.sqrt(np.maximum(variances, 1e-6))
                     if len(stds) != len(means): stds = np.full_like(means, np.sqrt(np.mean(variances)))
                 else: # Fallback for other types
                     stds = np.sqrt(np.diag(gmm.covariances_)) if gmm.covariances_.ndim == 2 else np.sqrt(gmm.covariances_)

                 if len(stds) == len(means): # Check successful std extraction
                     high_mean_idx = np.argmax(means)
                     high_mean = means[high_mean_idx]; high_std = stds[high_mean_idx]; high_weight = weights[high_mean_idx]
                     low_mean_idx = 1 - high_mean_idx
                     low_mean = means[low_mean_idx]; low_std = stds[low_mean_idx]
                     p95 = np.percentile(non_zero_valid, 95)
                     gmm_based_thresh = max(min_thresh_val, high_mean - 1.0 * high_std)
                     rec = max(min_thresh_val, gmm_based_thresh)
                     analysis_data['gmm'] = {'threshold_definite': rec, 'high_mean': high_mean, 'high_std': high_std,
                                            'low_mean': low_mean, 'low_std': low_std, 'high_weight': high_weight,
                                            'p95_ref': p95}
                     final_method = 'gmm'
                 else: # Handle std dev extraction failure
                     warnings.warn("无法解析 GMM 标准差，回退百分位法。")
                     method = 'percentile'

        except Exception as e:
            warnings.warn(f"GMM 失败: {e}。回退百分位法。")
            method = 'percentile'

    if method == 'percentile':
        if len(non_zero_valid) >= 2:
            p95 = np.percentile(non_zero_valid, 95)
            rec = max(p95, min_thresh_val)
            analysis_data['percentile'] = {'threshold_definite': rec, 'p95': p95}
            final_method = 'percentile'
        else:
             rec = default_thresh
             analysis_data['fallback'] = {'threshold_definite': rec}
             final_method = 'fallback'
             warnings.warn(f"百分位法也失败，使用默认阈值 {rec:.1f}°。")

    if not np.isfinite(rec) or rec <= 0:
         rec = default_thresh; final_method = 'default_fallback'
         warnings.warn(f"推荐阈值无效 ({rec})，使用默认值 {default_thresh:.1f}°。")
         analysis_data['fallback'] = {'threshold_definite': default_thresh}

    analysis_data['final_recommendation'] = {'threshold': rec, 'method_used': final_method}
    return rec, analysis_data

def plot_tilt_threshold_analysis_2d(tilt_angles_0_180, analysis_data, target_name=TARGET_NAME):
    """
    绘制 0-90 度倾斜角度分布与推荐阈值。
    (函数体与之前提供的一致，包含 GMM 分量绘制逻辑)
    """
    fig = plt.figure(figsize=(12, 7))
    tilt_angles_np = np.asarray(tilt_angles_0_180)
    if tilt_angles_np.size == 0 or np.all(np.isnan(tilt_angles_np)):
        plt.title(f"{target_name}: 无有效角度数据用于阈值分析"); return fig

    angles_relative_to_vertical = np.minimum(tilt_angles_np, 180.0 - tilt_angles_np)
    angles_relative_to_vertical[np.isnan(tilt_angles_np)] = np.nan
    valid_data = angles_relative_to_vertical[~np.isnan(angles_relative_to_vertical)]
    plot_data = valid_data[valid_data > 0.5]

    if len(plot_data) == 0:
        plt.title(f"{target_name}: 无有效非零角度(>0.5°)数据用于阈值分析"); return fig

    try: counts, bins, patches = plt.hist(plot_data, bins=60, density=True, alpha=0.75, color='mediumseagreen', label=f'倾斜幅度 (0-90°) 分布')
    except Exception as e: plt.title(f"{target_name}: 绘制直方图失败 - {e}"); return fig

    rec_data = analysis_data.get('final_recommendation', {})
    rec_thresh = rec_data.get('threshold'); rec_method = rec_data.get('method_used', '?').upper()
    if rec_thresh is not None and np.isfinite(rec_thresh):
        plt.axvline(rec_thresh, color='r', linestyle='--', linewidth=2, label=f"最终推荐阈值 ({rec_method}): {rec_thresh:.2f}°")
        try: y_max = plt.gca().get_ylim()[1]; plt.text(rec_thresh * 1.02, y_max * 0.9, f"{rec_thresh:.2f}", color='r', weight='bold', fontsize=10)
        except Exception: pass

    if analysis_data and 'gmm' in analysis_data and analysis_data['final_recommendation']['method_used'] == 'gmm':
        gmm_data = analysis_data['gmm']
        try: # Safely plot GMM components
            means = [gmm_data.get('low_mean'), gmm_data.get('high_mean')]
            stds = [gmm_data.get('low_std'), gmm_data.get('high_std')]
            weights = [1.0 - gmm_data.get('high_weight', 0.5), gmm_data.get('high_weight', 0.5)] # Estimate weights
            x_plot = np.linspace(bins[0], bins[-1], 500)
            colors = ['darkorange', 'dodgerblue']; labels = ['GMM 低角度分量', 'GMM 高角度分量']
            for i in range(len(means)):
                if means[i] is not None and stds[i] is not None and np.isfinite(means[i]) and np.isfinite(stds[i]) and stds[i] > 1e-6:
                    pdf = weights[i] * (1 / (stds[i] * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_plot - means[i]) / stds[i])**2)
                    plt.plot(x_plot, pdf, color=colors[i], linestyle=':', linewidth=1.5, label=f'{labels[i]} (μ={means[i]:.1f}, σ={stds[i]:.1f})')
        except Exception as e: warnings.warn(f"绘制 GMM 分量时出错: {e}")

    plt.title(f"{target_name} 倾斜幅度 (0-90°) 分布与阈值分析")
    plt.xlabel(f"躯干倾斜幅度 (度)")
    plt.ylabel("概率密度 / GMM分量")
    plt.legend(fontsize='small'); plt.grid(True, alpha=0.3); plt.xlim(left=-2)
    xmax_data = np.max(plot_data) if len(plot_data) > 0 else 90
    xmax_thresh = rec_thresh if rec_thresh is not None and np.isfinite(rec_thresh) else 0
    plt.xlim(right=max(xmax_data * 1.1, xmax_thresh * 1.2, 30))
    plt.tight_layout()
    return fig

def detect_tilt_events_2d(tilt_angles_0_180, threshold_0_90,
                          min_duration_frames=3, max_duration_frames=None,
                          min_gap_frames=15, fps=30.0):
    """
    检测倾斜事件。
    (函数体与之前提供的一致，包含事件合并和统计)
    """
    tilt_angles_np = np.asarray(tilt_angles_0_180)
    num_frames = len(tilt_angles_np)
    if num_frames == 0: return []
    if not isinstance(threshold_0_90, (int, float)) or threshold_0_90 <= 0: return []
    if not isinstance(fps, (int, float)) or fps <= 0: fps = 1

    angles_relative_to_vertical = np.minimum(tilt_angles_np, 180.0 - tilt_angles_np)
    angles_relative_to_vertical[np.isnan(tilt_angles_np)] = np.nan
    exceed_mask = np.where(np.isnan(angles_relative_to_vertical), False, angles_relative_to_vertical >= threshold_0_90)
    if not np.any(exceed_mask): return []

    idx = np.where(exceed_mask)[0]
    if len(idx) <= 1: groups = [idx] if len(idx) == 1 else []
    else: splits = np.where(np.diff(idx) > min_gap_frames)[0] + 1; groups = np.split(idx, splits)

    events = []
    for group_indices in groups:
        if group_indices.size == 0: continue
        start_frame = group_indices[0]; end_frame = group_indices[-1]
        duration_frames = (end_frame - start_frame) + 1

        if duration_frames < min_duration_frames: continue
        if max_duration_frames is not None and duration_frames > max_duration_frames: continue

        segment_angles_0_90 = angles_relative_to_vertical[start_frame : end_frame + 1]
        valid_segment_angles = segment_angles_0_90[~np.isnan(segment_angles_0_90)]
        if valid_segment_angles.size == 0: continue
        if not np.any(valid_segment_angles >= threshold_0_90): continue

        start_time_sec = start_frame / fps; end_time_sec = (end_frame + 1) / fps
        duration_sec = duration_frames / fps
        max_angle = float(np.nanmax(valid_segment_angles)) if valid_segment_angles.size > 0 else 0.0
        avg_angle = float(np.nanmean(valid_segment_angles)) if valid_segment_angles.size > 0 else 0.0

        events.append({
            'event_type': '倾斜', 'start_frame': start_frame, 'end_frame': end_frame,
            'start_time_str': frame_to_time_str(start_frame, fps), 'end_time_str': frame_to_time_str(end_frame, fps),
            'start_time_sec': start_time_sec, 'end_time_sec': end_time_sec,
            'duration_sec': duration_sec, 'duration_frames': duration_frames,
            'max_angle_0_90': max_angle, 'avg_angle_0_90': avg_angle,
            'units_angle': '度', 'threshold_0_90_used': threshold_0_90
        })
    return events

def export_tilt_events_to_csv(events, output_path):
    """
    导出倾斜事件到 CSV。
    (函数体与之前提供的一致)
    """
    output_path = Path(output_path)
    if not events: return None
    try:
        fieldnames = [
            'event_type', 'start_frame', 'end_frame', 'start_time_sec', 'end_time_sec',
            'duration_sec', 'duration_frames', 'max_angle_0_90', 'avg_angle_0_90',
            'units_angle', 'threshold_0_90_used', 'start_time_str', 'end_time_str',
        ]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open('w', newline='', encoding='utf-8-sig') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            for event in events:
                 row = event.copy()
                 for key in ['start_time_sec', 'end_time_sec', 'duration_sec']: row[key] = f"{event.get(key, 0):.3f}"
                 for key in ['max_angle_0_90', 'avg_angle_0_90', 'threshold_0_90_used']: row[key] = f"{event.get(key, 0):.2f}"
                 writer.writerow(row)
        return output_path
    except IOError as e: warnings.warn(f"CSV 文件写入权限错误: {output_path} - {e}"); return None
    except Exception as e: warnings.warn(f"CSV 导出失败: {e}"); return None

def plot_tilt_analysis_2d(angles_0_180_raw, smoothed_angles_0_180, angular_velocities, events,
                          threshold_0_90, fps, title_prefix="", target_name=TARGET_NAME):
    """
    绘制 2D 倾斜综合分析图。
    (函数体与之前提供的一致，包含双子图和时间轴格式化)
    """
    angles_np_raw = np.asarray(angles_0_180_raw)
    num_frames = len(angles_np_raw)
    if num_frames == 0: return None

    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    if not isinstance(axes, (list, np.ndarray)) or len(axes) != 2:
        plt.close(fig); return None
    ax1, ax2 = axes
    full_title = f"{title_prefix.strip()} {target_name} 检测分析".strip()
    fig.suptitle(full_title, fontsize=16, y=0.98)
    frames_axis = np.arange(num_frames)
    time_axis_func = lambda f: f / fps if isinstance(fps, (int, float)) and fps > 0 else np.nan

    ax1.set_ylabel('躯干倾斜角度 (度)')
    ax1.set_title('躯干倾斜角度 (与垂直向下 [0,1] 夹角) 随时间变化')
    ax1.grid(True, alpha=0.3); ax1.set_ylim(-5, 185)

    valid_raw_mask = ~np.isnan(angles_np_raw)
    if np.any(valid_raw_mask): ax1.plot(frames_axis[valid_raw_mask], angles_np_raw[valid_raw_mask], 'b-', alpha=0.5, linewidth=1.0, label='原始角度 (0-180°)')
    smoothed_np = np.asarray(smoothed_angles_0_180)
    if len(smoothed_np) == num_frames :
        valid_smooth_mask = ~np.isnan(smoothed_np)
        if np.any(valid_smooth_mask): ax1.plot(frames_axis[valid_smooth_mask], smoothed_np[valid_smooth_mask], 'g-', linewidth=1.5, label='平滑角度 (0-180°)')

    ax1.axhline(0, color='black', linestyle='-', linewidth=0.8, label='垂直向下 (0°)')
    ax1.axhline(90, color='grey', linestyle=':', linewidth=0.8, label='水平 (90°)')
    ax1.axhline(180, color='black', linestyle='-', linewidth=0.8, label='垂直向上 (180°)')
    if threshold_0_90 is not None and threshold_0_90 > 0:
        ax1.axhline(threshold_0_90, color='r', ls='--', lw=1.0, label=f'阈值线 (±{threshold_0_90:.1f}° 对应)')
        if not np.isclose(180.0 - threshold_0_90, threshold_0_90) : ax1.axhline(180.0 - threshold_0_90, color='r', ls='--', lw=1.0)

    event_label_added = False
    if events:
         for event in events:
             st = event.get('start_frame', -1); et = event.get('end_frame', -1)
             if 0 <= st <= et < num_frames:
                  label = '检测到的倾斜事件' if not event_label_added else "_nolegend_"
                  ax1.axvspan(st, et + 1, alpha=0.25, color='gold', label=label, zorder=-1); event_label_added = True

    ax2.set_title('绝对角速度 (基于平滑角度)'); ax2.set_ylabel('绝对角速度 (度/秒)'); ax2.grid(True, alpha=0.3)
    vel_np = np.asarray(angular_velocities)
    if len(vel_np) == num_frames :
        valid_vel_mask = ~np.isnan(vel_np)
        if np.any(valid_vel_mask):
            ax2.plot(frames_axis[valid_vel_mask], vel_np[valid_vel_mask], 'r-', alpha=0.7, linewidth=1.2, label='绝对角速度 (|ω|)')
            try: max_vel = np.nanmax(vel_np[valid_vel_mask]); min_vel = -0.05 * max_vel if max_vel > 0 else -0.1; top_lim = max(max_vel*1.1, 10); ax2.set_ylim(bottom=min_vel, top=top_lim)
            except Exception: ax2.set_ylim(bottom=-1, top=50)
        else: ax2.text(0.5, 0.5, '角速度全为无效值', ha='center', va='center', transform=ax2.transAxes, fontsize=12, color='gray'); ax2.set_ylim(bottom=-1, top=10)
    else: ax2.text(0.5, 0.5, '角速度数据不可用', ha='center', va='center', transform=ax2.transAxes, fontsize=12, color='gray'); ax2.set_ylim(bottom=-1, top=10)

    if events:
         for event in events:
             st = event.get('start_frame', -1); et = event.get('end_frame', -1)
             if 0 <= st <= et < num_frames: ax2.axvspan(st, et + 1, alpha=0.25, color='gold', zorder=-1)

    handles1, labels1 = ax1.get_legend_handles_labels(); handles2, labels2 = ax2.get_legend_handles_labels()
    if handles1: ax1.legend(loc='upper right', fontsize='small');
    if handles2: ax2.legend(loc='upper right', fontsize='small')
    ax2.set_xlabel('帧 / 时间')

    if isinstance(fps, (int, float)) and fps > 0:
        formatter = FuncFormatter(lambda x, pos: time_formatter(x, pos, fps))
        ax2.xaxis.set_major_formatter(formatter)
        try: secax_top = ax1.secondary_xaxis('top', functions=(time_axis_func, lambda t: t * fps)); secax_top.set_xlabel('时间 (秒)')
        except Exception: pass # Ignore secondary axis errors
    else: ax2.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(x)}"))

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]); return fig


# ==================================
#  == 主执行区域 (完全交互式) ==
# ==================================
if __name__ == "__main__":

    print("欢迎使用 2D 身体倾斜分析工具！")
    print("请按照提示输入参数。直接按 Enter 键可使用括号中的默认值。")
    print("-" * 30)

    # --- 1. 获取输入 JSON 文件路径 ---
    input_json_path = None
    while input_json_path is None:
        try:
            json_path_str = input(">>> 请输入 JSON 文件路径: ")
            if not json_path_str: continue
            temp_path = Path(json_path_str.strip())
            if temp_path.is_file(): input_json_path = temp_path
            else: print(f"错误：文件不存在或路径无效: '{json_path_str}'")
        except (KeyboardInterrupt, EOFError): print("\n操作取消。"); sys.exit(1)
        except Exception as e: print(f"读取路径时出错: {e}")

    # --- 2. 获取输出目录 ---
    default_output_dir = input_json_path.parent / ANALYSIS_FOLDER_TEMPLATE.format(input_json_path.stem)
    output_dir = default_output_dir
    try:
        output_dir_str = input(f">>> 请输入输出目录 (默认: '{default_output_dir}'): ").strip()
        if output_dir_str: output_dir = Path(output_dir_str)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"结果将保存到: {output_dir}")
    except (KeyboardInterrupt, EOFError): print("\n操作取消。"); sys.exit(1)
    except Exception as e:
        print(f"设置输出目录时出错 ({e})，将使用默认目录。")
        output_dir = default_output_dir
        try: output_dir.mkdir(parents=True, exist_ok=True)
        except Exception: print(f"错误：无法创建默认输出目录 {output_dir}"); sys.exit(1)


    # --- 3. 获取基础参数 (FPS, Min Confidence) ---
    fps = DEFAULT_FPS
    min_confidence = DEFAULT_MIN_KEYPOINT_CONFIDENCE
    try:
        fps_str = input(f">>> 请输入视频帧率 FPS (默认: {DEFAULT_FPS:.1f}): ").strip()
        if fps_str: fps = float(fps_str)
        if fps <= 0: raise ValueError("FPS 必须为正数")

        conf_str = input(f">>> 请输入关键点最小置信度 (0-1, 默认: {DEFAULT_MIN_KEYPOINT_CONFIDENCE:.2f}): ").strip()
        if conf_str: min_confidence = float(conf_str)
        if not (0.0 <= min_confidence <= 1.0): raise ValueError("置信度必须在 0 和 1 之间")
    except (KeyboardInterrupt, EOFError): print("\n操作取消。"); sys.exit(1)
    except ValueError as e:
        print(f"输入错误 ({e})，将使用默认值。")
        fps = DEFAULT_FPS; min_confidence = DEFAULT_MIN_KEYPOINT_CONFIDENCE
    print(f"使用参数: FPS={fps:.1f}, MinConf={min_confidence:.2f}")

    # --- 4. 获取平滑参数 ---
    smooth_window = DEFAULT_SMOOTH_WINDOW
    smooth_poly = DEFAULT_SMOOTH_POLY
    try:
        win_str = input(f">>> 请输入平滑窗口大小 (奇数>=3, 默认: {DEFAULT_SMOOTH_WINDOW}): ").strip()
        if win_str: smooth_window = int(win_str)
        if smooth_window < 3 or smooth_window % 2 == 0: raise ValueError("平滑窗口必须是 >=3 的奇数")

        poly_str = input(f">>> 请输入平滑多项式阶数 (1 <= poly < window, 默认: {DEFAULT_SMOOTH_POLY}): ").strip()
        if poly_str: smooth_poly = int(poly_str)
        if not (1 <= smooth_poly < smooth_window): raise ValueError("多项式阶数必须 >=1 且小于窗口大小")
    except (KeyboardInterrupt, EOFError): print("\n操作取消。"); sys.exit(1)
    except ValueError as e:
        print(f"输入错误 ({e})，将使用默认平滑参数。")
        smooth_window = DEFAULT_SMOOTH_WINDOW; smooth_poly = DEFAULT_SMOOTH_POLY
    print(f"使用平滑参数: 窗口={smooth_window}, 阶数={smooth_poly}")

    # --- 5. 获取异常值处理参数 ---
    outlier_thresh = DEFAULT_OUTLIER_DIFF_THRESHOLD
    try:
        outlier_str = input(f">>> 请输入角度突变阈值 (度/帧, 0 禁用, 默认: {DEFAULT_OUTLIER_DIFF_THRESHOLD:.1f}): ").strip()
        if outlier_str: outlier_thresh = float(outlier_str)
        if outlier_thresh < 0: raise ValueError("阈值不能为负数")
    except (KeyboardInterrupt, EOFError): print("\n操作取消。"); sys.exit(1)
    except ValueError as e:
        print(f"输入错误 ({e})，将使用默认突变阈值。")
        outlier_thresh = DEFAULT_OUTLIER_DIFF_THRESHOLD
    print(f"角度突变检测阈值: {'禁用' if outlier_thresh == 0 else f'{outlier_thresh:.1f} 度/帧'}")

    # --- 核心处理步骤 (加载、计算、推荐阈值) ---
    print("\n--- 正在加载和预处理数据 ---")
    final_threshold = DEFAULT_TILT_THRESHOLD # 初始化，以防后续步骤出错
    rec_thresh_angle = DEFAULT_TILT_THRESHOLD
    threshold_analysis = None
    detected_events = []
    results_summary = (0, final_threshold, 0.0)
    analysis_successful = False # 标记是否成功完成主要分析步骤

    try:
        coords_list, _, total_frames, valid_frames_mask = load_required_keypoints_2d(
            input_json_path, DEFAULT_REQUIRED_KP_INDICES, min_confidence) # 使用默认索引
        min_req_frames = max(10, smooth_window)
        if total_frames == 0 or np.sum(valid_frames_mask) < min_req_frames:
             raise ValueError(f"有效帧数 ({np.sum(valid_frames_mask)}) 过少 (需要 >={min_req_frames})。")

        ls_coords, rs_coords, lh_coords, rh_coords = interpolate_keypoint_positions(coords_list)
        mid_shoulder_pos = (ls_coords + rs_coords) / 2.0
        mid_hip_pos = (lh_coords + rh_coords) / 2.0
        mid_shoulder_pos[np.any(np.isnan(ls_coords), axis=1) | np.any(np.isnan(rs_coords), axis=1)] = np.nan
        mid_hip_pos[np.any(np.isnan(lh_coords), axis=1) | np.any(np.isnan(rh_coords), axis=1)] = np.nan
        tilt_angles_0_180_raw = calculate_torso_tilt_angle_2d(mid_hip_pos, mid_shoulder_pos)
        tilt_angles_0_180_for_plot = tilt_angles_0_180_raw.copy()
        if np.all(np.isnan(tilt_angles_0_180_raw)): raise ValueError("所有计算出的躯干角度均为无效值！")

        tilt_angles_0_180_corrected = tilt_angles_0_180_raw.copy()
        if outlier_thresh > 0 and len(tilt_angles_0_180_corrected) > 1:
            angles_series = pd.Series(tilt_angles_0_180_corrected)
            fill_limit = max(5, int(fps * 0.2)) if fps > 0 else 5
            angles_filled_temp = angles_series.interpolate(method='linear', limit_direction='both', limit=fill_limit).fillna(method='ffill').fillna(method='bfill')
            angles_for_diff = angles_filled_temp.to_numpy()
            if not np.all(np.isnan(angles_for_diff)):
                angle_diff = np.diff(angles_for_diff, prepend=np.nan)
                outlier_indices = np.where(np.abs(angle_diff) > outlier_thresh)[0]
                if len(outlier_indices) > 0:
                    print(f"  检测到 {len(outlier_indices)} 个角度突变点，进行插值修复...")
                    tilt_angles_series_corr = pd.Series(tilt_angles_0_180_corrected)
                    indices_to_nan = np.unique(outlier_indices)
                    indices_to_nan = indices_to_nan[indices_to_nan < len(tilt_angles_series_corr)]
                    if indices_to_nan.size > 0:
                        tilt_angles_series_corr.iloc[indices_to_nan] = np.nan
                        tilt_angles_0_180_corrected = tilt_angles_series_corr.interpolate(method='linear', limit_direction='both', limit_area=None, limit=10).to_numpy()

        smoothed_angles_0_180 = smooth_angles(tilt_angles_0_180_corrected, window=smooth_window, poly=smooth_poly)
        if np.all(np.isnan(smoothed_angles_0_180)):
            print("警告：角度平滑后全为无效值！")
            smoothed_angles_0_180 = tilt_angles_0_180_corrected

        angular_velocities_abs = compute_angular_velocity(smoothed_angles_0_180, fps)

        # --- 6. 阈值推荐与确认 ---
        print("\n--- 推荐倾斜阈值 (作用于 0-90 度范围) ---")
        try:
            rec_thresh_angle, threshold_analysis = recommend_tilt_threshold_2d(
                tilt_angles_0_180_corrected, default_thresh=DEFAULT_TILT_THRESHOLD) # Pass default
            if threshold_analysis:
                print(f"推荐方法: {threshold_analysis.get('final_recommendation', {}).get('method_used', '未知')}")
                threshold_plot_fig = plot_tilt_threshold_analysis_2d(
                    tilt_angles_0_180_corrected, threshold_analysis, target_name=f"{TARGET_NAME} - {input_json_path.stem}")
                if threshold_plot_fig:
                    threshold_plot_path = output_dir / f"{input_json_path.stem}_tilt_threshold_analysis.png"
                    try: threshold_plot_fig.savefig(threshold_plot_path, dpi=100)
                    except Exception as save_err: print(f"警告：保存阈值图失败: {save_err}")
                    plt.close(threshold_plot_fig)
            else: rec_thresh_angle = DEFAULT_TILT_THRESHOLD
        except Exception as e:
            print(f"错误：阈值推荐时出错: {e}")
            rec_thresh_angle = DEFAULT_TILT_THRESHOLD; threshold_analysis = None

        print(f"\n推荐阈值: ≈ {rec_thresh_angle:.2f} 度 (0-90范围)")
        try:
            thresh_input = input(f"  输入最终使用的阈值 (回车使用推荐值): ").strip()
            if thresh_input: final_threshold = float(thresh_input)
            else: final_threshold = rec_thresh_angle
            if final_threshold <= 0: raise ValueError("阈值必须为正数")
        except (KeyboardInterrupt, EOFError): print("\n操作取消。"); sys.exit(1)
        except ValueError as e:
            print(f"输入错误 ({e})，使用推荐/默认阈值 {rec_thresh_angle:.2f}")
            final_threshold = max(1.0, rec_thresh_angle)
        print(f"最终使用阈值: {final_threshold:.2f} 度")

        # --- 7. 获取事件时间约束 ---
        print("\n设置事件时间约束 (单位: 秒)")
        min_dur_sec = DEFAULT_MIN_DURATION_SEC; max_dur_sec = DEFAULT_MAX_DURATION_SEC; merge_gap_sec = DEFAULT_MERGE_GAP_SEC
        try:
            min_dur_input = input(f"  最短持续时间 (默认 {DEFAULT_MIN_DURATION_SEC:.2f}s): ").strip()
            if min_dur_input: min_dur_sec = float(min_dur_input)
            max_dur_input = input(f"  最长持续时间 (默认 {DEFAULT_MAX_DURATION_SEC:.1f}s, 0 无限制): ").strip()
            if max_dur_input: max_dur_sec = float(max_dur_input)
            merge_gap_input = input(f"  合并间隔 (默认 {DEFAULT_MERGE_GAP_SEC:.2f}s): ").strip()
            if merge_gap_input: merge_gap_sec = float(merge_gap_input)
            if min_dur_sec < 0 or max_dur_sec < 0 or merge_gap_sec < 0: raise ValueError("时间不能为负数")
        except (KeyboardInterrupt, EOFError): print("\n操作取消。"); sys.exit(1)
        except ValueError as e:
            print(f"输入错误 ({e})，使用默认时间约束。")
            min_dur_sec = DEFAULT_MIN_DURATION_SEC; max_dur_sec = DEFAULT_MAX_DURATION_SEC; merge_gap_sec = DEFAULT_MERGE_GAP_SEC

        min_df = max(1, int(min_dur_sec * fps)) if fps > 0 else 1
        max_df = int(max_dur_sec * fps) if max_dur_sec > 0 and fps > 0 else None
        min_gf = int(merge_gap_sec * fps) if fps > 0 else 0
        print(f"约束 (帧): MinDur={min_df}, MaxDur={'无限制' if max_df is None else max_df}, MergeGap={min_gf}")

        # --- 8. 事件检测 ---
        print(f"\n使用阈值 {final_threshold:.2f} 检测事件...")
        detected_events = detect_tilt_events_2d(
            smoothed_angles_0_180, final_threshold, min_duration_frames=min_df,
            max_duration_frames=max_df, min_gap_frames=min_gf, fps=fps
        )

        # --- 9. 显示 & 导出结果 ---
        if detected_events:
            print(f"\n检测到 {len(detected_events)} 个倾斜事件:")
            csv_fn = OUTPUT_CSV_TEMPLATE.format(input_json_path.stem, final_threshold)
            csv_path = output_dir / csv_fn
            export_path = export_tilt_events_to_csv(detected_events, csv_path)
            if export_path: print(f"事件已导出到: {export_path}")
            else: print("警告：CSV 导出失败。")
        else: print("\n未检测到符合条件的倾斜事件。")

        # --- 10. 最终绘图 ---
        create_plots = True
        try:
            plot_choice = input("\n是否生成并显示最终分析图? (Y/n): ").strip().lower()
            if plot_choice == 'n': create_plots = False
        except (KeyboardInterrupt, EOFError): print("\n操作取消。"); sys.exit(1)

        if create_plots:
            print("生成最终分析图...")
            analysis_plot_fig = plot_tilt_analysis_2d(
                angles_0_180_raw=tilt_angles_0_180_for_plot, smoothed_angles_0_180=smoothed_angles_0_180,
                angular_velocities=angular_velocities_abs, events=detected_events,
                threshold_0_90=final_threshold, fps=fps, title_prefix=f"{input_json_path.stem}\n", target_name=TARGET_NAME
            )
            if analysis_plot_fig:
                plot_path = output_dir / f"{input_json_path.stem}_tilt_analysis_final.png"
                try:
                    analysis_plot_fig.savefig(plot_path, dpi=150)
                    print(f"最终分析图已保存到: {plot_path}")
                    print("提示：关闭图形窗口后程序将结束。")
                    analysis_plot_fig.show(); plt.close(analysis_plot_fig)
                except Exception as e:
                    print(f"错误：保存或显示最终分析图失败: {e}")
                    plt.close(analysis_plot_fig)
            else: print("未能生成最终分析图。")
        else: print("已跳过生成最终分析图。")

        analysis_successful = True # 标记分析成功完成

    # --- 统一的错误处理 ---
    except FileNotFoundError as e: print(f"\n错误: {e}")
    except ValueError as e: print(f"\n处理错误: {e}")
    except Exception as e:
        print(f"\n分析过程中发生未预料的错误: {e}")
        import traceback; traceback.print_exc()
    finally:
        plt.close('all')

        # 计算总结信息
        num_events = len(detected_events)
        max_abs_angle_overall = 0.0
        # 仅在分析成功且相关变量存在时计算
        if analysis_successful and 'smoothed_angles_0_180' in locals():
            try:
                 angles_relative = np.minimum(smoothed_angles_0_180, 180 - smoothed_angles_0_180)
                 valid_angles_relative = angles_relative[~np.isnan(angles_relative)]
                 if valid_angles_relative.size > 0: max_abs_angle_overall = np.nanmax(valid_angles_relative)
            except Exception: pass # 忽略计算总结时的错误

        # 使用在主流程中最终确定的 final_threshold 值
        final_thresh_value = final_threshold if 'final_threshold' in locals() else DEFAULT_TILT_THRESHOLD
        results_summary = (num_events, final_thresh_value, max_abs_angle_overall)

        print(f"\n--- 分析流程结束 ({input_json_path.name if input_json_path else '未知文件'}) ---")
        print(f"总结: 检测到 {results_summary[0]} 个事件, 使用阈值 {results_summary[1]:.2f}°, 最大倾斜幅度约 {results_summary[2]:.1f}°")
        sys.exit(0)