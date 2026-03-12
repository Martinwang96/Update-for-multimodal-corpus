# -*- coding: utf-8 -*-
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.ticker import FuncFormatter # 用于格式化时间轴 (从你的第一个脚本引入)
from scipy.signal import savgol_filter
import pandas as pd
from sklearn.mixture import GaussianMixture
import csv
import warnings
import sys # 用于 sys.exit (从你的第一个脚本引入)
import math # 用于角度计算中的 math.degrees (从你的第一个脚本引入)

# Matplotlib settings & Warnings
rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'sans-serif']
rcParams['axes.unicode_minus'] = False
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')
warnings.filterwarnings("ignore", message="Graph is not weighted")
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ==================================
#  == 配置区域 (部分默认值) ==
# ==================================
TARGET_NAME_ORIENTATION = "身体朝向 (2D)"
OUTPUT_CSV_TEMPLATE_ORIENTATION = '{}_events_orientation2d_thresh{:.1f}.csv' # 更新CSV模板
ANALYSIS_FOLDER_TEMPLATE_ORIENTATION = '{}_orientation2d_analysis'

DEFAULT_FPS = 30.0
DEFAULT_MIN_KEYPOINT_CONFIDENCE = 0.3
DEFAULT_SMOOTH_WINDOW = 11
DEFAULT_SMOOTH_POLY = 3
DEFAULT_ROTATION_THRESHOLD = 5.0 # 默认的帧间角度变化阈值
DEFAULT_MIN_DURATION_SEC_EVENT = 0.1
DEFAULT_MAX_DURATION_SEC_EVENT = 2.0
DEFAULT_MERGE_GAP_SEC_EVENT = 0.3


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
#  == 核心函数 (2D 朝向分析) ==
# ==================================

def load_mmpose_json_2d(json_path, left_shoulder_idx, right_shoulder_idx, min_confidence=0.3):
    """
    从 mmpose JSON 加载和处理 2D 肩部关键点数据。
    """
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"文件未找到: {json_path}")

    try:
        with json_path.open('r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"无法解析 JSON 文件: {json_path} - {e}")
    except Exception as e:
        raise IOError(f"读取文件时出错: {json_path} - {e}")

    num_frames = len(data)
    if num_frames == 0:
        warnings.warn(f"JSON 文件 '{json_path.name}' 为空或格式不正确。")
        return np.full((0, 4), np.nan), np.zeros(0, dtype=bool), 0

    frames_coords = np.full((num_frames, 4), np.nan)
    valid_flags = np.zeros(num_frames, dtype=bool)
    processed_frames_count = 0

    for i, frame_info in enumerate(data):
        if not isinstance(frame_info, dict) or 'instances' not in frame_info:
            # warnings.warn(f"警告: 第 {i} 帧数据格式不正确或无实例。") # 可能过于冗余
            continue
        
        instances = frame_info.get("instances", [])
        if not isinstance(instances, list) or not instances:
            # warnings.warn(f"警告: 第 {i} 帧实例列表为空。")
            continue
        
        inst = instances[0] # 只取第一个检测到的人
        if not isinstance(inst, dict):
            # warnings.warn(f"警告: 第 {i} 帧的第一个实例格式不正确。")
            continue

        keypoints_raw = inst.get("keypoints")
        scores_raw = inst.get("keypoint_scores")

        if keypoints_raw is None:
            # warnings.warn(f"警告: 第 {i} 帧缺少 'keypoints'。")
            continue
        
        try:
            keypoints = np.array(keypoints_raw)
            if scores_raw is not None:
                scores = np.array(scores_raw)
            else: # 如果没有分数信息，我们无法进行置信度检查
                warnings.warn(f"警告: 第 {i} 帧缺少 'keypoint_scores'。假设关键点有效（置信度1.0）。")
                scores = np.ones(len(keypoints) if hasattr(keypoints, '__len__') else 0)
        except Exception:
            # warnings.warn(f"警告: 第 {i} 帧关键点或分数无法转换为Numpy数组。")
            continue
            
        if keypoints.size == 0: continue

        if keypoints.ndim == 1 and keypoints.size % 2 == 0 and keypoints.shape[0] > 0 :
            try: keypoints = keypoints.reshape(-1, 2)
            except ValueError: continue
        elif keypoints.ndim != 2 or keypoints.shape[1] < 2: # 至少需要x,y
             # warnings.warn(f"警告: 第 {i} 帧关键点格式不正确（维度/列数）。")
             continue
        
        num_kps_available = keypoints.shape[0]
        if num_kps_available == 0: continue

        processed_frames_count += 1

        if not (0 <= left_shoulder_idx < num_kps_available and \
                0 <= right_shoulder_idx < num_kps_available and \
                0 <= left_shoulder_idx < len(scores) and \
                0 <= right_shoulder_idx < len(scores)):
            # warnings.warn(f"警告: 第 {i} 帧的关键点/分数数量 ({num_kps_available}, {len(scores)}) 不足以获取索引...")
            continue

        L_coord_2d = keypoints[left_shoulder_idx, :2]
        R_coord_2d = keypoints[right_shoulder_idx, :2]
        L_score = scores[left_shoulder_idx]
        R_score = scores[right_shoulder_idx]

        # 只有当坐标有效时才填充
        if not (np.any(np.isnan(L_coord_2d)) or np.any(np.isnan(R_coord_2d))):
            frames_coords[i] = [L_coord_2d[0], L_coord_2d[1], R_coord_2d[0], R_coord_2d[1]]
            if L_score >= min_confidence and R_score >= min_confidence:
                valid_flags[i] = True
    
    if processed_frames_count == 0:
        warnings.warn(f"文件 '{json_path.name}': 未找到包含有效实例数据的帧。")

    return frames_coords, valid_flags, num_frames

def interpolate_invalid_frames_2d(frames_coords, valid_flags):
    interpolated_frames = frames_coords.copy()
    if np.all(valid_flags) or not np.any(valid_flags): # 全有效或全无效
        if not np.any(valid_flags) and frames_coords.shape[0]>0 : warnings.warn("所有帧数据均无效，无法插值。")
        return interpolated_frames

    # 将无效帧的坐标明确设置为 NaN 以便 Pandas 插值
    # 原始加载时，无效置信度的帧其 valid_flags[i] 为 False，但 frames_coords[i] 可能已有数据或NaN
    # 我们需要确保所有 valid_flags 为 False 的行的 frames_coords 都被视为 NaN 进行插值
    # 并且，如果 frames_coords 本身就有 NaN (例如原始数据缺失)，也应该被插值
    
    # 创建一个布尔掩码，标记需要被视为 NaN 的位置
    # 1. valid_flags 为 False 的行
    # 2. frames_coords 本身就是 NaN 的位置
    nan_mask = ~valid_flags # (num_frames,)
    # 扩展到 (num_frames, 4) 以便对所有坐标应用
    nan_mask_expanded = np.repeat(nan_mask[:, np.newaxis], frames_coords.shape[1], axis=1)
    
    # 将这些位置的数据设置为 NaN
    interpolated_frames_for_pd = interpolated_frames.copy()
    interpolated_frames_for_pd[nan_mask_expanded] = np.nan
    # 同时，如果原始数据中就有NaN，也应该被插值
    interpolated_frames_for_pd[np.isnan(frames_coords)] = np.nan


    df = pd.DataFrame(interpolated_frames_for_pd)
    # limit_area='inside' 仅插值被有效值包围的NaN，对开头结尾的大段NaN无效
    # limit=10 最多连续插值10个NaN
    df_interpolated = df.interpolate(method='linear', axis=0, limit_direction='both', limit_area='inside', limit=15)

    # 检查开头/结尾的NaN，并用最近的有效值填充
    if df_interpolated.isnull().values.any():
        df_interpolated = df_interpolated.fillna(method='ffill').fillna(method='bfill')

    final_result = df_interpolated.to_numpy()
    
    # 统计插值情况
    num_initially_invalid_or_nan = np.sum(np.isnan(interpolated_frames_for_pd))
    num_remaining_nan = np.sum(np.isnan(final_result))
    num_interpolated_values = num_initially_invalid_or_nan - num_remaining_nan
    
    if num_interpolated_values > 0:
        print(f"对约 {num_interpolated_values // frames_coords.shape[1]} 帧中的缺失/无效数据进行了插值/填充。")
    if num_remaining_nan > 0:
        warnings.warn(f"插值后仍有 {num_remaining_nan // frames_coords.shape[1]} 帧数据未能完全填充（可能因数据完全缺失）。")
        
    return final_result


def compute_shoulder_vectors_2d(frames_coords_2d):
    L_xy = frames_coords_2d[:, 0:2]
    R_xy = frames_coords_2d[:, 2:4]
    return R_xy - L_xy

def compute_orientation_change_angles_2d(shoulder_vectors_2d):
    num_vectors = len(shoulder_vectors_2d)
    if num_vectors < 2: return np.array([])
    orientation_changes = np.full(num_vectors - 1, np.nan)
    for i in range(num_vectors - 1):
        v1, v2 = shoulder_vectors_2d[i], shoulder_vectors_2d[i+1]
        if np.any(np.isnan(v1)) or np.any(np.isnan(v2)): continue
        norm_v1, norm_v2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if norm_v1 < 1e-9 or norm_v2 < 1e-9:
            orientation_changes[i] = 0.0
            continue
        dot_product = np.dot(v1, v2)
        cos_val = np.clip(dot_product / (norm_v1 * norm_v2), -1.0, 1.0)
        angle_deg = math.degrees(np.arccos(cos_val))
        cross_product_z = v1[0] * v2[1] - v1[1] * v2[0]
        orientation_changes[i] = angle_deg if cross_product_z >= 0 else -angle_deg
    return orientation_changes

def compute_shoulder_projection_length_changes_2d(shoulder_vectors_2d):
    num_vectors = len(shoulder_vectors_2d)
    if num_vectors < 2: return np.array([])
    lengths = np.full(num_vectors, np.nan)
    for i, vec in enumerate(shoulder_vectors_2d):
        if not np.any(np.isnan(vec)): lengths[i] = np.linalg.norm(vec)
    return np.diff(lengths)


def compute_body_orientation_2d(json_filepath, left_shoulder_idx, right_shoulder_idx,
                               min_confidence, smooth, smooth_window, smooth_poly):
    raw_frames_coords, valid_flags, total_frames = load_mmpose_json_2d(
        json_filepath, left_shoulder_idx, right_shoulder_idx, min_confidence
    )
    if total_frames == 0: return np.array([]), np.array([]), 0
    
    interpolated_frames_coords = interpolate_invalid_frames_2d(raw_frames_coords, valid_flags)
    if np.all(np.isnan(interpolated_frames_coords)):
        warnings.warn("插值后所有坐标数据仍为NaN，无法继续。"); return np.array([]), np.array([]), total_frames

    shoulder_vecs_2d = compute_shoulder_vectors_2d(interpolated_frames_coords)
    if np.all(np.isnan(shoulder_vecs_2d)):
        warnings.warn("计算的肩部向量全为NaN。"); return np.array([]), np.array([]), total_frames

    orientation_angles = compute_orientation_change_angles_2d(shoulder_vecs_2d)
    length_changes = compute_shoulder_projection_length_changes_2d(shoulder_vecs_2d)

    orientation_angles = np.nan_to_num(orientation_angles, nan=0.0)
    length_changes = np.nan_to_num(length_changes, nan=0.0)

    if smooth:
        if len(orientation_angles) >= smooth_window :
            orientation_angles = savgol_filter(orientation_angles, smooth_window, smooth_poly)
        # else: warnings.warn("朝向角数据不足以平滑。")
        if len(length_changes) >= smooth_window :
            length_changes = savgol_filter(length_changes, smooth_window, smooth_poly)
        # else: warnings.warn("长度变化数据不足以平滑。")
            
    return orientation_angles, length_changes, total_frames


def recommend_rotation_threshold(rot_angles, default_thresh, method='gmm', min_thresh_val=0.5):
    abs_angles = np.abs(np.asarray(rot_angles)[~np.isnan(rot_angles)]) # 使用有效值
    analysis_data = {}
    if len(abs_angles) < 10:
        warnings.warn(f"旋转角度有效数据点过少 ({len(abs_angles)} < 10)，使用默认阈值 {default_thresh:.1f}°。")
        return default_thresh, {'fallback': {'threshold_definite': default_thresh}}

    non_zero_abs_angles = abs_angles[abs_angles > 0.1] # 忽略非常小的变化
    if len(non_zero_abs_angles) < 20:
        warnings.warn(f"有效非零角度(>0.1°)数据不足 (<20)，回退到百分位法。")
        if len(non_zero_abs_angles) < 2:
             warnings.warn(f"有效非零角度(>0.1°)数据极少 (<2)，使用默认阈值 {default_thresh:.1f}°。")
             return default_thresh, {'fallback': {'threshold_definite': default_thresh}}
        method = 'percentile'

    rec_thresh = default_thresh
    final_method_used = 'default'

    if method == 'gmm':
        try:
            X = non_zero_abs_angles.reshape(-1, 1)
            # 调整GMM参数以适应可能的单峰或噪声数据
            gmm = GaussianMixture(n_components=2, random_state=42, covariance_type='spherical',
                                  n_init=5, reg_covar=1e-5, init_params='kmeans')
            gmm.fit(X)
            if not gmm.converged_:
                 warnings.warn("GMM 未收敛，回退百分位法。")
                 method = 'percentile'
            else:
                means = gmm.means_.flatten()
                # weights = gmm.weights_.flatten() # 权重可能不那么重要
                if gmm.covariance_type == 'spherical': stds = np.sqrt(np.maximum(gmm.covariances_, 1e-6))
                else: stds = np.sqrt(np.array([c[0,0] for c in gmm.covariances_])) # for diagonal

                if len(means) == 2 and len(stds) == 2:
                    # 假设一个低均值（噪声）和一个高均值（真实旋转）分量
                    low_mean_idx = np.argmin(means)
                    high_mean_idx = 1 - low_mean_idx
                    
                    noise_mean, noise_std = means[low_mean_idx], stds[low_mean_idx]
                    signal_mean, signal_std = means[high_mean_idx], stds[high_mean_idx]

                    # 阈值可以设在噪声分布的尾部，或者两个分布的交点附近
                    # 一个简单的策略：噪声均值 + k * 噪声标准差
                    gmm_based_thresh = noise_mean + 2.0 * noise_std 
                    # 或者，如果两个峰分离得很好，可以取它们之间的一个点
                    # if signal_mean > noise_mean + 2*noise_std : # 确保有足够的分离
                    #    gmm_based_thresh = (noise_mean + signal_mean) / 2
                    
                    rec_thresh = max(min_thresh_val, gmm_based_thresh)
                    analysis_data['gmm'] = {'threshold_definite': rec_thresh, 'noise_mean': noise_mean, 'noise_std': noise_std, 'signal_mean': signal_mean}
                    final_method_used = 'gmm'
                else: # GMM结果不符合预期
                    warnings.warn("GMM未能解析出两个有效分量，回退百分位法。")
                    method = 'percentile'
        except Exception as e:
            warnings.warn(f"GMM 分析失败: {e}。回退百分位法。")
            method = 'percentile'

    if method == 'percentile':
        if len(non_zero_abs_angles) >= 2:
            p_val = 90 # 可以尝试调整百分位值，如85, 90, 95
            percentile_based_thresh = np.percentile(non_zero_abs_angles, p_val)
            rec_thresh = max(min_thresh_val, percentile_based_thresh)
            analysis_data['percentile'] = {'threshold_definite': rec_thresh, f'p{p_val}': percentile_based_thresh}
            final_method_used = 'percentile'
        else: # 连百分位法都数据不足
            rec_thresh = default_thresh
            analysis_data['fallback'] = {'threshold_definite': rec_thresh}
            final_method_used = 'fallback_default'
            warnings.warn(f"百分位法数据不足，使用默认阈值 {rec_thresh:.1f}°。")

    if not np.isfinite(rec_thresh) or rec_thresh <= 0:
         rec_thresh = default_thresh; final_method_used = 'default_fallback_invalid'
         warnings.warn(f"推荐阈值无效 ({rec_thresh})，使用默认值 {default_thresh:.1f}°。")
         analysis_data['fallback'] = {'threshold_definite': default_thresh}

    analysis_data['final_recommendation'] = {'threshold': rec_thresh, 'method_used': final_method_used}
    return rec_thresh, analysis_data


def plot_threshold_analysis(rot_angles, analysis_data, target_name_plot="旋转"):
    fig = plt.figure(figsize=(12, 7))
    abs_angles = np.abs(np.asarray(rot_angles)[~np.isnan(rot_angles)])
    if len(abs_angles) == 0:
        plt.title(f"{target_name_plot}: 无有效数据用于阈值分析"); return fig

    plot_data = abs_angles[abs_angles > 0.01] # 忽略非常小的抖动以改善绘图
    if len(plot_data) == 0:
        plt.title(f"{target_name_plot}: 无有效非零角度(>0.01°)数据用于阈值分析"); return fig
    
    try:
        counts, bins, patches = plt.hist(plot_data, bins=60, density=True, alpha=0.75, color='mediumpurple', label=f'{target_name_plot}绝对值分布')
    except Exception as e:
        plt.title(f"{target_name_plot}: 绘制直方图失败 - {e}"); return fig

    rec_info = analysis_data.get('final_recommendation', {})
    rec_thresh_val = rec_info.get('threshold')
    rec_method_str = rec_info.get('method_used', '?').upper()

    if rec_thresh_val is not None and np.isfinite(rec_thresh_val):
        plt.axvline(rec_thresh_val, color='r', linestyle='--', linewidth=2, label=f"最终推荐阈值 ({rec_method_str}): {rec_thresh_val:.2f}°")
        try:
            y_max_plot = plt.gca().get_ylim()[1]
            plt.text(rec_thresh_val * 1.02, y_max_plot * 0.9, f"{rec_thresh_val:.2f}", color='r', weight='bold', fontsize=10)
        except Exception: pass # 忽略文本放置错误

    # 可选：绘制GMM分量 (如果GMM成功)
    if 'gmm' in analysis_data and analysis_data.get('final_recommendation',{}).get('method_used') == 'gmm':
        gmm_plot_data = analysis_data['gmm']
        # ... (这里可以添加GMM分量绘制逻辑，如果需要) ...

    plt.title(f"{target_name_plot} 绝对值分布与阈值分析")
    plt.xlabel(f"{target_name_plot} 角度变化绝对值 (°/帧)")
    plt.ylabel("概率密度")
    plt.legend(fontsize='small'); plt.grid(True, alpha=0.3); plt.xlim(left=-0.5)
    xmax_data_plot = np.max(plot_data) if len(plot_data) > 0 else 30
    xmax_thresh_plot = rec_thresh_val if rec_thresh_val is not None and np.isfinite(rec_thresh_val) else 0
    plt.xlim(right=max(xmax_data_plot * 1.1, xmax_thresh_plot * 1.2, 10)) # 动态调整X轴上限
    plt.tight_layout()
    return fig


def detect_orientation_change_events(orientation_angles, threshold, min_duration_frames, max_duration_frames,
                                     min_gap_frames, fps):
    if len(orientation_angles) == 0: return []
    significant_any = np.abs(orientation_angles) >= threshold
    significant_indices = np.where(significant_any)[0]
    if len(significant_indices) == 0: return []

    groups = []
    current_group = [significant_indices[0]]
    for i in range(1, len(significant_indices)):
        if significant_indices[i] - significant_indices[i-1] <= min_gap_frames + 1:
            current_group.append(significant_indices[i])
        else:
            if current_group: groups.append(current_group)
            current_group = [significant_indices[i]]
    if current_group: groups.append(current_group)

    events = []
    for group in groups:
        start_idx_angle_seq = group[0]
        end_idx_angle_seq = group[-1]
        duration_angle_seq = (end_idx_angle_seq - start_idx_angle_seq) + 1

        if not (min_duration_frames <= duration_angle_seq <= max_duration_frames):
            continue

        event_angles_segment = orientation_angles[start_idx_angle_seq : end_idx_angle_seq + 1]
        angle_sum_segment = np.sum(event_angles_segment)

        event_start_frame = start_idx_angle_seq # 视频帧号，从0开始
        event_end_frame = end_idx_angle_seq + 1 # 视频帧号（不含）或最后一个变化发生的后一帧

        start_time_sec = event_start_frame / fps
        end_time_sec = event_end_frame / fps
        duration_sec_event = (event_end_frame - event_start_frame) / fps # 持续的视频帧数 / fps

        events.append({
            'event_label': "转动",
            'start_frame': event_start_frame,
            'end_frame': event_end_frame,
            'start_time_sec': start_time_sec,
            'end_time_sec': end_time_sec,
            'duration_sec': duration_sec_event,
            'angle_sum_abs': abs(angle_sum_segment),
            # 'internal_direction_debug': "顺时针" if angle_sum_segment > 1e-6 else ("逆时针" if angle_sum_segment < -1e-6 else "混合/微小")
        })
    return events

def export_events_to_csv(events, output_path):
    if not events: print("没有检测到事件，不生成 CSV 文件。"); return None
    try:
        fieldnames = ['event_label', 'start_frame', 'end_frame', 'start_time_sec',
                      'end_time_sec', 'duration_sec', 'angle_sum_abs', 
                      'start_time_str', 'end_time_str'] # 添加时间字符串
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open('w', newline='', encoding='utf-8-sig') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            for event_dict in events:
                row = event_dict.copy()
                # 确保时间戳和数值格式化
                row['start_time_sec'] = f"{event_dict.get('start_time_sec', 0):.3f}"
                row['end_time_sec'] = f"{event_dict.get('end_time_sec', 0):.3f}"
                row['duration_sec'] = f"{event_dict.get('duration_sec', 0):.3f}"
                row['angle_sum_abs'] = f"{event_dict.get('angle_sum_abs', 0):.2f}"
                # 添加时间字符串 (假设 fps 在外部作用域可用或传入)
                # 为了简单，这里假设 fps 是一个已知值，或者从 event_dict 获取
                # 如果 fps 不是全局或易于访问，这个特性可能需要调整
                # current_fps = event_dict.get('fps_for_str_time', DEFAULT_FPS) # 假设事件字典里有FPS
                current_fps = DEFAULT_FPS # 使用全局默认值，如果interactive函数中设定了不同的，那这里可能不准
                                          # 更好的做法是在事件检测时就把fps传入并存入event_dict
                row['start_time_str'] = frame_to_time_str(event_dict['start_frame'], current_fps)
                row['end_time_str'] = frame_to_time_str(event_dict['end_frame']-1, current_fps) # end_frame 是不包含的帧，所以用-1

                writer.writerow(row)
        print(f"事件已成功导出到: {output_path}")
        return output_path
    except IOError as e: warnings.warn(f"CSV 文件写入权限错误: {output_path} - {e}"); return None
    except Exception as e: warnings.warn(f"CSV 导出失败: {e}"); return None


def plot_orientation_analysis_2d(orientation_changes, length_changes, fps_param,
                                 events_list=None, threshold_val=None,
                                 cumulative=True, original=True, title_prefix=""):
    num_series = 1
    if length_changes is not None and len(length_changes) > 0 : num_series +=1

    num_cols_per_series = (1 if original else 0) + (1 if cumulative else 0)
    if num_cols_per_series == 0: return None

    fig, axes = plt.subplots(num_series, num_cols_per_series, 
                             figsize=(7 * num_cols_per_series, 4 * num_series), 
                             squeeze=False, sharex=True if num_cols_per_series > 0 else False) # 共享X轴
    
    full_title = f"{title_prefix.strip()} {TARGET_NAME_ORIENTATION} 分析".strip()
    fig.suptitle(full_title, fontsize=16, y=0.98 if num_series > 1 else 0.95)


    time_axis_frames = np.arange(len(orientation_changes)) # 帧索引 (对应角度变化)
    time_axis_func = lambda f: f / fps_param if isinstance(fps_param, (int, float)) and fps_param > 0 else np.nan


    # --- 2D 朝向变化 ---
    current_row = 0
    plot_col_idx_orient = 0
    ax_orient_orig = None
    if original:
        ax_orient_orig = axes[current_row, plot_col_idx_orient]
        ax_orient_orig.plot(time_axis_frames, orientation_changes, label='每帧朝向角变化', linewidth=1.2, color='dodgerblue')
        ax_orient_orig.set_title("瞬时朝向角变化 (2D)")
        ax_orient_orig.set_ylabel("角度变化 (°/帧)")
        ax_orient_orig.grid(True, alpha=0.4)
        if threshold_val is not None:
            ax_orient_orig.axhline(threshold_val, color='r', ls='--', lw=1, label=f'阈值 ±{threshold_val:.1f}°')
            ax_orient_orig.axhline(-threshold_val, color='r', ls='--', lw=1)
        if events_list:
            event_label_added_o = False
            for evt in events_list:
                lbl = "检测到的'转动'事件" if not event_label_added_o else "_nolegend_"
                ax_orient_orig.axvspan(evt['start_frame'], evt['end_frame'], alpha=0.25, color='gold', label=lbl, zorder=-1)
                event_label_added_o = True
        ax_orient_orig.legend(fontsize='small')

    if cumulative:
        plot_col_idx_orient = 1 if original else 0
        ax_orient_cum = axes[current_row, plot_col_idx_orient]
        cum_orientation = np.cumsum(orientation_changes)
        ax_orient_cum.plot(time_axis_frames, cum_orientation, label='累计朝向角变化', color='darkorchid', linewidth=1.5)
        ax_orient_cum.set_title("累计朝向角变化 (2D)")
        ax_orient_cum.set_ylabel("累计角度 (°)")
        ax_orient_cum.grid(True, alpha=0.4)
        if events_list: # 也可在累计图上标记事件
            for evt in events_list:
                ax_orient_cum.axvspan(evt['start_frame'], evt['end_frame'], alpha=0.2, color='palegoldenrod', zorder=-1)
        ax_orient_cum.legend(fontsize='small')

    # --- 肩部投影长度变化 (可选) ---
    if length_changes is not None and len(length_changes) > 0:
        current_row +=1
        plot_col_idx_len = 0
        if original:
            ax_len_orig = axes[current_row, plot_col_idx_len]
            ax_len_orig.plot(time_axis_frames, length_changes, label='每帧肩部投影长度变化', color='seagreen', linewidth=1.2)
            ax_len_orig.set_title("瞬时肩部投影长度变化")
            ax_len_orig.set_ylabel("长度变化 (像素/帧)")
            ax_len_orig.grid(True, alpha=0.4)
            ax_len_orig.legend(fontsize='small')
        if cumulative:
            plot_col_idx_len = 1 if original else 0
            ax_len_cum = axes[current_row, plot_col_idx_len]
            cum_length = np.cumsum(length_changes)
            ax_len_cum.plot(time_axis_frames, cum_length, label='累计肩部投影长度变化', color='sienna', linewidth=1.5)
            ax_len_cum.set_title("累计肩部投影长度变化")
            ax_len_cum.set_ylabel("累计长度变化 (像素)")
            ax_len_cum.grid(True, alpha=0.4)
            ax_len_cum.legend(fontsize='small')

    # 设置共享X轴标签和时间格式化 (作用于最底部的图)
    bottom_ax = axes[-1, 0] # 最底部的左边图
    bottom_ax.set_xlabel("帧 / 时间")
    if isinstance(fps_param, (int, float)) and fps_param > 0:
        time_fmt = FuncFormatter(lambda x, pos: time_formatter(x, pos, fps_param))
        bottom_ax.xaxis.set_major_formatter(time_fmt)
        try: # 添加顶部时间轴 (只在第一个原始图上，如果存在)
            if ax_orient_orig :
                 secax_top = ax_orient_orig.secondary_xaxis('top', functions=(time_axis_func, lambda t: t * fps_param))
                 secax_top.set_xlabel('时间 (秒)')
        except Exception: pass # 忽略次轴错误

    plt.tight_layout(rect=[0, 0.03, 1, 0.94])
    return fig


# ==================================
#  == 主执行区域 (交互式) ==
# ==================================
def interactive_orientation_detection_2d(json_filepath_str, output_dir_str=None, fps_val=DEFAULT_FPS,
                                 left_shoulder_idx_val=5, right_shoulder_idx_val=6,
                                 min_confidence_val=DEFAULT_MIN_KEYPOINT_CONFIDENCE,
                                 smooth_window_val=DEFAULT_SMOOTH_WINDOW,
                                 smooth_poly_val=DEFAULT_SMOOTH_POLY,
                                 default_event_thresh=DEFAULT_ROTATION_THRESHOLD,
                                 min_event_dur_sec=DEFAULT_MIN_DURATION_SEC_EVENT,
                                 max_event_dur_sec=DEFAULT_MAX_DURATION_SEC_EVENT,
                                 merge_event_gap_sec=DEFAULT_MERGE_GAP_SEC_EVENT):
    
    json_filepath = Path(json_filepath_str)
    if not json_filepath.is_file():
        print(f"错误: 输入 JSON 文件未找到或无效: {json_filepath}")
        return [], default_event_thresh, (min_event_dur_sec, max_event_dur_sec, merge_event_gap_sec)

    if output_dir_str is None:
        output_dir_path = json_filepath.parent / ANALYSIS_FOLDER_TEMPLATE_ORIENTATION.format(json_filepath.stem)
    else:
        output_dir_path = Path(output_dir_str)
    
    try:
        output_dir_path.mkdir(parents=True, exist_ok=True)
        print(f"结果将保存到: {output_dir_path}")
    except Exception as e:
        print(f"创建输出目录失败 ({e})，将尝试在脚本同级目录保存。")
        output_dir_path = Path(".") # 回退到当前目录

    print(f"\n--- 开始处理文件 (2D {TARGET_NAME_ORIENTATION} 分析) ---")
    print(f"文件: {json_filepath.name}")
    print(f"参数: FPS={fps_val:.1f}, MinConf={min_confidence_val:.2f}, SmoothWin={smooth_window_val}, SmoothPoly={smooth_poly_val}")

    # 1. 计算核心数据
    try:
        orientation_changes, shoulder_len_changes, total_frames = compute_body_orientation_2d(
            json_filepath, left_shoulder_idx_val, right_shoulder_idx_val,
            min_confidence=min_confidence_val, smooth=True,
            smooth_window=smooth_window_val, smooth_poly=smooth_poly_val
        )
        if total_frames == 0:
            print("错误：文件不包含任何帧数据。")
            return [], default_event_thresh, (min_event_dur_sec, max_event_dur_sec, merge_event_gap_sec)
        if len(orientation_changes) == 0:
            print("警告：未能计算出任何有效的朝向变化角度数据。可能是由于所有帧的关键点均无效或数据不足。")
            # 允许继续，但后续步骤可能无法正常工作
    except FileNotFoundError: # load_mmpose_json_2d 内部已处理
        print(f"错误: 文件读取问题，请检查路径 {json_filepath}") # 不应到这里，但以防万一
        return [], default_event_thresh, (min_event_dur_sec, max_event_dur_sec, merge_event_gap_sec)
    except Exception as e:
        print(f"计算2D朝向数据时发生严重错误: {e}")
        import traceback; traceback.print_exc()
        return [], default_event_thresh, (min_event_dur_sec, max_event_dur_sec, merge_event_gap_sec)

    # 如果没有角度数据，很多后续步骤无意义
    if len(orientation_changes) == 0:
        print("由于缺乏有效的角度数据，无法进行事件检测和阈值推荐。")
        return [], default_event_thresh, (min_event_dur_sec, max_event_dur_sec, merge_event_gap_sec)
        
    print(f"数据加载与预处理完成。总帧数: {total_frames}, 有效角度变化数据点: {len(orientation_changes)}")

    # 2. 阈值推荐与确认
    plt_threshold_fig = None # 初始化图对象
    print("\n--- 推荐事件检测阈值 ---")
    try:
        rec_thresh, thresh_analysis_data = recommend_rotation_threshold(orientation_changes, default_thresh=default_event_thresh)
        print(f"推荐方法: {thresh_analysis_data.get('final_recommendation', {}).get('method_used', '未知')}")
        
        plot_choice_thresh = input("是否显示阈值分析图? (Y/n): ").strip().lower()
        if plot_choice_thresh != 'n':
            plt_threshold_fig = plot_threshold_analysis(orientation_changes, thresh_analysis_data, target_name_plot=TARGET_NAME_ORIENTATION)
            plt_threshold_fig.show()
            # plt_threshold_fig.show(block=False) # 非阻塞显示
            print("阈值分析图已显示 (非阻塞)。")
        else:
            print("已跳过显示阈值分析图。")

    except Exception as e:
        print(f"错误：阈值推荐或绘图时出错: {e}")
        rec_thresh = default_event_thresh # 回退到默认值

    final_event_threshold = default_event_thresh
    try:
        thresh_input_str = input(f"推荐阈值: ≈ {rec_thresh:.2f}°/帧. 输入最终使用的阈值 (回车使用推荐值): ").strip()
        if thresh_input_str: final_event_threshold = float(thresh_input_str)
        else: final_event_threshold = rec_thresh
        if final_event_threshold <= 0: raise ValueError("阈值必须为正数")
    except ValueError as e:
        print(f"输入错误 ({e})，使用推荐/默认阈值 {rec_thresh:.2f}")
        final_event_threshold = max(0.1, rec_thresh) # 确保至少是一个很小正数
    print(f"最终使用事件检测阈值: {final_event_threshold:.2f}°/帧")


    # 3. 获取事件时间约束
    print("\n--- 设置事件时间约束 (单位: 秒) ---")
    final_min_dur_s = min_event_dur_sec
    final_max_dur_s = max_event_dur_sec
    final_merge_gap_s = merge_event_gap_sec
    try:
        min_dur_input = input(f"  最短持续时间 (默认 {min_event_dur_sec:.2f}s): ").strip()
        if min_dur_input: final_min_dur_s = float(min_dur_input)
        max_dur_input = input(f"  最长持续时间 (默认 {max_event_dur_sec:.1f}s, 0 无限制): ").strip()
        if max_dur_input: final_max_dur_s = float(max_dur_input)
        merge_gap_input = input(f"  合并间隔 (默认 {merge_event_gap_sec:.2f}s): ").strip()
        if merge_gap_input: final_merge_gap_s = float(merge_gap_input)
        if final_min_dur_s < 0 or final_max_dur_s < 0 or final_merge_gap_s < 0: raise ValueError("时间不能为负数")
    except ValueError as e:
        print(f"输入错误 ({e})，使用默认时间约束。")
        final_min_dur_s, final_max_dur_s, final_merge_gap_s = min_event_dur_sec, max_event_dur_sec, merge_event_gap_sec

    min_dframes = max(1, int(final_min_dur_s * fps_val)) if fps_val > 0 else 1
    max_dframes = int(final_max_dur_s * fps_val) if final_max_dur_s > 0 and fps_val > 0 else (total_frames or 99999) # 如果无限制，则设为总帧数
    merge_gframes = int(final_merge_gap_s * fps_val) if fps_val > 0 else 0
    print(f"事件约束 (帧): MinDur={min_dframes}, MaxDur={max_dframes if final_max_dur_s > 0 else '无限制'}, MergeGap={merge_gframes}")

    # 4. 事件检测
    print(f"\n--- 使用阈值 {final_event_threshold:.2f}°/帧 检测事件 ---")
    detected_events_list = detect_orientation_change_events(
        orientation_changes, final_event_threshold,
        min_duration_frames=min_dframes, max_duration_frames=max_dframes,
        min_gap_frames=merge_gframes, fps=fps_val
    )

    # 5. 显示与导出结果
    if detected_events_list:
        print(f"\n检测到 {len(detected_events_list)} 个 '{detected_events_list[0]['event_label']}' 事件:")
        # for i, evt in enumerate(detected_events_list[:5]): # 最多显示前5个
        #     print(f"  事件 {i+1}: 帧 {evt['start_frame']}-{evt['end_frame']-1}, "
        #           f"持续 {evt['duration_sec']:.3f}s, 幅度 {evt['angle_sum_abs']:.1f}°")
        # if len(detected_events_list) > 5: print("  ...")
            
        csv_filename = OUTPUT_CSV_TEMPLATE_ORIENTATION.format(json_filepath.stem, final_event_threshold)
        csv_filepath = output_dir_path / csv_filename
        export_path = export_events_to_csv(detected_events_list, csv_filepath)
        # if export_path: print(f"事件已导出到: {export_path}") # export_events_to_csv 内部已有打印
    else:
        print(f"\n未检测到符合条件的 '{TARGET_NAME_ORIENTATION}' 事件。")

    # 6. 最终绘图
    plt_analysis_fig = None
    plot_choice_final = input("\n是否生成并显示最终分析图? (Y/n): ").strip().lower()
    if plot_choice_final != 'n':
        print("生成最终分析图...")
        try:
            plt_analysis_fig = plot_orientation_analysis_2d(
                orientation_changes, shoulder_len_changes, fps_val,
                events_list=detected_events_list, threshold_val=final_event_threshold,
                title_prefix=f"{json_filepath.stem}\n"
            )
            if plt_analysis_fig:
                plot_final_path = output_dir_path / f"{json_filepath.stem}_orientation_analysis_final.png"
                try:
                    plt_analysis_fig.savefig(plot_final_path, dpi=120)
                    print(f"最终分析图已保存到: {plot_final_path}")
                except Exception as save_err:
                    print(f"警告：保存最终分析图失败: {save_err}")
                print("提示：关闭所有图形窗口后程序将结束。")
                plt_threshold_fig.show()
                # plt_analysis_fig.show(block=True) # 阻塞显示，直到用户关闭
            else:
                print("未能生成最终分析图。")
        except Exception as e:
            print(f"错误：绘制最终分析图时出错: {e}")
            if plt_analysis_fig: plt.close(plt_analysis_fig) # 清理可能已创建的图
    else:
        print("已跳过生成最终分析图。")

    # 清理阈值分析图 (如果之前是非阻塞显示的)
    if plt_threshold_fig:
        try: plt.close(plt_threshold_fig.gcf())
        except: pass
    
    print(f"\n--- 分析流程结束 ({json_filepath.name}) ---")
    return detected_events_list, final_event_threshold, (final_min_dur_s, final_max_dur_s, final_merge_gap_s)


if __name__ == "__main__":
    print(f"欢迎使用 2D {TARGET_NAME_ORIENTATION} 分析工具！")
    print("请按照提示输入参数。直接按 Enter 键可使用括号中的默认值。")
    print("-" * 40)

    # --- 1. 获取输入 JSON 文件路径 ---
    input_json_path_main = None
    while input_json_path_main is None:
        try:
            json_path_str_main = input(">>> 请输入 JSON 文件路径: ").strip()
            if not json_path_str_main: print("路径不能为空，请重新输入。"); continue
            temp_path_main = Path(json_path_str_main)
            if temp_path_main.is_file(): input_json_path_main = temp_path_main
            else: print(f"错误：文件不存在或路径无效: '{json_path_str_main}'")
        except (KeyboardInterrupt, EOFError): print("\n操作取消。"); sys.exit(1)
        except Exception as e_json: print(f"读取路径时出错: {e_json}")

    # --- 2. 获取输出目录 ---
    default_output_dir_main = input_json_path_main.parent / ANALYSIS_FOLDER_TEMPLATE_ORIENTATION.format(input_json_path_main.stem)
    output_dir_main = default_output_dir_main
    try:
        output_dir_str_main = input(f">>> 请输入输出目录 (默认: '{default_output_dir_main}'): ").strip()
        if output_dir_str_main: output_dir_main = Path(output_dir_str_main)
    except (KeyboardInterrupt, EOFError): print("\n操作取消。"); sys.exit(1)
    except Exception as e_outdir: # Path errors
        print(f"设置输出目录时出错 ({e_outdir})，将使用默认目录。")
        output_dir_main = default_output_dir_main
    # (目录创建移到 interactive 函数内部)

    # --- 3. 获取基础参数 (FPS, Min Confidence) ---
    fps_main = DEFAULT_FPS
    min_conf_main = DEFAULT_MIN_KEYPOINT_CONFIDENCE
    try:
        fps_str_main = input(f">>> 请输入视频帧率 FPS (默认: {DEFAULT_FPS:.1f}): ").strip()
        if fps_str_main: fps_main = float(fps_str_main)
        if fps_main <= 0: raise ValueError("FPS 必须为正数")

        conf_str_main = input(f">>> 请输入关键点最小置信度 (0-1, 默认: {DEFAULT_MIN_KEYPOINT_CONFIDENCE:.2f}): ").strip()
        if conf_str_main: min_conf_main = float(conf_str_main)
        if not (0.0 <= min_conf_main <= 1.0): raise ValueError("置信度必须在 0 和 1 之间")
    except (KeyboardInterrupt, EOFError): print("\n操作取消。"); sys.exit(1)
    except ValueError as e_param:
        print(f"输入错误 ({e_param})，将使用默认值。")
        fps_main = DEFAULT_FPS; min_conf_main = DEFAULT_MIN_KEYPOINT_CONFIDENCE
    
    # --- 4. 获取平滑参数 ---
    smooth_win_main = DEFAULT_SMOOTH_WINDOW
    smooth_poly_main = DEFAULT_SMOOTH_POLY
    try:
        win_str_main = input(f">>> 请输入平滑窗口大小 (奇数>=3, 默认: {DEFAULT_SMOOTH_WINDOW}): ").strip()
        if win_str_main: smooth_win_main = int(win_str_main)
        if smooth_win_main < 3 or smooth_win_main % 2 == 0: raise ValueError("平滑窗口必须是 >=3 的奇数")

        poly_str_main = input(f">>> 请输入平滑多项式阶数 (1 <= poly < window, 默认: {DEFAULT_SMOOTH_POLY}): ").strip()
        if poly_str_main: smooth_poly_main = int(poly_str_main)
        if not (1 <= smooth_poly_main < smooth_win_main): raise ValueError("多项式阶数必须 >=1 且小于窗口大小")
    except (KeyboardInterrupt, EOFError): print("\n操作取消。"); sys.exit(1)
    except ValueError as e_smooth:
        print(f"输入错误 ({e_smooth})，将使用默认平滑参数。")
        smooth_win_main = DEFAULT_SMOOTH_WINDOW; smooth_poly_main = DEFAULT_SMOOTH_POLY
    
    # 调用主分析函数
    # 其他参数如关键点索引、事件默认阈值等可以使用其默认值
    try:
        events_summary, threshold_summary, constraints_summary = interactive_orientation_detection_2d(
            json_filepath_str=str(input_json_path_main), #确保是字符串
            output_dir_str=str(output_dir_main), #确保是字符串
            fps_val=fps_main,
            min_confidence_val=min_conf_main,
            smooth_window_val=smooth_win_main,
            smooth_poly_val=smooth_poly_main
            # left_shoulder_idx_val, right_shoulder_idx_val 等使用函数内默认
            # default_event_thresh, min_event_dur_sec 等也使用函数内默认
        )
        
        # 打印最终总结
        print("\n--- 主流程结束 ---")
        if events_summary:
            print(f"最终使用的事件检测阈值: {threshold_summary:.2f}°/帧")
            print(f"最终时间约束 (秒): MinDur={constraints_summary[0]:.2f}, MaxDur={constraints_summary[1]:.2f}, MergeGap={constraints_summary[2]:.2f}")
            print(f"共检测到 {len(events_summary)} 个 '{events_summary[0]['event_label']}' 事件。")
        elif threshold_summary != DEFAULT_ROTATION_THRESHOLD : #即使没事件，也可能改了阈值
            print(f"最终使用的事件检测阈值为: {threshold_summary:.2f}°/帧，但未检测到事件。")
        else:
            print("未检测到符合条件的事件。")
            
    except Exception as e_main_call:
        print(f"\n在执行主分析函数时发生未预料的错误: {e_main_call}")
        import traceback
        traceback.print_exc()
    finally:
        plt.close('all') # 关闭所有可能残留的 matplotlib 窗口
        print("程序退出。")
        sys.exit(0)       