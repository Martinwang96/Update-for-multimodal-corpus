'''
Author: Martinwang96 -git
Date: 2025-04-15 18:42:22
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
from sklearn.preprocessing import StandardScaler # GMM需要
import csv
import warnings
import sys # 用于 sys.exit
import math # 用于角度计算

# --- Matplotlib 设置 ---
rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'sans-serif']
rcParams['axes.unicode_minus'] = False
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="Graph is not weighted")

# --- 辅助函数 ---
def frame_to_time_str(frame_number, fps=30):
    """将帧号转换为 MM:SS.ms 格式字符串"""
    if not isinstance(fps, (int, float)) or fps <= 0: return f"Frame {frame_number} (无效 FPS)"
    total_seconds = frame_number / fps
    minutes = int(total_seconds // 60)
    seconds = total_seconds % 60
    return f"{minutes:02d}:{seconds:06.3f}"

def time_formatter(x, pos, fps=30):
    """Matplotlib FuncFormatter for time axis"""
    return frame_to_time_str(x, fps)

# ==================================
#  == 配置区域 ==
# ==================================

# 1. 检测目标
TARGET_NAME = "耸肩 (2D相对抬升距离)"
OUTPUT_CSV_TEMPLATE = '{}_events_shrug_2d_relDist_thresh{:.1f}px.csv'
ANALYSIS_FOLDER_TEMPLATE = '{}_shrug_2d_relDist_analysis'

# 2. 定义关键点索引 (COCO 17)
DEFAULT_LEFT_SHOULDER_IDX = 5
DEFAULT_RIGHT_SHOULDER_IDX = 6
DEFAULT_LEFT_HIP_IDX = 11
DEFAULT_RIGHT_HIP_IDX = 12
DEFAULT_REQUIRED_KP_INDICES = (DEFAULT_LEFT_SHOULDER_IDX, DEFAULT_RIGHT_SHOULDER_IDX, DEFAULT_LEFT_HIP_IDX, DEFAULT_RIGHT_HIP_IDX)

# 3. 默认计算参数
DEFAULT_FPS = 30.0
DEFAULT_MIN_KEYPOINT_CONFIDENCE = 0.3
# 平滑 Y 坐标用
DEFAULT_SMOOTH_Y_WINDOW = 7
DEFAULT_SMOOTH_Y_POLY = 2
# 计算基准线用的低通滤波器窗口 (需要比较大，例如1-3秒)
DEFAULT_BASELINE_WINDOW_SEC = 1.5 # 秒
# 默认的相对抬升距离阈值 (像素) - 高度依赖图像尺度
DEFAULT_SHRUG_DISTANCE_THRESHOLD = 5.0 # 像素

# 默认事件检测参数
DEFAULT_MIN_DURATION_SEC = 0.15
DEFAULT_MAX_DURATION_SEC = 2.0
DEFAULT_MERGE_GAP_SEC = 0.3

# ==================================
#  == 核心函数 ==
# ==================================

# load_shoulder_hip_y_coordinates_2d, interpolate_single_coordinate, smooth_single_coordinate 函数不变，从上面完整代码复制

def load_shoulder_hip_y_coordinates_2d(json_path, kp_indices, min_confidence):
    """
    从 JSON 加载指定的 2D 关键点 Y 坐标和置信度。
    返回: Y坐标列表, 分数列表, 有效标志列表, 总帧数, 坐标单位
    """
    json_path = Path(json_path)
    if not json_path.exists(): raise FileNotFoundError(f"文件未找到: {json_path}")
    num_required_kps = len(kp_indices)
    if num_required_kps != 4: raise ValueError("kp_indices 必须包含4个索引 (LS, RS, LH, RH)。")

    try:
        with json_path.open('r', encoding='utf-8') as f: data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"无法解析 JSON 文件: {json_path} - {e}")
    except Exception as e:
        raise IOError(f"读取文件时出错: {json_path} - {e}")

    total_frames = len(data)
    if total_frames == 0:
        warnings.warn(f"JSON 文件 '{json_path.name}' 为空或格式不正确。")
        coords_y_list = [np.full(0, np.nan) for _ in range(num_required_kps)]
        scores_list = [np.full(0, np.nan) for _ in range(num_required_kps)]
        valid_flags_list = [np.zeros(0, dtype=bool) for _ in range(num_required_kps)]
        return coords_y_list, scores_list, valid_flags_list, 0, "像素"

    coords_y_list = [np.full(total_frames, np.nan) for _ in range(num_required_kps)]
    scores_list = [np.full(total_frames, np.nan) for _ in range(num_required_kps)]
    valid_flags_list = [np.zeros(total_frames, dtype=bool) for _ in range(num_required_kps)]
    processed_frames = 0
    coord_units = "像素"

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

        processed_frames += 1

        for list_idx, kp_idx in enumerate(kp_indices):
            if 0 <= kp_idx < num_kps_available and 0 <= kp_idx < len(scores):
                if scores[kp_idx] >= min_confidence:
                    coords_y_list[list_idx][frame_idx] = keypoints[kp_idx, 1]
                    scores_list[list_idx][frame_idx] = scores[kp_idx]
                    valid_flags_list[list_idx][frame_idx] = True

    if processed_frames == 0:
        warnings.warn(f"文件 '{json_path.name}': 未找到包含有效实例数据的帧。")

    num_valid = [np.sum(v) for v in valid_flags_list]
    print(f"有效帧数: LS={num_valid[0]}, RS={num_valid[1]}, LH={num_valid[2]}, RH={num_valid[3]}")

    return coords_y_list, scores_list, valid_flags_list, total_frames, coord_units

def interpolate_single_coordinate(coords, valid_flags, limit_gap_frames=15):
    """对单个坐标序列进行线性插值以填充 NaN 值 (基于有效性标志)。"""
    interpolated_coords = coords.copy()
    num_frames = len(coords)

    if np.all(valid_flags):
        return interpolated_coords
    if not np.any(valid_flags):
        warnings.warn("警告：所有帧数据均无效，无法进行插值！将返回原始数据。")
        return interpolated_coords

    interpolated_coords[~valid_flags] = np.nan
    s = pd.Series(interpolated_coords)
    s_interpolated = s.interpolate(method='linear', axis=0, limit=limit_gap_frames,
                                     limit_direction='both', limit_area=None)
    s_interpolated = s_interpolated.fillna(method='ffill').fillna(method='bfill')

    final_nan = s_interpolated.isnull().sum()
    if final_nan > 0:
        warnings.warn(f"警告：插值和填充后仍有 {final_nan} 个 NaN 数据点。")

    return s_interpolated.to_numpy()

def smooth_single_coordinate(coords, window_length=7, poly_order=2):
    """使用 Savitzky-Golay 滤波器平滑单个坐标数组。"""
    coords_arr = np.asarray(coords)
    num_frames = len(coords_arr)
    valid_mask = ~np.isnan(coords_arr)
    num_valid = np.sum(valid_mask)

    try:
        eff_window = max(3, int(window_length))
        if eff_window % 2 == 0: eff_window += 1
        eff_poly = min(int(poly_order), eff_window - 1); eff_poly = max(1, eff_poly)
    except (TypeError, ValueError):
        warnings.warn("平滑窗口或阶数无效，使用默认值。")
        eff_window = max(3, int(DEFAULT_SMOOTH_Y_WINDOW) | 1)
        eff_poly = min(int(DEFAULT_SMOOTH_Y_POLY), eff_window - 1); eff_poly = max(1, eff_poly)

    if num_valid < eff_window or num_frames < eff_window:
        return coords_arr

    coords_series = pd.Series(coords_arr)
    coords_filled = coords_series.interpolate(method='linear', limit_direction='both', limit_area='inside', limit=int(eff_window*2))
    coords_filled = coords_filled.fillna(method='ffill').fillna(method='bfill')
    coords_to_smooth = coords_filled.to_numpy()

    if np.any(np.isnan(coords_to_smooth)):
         warnings.warn("平滑前坐标序列仍含 NaN，返回原始数据。")
         return coords_arr

    try:
        smoothed = savgol_filter(coords_to_smooth, eff_window, eff_poly)
        smoothed[~valid_mask] = np.nan
        return smoothed
    except ValueError as ve:
        warnings.warn(f"SavGol 平滑失败 (可能参数问题: {ve})。返回插值后的数据。")
        return coords_to_smooth
    except Exception as e:
        warnings.warn(f"SavGol 平滑失败: {e}。返回插值后的数据。")
        return coords_to_smooth


def calculate_relative_y_and_shrug_distance(ls_y, rs_y, lh_y, rh_y, fps, baseline_window_sec):
    """
    计算肩部中点相对于臀部中点的 Y 坐标 (relative_y)，
    计算其低通滤波后的基准线 (baseline_relative_y)，
    并计算耸肩抬升距离 (shrug_distance = baseline - relative)。
    输入是已经插值和平滑过的 Y 坐标。
    返回: relative_y, baseline_relative_y, shrug_distance
    """
    n_frames = len(ls_y)
    if not (len(rs_y) == n_frames and len(lh_y) == n_frames and len(rh_y) == n_frames):
        raise ValueError("所有输入的Y坐标数组长度必须一致。")

    # 1. 计算相对 Y 坐标
    mid_shoulder_y = (ls_y + rs_y) / 2.0
    mid_hip_y = (lh_y + rh_y) / 2.0
    mid_shoulder_y[np.isnan(ls_y) | np.isnan(rs_y)] = np.nan
    mid_hip_y[np.isnan(lh_y) | np.isnan(rh_y)] = np.nan
    relative_y = mid_shoulder_y - mid_hip_y
    relative_y[np.isnan(mid_shoulder_y) | np.isnan(mid_hip_y)] = np.nan

    # --- 2. 计算基准线 (低通滤波) ---
    baseline_relative_y = np.full(n_frames, np.nan)
    valid_relative_y_mask = ~np.isnan(relative_y)

    if np.any(valid_relative_y_mask):
        relative_y_series = pd.Series(relative_y)
        # 填充 NaN 以便滤波
        fill_limit = max(5, int(fps * 0.5)) if fps > 0 else 5 # 允许填充稍大间隙
        relative_y_filled = relative_y_series.interpolate(method='linear', limit_direction='both', limit=fill_limit).fillna(method='ffill').fillna(method='bfill')
        relative_y_ready = relative_y_filled.to_numpy()

        if not np.any(np.isnan(relative_y_ready)):
            # 计算基准线滤波窗口大小 (帧数)
            baseline_window_frames = int(baseline_window_sec * fps) if fps > 0 else 31 # 默认值
            # 确保窗口为奇数且不大于数据长度
            baseline_window_frames = max(3, baseline_window_frames)
            if baseline_window_frames % 2 == 0: baseline_window_frames += 1
            baseline_window_frames = min(baseline_window_frames, n_frames) # 窗口不能超过总帧数
            if baseline_window_frames < 3: # 修正：如果窗口太小无法滤波
                 warnings.warn(f"基准线窗口 ({baseline_window_frames}) 过小，无法计算基准线。")
                 baseline_relative_y = relative_y_ready # 使用填充后的原始数据作为基准
            else:
                 baseline_poly = 1 # 低通滤波用低阶多项式
                 if baseline_poly >= baseline_window_frames:
                      baseline_poly = max(0, baseline_window_frames - 1) # 修正：确保 poly < window
                 if baseline_poly < 0: # 不应该发生，但做个检查
                      warnings.warn(f"基准线多项式阶数计算错误 ({baseline_poly})，无法计算基准线。")
                      baseline_relative_y = relative_y_ready
                 else:
                     try:
                         baseline_calc = savgol_filter(relative_y_ready, baseline_window_frames, baseline_poly)
                         # 恢复原始 NaN 位置
                         baseline_calc[~valid_relative_y_mask] = np.nan
                         baseline_relative_y = baseline_calc
                         print(f"  计算基准线使用窗口: {baseline_window_frames} 帧 ({baseline_window_sec}s), 阶数: {baseline_poly}")
                     except Exception as e:
                         warnings.warn(f"计算基准线失败: {e}。将使用填充后的相对Y坐标作为基准。")
                         baseline_relative_y = relative_y_ready # 出错时回退
                         baseline_relative_y[~valid_relative_y_mask] = np.nan # 恢复 NaN
        else:
             warnings.warn("相对 Y 坐标插值后仍含 NaN，无法计算基准线。")
             baseline_relative_y = relative_y # 使用原始（含NaN）数据
    else:
        warnings.warn("无有效的相对 Y 坐标数据，无法计算基准线。")

    # 3. 计算耸肩抬升距离
    shrug_distance = baseline_relative_y - relative_y
    # 抬升距离为负值或基准无效时，认为抬升距离为0或NaN
    shrug_distance[shrug_distance < 0] = 0
    shrug_distance[np.isnan(baseline_relative_y)] = np.nan

    return relative_y, baseline_relative_y, shrug_distance


# --- GMM 辅助函数 (需要) ---
def robust_gmm_clustering_1d(feature, feature_name, n_components=2, n_init=10, random_state=42, use_outlier_removal=False):
    """对一维特征进行 GMM 聚类（需要 StandardScaler）。返回分析字典。"""
    feature = np.asarray(feature)
    if feature.ndim != 1 or feature.size == 0 or np.all(np.isnan(feature)):
        warnings.warn(f"{feature_name} 数据无效或为空，无法进行 GMM 聚类。")
        return None, feature, None, {}
    feature = feature[~np.isnan(feature)]

    X = feature.reshape(-1, 1)
    if X.shape[0] < n_components * 5:
        warnings.warn(f"{feature_name} 数据点 ({X.shape[0]}) 过少，无法进行 GMM 聚类。")
        return None, feature, None, {}

    scaler = StandardScaler()
    try:
        X_scaled = scaler.fit_transform(X)
    except ValueError:
        warnings.warn(f"{feature_name} 数据无法标准化 (可能值都相同)，无法 GMM 聚类。")
        return None, feature, None, {}

    X_filtered = X_scaled
    if use_outlier_removal:
        lower, upper = np.percentile(X_scaled, [1, 99])
        mask = (X_scaled >= lower) & (X_scaled <= upper)
        if 0 < np.sum(~mask) < len(X_scaled) * 0.1:
             X_filtered = X_scaled[mask.flatten()]

    if X_filtered.shape[0] < n_components * 5:
        warnings.warn(f"过滤后 {feature_name} 数据点 ({X_filtered.shape[0]}) 过少，无法 GMM。")
        return None, scaler.inverse_transform(X_filtered).flatten(), None, {}

    best_gmm = None; best_bic = np.inf; best_cov_type = None
    for cov_type in ['full', 'spherical']:
        try:
            gmm = GaussianMixture(n_components=n_components, covariance_type=cov_type,
                                  n_init=n_init, random_state=random_state, reg_covar=1e-6)
            gmm.fit(X_filtered)
            bic = gmm.bic(X_filtered)
            if gmm.converged_ and bic < best_bic:
                best_bic = bic; best_gmm = gmm; best_cov_type = cov_type
        except Exception: pass

    if best_gmm is None:
        warnings.warn(f"GMM 模型未能成功拟合 {feature_name}。")
        return None, scaler.inverse_transform(X_filtered).flatten(), None, {}

    labels = best_gmm.predict(X_filtered)
    means_scaled = best_gmm.means_
    if best_cov_type == 'full': variances_scaled = best_gmm.covariances_.flatten()
    elif best_cov_type == 'spherical': variances_scaled = np.full(means_scaled.shape[0], best_gmm.covariances_[0])
    else: variances_scaled = np.ones(means_scaled.shape[0]); warnings.warn(f"未知GMM类型 '{best_cov_type}'")
    weights = best_gmm.weights_
    means_original = scaler.inverse_transform(means_scaled).flatten()
    stds_original = np.sqrt(np.maximum(variances_scaled, 1e-9)) * scaler.scale_[0]

    # 在 GMM 分析中，通常将均值较小的视为“低”分量
    low_idx = np.argmin(means_original); high_idx = 1 - low_idx
    analysis = {
        'best_covariance_type': best_cov_type, 'bic': best_bic, 'converged': best_gmm.converged_,
        'feature_name': feature_name, 'scaler_mean': scaler.mean_[0], 'scaler_std': scaler.scale_[0],
        'components': {
            'low_distance': {'index': int(low_idx), 'mean_original': means_original[low_idx], 'std_original': stds_original[low_idx], 'weight': float(weights[low_idx])},
            'high_distance': {'index': int(high_idx), 'mean_original': means_original[high_idx], 'std_original': stds_original[high_idx], 'weight': float(weights[high_idx])}
        }
    }
    return best_gmm, scaler.inverse_transform(X_filtered).flatten(), labels, analysis


# --- 阈值推荐函数 (基于距离) ---
def recommend_shrug_distance_threshold(shrug_distance, coord_units="像素", method='gmm'):
    """基于耸肩抬升距离 (shrug_distance) 推荐检测阈值。"""
    analysis_data = {}
    signal_name = "肩部相对抬升距离"
    signal_units = coord_units # 单位是像素

    # 最小阈值和默认阈值 (像素单位)
    min_thresh_val = 1.0 # 至少抬升 1 像素才有意义
    default_thresh = DEFAULT_SHRUG_DISTANCE_THRESHOLD
    analysis_data['min_thresh_val'] = min_thresh_val

    shrug_dist_np = np.asarray(shrug_distance)
    valid_dist_mask = ~np.isnan(shrug_dist_np)
    dist_to_analyze_all = shrug_dist_np[valid_dist_mask]

    # 只分析正值的抬升距离
    dist_to_analyze = dist_to_analyze_all[dist_to_analyze_all > min_thresh_val]

    if len(dist_to_analyze) < 10:
        warnings.warn(f"{signal_name} 有效数据点过少 (<10)，无法推荐阈值。返回默认值 {default_thresh:.1f} {signal_units}。")
        return default_thresh, {'fallback': {'threshold_shrug': default_thresh}}

    if len(dist_to_analyze) < 20:
        warnings.warn(f"显著抬升(>{min_thresh_val:.1f}{signal_units}) 的数据点不足 ({len(dist_to_analyze)} < 20)，回退到百分位法。")
        method = 'percentile'
        if len(dist_to_analyze) < 2:
             warnings.warn(f"显著抬升的数据点极少 (<2)，使用默认阈值 {default_thresh:.1f} {signal_units}。")
             return default_thresh, {'fallback': {'threshold_shrug': default_thresh}}

    recommended_threshold = default_thresh
    final_method = 'default'

    if method == 'gmm':
        try:
            # 尝试用 GMM 分离 "小幅抬升" 和 "大幅抬升"
            gmm_model, dist_clustered, labels, gmm_analysis = robust_gmm_clustering_1d(
                dist_to_analyze, signal_name, n_components=2
            )
            if gmm_model is None or not gmm_analysis:
                warnings.warn("GMM 聚类失败，回退到百分位法。")
                method = 'percentile'
            else:
                analysis_data['gmm'] = gmm_analysis
                # 在 GMM 分析结果中，'low_distance' 对应均值较小的簇，'high_distance' 对应均值较大的簇
                low_mean = gmm_analysis['components']['low_distance']['mean_original']
                high_mean = gmm_analysis['components']['high_distance']['mean_original']
                low_std = gmm_analysis['components']['low_distance']['std_original']
                high_std = gmm_analysis['components']['high_distance']['std_original']

                # 阈值策略：倾向于选择能分开两个簇，且位于“高抬升”簇开始的位置
                threshold_mid = (low_mean + high_mean) / 2.0
                # 高抬升簇的起始大概位置 (mean - std)
                threshold_high_start = high_mean - 1.0 * high_std if high_std > 1e-6 else high_mean * 0.7
                # 低抬升簇的结束大概位置 (mean + 2*std)
                threshold_low_end = low_mean + 2.0 * low_std if low_std > 1e-6 else low_mean * 1.5
                threshold_percentile = np.percentile(dist_to_analyze, 85) # 例如 P85

                # 根据分布权重和分离度选择
                low_weight = gmm_analysis['components']['low_distance']['weight']
                if (high_mean - low_mean) > 1.5 * (low_std + high_std): # 分离度尚可
                     rec_thresh = max(threshold_mid, threshold_low_end, threshold_high_start) # 取几个估计值中较大的
                elif low_weight > 0.8: # 大部分是小幅抬升
                     rec_thresh = threshold_percentile # 使用百分位捕捉显著抬升
                else: # 重叠较多
                     rec_thresh = max(threshold_high_start, threshold_percentile)

                rec_thresh = max(rec_thresh, min_thresh_val) # 不低于最小合理值
                recommended_threshold = rec_thresh
                final_method = 'gmm'
                analysis_data['gmm']['recommended_threshold_shrug'] = recommended_threshold

        except Exception as e:
            warnings.warn(f"GMM ({signal_name}) 分析失败: {e}。回退到百分位法。")
            method = 'percentile'

    if method == 'percentile':
        if len(dist_to_analyze) < 10:
             warnings.warn(f"{signal_name} 数据过少 (<10)，无法百分位法。使用默认值 {default_thresh:.1f}。")
             return default_thresh, {'fallback': {'threshold_shrug': default_thresh}}

        # 使用较高百分位捕捉显著抬升，例如 P85 或 P90
        rec_thresh = np.percentile(dist_to_analyze, 85)
        rec_thresh = max(rec_thresh, min_thresh_val)
        recommended_threshold = rec_thresh
        final_method = 'percentile'
        analysis_data['percentile'] = {'threshold_shrug': recommended_threshold, 'percentile_used': 85}

    if not np.isfinite(recommended_threshold) or recommended_threshold <= 0:
         recommended_threshold = default_thresh; final_method = 'default_fallback'
         warnings.warn(f"推荐阈值无效 ({recommended_threshold})，使用默认值 {default_thresh:.1f}。")
         analysis_data['fallback'] = {'threshold_shrug': default_thresh}

    analysis_data['final_recommendation'] = {'threshold': recommended_threshold, 'method_used': final_method}
    return recommended_threshold, analysis_data


# --- 阈值分析绘图 (基于距离) ---
def plot_shrug_distance_threshold_analysis(shrug_distance, analysis_data, signal_name, signal_units):
    """绘制耸肩抬升距离的分布和推荐的单一阈值。"""
    fig = plt.figure(figsize=(12, 7))
    dist_signal = np.asarray(shrug_distance)
    valid_dist_mask = ~np.isnan(dist_signal)
    plot_data_all = dist_signal[valid_dist_mask]

    if len(plot_data_all) == 0:
        plt.title(f"无有效 {signal_name} 数据可供分析")
        plt.close(fig); return None

    # 只绘制正值距离的直方图
    plot_data = plot_data_all[plot_data_all > 1e-6] # 忽略非常接近0的值
    if len(plot_data) == 0:
         plt.title(f"{TARGET_NAME}: 无正值 {signal_name} 数据，无法绘制直方图。")
         plt.close(fig); return None

    try:
        counts, bins, patches = plt.hist(plot_data, bins=60, density=True, alpha=0.75, color='skyblue', label=f'{signal_name} 分布 (>0)')
    except Exception as e:
         plt.title(f"{TARGET_NAME}: 绘制直方图失败 - {e}")
         plt.close(fig); return None

    rec_data = analysis_data.get('final_recommendation', {})
    rec_thresh = rec_data.get('threshold')
    rec_method = rec_data.get('method_used', '?').upper()
    min_thresh_val = analysis_data.get('min_thresh_val', 0.1)

    if rec_thresh is not None and np.isfinite(rec_thresh):
        plt.axvline(rec_thresh, color='r', linestyle='--', linewidth=2, label=f"最终推荐阈值 ({rec_method}): {rec_thresh:.2f} {signal_units}")
        try:
            y_max = plt.gca().get_ylim()[1]
            text_x = rec_thresh * 1.02
            text_y = y_max * 0.9
            plot_xmax = plt.gca().get_xlim()[1]
            ha = 'left'
            if text_x > plot_xmax * 0.9: text_x = rec_thresh * 0.98; ha = 'right'
            plt.text(text_x, text_y, f"{rec_thresh:.2f}", color='r', weight='bold', fontsize=10, horizontalalignment=ha)
        except Exception: pass

    # 可选：绘制 GMM 分量
    if analysis_data and 'gmm' in analysis_data and analysis_data['gmm'] and analysis_data['final_recommendation']['method_used'] == 'gmm':
        gmm_data = analysis_data['gmm']
        if 'components' in gmm_data and 'low_distance' in gmm_data['components'] and 'high_distance' in gmm_data['components']:
             try:
                # 使用 GMM 分析结果中的键名 'low_distance' 和 'high_distance'
                means = [gmm_data['components']['low_distance']['mean_original'], gmm_data['components']['high_distance']['mean_original']]
                stds = [gmm_data['components']['low_distance']['std_original'], gmm_data['components']['high_distance']['std_original']]
                weights = [gmm_data['components']['low_distance']['weight'], gmm_data['components']['high_distance']['weight']]
                if 'bins' in locals() and bins is not None and len(bins) > 1:
                    x_plot = np.linspace(max(0, bins[0]), bins[-1], 500)
                    colors = ['darkorange', 'dodgerblue']; labels = ['GMM 低抬升分量', 'GMM 高抬升分量']
                    for i in range(len(means)):
                        if means[i] is not None and stds[i] is not None and np.isfinite(means[i]) and np.isfinite(stds[i]) and stds[i] > 1e-6 and weights[i] > 1e-3:
                            pdf = weights[i] * (1 / (stds[i] * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_plot - means[i]) / stds[i])**2)
                            plt.plot(x_plot, pdf, color=colors[i], linestyle=':', linewidth=1.5, label=f'{labels[i]} (μ={means[i]:.1f}, σ={stds[i]:.1f})')
                else: warnings.warn("无法获取有效的 bins 信息来绘制 GMM 分量。")
             except KeyError as ke: warnings.warn(f"绘制 GMM 分量时缺少键: {ke}")
             except Exception as e: warnings.warn(f"绘制 GMM 分量时出错: {e}")


    plt.title(f"{TARGET_NAME} - {signal_name} 分布与阈值分析")
    plt.xlabel(f"{signal_name} ({signal_units})")
    plt.ylabel("概率密度 / GMM分量")
    plt.legend(fontsize='small'); plt.grid(True, alpha=0.3)

    if len(plot_data) > 0:
        p99 = np.percentile(plot_data, 99.5)
        xmax_thresh = rec_thresh if rec_thresh is not None and np.isfinite(rec_thresh) else 0
        plt.xlim(left= -0.05 * p99 , right=max(p99 * 1.1, xmax_thresh * 1.2, min_thresh_val * 5))
    else:
        plt.xlim(left=0, right=max(DEFAULT_SHRUG_DISTANCE_THRESHOLD * 1.2, 5.0)) # 默认上限

    plt.tight_layout()
    return fig


# --- 事件检测函数 (通用) ---
def detect_events_from_signal_single_threshold(signal,
                                              threshold, # 现在是距离阈值
                                              min_duration_frames=3, max_duration_frames=None,
                                              min_gap_frames=15, fps=30):
    """检测信号超过单一阈值的时段。"""
    signal_np = np.asarray(signal)
    num_frames = len(signal_np)
    if num_frames == 0: return []
    if not isinstance(threshold, (int, float)) or threshold <= 0:
        warnings.warn(f"检测阈值无效 ({threshold}) 或非正，将不检测任何事件。")
        return []

    exceed_mask = np.where(np.isnan(signal_np), False, signal_np >= threshold)
    exceed_indices = np.where(exceed_mask)[0]
    if len(exceed_indices) == 0: return []

    if len(exceed_indices) <= 1: groups = [exceed_indices] if len(exceed_indices) == 1 else []
    else: splits = np.where(np.diff(exceed_indices) > min_gap_frames)[0] + 1; groups = np.split(exceed_indices, splits)

    detected_events = []
    for group_indices in groups:
        if group_indices.size == 0: continue
        start_frame = group_indices[0]
        end_frame = group_indices[-1]
        duration_frames = (end_frame - start_frame) + 1

        if duration_frames < min_duration_frames: continue
        if max_duration_frames is not None and duration_frames > max_duration_frames: continue

        detected_events.append({'start_frame': start_frame, 'end_frame': end_frame})
    return detected_events


# --- 事件分析函数 (基于距离) ---
def analyze_shrug_events_distance(detected_events,
                                  shrug_distance_signal, # 检测信号
                                  relative_y_pos,        # 相对 Y 坐标 (用于上下文)
                                  fps=30, coord_units="像素"):
    """分析耸肩事件的抬升距离等指标。"""
    analyzed_events = []
    num_frames_pos = len(relative_y_pos)
    num_frames_signal = len(shrug_distance_signal)
    dist_units = coord_units

    for event in detected_events:
        start = event['start_frame']
        end = event['end_frame'] # inclusive

        if start > end or start < 0 or end >= num_frames_pos or end >= num_frames_signal:
            warnings.warn(f"事件索引 ({start}-{end}) 无效或超出范围，跳过分析。")
            continue

        seg_indices = np.arange(start, end + 1)
        try:
            seg_dist = shrug_distance_signal[seg_indices]
            seg_rel_y = relative_y_pos[seg_indices]
        except IndexError:
            warnings.warn(f"分析事件 {start}-{end} 时索引超出边界，跳过。")
            continue

        if np.all(np.isnan(seg_dist)):
             warnings.warn(f"事件 {start}-{end} 内抬升距离全为 NaN，跳过分析。")
             continue

        event_type = "耸肩"
        duration_frames = (end - start) + 1
        duration_sec = duration_frames / fps if fps > 0 else duration_frames

        # 分析指标：峰值抬升距离、平均抬升距离
        peak_shrug_distance = np.nanmax(seg_dist) if not np.all(np.isnan(seg_dist)) else 0
        avg_shrug_distance = np.nanmean(seg_dist) if not np.all(np.isnan(seg_dist)) else 0
        # 相对 Y 坐标的最小值（表示抬升最高点的位置）
        min_relative_y = np.nanmin(seg_rel_y) if not np.all(np.isnan(seg_rel_y)) else np.nan

        analyzed_event_data = {
            'event_type': event_type,
            'start_frame': start,
            'end_frame': end,
            'start_time_sec': start / fps if fps > 0 else start,
            'end_time_sec': (end + 1) / fps if fps > 0 else (end + 1),
            'duration_sec': duration_sec,
            'duration_frames': duration_frames,
            'peak_shrug_distance': peak_shrug_distance, # 峰值抬升距离
            'avg_shrug_distance': avg_shrug_distance,   # 平均抬升距离
            'min_relative_y': min_relative_y,         # 抬升最高点的相对Y坐标
            'units_distance': dist_units,
            'units_relative_y': coord_units,
            'start_time_str': frame_to_time_str(start, fps),
            'end_time_str': frame_to_time_str(end, fps),
        }
        analyzed_events.append(analyzed_event_data)

    return analyzed_events


# --- CSV 导出函数 (基于距离) ---
def export_shrug_events_to_csv(events, output_path):
    """导出检测到的耸肩事件及其距离指标到 CSV。"""
    output_path = Path(output_path)
    if not events:
        print("没有检测到耸肩事件，不生成 CSV 文件。")
        return None
    try:
        # 更新列名
        fieldnames = [
            'event_type', 'start_frame', 'end_frame', 'start_time_sec', 'end_time_sec',
            'duration_sec', 'duration_frames',
            'peak_shrug_distance', 'avg_shrug_distance', 'min_relative_y',
            'units_distance', 'units_relative_y',
            'start_time_str', 'end_time_str',
        ]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open('w', newline='', encoding='utf-8-sig') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            for event in events:
                row = event.copy()
                for key in ['start_time_sec', 'end_time_sec', 'duration_sec']:
                    row[key] = f"{event.get(key, 0):.3f}"
                for key in ['peak_shrug_distance', 'avg_shrug_distance', 'min_relative_y']:
                     row[key] = f"{event.get(key, np.nan):.2f}" # 保留2位小数
                writer.writerow(row)
        print(f"包含距离指标的耸肩事件已成功导出到: {output_path}")
        return output_path
    except IOError as e: warnings.warn(f"CSV 文件写入权限错误: {output_path} - {e}"); return None
    except Exception as e: warnings.warn(f"CSV 导出失败: {e}"); return None


# --- 最终绘图函数 (基于距离) ---
def plot_shrug_analysis_distance(shrug_distance_signal, # 检测信号
                                relative_y_pos,        # 相对 Y 坐标
                                baseline_relative_y,   # 基准线 Y 坐标
                                events,
                                threshold_shrug,       # 距离阈值
                                fps=30, coord_units="像素", title_prefix=""):
    """绘制耸肩分析图，显示抬升距离和相对 Y 坐标。"""
    num_signal_frames = len(shrug_distance_signal)
    num_pos_frames = len(relative_y_pos)
    if num_signal_frames == 0 or num_pos_frames == 0:
        print("数据不足，无法绘制耸肩分析图。")
        return None

    dist_units = coord_units

    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True, gridspec_kw={'height_ratios': [1, 1.5]})
    if not isinstance(axes, (list, np.ndarray)) or len(axes) != 2:
        plt.close(fig); return None
    ax1, ax2 = axes
    full_title = f"{title_prefix.strip()} {TARGET_NAME} 检测分析".strip()
    fig.suptitle(full_title, fontsize=16, y=0.98)

    # --- 1. 耸肩抬升距离图 ---
    frames_axis_signal = np.arange(num_signal_frames)
    ax1.plot(frames_axis_signal, shrug_distance_signal, label=f'肩部相对抬升距离 ({dist_units})', linewidth=1.5, alpha=0.9, color='dodgerblue')
    ax1.axhline(y=threshold_shrug, color='red', linestyle='--', alpha=0.8, linewidth=1.5, label=f'检测阈值={threshold_shrug:.2f} {dist_units}')
    ax1.set_title("肩部相对抬升距离 与 检测阈值")
    ax1.set_ylabel(f"抬升距离 ({dist_units})")
    # Y轴从0开始比较好
    ymin, ymax = ax1.get_ylim()
    ax1.set_ylim(bottom = -0.05 * ymax , top=ymax * 1.1) # 留一点空间
    ax1.grid(True, alpha=0.3)

    shrug_label_added = False
    if events:
        for event in events:
            start_f, end_f = event['start_frame'], event['end_frame']
            if 0 <= start_f <= end_f < num_signal_frames:
                label = "耸肩事件" if not shrug_label_added else "_nolegend_"
                color = 'lightcoral'
                ax1.axvspan(start_f, end_f + 1, color=color, alpha=0.35, label=label, zorder=-1)
                shrug_label_added = True
    ax1.legend(fontsize='small', loc='upper right')

    # --- 2. 相对 Y 坐标 和 基准线 图 ---
    frames_axis_pos = np.arange(num_pos_frames)
    # 绘制相对 Y 坐标 (Y_shoulder - Y_hip)
    ln1 = ax2.plot(frames_axis_pos, relative_y_pos, label=f'肩臀相对 Y 坐标 ({coord_units})', color='forestgreen', linewidth=1.5, alpha=0.8)
    # 绘制基准线
    ln2 = ax2.plot(frames_axis_pos, baseline_relative_y, label=f'相对 Y 坐标基准线 ({coord_units})', color='black', linestyle='--', linewidth=1.0, alpha=0.7)

    ax2.set_ylabel(f"相对 Y 坐标 ({coord_units})")
    ax2.tick_params(axis='y', labelcolor='black')
    ax2.set_title("肩臀相对 Y 坐标 与 基准线")
    ax2.set_xlabel("帧 / 时间")
    ax2.grid(True, alpha=0.3)

    # 合并图例
    lns = ln1 + ln2
    labs = [l.get_label() for l in lns]
    ax2.legend(lns, labs, loc='upper right', fontsize='small')

    # 在第二个图上也标记事件区域
    if events:
        for event in events:
            start_f, end_f = event['start_frame'], event['end_frame']
            if 0 <= start_f <= end_f < num_pos_frames:
                ax2.axvspan(start_f, end_f + 1, color='lightcoral', alpha=0.25, zorder=-1)

    if isinstance(fps, (int, float)) and fps > 0:
        formatter = FuncFormatter(lambda x, pos: time_formatter(x, pos, fps))
        ax2.xaxis.set_major_formatter(formatter)
    else:
        ax2.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(x)}"))

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig


# ==================================
#  == 主执行区域 ==
# ==================================
if __name__ == "__main__":

    print(f"欢迎使用 {TARGET_NAME} 检测工具！")
    print("请按照提示输入参数。直接按 Enter 键可使用括号中的默认值。")
    print("-" * 30)

    # --- 1. 获取输入 JSON 文件路径 ---
    input_json_path = None
    while input_json_path is None:
        try:
            # <<< --- 在这里修改你的 JSON 文件路径 --- >>>
            json_path_str = input(">>> 请输入 JSON 文件路径: ")
            # json_path_str = r"E:\视频标注\东华-苟亚军\苟亚军-躯体\results_3d_jupyter_cn\predictions\0425-东华苟亚军-1-tonly.json" # 调试路径
            print(f">>> JSON 文件路径: {json_path_str}")
            # <<< --- 修改结束 --- >>>

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

    # --- 4. 获取平滑和基准线参数 ---
    smooth_y_window = DEFAULT_SMOOTH_Y_WINDOW
    smooth_y_poly = DEFAULT_SMOOTH_Y_POLY
    baseline_window_sec = DEFAULT_BASELINE_WINDOW_SEC
    try:
        print("\n--- 设置平滑与基准线参数 ---")
        win_y_str = input(f"  Y坐标平滑窗口 (奇数>=3, 默认: {DEFAULT_SMOOTH_Y_WINDOW}): ").strip()
        if win_y_str: smooth_y_window = int(win_y_str)
        if smooth_y_window < 3 or smooth_y_window % 2 == 0: raise ValueError("Y坐标平滑窗口必须是 >=3 的奇数")

        poly_y_str = input(f"  Y坐标平滑阶数 (1 <= poly < window, 默认: {DEFAULT_SMOOTH_Y_POLY}): ").strip()
        if poly_y_str: smooth_y_poly = int(poly_y_str)
        if not (1 <= smooth_y_poly < smooth_y_window): raise ValueError("Y坐标多项式阶数必须 >=1 且小于窗口大小")

        baseline_sec_str = input(f"  基准线计算窗口时长 (秒, 默认: {DEFAULT_BASELINE_WINDOW_SEC:.1f}s): ").strip()
        if baseline_sec_str: baseline_window_sec = float(baseline_sec_str)
        if baseline_window_sec <= 0: raise ValueError("基准线窗口时长必须为正数")

    except (KeyboardInterrupt, EOFError): print("\n操作取消。"); sys.exit(1)
    except ValueError as e:
        print(f"输入错误 ({e})，将使用默认参数。")
        smooth_y_window = DEFAULT_SMOOTH_Y_WINDOW; smooth_y_poly = DEFAULT_SMOOTH_Y_POLY
        baseline_window_sec = DEFAULT_BASELINE_WINDOW_SEC
    print(f"使用Y坐标平滑: 窗口={smooth_y_window}, 阶数={smooth_y_poly}")
    print(f"使用基准线窗口: {baseline_window_sec:.1f} 秒")

    # --- 核心处理步骤 ---
    print("\n--- 正在加载和预处理数据 ---")
    final_threshold = DEFAULT_SHRUG_DISTANCE_THRESHOLD # 初始化默认距离阈值
    rec_thresh_dist = DEFAULT_SHRUG_DISTANCE_THRESHOLD
    threshold_analysis_dist = None
    analyzed_events = []
    analysis_successful = False
    coord_units_final = "像素"

    try:
        # 1. 加载数据
        coords_y_list, _, valid_flags_list, total_frames, coord_units_final = load_shoulder_hip_y_coordinates_2d(
            input_json_path, DEFAULT_REQUIRED_KP_INDICES, min_confidence)
        ls_y_raw, rs_y_raw, lh_y_raw, rh_y_raw = coords_y_list
        valid_ls, valid_rs, valid_lh, valid_rh = valid_flags_list

        min_req_frames = max(10, smooth_y_window, int(baseline_window_sec * fps * 1.1) if fps > 0 else 10) # 基准线需要足够数据
        num_both_shoulders_valid = np.sum(valid_ls & valid_rs)
        num_both_hips_valid = np.sum(valid_lh & valid_rh)
        if total_frames == 0 or num_both_shoulders_valid < min_req_frames or num_both_hips_valid < min_req_frames:
             raise ValueError(f"有效帧数不足 (需至少 {min_req_frames} 帧同时检测到双肩和双臀)。")

        # 2. 插值
        ls_y_interp = interpolate_single_coordinate(ls_y_raw, valid_ls)
        rs_y_interp = interpolate_single_coordinate(rs_y_raw, valid_rs)
        lh_y_interp = interpolate_single_coordinate(lh_y_raw, valid_lh)
        rh_y_interp = interpolate_single_coordinate(rh_y_raw, valid_rh)

        # 3. 平滑
        ls_y_smooth = smooth_single_coordinate(ls_y_interp, smooth_y_window, smooth_y_poly)
        rs_y_smooth = smooth_single_coordinate(rs_y_interp, smooth_y_window, smooth_y_poly)
        lh_y_smooth = smooth_single_coordinate(lh_y_interp, smooth_y_window, smooth_y_poly)
        rh_y_smooth = smooth_single_coordinate(rh_y_interp, smooth_y_window, smooth_y_poly)

        # 4. 计算相对距离信号
        print("计算相对坐标、基准线和抬升距离...")
        relative_y, baseline_relative_y, shrug_distance = calculate_relative_y_and_shrug_distance(
            ls_y_smooth, rs_y_smooth, lh_y_smooth, rh_y_smooth,
            fps, baseline_window_sec
        )

        if np.all(np.isnan(shrug_distance)):
            raise ValueError("所有计算出的耸肩抬升距离均为无效值！")

        signal_name = "肩部相对抬升距离"
        signal_units = coord_units_final # 像素

        # 5. 推荐距离阈值
        print(f"\n--- 推荐耸肩检测阈值 (基于 {signal_name}) ---")
        threshold_plot_fig = None
        try:
            rec_thresh_dist, threshold_analysis_dist = recommend_shrug_distance_threshold(
                shrug_distance, coord_units=coord_units_final, method='gmm' # fps 不再直接影响距离阈值
            )
            if threshold_analysis_dist:
                print(f"  推荐方法: {threshold_analysis_dist.get('final_recommendation', {}).get('method_used', '未知').upper()}")
                threshold_plot_fig = plot_shrug_distance_threshold_analysis(
                    shrug_distance, threshold_analysis_dist, signal_name, signal_units)
                if threshold_plot_fig:
                    threshold_plot_path = output_dir / f"{input_json_path.stem}_shrug_dist_threshold_analysis.png"
                    try:
                        threshold_plot_fig.savefig(threshold_plot_path, dpi=100)
                        print(f"  距离阈值分析图已保存: {threshold_plot_path}")
                    except Exception as save_err: print(f"  警告：保存阈值图失败: {save_err}")
                    plt.close(threshold_plot_fig)
                else:
                    print("  未能生成距离阈值分析图。")
            else:
                 rec_thresh_dist = DEFAULT_SHRUG_DISTANCE_THRESHOLD
                 print("  未能生成阈值分析数据，将使用默认阈值。")

        except Exception as e:
            print(f"  错误：距离阈值推荐时出错: {e}")
            rec_thresh_dist = DEFAULT_SHRUG_DISTANCE_THRESHOLD; threshold_analysis_dist = None

        # 6. 用户确认阈值
        print(f"\n推荐阈值: ≈ {rec_thresh_dist:.2f} {signal_units}")
        try:
            thresh_input = input(f"  输入最终使用的耸肩检测阈值 ({signal_units}, 回车使用推荐值): ").strip()
            if thresh_input: final_threshold = float(thresh_input)
            else: final_threshold = rec_thresh_dist
            if final_threshold <= 0: raise ValueError("阈值必须为正数")
        except (KeyboardInterrupt, EOFError): print("\n操作取消。"); sys.exit(1)
        except ValueError as e:
            print(f"  输入错误 ({e})，使用推荐/默认阈值 {rec_thresh_dist:.2f}")
            final_threshold = max(0.1, rec_thresh_dist) # 至少 0.1 像素
        print(f"最终使用阈值: {final_threshold:.2f} {signal_units}")

        # 7. 获取事件时间约束 (不变)
        print("\n设置事件时间约束 (单位: 秒)")
        min_dur_sec = DEFAULT_MIN_DURATION_SEC; max_dur_sec = DEFAULT_MAX_DURATION_SEC; merge_gap_sec = DEFAULT_MERGE_GAP_SEC
        try:
            min_dur_input = input(f"  最短持续时间 (默认 {DEFAULT_MIN_DURATION_SEC:.2f}s): ").strip()
            if min_dur_input: min_dur_sec = float(min_dur_input)
            max_dur_input = input(f"  最长持续时间 (默认 {DEFAULT_MAX_DURATION_SEC:.1f}s): ").strip()
            if max_dur_input: max_dur_sec = float(max_dur_input)
            merge_gap_input = input(f"  合并间隔 (默认 {DEFAULT_MERGE_GAP_SEC:.2f}s): ").strip()
            if merge_gap_input: merge_gap_sec = float(merge_gap_input)
            if min_dur_sec < 0 or max_dur_sec < 0 or merge_gap_sec < 0: raise ValueError("时间不能为负数")
        except (KeyboardInterrupt, EOFError): print("\n操作取消。"); sys.exit(1)
        except ValueError as e:
            print(f"  输入错误 ({e})，使用默认时间约束。")
            min_dur_sec = DEFAULT_MIN_DURATION_SEC; max_dur_sec = DEFAULT_MAX_DURATION_SEC; merge_gap_sec = DEFAULT_MERGE_GAP_SEC

        min_df = max(1, int(min_dur_sec * fps)) if fps > 0 else 1
        max_df = int(max_dur_sec * fps) if max_dur_sec > 0 and fps > 0 else None
        min_gf = int(merge_gap_sec * fps) if fps > 0 else 0
        print(f"约束 (帧): MinDur={min_df}, MaxDur={'无限制' if max_df is None else max_df}, MergeGap={min_gf}")

        # 8. 事件检测 (使用距离信号)
        print(f"\n使用阈值 {final_threshold:.2f} {signal_units} 检测事件...")
        detected_event_indices = detect_events_from_signal_single_threshold(
            shrug_distance, final_threshold, min_duration_frames=min_df,
            max_duration_frames=max_df, min_gap_frames=min_gf, fps=fps
        )

        # 9. 事件分析 (基于距离)
        print("分析检测到的事件...")
        analyzed_events = analyze_shrug_events_distance(
             detected_event_indices,
             shrug_distance, relative_y,
             fps, coord_units_final
        )

        # 10. 显示 & 导出结果
        if analyzed_events:
            print(f"\n检测到 {len(analyzed_events)} 个耸肩事件:")
            for i, ev in enumerate(analyzed_events[:5]):
                 print(f"  事件 {i+1}: {ev['start_time_str']} - {ev['end_time_str']} ({ev['duration_sec']:.2f}s), "
                       f"峰值抬升≈{ev['peak_shrug_distance']:.1f}{signal_units}")
            if len(analyzed_events) > 5: print("  ...")

            csv_fn = OUTPUT_CSV_TEMPLATE.format(input_json_path.stem, final_threshold)
            csv_path = output_dir / csv_fn
            export_path = export_shrug_events_to_csv(analyzed_events, csv_path)
            if not export_path: print("  警告：CSV 导出失败。")
        else: print("\n未检测到符合条件的耸肩事件。")

        # 11. 最终绘图 (基于距离)
        create_plots = True
        analysis_plot_fig = None
        try:
            plot_choice = input("\n是否生成并显示最终分析图? (Y/n): ").strip().lower()
            if plot_choice == 'n': create_plots = False
        except (KeyboardInterrupt, EOFError): print("\n操作取消。"); sys.exit(1)

        if create_plots:
            print("生成最终分析图...")
            analysis_plot_fig = plot_shrug_analysis_distance(
                shrug_distance_signal=shrug_distance,
                relative_y_pos=relative_y,
                baseline_relative_y=baseline_relative_y, # 传递基准线用于绘图
                events=analyzed_events,
                threshold_shrug=final_threshold,
                fps=fps, coord_units=coord_units_final,
                title_prefix=f"{input_json_path.stem}\n"
            )
            if analysis_plot_fig:
                plot_path = output_dir / f"{input_json_path.stem}_shrug_analysis_final.png"
                try:
                    analysis_plot_fig.savefig(plot_path, dpi=150)
                    print(f"最终分析图已保存到: {plot_path}")
                    print("提示：关闭图形窗口后程序将结束。")
                    plt.show()
                except Exception as e:
                    print(f"错误：保存或显示最终分析图失败: {e}")
                finally:
                    plt.close(analysis_plot_fig) # 确保关闭
            else: print("未能生成最终分析图。")
        else: print("已跳过生成最终分析图。")

        analysis_successful = True

    # --- 统一的错误处理 ---
    except FileNotFoundError as e: print(f"\n错误: {e}")
    except ValueError as e: print(f"\n处理错误: {e}")
    except Exception as e:
        print(f"\n分析过程中发生未预料的错误: {e}")
        import traceback; traceback.print_exc()
    finally:
        plt.close('all')

        # 计算总结信息
        num_events = len(analyzed_events) if 'analyzed_events' in locals() else 0
        max_shrug_dist_overall = 0.0
        if analysis_successful and 'shrug_distance' in locals() and shrug_distance is not None:
            try:
                 valid_dist = shrug_distance[~np.isnan(shrug_distance)]
                 if valid_dist.size > 0: max_shrug_dist_overall = np.nanmax(valid_dist)
            except Exception: pass

        final_thresh_value = final_threshold if 'final_threshold' in locals() else DEFAULT_SHRUG_DISTANCE_THRESHOLD
        results_summary = (num_events, final_thresh_value, max_shrug_dist_overall)
        units_str = coord_units_final # 单位是像素

        print(f"\n--- 分析流程结束 ({input_json_path.name if input_json_path else '未知文件'}) ---")
        print(f"总结: 检测到 {results_summary[0]} 个耸肩事件, 使用阈值 {results_summary[1]:.2f} {units_str}, 最大抬升距离约 {results_summary[2]:.1f} {units_str}")
        sys.exit(0)