'''
Author: Martinwang96 -git
Date: 2025-04-16 10:25:53
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
# from mpl_toolkits.mplot3d import Axes3D # 不再需要
from scipy.signal import savgol_filter
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler # GMM需要
# from sklearn.cluster import DBSCAN # 可选
import csv
import warnings
import sys # 用于 sys.exit

# --- Matplotlib 设置 ---
rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'sans-serif']
rcParams['axes.unicode_minus'] = False
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="Graph is not weighted")

# ==================================
#  == 配置区域 ==
# ==================================

# 1. 追踪点模式 (2D)
TRACKING_POINT_MODE = 'body_center' # 'mid_hip', 'mid_shoulder', 'body_center', 'single' (不再支持 pelvis, 因为通常没有 2D pelvis 点)

# 2. 关键点索引 (COCO 17)
LEFT_HIP_IDX = 11
RIGHT_HIP_IDX = 12
LEFT_SHOULDER_IDX = 5
RIGHT_SHOULDER_IDX = 6
SINGLE_KP_IDX = 0 # 仅在 'single' 模式下设置

# --- 自动配置 (2D) ---
CENTROID_KP_INDICES = None; CENTROID_NAME = ""; LOAD_MODE = 'single'; REQUIRED_INDICES = 1
if TRACKING_POINT_MODE == 'mid_hip':
    CENTROID_KP_INDICES = (LEFT_HIP_IDX, RIGHT_HIP_IDX); CENTROID_NAME = "髋部中点(2D)"; LOAD_MODE = 'mid_point'; REQUIRED_INDICES = 2
elif TRACKING_POINT_MODE == 'mid_shoulder':
    CENTROID_KP_INDICES = (LEFT_SHOULDER_IDX, RIGHT_SHOULDER_IDX); CENTROID_NAME = "肩部中点(2D)"; LOAD_MODE = 'mid_point'; REQUIRED_INDICES = 2
elif TRACKING_POINT_MODE == 'body_center':
    CENTROID_KP_INDICES = (LEFT_HIP_IDX, RIGHT_HIP_IDX, LEFT_SHOULDER_IDX, RIGHT_SHOULDER_IDX); CENTROID_NAME = "身体中心(2D)"; LOAD_MODE = 'body_center'; REQUIRED_INDICES = 4
elif TRACKING_POINT_MODE == 'single':
    CENTROID_KP_INDICES = (SINGLE_KP_IDX,); CENTROID_NAME = f"关键点({SINGLE_KP_IDX})(2D)"; LOAD_MODE = 'single'; REQUIRED_INDICES = 1
else: raise ValueError(f"未知模式: {TRACKING_POINT_MODE}")
if CENTROID_KP_INDICES is None or len(CENTROID_KP_INDICES) != REQUIRED_INDICES: raise ValueError("索引配置错误")

# 3. 默认计算参数 (2D)
DEFAULT_FPS = 30.0
DEFAULT_MIN_KEYPOINT_CONFIDENCE = 0.3
# 位置平滑参数
DEFAULT_SMOOTH_POS_WINDOW = 7
DEFAULT_SMOOTH_POS_POLY = 2
# 基准线计算窗口时长 (秒)
DEFAULT_BASELINE_WINDOW_SEC = 2.0
# 运动学计算参数 (X方向)
DEFAULT_KINEMATICS_WINDOW = 9
DEFAULT_KINEMATICS_POLY = 2
# 默认的水平偏差距离阈值 (像素) - 需要根据图像分辨率调整
DEFAULT_DEVIATION_X_PA_THRESHOLD = 3.0 # 姿态调整偏差距离 (像素)
DEFAULT_DEVIATION_X_MOVE_THRESHOLD = 10.0 # 移动偏差距离 (像素)

# 默认事件检测参数
DEFAULT_MIN_DURATION_SEC = 0.2
DEFAULT_MAX_DURATION_SEC = 10.0
DEFAULT_MERGE_GAP_SEC = 0.4

# ==================================
#  == 核心函数 (2D 版本) ==
# ==================================

# --- 辅助函数 ---
def frame_to_time_str(frame_number, fps=30):
    """将帧号转换为 MM:SS.ms 格式字符串"""
    if not isinstance(fps, (int, float)) or fps <= 0: return f"Frame {frame_number} (无效 FPS)"
    total_seconds = frame_number / fps; minutes = int(total_seconds // 60); seconds = total_seconds % 60
    return f"{minutes:02d}:{seconds:06.3f}"

def time_formatter(x, pos, fps=30):
    """Matplotlib FuncFormatter for time axis"""
    return frame_to_time_str(x, fps)

# --- 数据加载与处理 (2D) ---
def load_and_calculate_tracked_point_2d(json_path, kp_indices, mode='mid_point', min_confidence=0.3):
    """加载 2D 关键点 JSON，计算追踪点 2D 坐标 (x, y)。"""
    json_path = Path(json_path)
    if not json_path.exists(): raise FileNotFoundError(f"文件未找到: {json_path}")

    try: data = json.load(json_path.open('r', encoding='utf-8'))
    except Exception as e: raise IOError(f"读取或解析JSON失败: {json_path} - {e}")

    num_frames = len(data)
    tracked_positions_2d = np.full((num_frames, 2), np.nan, dtype=float) # (x, y)
    valid_flags = np.zeros(num_frames, dtype=bool)
    coord_units = "像素" # 2D 单位固定为像素

    required = {'single': 1, 'mid_point': 2, 'body_center': 4}.get(mode)
    if required is None or len(kp_indices) != required: raise ValueError(f"模式 '{mode}' 索引配置错误")

    processed_frames = 0
    for i, frame_info in enumerate(data):
        if not isinstance(frame_info, dict) or 'instances' not in frame_info: continue
        instances = frame_info.get("instances", [])
        if not instances or not isinstance(instances[0], dict): continue
        inst = instances[0]
        kps = np.array(inst.get("keypoints") or [])
        scores = np.array(inst.get("keypoint_scores") or [])
        if kps.size == 0 or scores.size == 0: continue

        # 确保 keypoints 是 N x 2
        if kps.ndim == 1 and kps.size % 2 == 0: kps = kps.reshape(-1, 2)
        elif kps.ndim != 2 or kps.shape[1] != 2: continue
        if kps.shape[0] <= max(kp_indices): continue

        processed_frames += 1
        pts = []; ok = True
        for idx in kp_indices:
            if scores[idx] < min_confidence: ok = False
            pts.append(kps[idx]) # 获取 (x, y)
        valid_flags[i] = ok

        # 计算追踪点 (即使无效也计算)
        if not all(np.all(np.isfinite(p)) for p in pts): pt = np.full(2, np.nan)
        elif mode == 'single': pt = pts[0]
        elif mode == 'mid_point': pt = (pts[0] + pts[1]) / 2.0
        else: # body_center
            hip_mid = (pts[0] + pts[1]) / 2.0; sh_mid = (pts[2] + pts[3]) / 2.0
            pt = (hip_mid + sh_mid) / 2.0
        tracked_positions_2d[i] = pt

        if np.any(np.isnan(pt)): valid_flags[i] = False

    if processed_frames == 0: warnings.warn(f"文件 '{json_path.name}' 未找到有效实例帧。")
    print(f"加载/计算完成: {num_frames} 帧, {np.sum(valid_flags)} 有效追踪点帧。单位: {coord_units}。")
    return tracked_positions_2d, valid_flags, num_frames, coord_units


def interpolate_invalid_frames_2d(positions_2d, valid_flags, limit_gap_frames=15):
    """线性插值无效的 2D 帧。"""
    interpolated_positions = positions_2d.copy()
    if np.all(valid_flags): return interpolated_positions
    if not np.any(valid_flags): warnings.warn("警告：无有效帧，无法插值！"); return interpolated_positions

    interpolated_positions[~valid_flags, :] = np.nan
    df = pd.DataFrame(interpolated_positions, columns=['x', 'y'])
    df_interpolated = df.interpolate(method='linear', axis=0, limit=limit_gap_frames, limit_direction='both', limit_area=None)
    df_interpolated = df_interpolated.fillna(method='ffill').fillna(method='bfill')

    final_nan = df_interpolated.isnull().sum().sum()
    if final_nan > 0: warnings.warn(f"警告：插值后仍有 {final_nan // 2} 帧数据未能完全填充。")
    return df_interpolated.to_numpy()


def smooth_positions_2d(positions_2d, window_length=7, poly_order=2):
    """使用 SavGol 平滑 2D 位置数据。"""
    # 填充内部 NaN
    pos_df = pd.DataFrame(positions_2d, columns=['x', 'y'])
    pos_filled = pos_df.interpolate(method='linear', limit_direction='both', limit_area='inside', limit=int(window_length*2))
    pos_filled = pos_filled.fillna(method='ffill').fillna(method='bfill')
    pos_ready = pos_filled.to_numpy()

    if pos_ready.shape[0] <= window_length:
        warnings.warn(f"数据点({pos_ready.shape[0]})不足以使用窗口{window_length}平滑。")
        return positions_2d # 返回原始含NaN数据

    if np.any(np.isnan(pos_ready)):
        warnings.warn("位置数据填充后仍含NaN，跳过平滑。")
        return positions_2d

    if window_length % 2 == 0: window_length += 1
    window_length = max(3, window_length)
    poly_order = min(max(1, poly_order), window_length - 1)

    smoothed = np.zeros_like(pos_ready)
    try:
        for i in range(pos_ready.shape[1]): # 对 x 和 y 分别滤波
            smoothed[:, i] = savgol_filter(pos_ready[:, i], window_length, poly_order)
        smoothed[np.isnan(positions_2d)] = np.nan # 恢复原始NaN
        # print(f"已应用 SavGol 平滑 2D 位置 (窗口={window_length}, 阶数={poly_order})。")
        return smoothed
    except Exception as e:
        warnings.warn(f"SavGol 平滑失败: {e}。返回插值后的数据。")
        return pos_ready


def calculate_baseline_and_deviation_x(positions_x, fps, baseline_window_sec):
    """计算 X 坐标的基准线和与基准线的偏差距离。"""
    n_frames = len(positions_x)
    baseline_x = np.full(n_frames, np.nan)
    deviation_distance_x = np.full(n_frames, np.nan)

    # 填充内部 NaN 以进行基准线滤波
    x_series = pd.Series(positions_x)
    fill_limit_base = max(10, int(fps * 1.0)) if fps > 0 else 10
    x_filled = x_series.interpolate(method='linear', limit_direction='both', limit_area='inside', limit=fill_limit_base)
    x_filled = x_filled.fillna(method='ffill').fillna(method='bfill')
    x_ready = x_filled.to_numpy()

    if np.any(np.isnan(x_ready)):
        warnings.warn("X坐标数据填充后仍含NaN，无法计算基准线。")
        return baseline_x, deviation_distance_x

    baseline_window_frames = int(baseline_window_sec * fps) if fps > 0 else 61
    baseline_window_frames = max(5, baseline_window_frames)
    if baseline_window_frames % 2 == 0: baseline_window_frames += 1
    baseline_window_frames = min(baseline_window_frames, n_frames)

    if baseline_window_frames < 5:
        warnings.warn(f"基准线窗口({baseline_window_frames})过小，使用原始平滑X坐标。")
        baseline_x = positions_x
    else:
        baseline_poly = 1
        if baseline_poly >= baseline_window_frames: baseline_poly = max(0, baseline_window_frames - 2)
        try:
            baseline_calc = savgol_filter(x_ready, baseline_window_frames, baseline_poly)
            baseline_calc[np.isnan(positions_x)] = np.nan # 恢复原始NaN
            baseline_x = baseline_calc
            print(f"  计算X轴基准线: 窗口={baseline_window_frames}帧 ({baseline_window_sec:.1f}s), 阶数={baseline_poly}")
        except Exception as e:
            warnings.warn(f"计算X轴基准线失败: {e}。")
            baseline_x = positions_x # 出错时回退

    # 计算水平偏差距离
    deviation_distance_x = np.abs(positions_x - baseline_x)
    # 如果原始位置或基准线是 NaN，偏差也是 NaN
    deviation_distance_x[np.isnan(positions_x) | np.isnan(baseline_x)] = np.nan

    return baseline_x, deviation_distance_x

def compute_kinematics_x(positions_x, fps, smooth_window=9, smooth_poly=2):
    """计算 X 方向的速度和加速度。"""
    # 内部处理 NaN
    x_series = pd.Series(positions_x)
    fill_limit_kin = max(5, int(fps * 0.2)) if fps > 0 else 5
    x_filled = x_series.interpolate(method='linear', limit_direction='both', limit_area='inside', limit=fill_limit_kin)
    x_filled = x_filled.fillna(method='ffill').fillna(method='bfill')
    x_ready = x_filled.to_numpy()

    n_frames = len(x_ready)
    nan_placeholder = np.full(n_frames, np.nan)
    zeros_placeholder = np.zeros(n_frames)

    if np.any(np.isnan(x_ready)):
        warnings.warn("X方向运动学计算前数据填充失败，返回 NaN。")
        return nan_placeholder, nan_placeholder

    if n_frames <= smooth_window:
        warnings.warn(f"数据点({n_frames})不足以计算X方向运动学(窗口 {smooth_window})。")
        return zeros_placeholder, zeros_placeholder

    if smooth_window % 2 == 0: smooth_window += 1
    smooth_window = max(3, smooth_window)
    smooth_poly_deriv1 = min(max(1, smooth_poly), smooth_window - 1)
    smooth_poly_deriv2 = min(max(2, smooth_poly), smooth_window - 1)
    dt = 1.0 / fps if fps > 0 else 1.0

    vx = zeros_placeholder.copy() # 初始化为 0
    ax = zeros_placeholder.copy()
    try:
        vx = savgol_filter(x_ready, smooth_window, smooth_poly_deriv1, deriv=1, delta=dt)
        ax = savgol_filter(x_ready, smooth_window, smooth_poly_deriv2, deriv=2, delta=dt**2)
        # 恢复原始NaN
        vx[np.isnan(positions_x)] = np.nan
        ax[np.isnan(positions_x)] = np.nan
        return vx, ax
    except Exception as e:
        warnings.warn(f"计算X方向运动学参数时出错: {e}。返回 NaN。")
        return nan_placeholder, nan_placeholder

def compute_frame_displacements_x(positions_x):
    """计算 X 方向的帧间位移绝对值 (用于X轴路径长度)。"""
    if not isinstance(positions_x, np.ndarray) or positions_x.ndim != 1:
         warnings.warn("计算X轴帧间位移的输入格式不正确。")
         return np.array([])
    if len(positions_x) < 2: return np.array([])

    valid_mask = ~np.isnan(positions_x)
    displacements_x = np.full(len(positions_x) - 1, np.nan)
    for i in range(len(positions_x) - 1):
        if valid_mask[i] and valid_mask[i+1]:
            displacements_x[i] = abs(positions_x[i+1] - positions_x[i]) # 只关心距离，不关心方向
    return displacements_x


# robust_gmm_clustering_1d 函数需要保留 (代码与之前相同，不再重复)
# --- GMM 辅助函数 (需要) ---
def robust_gmm_clustering_1d(feature, feature_name, n_components=2, n_init=10, random_state=42, use_outlier_removal=False):
    """对一维特征进行 GMM 聚类（需要 StandardScaler）。返回分析字典。"""
    feature = np.asarray(feature)
    if feature.ndim != 1 or feature.size == 0 or np.all(np.isnan(feature)):
        warnings.warn(f"{feature_name} 数据无效或为空，无法进行 GMM 聚类。")
        return None, feature, None, {}
    feature = feature[~np.isnan(feature)] # 去除 NaN

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

    noise_idx = np.argmin(means_original); move_idx = 1 - noise_idx
    analysis = {
        'best_covariance_type': best_cov_type, 'bic': best_bic, 'converged': best_gmm.converged_,
        'feature_name': feature_name, 'scaler_mean': scaler.mean_[0], 'scaler_std': scaler.scale_[0],
        'components': {
            'noise_or_adjust': {'index': int(noise_idx), 'mean_original': means_original[noise_idx], 'std_original': stds_original[noise_idx], 'weight': float(weights[noise_idx])},
            'movement': {'index': int(move_idx), 'mean_original': means_original[move_idx], 'std_original': stds_original[move_idx], 'weight': float(weights[move_idx])}
        }
    }
    return best_gmm, scaler.inverse_transform(X_filtered).flatten(), labels, analysis


# --- 阈值推荐 (基于水平偏差距离) ---
def recommend_deviation_thresholds_x(deviation_distance_x, coord_units="像素", method='gmm'):
    """基于 X 轴偏差距离推荐 PA 和 Movement 阈值。"""
    analysis_data = {}
    signal_name = "追踪点 X 轴与基准线偏差距离"
    signal_units = coord_units # 像素

    # 默认阈值 (像素)
    min_thresh_pa_val = 1.0 # 至少偏离1像素
    min_thresh_move_val = 3.0
    default_thresh_pa = DEFAULT_DEVIATION_X_PA_THRESHOLD
    default_thresh_move = DEFAULT_DEVIATION_X_MOVE_THRESHOLD
    analysis_data['min_thresh_pa_val'] = min_thresh_pa_val
    analysis_data['min_thresh_move_val'] = min_thresh_move_val
    analysis_data['default_threshold_pa'] = default_thresh_pa
    analysis_data['default_threshold_move'] = default_thresh_move

    dev_dist_np = np.asarray(deviation_distance_x)
    valid_dist_mask = ~np.isnan(dev_dist_np)
    dist_to_analyze_all = dev_dist_np[valid_dist_mask]
    dist_to_analyze = dist_to_analyze_all[dist_to_analyze_all > 1e-6] # 过滤零值

    if len(dist_to_analyze) < 10:
        warnings.warn(f"{signal_name} 有效数据点过少 (<10)，无法推荐。返回默认值。")
        return default_thresh_pa, default_thresh_move, {'fallback': {'threshold_pa': default_thresh_pa, 'threshold_move': default_thresh_move}}
    if len(dist_to_analyze) < 20:
        warnings.warn(f"非零 {signal_name} 数据不足 ({len(dist_to_analyze)} < 20)，回退百分位法。")
        method = 'percentile'

    rec_thresh_pa = default_thresh_pa; rec_thresh_move = default_thresh_move
    final_method = 'default'

    if method == 'gmm':
        try:
            gmm_model, dist_clustered, labels, gmm_analysis = robust_gmm_clustering_1d(
                dist_to_analyze, signal_name, n_components=2
            )
            if gmm_model is None or not gmm_analysis: method = 'percentile'; warnings.warn("GMM失败，回退百分位法。")
            else:
                analysis_data['gmm'] = gmm_analysis
                noise_mean = gmm_analysis['components']['noise_or_adjust']['mean_original']
                move_mean = gmm_analysis['components']['movement']['mean_original']
                noise_std = gmm_analysis['components']['noise_or_adjust']['std_original']
                move_std = gmm_analysis['components']['movement']['std_original']

                rec_thresh_pa = noise_mean + 2.0 * noise_std
                rec_thresh_pa = min(rec_thresh_pa, noise_mean + 3.0*noise_std, move_mean - 0.5*move_std if move_std > 1e-6 else move_mean * 0.8)

                threshold_mid = (noise_mean + move_mean) / 2.0
                threshold_move_start = move_mean - 1.0 * move_std
                threshold_percentile = np.percentile(dist_to_analyze, 90)

                noise_weight = gmm_analysis['components']['noise_or_adjust']['weight']
                if (move_mean - noise_mean) > 2.0 * (noise_std + move_std): rec_thresh_move = max(threshold_mid, threshold_move_start)
                elif noise_weight > 0.8: rec_thresh_move = threshold_percentile
                else: rec_thresh_move = max(threshold_move_start, threshold_percentile)

                rec_thresh_pa = max(rec_thresh_pa, min_thresh_pa_val)
                rec_thresh_move = max(rec_thresh_move, min_thresh_move_val)
                if rec_thresh_move <= rec_thresh_pa * 1.2:
                    rec_thresh_move = max(rec_thresh_pa * 1.5, np.percentile(dist_to_analyze, 90)); print("警告：GMM双阈值接近，已调整。")
                rec_thresh_move = max(rec_thresh_move, min_thresh_move_val)
                rec_thresh_pa = min(rec_thresh_pa, rec_thresh_move * 0.9); rec_thresh_pa = max(rec_thresh_pa, min_thresh_pa_val)

                analysis_data['gmm']['recommended_threshold_pa'] = rec_thresh_pa
                analysis_data['gmm']['recommended_threshold_move'] = rec_thresh_move
                final_method = 'gmm'
        except Exception as e: warnings.warn(f"GMM分析失败: {e}。回退百分位法。"); method = 'percentile'

    if method == 'percentile':
        if len(dist_to_analyze) < 10:
             warnings.warn(f"数据过少无法用百分位法。返回默认值。")
             return default_thresh_pa, default_thresh_move, {'fallback': {'threshold_pa': default_thresh_pa, 'threshold_move': default_thresh_move}}

        rec_thresh_pa = np.percentile(dist_to_analyze, 70)
        rec_thresh_move = np.percentile(dist_to_analyze, 92)

        rec_thresh_pa = max(rec_thresh_pa, min_thresh_pa_val)
        rec_thresh_move = max(rec_thresh_move, min_thresh_move_val)
        if rec_thresh_move <= rec_thresh_pa * 1.2: rec_thresh_move = max(rec_thresh_pa * 1.5, np.percentile(dist_to_analyze, 92))
        rec_thresh_move = max(rec_thresh_move, min_thresh_move_val)
        rec_thresh_pa = min(rec_thresh_pa, rec_thresh_move * 0.9); rec_thresh_pa = max(rec_thresh_pa, min_thresh_pa_val)

        analysis_data['percentile'] = {'threshold_pa': rec_thresh_pa, 'threshold_move': rec_thresh_move}
        final_method = 'percentile'

    final_pa = analysis_data.get('gmm', {}).get('recommended_threshold_pa') or analysis_data.get('percentile', {}).get('threshold_pa') or default_thresh_pa
    final_move = analysis_data.get('gmm', {}).get('recommended_threshold_move') or analysis_data.get('percentile', {}).get('threshold_move') or default_thresh_move
    if final_move <= final_pa: final_move = final_pa * 1.5

    analysis_data['final_recommendation'] = {'threshold_pa': final_pa, 'threshold_move': final_move, 'method_used': final_method}
    print(f"{final_method.upper()} 推荐: PA 阈值 ≈ {final_pa:.2f}, Move 阈值 ≈ {final_move:.2f} {signal_units}")
    return final_pa, final_move, analysis_data


# --- 阈值分析绘图 (基于水平偏差距离) ---
def plot_deviation_threshold_analysis_x(deviation_distance_x, analysis_data, coord_units="像素"):
    """绘制 X 轴偏差距离直方图和推荐的 PA/Movement 阈值。"""
    fig = plt.figure(figsize=(12, 7))
    signal_name = "追踪点 X 轴与基准线偏差距离"
    signal_units = coord_units

    dev_dist_np = np.asarray(deviation_distance_x)
    valid_dist_mask = ~np.isnan(dev_dist_np)
    plot_data_all = dev_dist_np[valid_dist_mask]
    if len(plot_data_all) == 0: plt.title(f"无有效 {signal_name} 数据"); plt.close(fig); return None
    plot_data = plot_data_all[plot_data_all > 1e-6]
    if len(plot_data) == 0: plt.title(f"{CENTROID_NAME}: 无正值 {signal_name} 数据"); plt.close(fig); return None

    try: counts, bins, patches = plt.hist(plot_data, bins=60, density=True, alpha=0.75, color='lightgreen', label=f'{signal_name} 分布 (>0)')
    except Exception as e: plt.title(f"{CENTROID_NAME}: 绘制直方图失败 - {e}"); plt.close(fig); return None

    final_rec = analysis_data.get('final_recommendation', {})
    thresh_pa = final_rec.get('threshold_pa'); thresh_move = final_rec.get('threshold_move')
    method_used = final_rec.get('method_used', 'unknown').upper()
    min_pa_val = analysis_data.get('min_thresh_pa_val', 0)
    default_thresh_move_plot = analysis_data.get('default_threshold_move', DEFAULT_DEVIATION_X_MOVE_THRESHOLD)

    colors = {'PA': 'orange', 'MOVE': 'red', 'GMM_MEAN_NOISE': 'blue', 'GMM_MEAN_MOVE': 'magenta'}
    styles = {'PA': ':', 'MOVE': '--', 'MEAN': ':'}
    y_max = plt.gca().get_ylim()[1]; text_y_pos = y_max * 0.95; text_offset = y_max * 0.07

    def plot_vline_with_text(value, label, color, linestyle, y_start):
        if value is not None and np.isfinite(value):
            plt.axvline(x=value, color=color, linestyle=linestyle, linewidth=1.5, label=f"{label} ({method_used}): {value:.2f}")
            ha = 'left'; text_x = value * 1.02; plot_xmax = plt.gca().get_xlim()[1]
            if text_x > plot_xmax * 0.9: text_x = value * 0.98; ha = 'right'
            plt.text(text_x, y_start, f"{value:.2f}", color=color, fontsize=9, ha=ha, va='top')
            return y_start - text_offset
        return y_start

    text_y_pos = plot_vline_with_text(thresh_pa, '姿态调整阈值', colors['PA'], styles['PA'], text_y_pos)
    text_y_pos = plot_vline_with_text(thresh_move, '移动阈值', colors['MOVE'], styles['MOVE'], text_y_pos)

    if analysis_data and 'gmm' in analysis_data and analysis_data['gmm']:
        gmm_data = analysis_data['gmm']
        if 'components' in gmm_data:
            try:
                noise_mean = gmm_data['components']['noise_or_adjust']['mean_original']
                move_mean = gmm_data['components']['movement']['mean_original']
                text_y_pos = plot_vline_with_text(noise_mean, 'GMM 调整均值', colors['GMM_MEAN_NOISE'], styles['MEAN'], text_y_pos)
                text_y_pos = plot_vline_with_text(move_mean, 'GMM 移动均值', colors['GMM_MEAN_MOVE'], styles['MEAN'], text_y_pos)
            except KeyError: pass

    plt.title(f"{CENTROID_NAME} {signal_name} 分布与推荐阈值")
    plt.xlabel(f"{signal_name} ({signal_units})"); plt.ylabel("概率密度")
    plt.legend(fontsize='small', loc='best', framealpha=0.9); plt.grid(True, alpha=0.3)
    if len(plot_data) > 0:
        p99 = np.percentile(plot_data, 99.5); xmax = p99 * 1.1
        if thresh_move is not None and np.isfinite(thresh_move): xmax = max(xmax, thresh_move * 1.2)
        plt.xlim(left=-0.05 * p99, right=max(xmax, min_pa_val * 5))
    else: xmax = 1.0; 
    if thresh_move is not None and np.isfinite(thresh_move): xmax = max(xmax, thresh_move * 1.2)
    else: xmax = max(xmax, default_thresh_move_plot * 1.2) # 使用变量
    plt.xlim(left=0, right=xmax)
    plt.tight_layout()
    return fig

# --- 事件检测 (基于水平偏差距离阈值) ---
def detect_events_deviation_multi_threshold_x(deviation_distance_x,
                                           threshold_pose_adj_dev_x, threshold_movement_dev_x,
                                           min_duration_frames=3, max_duration_frames=None,
                                           min_gap_frames=15):
    """检测 X 轴偏差距离超过 PA 阈值的时段。"""
    # (代码与 detect_events_deviation_multi_threshold 完全相同，只是输入信号和阈值不同)
    signal_np = np.asarray(deviation_distance_x)
    num_frames = len(signal_np)
    if num_frames == 0: return []
    if not isinstance(threshold_pose_adj_dev_x, (int, float)) or threshold_pose_adj_dev_x < 0: return []
    if not isinstance(threshold_movement_dev_x, (int, float)) or threshold_movement_dev_x <= threshold_pose_adj_dev_x: threshold_movement_dev_x = threshold_pose_adj_dev_x * 1.5

    exceed_mask = np.where(np.isnan(signal_np), False, signal_np >= threshold_pose_adj_dev_x)
    exceed_indices = np.where(exceed_mask)[0]
    if len(exceed_indices) == 0: return []

    if len(exceed_indices) <= 1: groups = [exceed_indices] if len(exceed_indices) == 1 else []
    else: splits = np.where(np.diff(exceed_indices) > min_gap_frames)[0] + 1; groups = np.split(exceed_indices, splits)

    candidate_events = []
    for group_indices in groups:
        if group_indices.size == 0: continue
        start_frame = group_indices[0]; end_frame = group_indices[-1]; duration_frames = (end_frame - start_frame) + 1
        if duration_frames < min_duration_frames: continue
        if max_duration_frames is not None and duration_frames > max_duration_frames: continue
        candidate_events.append({'start_frame': start_frame, 'end_frame': end_frame})
    return candidate_events

# --- 事件分析与分类 (基于水平偏差距离分类, 分析X轴运动学) ---
def analyze_and_classify_events_deviation_x(candidate_events,
                                         deviation_distance_x, # 用于分类
                                         threshold_movement_dev_x, # 移动分类阈值
                                         displacements_x, # 用于计算X轴路径长度
                                         positions_x, vx, ax, # X轴运动学数据
                                         fps=30, coord_units="像素"):
    """分析 X 轴运动学指标，并根据 X 轴偏差距离峰值分类事件。"""
    analyzed_events = []
    num_pos_frames = len(positions_x); num_dev_frames = len(deviation_distance_x); num_disp_frames = len(displacements_x)
    vel_units = f"{coord_units}/s" if fps > 0 else f"{coord_units}/帧"
    acc_units = f"{coord_units}/s²" if fps > 0 else f"{coord_units}/帧²"

    for event in candidate_events:
        start = event['start_frame']; end = event['end_frame']
        if start > end or start < 0 or end >= num_pos_frames or end >= num_dev_frames: continue

        # --- 分类 ---
        seg_indices_dev = np.arange(start, end + 1)
        try: event_deviations_x = deviation_distance_x[seg_indices_dev]
        except IndexError: continue
        peak_deviation_x = np.nanmax(event_deviations_x) if not np.all(np.isnan(event_deviations_x)) else 0
        is_movement_event = peak_deviation_x >= threshold_movement_dev_x
        event_type = "Movement" if is_movement_event else "Pose Adjustment"

        # --- X 轴运动学分析 ---
        seg_indices_kin = np.arange(start, end + 1)
        try:
            seg_pos_x = positions_x[seg_indices_kin]; seg_vx = vx[seg_indices_kin]; seg_ax = ax[seg_indices_kin]
        except IndexError: continue

        duration_frames = (end - start) + 1; duration_sec = duration_frames / fps if fps > 0 else duration_frames

        # X 轴路径长度 (帧间位移绝对值之和)
        event_disp_indices = np.arange(start, end); valid_disp_indices = event_disp_indices[event_disp_indices < num_disp_frames]
        path_length_x = 0.0; max_inst_displacement_x = 0.0
        if valid_disp_indices.size > 0:
            event_displacements_x = displacements_x[valid_disp_indices]
            path_length_x = np.nansum(event_displacements_x)
            max_inst_displacement_x = np.nanmax(event_displacements_x) if not np.all(np.isnan(event_displacements_x)) else 0.0

        # X 轴净位移
        start_x = positions_x[start]; end_x = positions_x[end]
        net_displacement_x = np.nan
        if not np.isnan(start_x) and not np.isnan(end_x): net_displacement_x = end_x - start_x

        # X 轴速度/加速度统计
        speed_x_abs = np.abs(seg_vx)
        speed_x_stats = {'mean': np.nanmean(speed_x_abs), 'max': np.nanmax(speed_x_abs)}
        accel_x_abs = np.abs(seg_ax)
        accel_x_stats = {'mean': np.nanmean(accel_x_abs), 'max': np.nanmax(accel_x_abs)}

        analyzed_event_data = {
            'event_type': event_type, 'start_frame': start, 'end_frame': end,
            'start_time_sec': start / fps if fps > 0 else start, 'end_time_sec': (end + 1) / fps if fps > 0 else (end + 1),
            'duration_sec': duration_sec, 'duration_frames': duration_frames,
            'peak_deviation_distance_x': peak_deviation_x,
            'path_length_x': path_length_x, 'net_displacement_x': net_displacement_x,
            'max_inst_displacement_x': max_inst_displacement_x,
            'speed_x_abs_mean': speed_x_stats['mean'], 'speed_x_abs_max': speed_x_stats['max'],
            'accel_x_abs_mean': accel_x_stats['mean'], 'accel_x_abs_max': accel_x_stats['max'],
            'units_pos': coord_units, 'units_vel': vel_units, 'units_acc': acc_units,
            'start_time_str': frame_to_time_str(start, fps), 'end_time_str': frame_to_time_str(end, fps),
        }
        analyzed_events.append(analyzed_event_data)
    return analyzed_events


# --- CSV 导出 (X轴版本) ---
def export_events_to_csv_x(events, output_path):
    """导出 X 轴移动事件及指标到 CSV。"""
    output_path = Path(output_path)
    if not events: print("无事件可导出。"); return None
    try:
        fieldnames = [ # 更新列名
            'event_type', 'start_frame', 'end_frame', 'start_time_sec', 'end_time_sec',
            'duration_sec', 'duration_frames', 'peak_deviation_distance_x',
            'path_length_x', 'net_displacement_x', 'max_inst_displacement_x',
            'speed_x_abs_mean', 'speed_x_abs_max', 'accel_x_abs_mean', 'accel_x_abs_max',
            'units_pos', 'units_vel', 'units_acc', 'start_time_str', 'end_time_str',
        ]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open('w', newline='', encoding='utf-8-sig') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            for event in events:
                row = event.copy()
                if row['event_type'] == 'Movement': row['event_type'] = '移动'
                elif row['event_type'] == 'Pose Adjustment': row['event_type'] = '姿态调整'
                for key in ['start_time_sec', 'end_time_sec', 'duration_sec']: row[key] = f"{event.get(key, 0):.3f}"
                for key in fieldnames: # 格式化所有数值型指标
                    if key not in ['event_type', 'start_frame', 'end_frame', 'duration_frames', 'units_pos', 'units_vel', 'units_acc', 'start_time_str', 'end_time_str']:
                        row[key] = f"{event.get(key, np.nan):.3f}" # 保留3位小数
                writer.writerow(row)
        print(f"包含X轴指标的事件已导出到: {output_path}")
        return output_path
    except Exception as e: warnings.warn(f"CSV导出失败: {e}"); return None


# --- 最终绘图 (2D X轴版本) ---
def plot_deviation_analysis_x(deviation_distance_x, # X轴偏差信号
                             positions_x,          # X轴坐标
                             baseline_x,           # X轴基准线
                             events,               # 分析和分类后的事件
                             threshold_pose_adj_dev_x, threshold_movement_dev_x, # X轴偏差阈值
                             fps=30, coord_units="像素", title_prefix=""):
    """绘制基于 X 轴偏差距离的分析图。"""
    num_dev_frames = len(deviation_distance_x); num_pos_frames = len(positions_x)
    if num_dev_frames == 0 or num_pos_frames == 0: print("数据不足，无法绘图。"); return None

    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True) # 改为 2 行 1 列
    if not isinstance(axes, (list, np.ndarray)) or len(axes) != 2: plt.close(fig); return None
    ax1, ax2 = axes
    full_title = f"{title_prefix.strip()} {CENTROID_NAME} 水平移动分析 (基于X轴与基准线偏差)"
    fig.suptitle(full_title, fontsize=16, y=0.98)

    # --- 1. X轴偏差距离图 ---
    frames_axis_dev = np.arange(num_dev_frames)
    ax1.plot(frames_axis_dev, deviation_distance_x, label=f'X轴与基准线偏差距离 ({coord_units})', linewidth=1, alpha=0.8, color='darkcyan')
    ax1.axhline(y=threshold_pose_adj_dev_x, color='orange', linestyle=':', alpha=0.8, linewidth=1.5, label=f'PA阈值={threshold_pose_adj_dev_x:.2f}')
    ax1.axhline(y=threshold_movement_dev_x, color='red', linestyle='--', alpha=0.8, linewidth=1.5, label=f'移动阈值={threshold_movement_dev_x:.2f}')
    ax1.set_title("X轴与基准线偏差距离 及 阈值")
    ax1.set_ylabel(f"偏差距离 ({coord_units})")
    ymin, ymax = ax1.get_ylim(); ax1.set_ylim(bottom= -0.05 * ymax, top=ymax * 1.1)
    ax1.grid(True, alpha=0.3)

    pa_label_added = False; move_label_added = False
    if events:
        for event in events:
            start_f, end_f = event['start_frame'], event['end_frame']
            if 0 <= start_f <= end_f < num_dev_frames:
                label = ""; color = ""; alpha = 0.35
                if event['event_type'] == 'Movement': label = "移动" if not move_label_added else "_nolegend_"; color = 'lightcoral'; move_label_added = True
                else: label = "姿态调整" if not pa_label_added else "_nolegend_"; color = 'lightblue'; pa_label_added = True
                ax1.axvspan(start_f, end_f + 1, color=color, alpha=alpha, label=label, zorder=-1)
    ax1.legend(fontsize='small', loc='upper right')

    # --- 2. X坐标 与 基准线 图 ---
    frames_axis_pos = np.arange(num_pos_frames)
    ln1 = ax2.plot(frames_axis_pos, positions_x, label=f'X 坐标 ({coord_units})', color='forestgreen', linewidth=1.5, alpha=0.8)
    ln2 = ax2.plot(frames_axis_pos, baseline_x, label=f'X 坐标基准线 ({coord_units})', color='black', linestyle='--', linewidth=1.0, alpha=0.7)
    ax2.set_ylabel(f"X 坐标 ({coord_units})")
    ax2.tick_params(axis='y', labelcolor='black')
    ax2.set_title("追踪点 X 坐标 与 基准线")
    ax2.set_xlabel("帧 / 时间")
    ax2.grid(True, alpha=0.3)

    lns = ln1 + ln2; labs = [l.get_label() for l in lns]
    ax2.legend(lns, labs, loc='upper right', fontsize='small')

    if events:
        for event in events:
            start_f, end_f = event['start_frame'], event['end_frame']
            if 0 <= start_f <= end_f < num_pos_frames:
                 color = 'lightcoral' if event['event_type'] == 'Movement' else 'lightblue'
                 ax2.axvspan(start_f, end_f + 1, color=color, alpha=0.20, zorder=-1) # 透明度降低

    # 添加时间轴格式
    if isinstance(fps, (int, float)) and fps > 0:
        formatter = FuncFormatter(lambda x, pos: time_formatter(x, pos, fps))
        ax2.xaxis.set_major_formatter(formatter) # 应用于底部X轴
    else: ax2.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(x)}"))

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig


# ==================================
#  == 主执行区域 (2D X轴版本) ==
# ==================================
if __name__ == "__main__":

    print(f"欢迎使用 {CENTROID_NAME} 水平移动分析工具 (基于X轴与基准线偏差)")
    print("请按照提示输入参数。直接按 Enter 键可使用括号中的默认值。")
    print("-" * 30)

    # --- 1. 获取输入 JSON 文件路径 ---
    input_json_path = None
    while input_json_path is None:
        try:
            # <<< --- 在这里修改你的 JSON 文件路径 --- >>>
            json_path_str = input(">>> 请输入 JSON 文件路径: ")
            # json_path_str = r"E:\视频标注\东华-苟亚军\0425-东华苟亚军-1_results\苟亚军-1-merged.json" # 调试路径
            print(f">>> JSON 文件路径: {json_path_str}")
            # <<< --- 修改结束 --- >>>
            if not json_path_str: continue
            temp_path = Path(json_path_str.strip())
            if temp_path.is_file(): input_json_path = temp_path
            else: print(f"错误：文件不存在: '{json_path_str}'")
        except (KeyboardInterrupt, EOFError): print("\n操作取消。"); sys.exit(1)
        except Exception as e: print(f"读取路径时出错: {e}")

    # --- 2. 获取输出目录 ---
    analysis_folder_name = f"{input_json_path.stem}_deviationX_analysis_{TRACKING_POINT_MODE}"
    default_output_dir = input_json_path.parent / analysis_folder_name
    output_dir = default_output_dir
    try:
        output_dir_str = input(f">>> 请输入输出目录 (默认: '{default_output_dir}'): ").strip()
        if output_dir_str: output_dir = Path(output_dir_str)
        output_dir.mkdir(parents=True, exist_ok=True); print(f"结果将保存到: {output_dir}")
    except (KeyboardInterrupt, EOFError): print("\n操作取消。"); sys.exit(1)
    except Exception as e:
        print(f"设置输出目录时出错 ({e})，将使用默认目录。")
        output_dir = default_output_dir
        try: output_dir.mkdir(parents=True, exist_ok=True)
        except Exception: print(f"错误：无法创建目录 {output_dir}"); sys.exit(1)

    # --- 3. 获取基础参数 ---
    fps = DEFAULT_FPS; min_confidence = DEFAULT_MIN_KEYPOINT_CONFIDENCE
    try:
        fps_str = input(f">>> 请输入 FPS (默认: {DEFAULT_FPS:.1f}): ").strip()
        if fps_str: fps = float(fps_str)
        if fps <= 0: raise ValueError("FPS 必须 > 0")
        conf_str = input(f">>> 请输入最小置信度 (0-1, 默认: {DEFAULT_MIN_KEYPOINT_CONFIDENCE:.2f}): ").strip()
        if conf_str: min_confidence = float(conf_str)
        if not (0.0 <= min_confidence <= 1.0): raise ValueError("置信度需在 0-1")
    except (KeyboardInterrupt, EOFError): print("\n操作取消。"); sys.exit(1)
    except ValueError as e: print(f"输入错误 ({e})，使用默认值。"); fps = DEFAULT_FPS; min_confidence = DEFAULT_MIN_KEYPOINT_CONFIDENCE
    print(f"使用参数: FPS={fps:.1f}, MinConf={min_confidence:.2f}")

    # --- 4. 获取计算参数 ---
    smooth_pos_window = DEFAULT_SMOOTH_POS_WINDOW; smooth_pos_poly = DEFAULT_SMOOTH_POS_POLY
    baseline_window_sec = DEFAULT_BASELINE_WINDOW_SEC
    kinematics_window = DEFAULT_KINEMATICS_WINDOW; kinematics_poly = DEFAULT_KINEMATICS_POLY
    try:
        print("\n--- 设置计算参数 ---")
        win_p_str = input(f"  位置平滑窗口 (奇数>=3, 默认: {DEFAULT_SMOOTH_POS_WINDOW}): ").strip()
        if win_p_str: smooth_pos_window = int(win_p_str)
        if smooth_pos_window < 3 or smooth_pos_window % 2 == 0: raise ValueError("位置平滑窗口错误")
        poly_p_str = input(f"  位置平滑阶数 (1<=poly<win, 默认: {DEFAULT_SMOOTH_POS_POLY}): ").strip()
        if poly_p_str: smooth_pos_poly = int(poly_p_str)
        if not (1 <= smooth_pos_poly < smooth_pos_window): raise ValueError("位置平滑阶数错误")
        baseline_sec_str = input(f"  基准线窗口时长 (秒, 默认: {DEFAULT_BASELINE_WINDOW_SEC:.1f}s): ").strip()
        if baseline_sec_str: baseline_window_sec = float(baseline_sec_str)
        if baseline_window_sec <= 0: raise ValueError("基准线窗口时长错误")
        win_k_str = input(f"  X轴运动学窗口 (奇数>=3, 默认: {DEFAULT_KINEMATICS_WINDOW}): ").strip()
        if win_k_str: kinematics_window = int(win_k_str)
        if kinematics_window < 3 or kinematics_window % 2 == 0: raise ValueError("运动学窗口错误")
        poly_k_str = input(f"  X轴运动学阶数 (>=1/<win, 默认: {DEFAULT_KINEMATICS_POLY}): ").strip()
        if poly_k_str: kinematics_poly = int(poly_k_str)
        if not (1 <= kinematics_poly < kinematics_window): raise ValueError("运动学阶数错误")
    except (KeyboardInterrupt, EOFError): print("\n操作取消。"); sys.exit(1)
    except ValueError as e:
        print(f"输入错误 ({e})，使用默认参数。")
        smooth_pos_window=DEFAULT_SMOOTH_POS_WINDOW; smooth_pos_poly=DEFAULT_SMOOTH_POS_POLY; baseline_window_sec=DEFAULT_BASELINE_WINDOW_SEC
        kinematics_window=DEFAULT_KINEMATICS_WINDOW; kinematics_poly=DEFAULT_KINEMATICS_POLY
    print(f"位置平滑: 窗口={smooth_pos_window}, 阶数={smooth_pos_poly}")
    print(f"基准线窗口: {baseline_window_sec:.1f} 秒")
    print(f"X轴运动学计算: 窗口={kinematics_window}, 阶数={kinematics_poly}")

    # --- 核心处理步骤 ---
    print("\n--- 正在加载和预处理数据 ---")
    final_thresh_pa = DEFAULT_DEVIATION_X_PA_THRESHOLD; final_thresh_move = DEFAULT_DEVIATION_X_MOVE_THRESHOLD
    analyzed_events = []; analysis_successful = False; coord_units_final = "像素"

    try:
        # 1. 加载 2D 数据
        positions_2d, valid_flags, total_frames, coord_units_final = load_and_calculate_tracked_point_2d(
            input_json_path, CENTROID_KP_INDICES, LOAD_MODE, min_confidence)

        # 检查有效帧
        min_req_frames = max(10, smooth_pos_window, int(baseline_window_sec * fps * 1.1) if fps > 0 else 10, kinematics_window)
        if np.sum(valid_flags) < min_req_frames + 1: raise ValueError(f"有效帧数({np.sum(valid_flags)})不足(需>{min_req_frames})。")

        # 2. 插值
        interpolated_positions_2d = interpolate_invalid_frames_2d(positions_2d, valid_flags)

        # 3. 平滑 2D 位置
        positions_smoothed_2d = smooth_positions_2d(interpolated_positions_2d, smooth_pos_window, smooth_pos_poly)
        positions_x_smooth = positions_smoothed_2d[:, 0] # 提取平滑后的 X 坐标

        # 4. 计算 X 轴基准线和偏差距离
        print("计算X轴基准线和偏差距离...")
        baseline_x, deviation_distance_x = calculate_baseline_and_deviation_x(
            positions_x_smooth, fps, baseline_window_sec
        )
        if np.all(np.isnan(deviation_distance_x)): raise ValueError("所有计算出的X轴偏差距离均为无效值！")

        # 5. 计算 X 轴运动学
        print("计算X轴运动学参数...")
        vx, ax = compute_kinematics_x(positions_x_smooth, fps, kinematics_window, kinematics_poly)

        # 6. 计算 X 轴帧间位移绝对值 (用于路径长度)
        displacements_x_abs = compute_frame_displacements_x(positions_x_smooth)

        # 7. 推荐 X 轴偏差阈值
        print(f"\n--- 推荐X轴偏差距离阈值 ({coord_units_final}) ---")
        threshold_plot_fig = None
        rec_thresh_pa, rec_thresh_move = final_thresh_pa, final_thresh_move # 使用默认值初始化推荐值
        try:
            rec_thresh_pa, rec_thresh_move, threshold_analysis_dev_x = recommend_deviation_thresholds_x(
                deviation_distance_x, coord_units=coord_units_final, method='gmm'
            )
            if threshold_analysis_dev_x:
                threshold_plot_fig = plot_deviation_threshold_analysis_x(
                    deviation_distance_x, threshold_analysis_dev_x, coord_units_final)
                if threshold_plot_fig:
                    plot_path = output_dir / f"{input_json_path.stem}_deviationX_threshold_analysis.png"
                    try: threshold_plot_fig.savefig(plot_path, dpi=100); print(f"  X轴偏差阈值分析图已保存: {plot_path}")
                    except Exception as e: print(f"  警告：保存X轴偏差阈值图失败: {e}")
                    plt.close(threshold_plot_fig)
                else: print("  未能生成X轴偏差阈值分析图。")
            else: print("  未能生成阈值分析数据，将使用默认阈值。")
        except Exception as e: print(f"  错误：X轴偏差阈值推荐时出错: {e}")

        # 8. 用户确认阈值
        print(f"\n推荐阈值: PA ≈ {rec_thresh_pa:.2f}, Move ≈ {rec_thresh_move:.2f} {coord_units_final}")
        try:
            pa_input = input(f"  输入最终 PA 阈值 ({coord_units_final}, 回车用推荐值): ").strip()
            final_thresh_pa = float(pa_input) if pa_input else rec_thresh_pa
            move_input = input(f"  输入最终 Move 阈值 ({coord_units_final}, 回车用推荐值): ").strip()
            final_thresh_move = float(move_input) if move_input else rec_thresh_move
            if final_thresh_pa < 0 or final_thresh_move <= 0: raise ValueError("阈值需>0")
            if final_thresh_move <= final_thresh_pa: raise ValueError("Move阈值需>PA阈值")
        except (KeyboardInterrupt, EOFError): print("\n操作取消。"); sys.exit(1)
        except ValueError as e:
            print(f"  输入错误 ({e})，使用推荐/默认阈值。")
            final_thresh_pa = max(0, rec_thresh_pa); final_thresh_move = max(final_thresh_pa * 1.1, rec_thresh_move)
        print(f"最终使用阈值: PA = {final_thresh_pa:.2f}, Move = {final_thresh_move:.2f} {coord_units_final}")

        # 9. 获取事件时间约束
        print("\n设置事件时间约束 (单位: 秒)")
        min_dur_sec=DEFAULT_MIN_DURATION_SEC; max_dur_sec=DEFAULT_MAX_DURATION_SEC; merge_gap_sec=DEFAULT_MERGE_GAP_SEC
        try:
            min_dur_input = input(f"  最短持续时间 (默认 {DEFAULT_MIN_DURATION_SEC:.2f}s): ").strip()
            if min_dur_input: min_dur_sec = float(min_dur_input)
            max_dur_input = input(f"  最长持续时间 (默认 {DEFAULT_MAX_DURATION_SEC:.1f}s): ").strip()
            if max_dur_input: max_dur_sec = float(max_dur_input)
            merge_gap_input = input(f"  合并间隔 (默认 {DEFAULT_MERGE_GAP_SEC:.2f}s): ").strip()
            if merge_gap_input: merge_gap_sec = float(merge_gap_input)
            if min_dur_sec < 0 or max_dur_sec < 0 or merge_gap_sec < 0: raise ValueError("时间不能<0")
        except (KeyboardInterrupt, EOFError): print("\n操作取消。"); sys.exit(1)
        except ValueError as e: print(f"输入错误 ({e})，用默认值。"); min_dur_sec=DEFAULT_MIN_DURATION_SEC; max_dur_sec=DEFAULT_MAX_DURATION_SEC; merge_gap_sec=DEFAULT_MERGE_GAP_SEC
        min_df = max(1, int(min_dur_sec * fps)) if fps > 0 else 1
        max_df_frames = int(max_dur_sec * fps) if max_dur_sec > 0 and fps > 0 else None
        min_gf = int(merge_gap_sec * fps) if fps > 0 else 0
        print(f"约束 (帧): MinDur={min_df}, MaxDur={'无限制' if max_df_frames is None else max_df_frames}, MergeGap={min_gf}")

        # 10. 事件检测 (基于 X 轴偏差距离)
        print(f"\n使用阈值 PA={final_thresh_pa:.2f}, Move={final_thresh_move:.2f} 检测事件...")
        candidate_events = detect_events_deviation_multi_threshold_x(
            deviation_distance_x, final_thresh_pa, final_thresh_move,
            min_duration_frames=min_df, max_duration_frames=max_df_frames, min_gap_frames=min_gf
        )

        # 11. 事件分析与分类 (基于 X 轴偏差分类, 分析 X 轴运动学)
        print("分析和分类检测到的事件...")
        analyzed_events = analyze_and_classify_events_deviation_x(
             candidate_events,
             deviation_distance_x, final_thresh_move, # 用于分类
             displacements_x_abs, # 用于X轴路径长度
             positions_x_smooth, vx, ax, # X轴运动学数据
             fps, coord_units_final
        )

        # 12. 显示 & 导出结果
        if analyzed_events:
            analyzed_events.sort(key=lambda e: e['start_frame'])
            count_pa = sum(1 for e in analyzed_events if e['event_type'] == 'Pose Adjustment'); count_move = sum(1 for e in analyzed_events if e['event_type'] == 'Movement')
            print(f"\n检测到 {len(analyzed_events)} 个事件 ({count_pa} Pose Adjustment, {count_move} Movement):")
            for i, ev in enumerate(analyzed_events[:10]):
                 print(f"  事件 {i+1} [{ev['event_type']}]: {ev['start_time_str']}-{ev['end_time_str']} ({ev['duration_sec']:.2f}s), "
                       f"峰值X偏≈{ev['peak_deviation_distance_x']:.1f}, X路径≈{ev['path_length_x']:.1f}{coord_units_final}")
            if len(analyzed_events) > 10: print("  ...")

            csv_filename_template = '{}_events_{}_devX_pa{:.1f}_mv{:.1f}.csv' # 更新文件名
            csv_fn = csv_filename_template.format(input_json_path.stem, TRACKING_POINT_MODE, final_thresh_pa, final_thresh_move)
            csv_path = output_dir / csv_fn
            export_path = export_events_to_csv_x(analyzed_events, csv_path) # 使用X轴导出函数
            if not export_path: print("  警告：CSV 导出失败。")
        else: print("\n未检测到符合条件的事件。")

        # 13. 最终绘图 (X轴版本)
        create_plots = True; analysis_plot_fig = None
        try: plot_choice = input("\n是否生成并显示最终分析图? (Y/n): ").strip().lower();
        except (KeyboardInterrupt, EOFError): print("\n操作取消。"); sys.exit(1)
        if plot_choice == 'n': create_plots = False

        if create_plots:
            print("生成最终分析图...")
            analysis_plot_fig = plot_deviation_analysis_x( # 使用X轴绘图函数
                deviation_distance_x=deviation_distance_x,
                positions_x=positions_x_smooth, # 绘图用平滑X坐标
                baseline_x=baseline_x,          # 传递X基准线
                events=analyzed_events,
                threshold_pose_adj_dev_x=final_thresh_pa,
                threshold_movement_dev_x=final_thresh_move,
                fps=fps, coord_units=coord_units_final,
                title_prefix=f"{input_json_path.stem}\n"
            )
            if analysis_plot_fig:
                plot_path = output_dir / f"{input_json_path.stem}_deviationX_analysis_final.png"
                try:
                    analysis_plot_fig.savefig(plot_path, dpi=150); print(f"最终分析图已保存: {plot_path}"); print("提示：关闭图形窗口后结束。"); plt.show()
                except Exception as e: print(f"错误：保存或显示图失败: {e}")
                finally: plt.close(analysis_plot_fig)
            else: print("未能生成最终分析图。")
        else: print("已跳过生成最终分析图。")
        analysis_successful = True

    # --- 统一的错误处理 ---
    except FileNotFoundError as e: print(f"\n错误: {e}")
    except ValueError as e: print(f"\n处理错误: {e}")
    except Exception as e: print(f"\n分析中发生意外错误: {e}"); import traceback; traceback.print_exc()
    finally:
        plt.close('all')
        num_events = len(analyzed_events) if 'analyzed_events' in locals() else 0
        peak_dev_overall = 0.0
        if analysis_successful and 'deviation_distance_x' in locals() and deviation_distance_x is not None:
            try: valid_dev = deviation_distance_x[~np.isnan(deviation_distance_x)];
            except NameError: valid_dev = np.array([]) # deviation_distance_x 可能未定义
            if valid_dev.size > 0: peak_dev_overall = np.nanmax(valid_dev)

        thresh_pa_final = final_thresh_pa if 'final_thresh_pa' in locals() and final_thresh_pa is not None else DEFAULT_DEVIATION_X_PA_THRESHOLD
        thresh_move_final = final_thresh_move if 'final_thresh_move' in locals() and final_thresh_move is not None else DEFAULT_DEVIATION_X_MOVE_THRESHOLD
        results_summary = (num_events, thresh_pa_final, thresh_move_final, peak_dev_overall)
        units_str = coord_units_final

        print(f"\n--- 分析流程结束 ({input_json_path.name if input_json_path else '未知'}) ---")
        print(f"总结: 基于 {CENTROID_NAME} X轴与基准线偏差检测到 {results_summary[0]} 个事件。")
        print(f"  使用阈值: PA ≈ {results_summary[1]:.2f} {units_str}, Move ≈ {results_summary[2]:.2f} {units_str}")
        print(f"  最大X轴偏差距离约: {results_summary[3]:.2f} {units_str}")
        sys.exit(0)