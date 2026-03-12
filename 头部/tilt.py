'''
Author: Martinwang96 -git
Date: 2025-04-13 13:48:17
Contact: martingwang01@163.com
LONG LIVE McDonald's
Copyright (c) 2025 by Martin Wang in Language of Sciences, Shanghai International Studies University, All Rights Reserved. 
'''

import pandas as pd
import numpy as np
from scipy import signal
from sklearn.cluster import KMeans
# Removed matplotlib and seaborn imports
import os
from datetime import datetime
import tkinter as tk # Optional for file dialog
from tkinter import filedialog # Optional for file dialog
# Note: No plotting libraries are imported or used.

# ==============================================================
# 1. 加载和准备数据 (Adapted for Roll - pose_Rz)
# ==============================================================
def load_and_prepare_data(file_path):
    """Loads OpenFace CSV data, prepares roll angle (pose_Rz), and calculates frame rate."""
    print(f"尝试从以下路径加载数据: {file_path}")
    try:
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip()
        print("CSV 文件加载成功。")
    except FileNotFoundError:
        print(f"错误: 文件未找到 {file_path}")
        return None, None
    except Exception as e:
        print(f"加载 CSV 文件时出错: {e}")
        return None, None

    # Check for required Roll column
    required_col = 'pose_Rz'
    if required_col not in df.columns:
        print(f"错误: CSV中未找到必需的列 '{required_col}'。")
        print(f"可用列: {list(df.columns)}")
        return None, None

    # Frame rate calculation (same as yaw code)
    frame_rate = 30
    if 'timestamp' in df.columns and len(df) > 1 and df['timestamp'].nunique() > 1:
        df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
        df = df.dropna(subset=['timestamp']).sort_values(by='timestamp')
        if len(df) > 1: # 确保排序和去NaN后仍有足够数据
            avg_time_diff = np.mean(np.diff(df['timestamp']))
            if avg_time_diff > 0:
                calculated_rate = 1 / avg_time_diff
                if 5 < calculated_rate < 200:
                     frame_rate = calculated_rate
                     print(f"计算得到的帧率: {frame_rate:.2f} FPS")
                else:
                     print(f"警告: 计算得到的帧率 ({calculated_rate:.2f}) 异常。使用默认 30 FPS。")
            else:
                print("警告: 无法从时间戳计算帧率 (时间差非正)。使用默认 30 FPS。")
        else:
             print("警告: 时间戳数据不足以计算帧率。使用默认 30 FPS。")
    else:
        print("警告: 未找到 'timestamp' 列或数据不足以计算帧率。使用默认 30 FPS。")

    # Roll angle conversion (pose_Rz is typically radians)
    # Assuming pose_Rz is in radians, convert to degrees.
    # Add check similar to yaw, though degrees for roll are less common directly in OpenFace
    if abs(df[required_col]).max() < np.pi * 2:
         print(f"假设 '{required_col}' 是弧度单位, 转换为角度。")
         df['roll_deg'] = df[required_col] * 180 / np.pi
    else:
         print(f"警告: '{required_col}' 值域较大，假设已经是角度单位。")
         df['roll_deg'] = df[required_col]

    df = df.reset_index(drop=True)
    df['frame'] = df.index
    df['second'] = (df['frame'] / frame_rate).round(3)

    # Filter by 'success' column if exists (same as yaw code)
    if 'success' in df.columns:
        initial_rows = len(df)
        df = df[df['success'] == 1].copy()
        removed_rows = initial_rows - len(df)
        if removed_rows > 0:
            print(f"根据 'success' 列过滤掉了 {removed_rows} 帧 (success != 1)。")

    df = df.dropna(subset=['roll_deg'])
    if len(df) == 0:
        print("错误: 处理后无有效数据。")
        return None, None

    print(f"数据加载和准备完成。有效总帧数: {len(df)}")
    return df, frame_rate

# ==============================================================
# 2. 描述性统计 (Copied directly)
# ==============================================================
def compute_statistics(df, column='roll_deg'):
    """Calculates basic statistics for a given column."""
    if column not in df.columns:
        print(f"错误: 用于统计的列 '{column}' 未找到。")
        return None
    if df[column].isnull().all():
        print(f"警告: 列 '{column}' 只包含空值。")
        return None

    stats_dict = {
        '平均值': df[column].mean(),
        '标准差': df[column].std(),
        '最小值': df[column].min(),
        '25%分位数': df[column].quantile(0.25),
        '中位数': df[column].median(),
        '75%分位数': df[column].quantile(0.75),
        '最大值': df[column].max(),
        '范围': df[column].max() - df[column].min(),
        '偏度': df[column].skew(),
        '峰度': df[column].kurt()
    }
    return stats_dict

# ==============================================================
# 3. Savitzky-Golay 滤波器 (Copied directly)
# ==============================================================
def apply_savgol_filter(df, column='roll_deg', window_length=11, polyorder=2):
    """Applies Savitzky-Golay filter to smooth data."""
    if column not in df.columns:
        print(f"错误: 用于 SavGol 滤波的列 '{column}' 未找到。")
        return df

    filtered_df = df.copy()
    filtered_col_name = f'{column}_filtered'
    valid_data = df[column].dropna()

    if len(valid_data) <= window_length:
        print(f"警告: 数据长度 ({len(valid_data)}) 过短，无法应用窗口大小 ({window_length})。跳过 SavGol 滤波。")
        filtered_df[filtered_col_name] = df[column]
        return filtered_df
    if window_length % 2 == 0:
        window_length += 1
        print(f"调整 SavGol 窗口长度为奇数: {window_length}")

    try:
        filtered_data = signal.savgol_filter(valid_data.values, window_length, polyorder)
        filtered_series = pd.Series(filtered_data, index=valid_data.index)
        filtered_df[filtered_col_name] = filtered_series
        print(f"已对 '{column}' 应用 Savitzky-Golay 滤波器 (窗口={window_length}, 阶数={polyorder})。结果在 '{filtered_col_name}'。")
    except Exception as e:
        print(f"应用 SavGol 滤波器时出错: {e}。在新列中返回原始数据。")
        filtered_df[filtered_col_name] = df[column]
    return filtered_df

# ==============================================================
# 4. 时间一致性检查 (可选的异常值校正) (Copied directly)
# ==============================================================
def apply_temporal_consistency(df, column='roll_deg_filtered', window_size=5, std_multiplier=3.0):
    """Applies temporal consistency check to correct potential outliers."""
    if column not in df.columns:
        print(f"错误: 用于时间一致性检查的列 '{column}' 未找到。")
        return df

    corrected_df = df.copy()
    corrected_col_name = f'{column}_corrected'
    outlier_col_name = f'{column}_is_outlier' # Keep track of outliers

    corrected_df[corrected_col_name] = df[column]
    corrected_df[outlier_col_name] = False

    if len(df) < window_size:
        print(f"警告: 数据长度 ({len(df)}) 小于窗口大小 ({window_size})。跳过时间一致性检查。")
        return corrected_df

    num_corrected = 0
    # Use rolling window for efficiency if possible, otherwise loop (using loop for consistency)
    for i in range(window_size, len(df)):
        window_indices = df.index[i-window_size : i]
        current_index = df.index[i]
        window_data = df.loc[window_indices, column].dropna()
        current_value = df.loc[current_index, column]

        if pd.isna(current_value) or window_data.empty:
            continue

        window_mean = np.mean(window_data.values)
        window_std = np.std(window_data.values)
        tolerance = 0.1 # degrees tolerance for low std dev

        if window_std > 1e-6: # Check if std dev is meaningful
            lower_bound = window_mean - std_multiplier * window_std
            upper_bound = window_mean + std_multiplier * window_std
        else: # If std dev is tiny, use a fixed tolerance around the mean
            lower_bound = window_mean - tolerance
            upper_bound = window_mean + tolerance

        if current_value < lower_bound or current_value > upper_bound:
            corrected_df.loc[current_index, outlier_col_name] = True
            # Replace outlier with the window mean
            corrected_df.loc[current_index, corrected_col_name] = window_mean
            num_corrected += 1

    if num_corrected > 0:
        print(f"已对 '{column}' 应用时间一致性检查。校正了 {num_corrected} 个潜在异常值。结果在 '{corrected_col_name}'。")
    else:
        print(f"已对 '{column}' 应用时间一致性检查。未检测到或校正异常值。结果在 '{corrected_col_name}'。")
    return corrected_df

# ==============================================================
# 5. 确定倾斜阈值 (Adapted from Yaw version)
# ==============================================================
def determine_tilt_thresholds(df, column='roll_deg_filtered_corrected', n_clusters=2):
    """Calculates head tilt (Roll) thresholds using statistics and optional clustering."""
    if column not in df.columns:
        print(f"错误: 用于阈值计算的列 '{column}' 未找到。")
        return None
    if df[column].isnull().all():
         print(f"错误: 列 '{column}' 全为空。无法计算阈值。")
         return None

    data_series = df[column].dropna()
    if len(data_series) < 10:
        print(f"警告: '{column}' 中的数据点 ({len(data_series)}) 不足，无法进行稳健的阈值计算。结果可能不可靠。")
        if len(data_series) == 0: return None

    # For Roll (tilt), the mean is often close to 0 (neutral head position).
    # We are interested in the *deviation* from this neutral position.
    # Instead of mean +/- delta, we might consider thresholds directly based on deviations.
    # However, keeping the mean +/- delta structure for consistency with yaw code.
    mean_val = data_series.mean()
    std_val = data_series.std()
    if std_val < 1e-6: # Handle case with no variation
        print(f"警告: 列 '{column}' 的标准差接近 0。将使用小的固定偏移量作为阈值。")
        std_val = 1.0 # Assign a small default std dev for calculation

    # Method 1: Statistical Threshold (deviation based on std dev)
    statistical_threshold_delta = 1.5 * std_val # Reduced multiplier for tilt? Or keep 2? Let's try 1.5
                                               # Tilt might have smaller natural std dev than yaw.

    # Method 2: Clustering Threshold (find clusters like 'neutral', 'tilted left', 'tilted right')
    # For tilt, 3 clusters might be more appropriate if significant left/right tilt exists.
    # Let's stick to 2 for simplicity, aiming to separate 'neutral' from 'tilted'.
    cluster_threshold_delta = statistical_threshold_delta # Default
    if len(data_series) >= n_clusters * 5:
        try:
            data_for_clustering = data_series.values.reshape(-1, 1)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
            clusters = kmeans.fit_predict(data_for_clustering)
            centers = kmeans.cluster_centers_.flatten()
            centers.sort()
            if len(centers) >= 2:
                 # Delta could be half the distance between centers, or distance from mean to outer center
                 # Let's use half distance between centers for symmetry.
                 cluster_threshold_delta = abs(centers[0] - centers[-1]) / 2
                 print(f"  - 聚类分析建议的中心点大约在: {centers}")
            else:
                 print("  - 聚类分析产生的中心点少于 2 个，使用统计 Delta。")
        except Exception as e:
            print(f"  - 聚类分析失败: {e}。使用统计 Delta。")
    else:
        print(f"  - 数据不足 ({len(data_series)}) 进行 {n_clusters}-均值聚类，使用统计 Delta。")

    # Method 3: IQR Threshold
    Q1 = data_series.quantile(0.25)
    Q3 = data_series.quantile(0.75)
    IQR = Q3 - Q1
    iqr_threshold_delta = 1.5 * IQR if IQR > 1e-6 else statistical_threshold_delta * 0.75 # Fallback if IQR is zero

    # Method 4: Weighted Average Threshold
    # Adjust weights if needed, e.g., give more weight to stats/IQR for tilt?
    weights = {'statistical': 0.4, 'cluster': 0.3, 'iqr': 0.3}
    weighted_threshold_delta = (weights['statistical'] * statistical_threshold_delta +
                                weights['cluster'] * cluster_threshold_delta +
                                weights['iqr'] * iqr_threshold_delta)

    print("\n--- 倾斜 (Roll) 阈值计算 ---")
    print(f"使用来自列的数据: '{column}'")
    print(f"基准平均角度: {mean_val:.2f}°")
    print("-" * 20)
    print(f"1. 统计 Delta (±1.5σ): {statistical_threshold_delta:.2f}°") # Adjusted description
    print(f"2. 聚类 Delta (或备用值): {cluster_threshold_delta:.2f}°")
    print(f"3. IQR Delta (±1.5*IQR): {iqr_threshold_delta:.2f}°")
    print("-" * 20)
    print(f"【综合加权 Delta】: {weighted_threshold_delta:.2f}°")

    # Define thresholds relative to the mean
    final_upper_threshold = mean_val + weighted_threshold_delta
    final_lower_threshold = mean_val - weighted_threshold_delta
    print(f"【最终建议阈值】: Roll < {final_lower_threshold:.2f}° (左倾) 或 Roll > {final_upper_threshold:.2f}° (右倾)")
    print("-" * 20)

    return {
        'mean': mean_val,
        'upper_threshold': final_upper_threshold, # Threshold for right tilt
        'lower_threshold': final_lower_threshold, # Threshold for left tilt
        'weighted_delta': weighted_threshold_delta,
        'statistical_delta': statistical_threshold_delta,
        'cluster_delta': cluster_threshold_delta,
        'iqr_delta': iqr_threshold_delta,
    }

# ==============================================================
# 6. 检测倾斜区间 (Adapted from Yaw version)
# ==============================================================
def detect_tilt_intervals(df, column, upper_thresh, lower_thresh, mean_roll, min_frames, max_frames, frame_rate):
    """
    检测头部倾斜角 (Roll) 超出阈值并满足持续时间标准的连续区间。
    增加返回每个区间的方向 ('left' 或 'right')。
    """
    if column not in df.columns:
        print(f"错误: 用于区间检测的列 '{column}' 未找到。")
        return []
    if df[column].isnull().all():
        print(f"警告: 列 '{column}' 全为空。未检测到区间。")
        return []
    if 'frame' not in df.columns or 'second' not in df.columns:
         print("错误: 'frame' 或 'second' 列缺失。无法计算区间。")
         return []

    intervals = []
    in_interval = False
    start_idx = None
    current_direction = None # Track direction of the current interval
    temp_df = df.reset_index(drop=True) # Use reset index for sequential access

    for idx in range(len(temp_df)):
        current_value = temp_df.loc[idx, column]
        current_frame = temp_df.loc[idx, 'frame']

        # Determine if current value is outside thresholds
        is_outside_upper = pd.notna(current_value) and current_value > upper_thresh # Right tilt
        is_outside_lower = pd.notna(current_value) and current_value < lower_thresh # Left tilt
        is_outside = is_outside_upper or is_outside_lower

        if is_outside and not in_interval:
            # Start a potential interval
            in_interval = True
            start_idx = idx
            # Determine initial direction based on which threshold was crossed
            current_direction = 'right' if is_outside_upper else 'left'

        elif in_interval:
            # Continue or end the interval
            current_duration_frames = idx - start_idx + 1

            # End conditions:
            # 1. Value goes back within the *original* direction's threshold band
            #    (e.g., if started as 'right', ends if value <= upper_thresh)
            # 2. Reaches max duration
            # 3. Reaches end of data
            # 4. *Crucially for tilt*: Switches direction (crosses mean significantly)
            #    We treat a continuous L->R or R->L tilt as potentially two events, split by returning to neutral zone.

            end_loop = (idx == len(temp_df) - 1)
            interval_should_end = False

            # Condition 1: No longer outside the threshold for the *current* interval's direction
            if current_direction == 'right' and not is_outside_upper:
                interval_should_end = True
            elif current_direction == 'left' and not is_outside_lower:
                 interval_should_end = True

            # Condition 2: Max duration exceeded
            if current_duration_frames > max_frames:
                 interval_should_end = True

            # Condition 3: End of data
            if end_loop:
                 interval_should_end = True

            # Optional Condition 4: Switched side (e.g., crossed mean significantly) - handled by condition 1 mostly

            if interval_should_end:
                 # Determine the actual end frame
                 # If ended because value returned towards mean, end frame is the previous one.
                 # If ended due to max duration or end of loop while still tilted, end frame is current one.
                 end_idx = idx -1 # Default assumes value returned to neutral
                 if (current_direction == 'right' and is_outside_upper and (current_duration_frames > max_frames or end_loop)) or \
                    (current_direction == 'left' and is_outside_lower and (current_duration_frames > max_frames or end_loop)):
                      end_idx = idx # Ended due to time/end while still tilted

                 # Ensure end_idx is not before start_idx (handles short intervals)
                 if end_idx < start_idx:
                     end_idx = start_idx

                 interval_duration_frames = end_idx - start_idx + 1

                 # Check if valid duration
                 if min_frames <= interval_duration_frames <= max_frames:
                     interval_data = temp_df.iloc[start_idx:end_idx+1][column]
                     if not interval_data.empty:
                         start_time = temp_df.iloc[start_idx]['second']
                         end_time = temp_df.iloc[end_idx]['second']
                         start_frame = temp_df.iloc[start_idx]['frame']
                         end_frame = temp_df.iloc[end_idx]['frame']
                         duration_sec = (interval_duration_frames / frame_rate)
                         avg_angle = interval_data.mean()
                         # Max/Min absolute angle might be more meaningful for tilt sometimes
                         max_abs_angle = interval_data.abs().max()
                         # Final direction based on the average angle during the interval
                         interval_final_direction = 'right' if avg_angle > mean_roll else 'left'
                         # Use the originally detected direction for consistency? Or the final average? Let's use final avg.
                         # interval_final_direction = current_direction # Use initial direction

                         intervals.append({
                             'start_frame': start_frame,
                             'end_frame': end_frame,
                             'start_time': start_time,
                             'end_time': end_time,
                             'duration_frames': interval_duration_frames,
                             'duration_seconds': round(duration_sec, 3),
                             'max_angle': round(interval_data.max(), 2), # Furthest right tilt
                             'min_angle': round(interval_data.min(), 2), # Furthest left tilt
                             'avg_angle': round(avg_angle, 2),
                             'max_abs_angle': round(max_abs_angle, 2), # Max deviation from 0
                             'direction': interval_final_direction # 'left' or 'right'
                         })

                 # Reset for next potential interval
                 in_interval = False
                 start_idx = None
                 current_direction = None

                 # Handle case where the current point *starts* a new interval immediately after ending one
                 # (e.g., due to max_frames ending, but current point is still tilted)
                 # Let the next loop iteration handle this naturally by checking `is_outside` again.

    print(f"\n检测到 {len(intervals)} 个满足条件的 *单个* 倾斜区间。")
    return intervals


# ==============================================================
# 7. 生成报告 (Adapted from Yaw version, removed Shake part)
# ==============================================================
def generate_report(tilt_intervals, output_dir="头部倾斜分析结果"):
    """Generates console output and saves CSV report for head tilts."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # --- 倾斜报告 ---
    tilt_report_path = os.path.join(output_dir, f"倾斜事件报告_{timestamp}.csv")
    if not tilt_intervals:
        print("\n未检测到满足条件的倾斜事件。")
        # Optionally create an empty file or just skip
        try: # Create empty file for consistency maybe?
            with open(tilt_report_path, 'w') as f:
                f.write('事件编号,类型,开始时间(秒),结束时间(秒),持续时间(秒),持续时间(帧),方向,平均角度(度),最小角度(度),最大角度(度),最大绝对角度(度),开始帧,结束帧\n')
            print(f"已创建空的倾斜事件报告: {tilt_report_path}")
        except Exception as e:
            print(f"创建空报告时出错: {e}")
    else:
        print("\n--- 检测到的倾斜事件 ---")
        tilt_report_data = []
        for i, tilt in enumerate(tilt_intervals):
            direction_zh = "向右" if tilt['direction'] == 'right' else "向左"
            print(f"倾斜事件 {i+1}:")
            print(f"  时间: {tilt['start_time']:.2f}s - {tilt['end_time']:.2f}s")
            print(f"  持续时间: {tilt['duration_seconds']:.2f}s ({tilt['duration_frames']} 帧)")
            print(f"  方向: {direction_zh} (平均角度: {tilt['avg_angle']:.1f}°)")
            print(f"  角度范围: Min={tilt['min_angle']:.1f}°, Max={tilt['max_angle']:.1f}° (Max Abs: {tilt['max_abs_angle']:.1f}°)")
            print("-" * 10)

            tilt_report_data.append({
                '事件编号': i + 1,
                '类型': '倾斜',
                '开始时间(秒)': tilt['start_time'],
                '结束时间(秒)': tilt['end_time'],
                '持续时间(秒)': tilt['duration_seconds'],
                '持续时间(帧)': tilt['duration_frames'],
                '方向': direction_zh,
                '平均角度(度)': tilt['avg_angle'],
                '最小角度(度)': tilt['min_angle'], # Furthest left
                '最大角度(度)': tilt['max_angle'], # Furthest right
                '最大绝对角度(度)': tilt['max_abs_angle'], # Max deviation
                '开始帧': tilt['start_frame'],
                '结束帧': tilt['end_frame'],
            })

        try:
            tilt_report_df = pd.DataFrame(tilt_report_data)
            # Define column order for clarity
            column_order = ['事件编号', '类型', '开始时间(秒)', '结束时间(秒)', '持续时间(秒)',
                            '持续时间(帧)', '方向', '平均角度(度)', '最小角度(度)',
                            '最大角度(度)', '最大绝对角度(度)', '开始帧', '结束帧']
            tilt_report_df = tilt_report_df[column_order]
            tilt_report_df.to_csv(tilt_report_path, index=False, encoding='utf-8-sig')
            print(f"\n倾斜事件报告已保存至: {tilt_report_path}")
        except Exception as e:
            print(f"\n保存倾斜事件报告 CSV 时出错: {e}")


# ==============================================================
# 8. 主分析函数 (Adapted from Yaw version)
# ==============================================================
def analyze_head_tilt(file_path):
    """主函数，运行头部倾斜 (Roll) 分析流程。"""
    print("=" * 50)
    print(" 开始头部倾斜 (Roll) 分析 ")
    print("=" * 50)

    # --- 1. 加载数据 ---
    df, frame_rate = load_and_prepare_data(file_path)
    if df is None:
        return

    # --- 2. 原始数据统计 ---
    print("\n--- 原始 roll_deg 基本统计 ---")
    stats_original = compute_statistics(df, 'roll_deg')
    if stats_original:
        for key, value in stats_original.items():
            print(f"  {key}: {value:.3f}")
    else:
        print("无法计算原始数据的统计信息。")


    # --- 3. 数据平滑与校正 ---
    # Adjust window length based on frame rate, e.g., 0.3-0.5 seconds for smoothing
    window_len_sec = 0.4 # Smoothing window
    window_length = int(window_len_sec * frame_rate)
    window_length = max(5, min(len(df) // 10 if len(df) > 50 else 5, window_length)) # Safety checks
    if window_length % 2 == 0: window_length += 1 # Ensure odd

    print(f"\n--- 应用数据平滑 (Savitzky-Golay 滤波器) ---")
    df_filtered = apply_savgol_filter(df, 'roll_deg', window_length=window_length, polyorder=2)
    analysis_col_smooth = 'roll_deg_filtered'

    print(f"\n--- 应用异常值校正 (时间一致性) ---")
    # Adjust consistency window, e.g., 0.1-0.2 seconds
    consistency_window_sec = 0.15
    consistency_window = max(3, int(consistency_window_sec * frame_rate))
    df_corrected = apply_temporal_consistency(df_filtered, analysis_col_smooth, window_size=consistency_window, std_multiplier=3.0)
    analysis_col_final = 'roll_deg_filtered_corrected' # Final column for analysis

    print(f"\n--- 处理后数据 ({analysis_col_final}) 基本统计 ---")
    stats_final = compute_statistics(df_corrected, analysis_col_final)
    if stats_final:
        for key, value in stats_final.items():
            print(f"  {key}: {value:.3f}")
        if stats_original and stats_original.get('标准差', 0) > 1e-6:
             std_reduction = (1 - stats_final.get('标准差', 0) / stats_original['标准差']) * 100
             print(f"  标准差减少率: {std_reduction:.2f}%")
    else:
        print(f"无法计算 {analysis_col_final} 的统计信息。")


    # --- 4. 确定动态阈值 ---
    print("\n--- 确定倾斜检测阈值 ---")
    suggested_thresholds = determine_tilt_thresholds(df_corrected, analysis_col_final)
    if suggested_thresholds is None:
        print("错误: 无法确定头部倾斜阈值。中止分析。")
        return

    mean_roll_angle = suggested_thresholds['mean'] # Save the calculated mean roll

    print("\n【建议阈值】")
    # For tilt, it's often clearer to talk about the absolute deviation threshold
    print(f"  基准平均角度: {suggested_thresholds['mean']:.2f}°")
    print(f"  建议倾斜阈值 (Delta): ±{suggested_thresholds['weighted_delta']:.2f}°")
    print(f"  => 左倾阈值 (Roll <): {suggested_thresholds['lower_threshold']:.2f}°")
    print(f"  => 右倾阈值 (Roll >): {suggested_thresholds['upper_threshold']:.2f}°")

    try:
        # Ask user for the delta or the specific upper/lower thresholds
        # Asking for delta might be simpler for tilt
        user_delta_input = input(f"请输入最终倾斜检测的绝对阈值 Delta (度, 回车保留建议值 {suggested_thresholds['weighted_delta']:.2f}°): ").strip()
        if user_delta_input:
            final_delta = abs(float(user_delta_input)) # Ensure positive delta
            # Recalculate thresholds based on user delta and the *original mean*
            final_upper_threshold = mean_roll_angle + final_delta
            final_lower_threshold = mean_roll_angle - final_delta
            print(f"使用用户输入的 Delta: {final_delta:.2f}°")
        else:
            # Use suggested thresholds directly
            final_upper_threshold = suggested_thresholds['upper_threshold']
            final_lower_threshold = suggested_thresholds['lower_threshold']
            final_delta = suggested_thresholds['weighted_delta'] # Keep track of the delta used
            print("使用建议阈值。")

    except ValueError:
        print("输入阈值无效，将使用建议值。")
        final_upper_threshold = suggested_thresholds['upper_threshold']
        final_lower_threshold = suggested_thresholds['lower_threshold']
        final_delta = suggested_thresholds['weighted_delta']

    final_thresholds = {
        'mean': mean_roll_angle, # Use the calculated mean
        'upper_threshold': final_upper_threshold,
        'lower_threshold': final_lower_threshold,
        'delta': final_delta # Store the effective delta used
    }

    print(f"\n【最终确定的阈值】: Roll < {final_thresholds['lower_threshold']:.2f}° (左倾) 或 Roll > {final_thresholds['upper_threshold']:.2f}° (右倾) 时判定为显著倾斜")


    # --- 5. 获取用户输入: 倾斜持续时间 ---
    print("\n--- 设置 *单个倾斜* 的持续时间标准 ---")
    try:
        # Default durations might be slightly shorter for tilt than yaw?
        min_tilt_duration_sec = float(input(f"输入单个倾斜事件的【最小】持续时间 (秒, 例如 0.25, 默认: 0.25): ") or 0.25)
        max_tilt_duration_sec = float(input(f"输入单个倾斜事件的【最大】持续时间 (秒, 例如 2.5, 默认: 2.5): ") or 2.5)
    except ValueError:
        print("输入无效。使用默认值 (最小: 0.25s, 最大: 2.5s)。")
        min_tilt_duration_sec = 0.25
        max_tilt_duration_sec = 2.5

    min_tilt_frames = max(1, int(min_tilt_duration_sec * frame_rate))
    max_tilt_frames = int(max_tilt_duration_sec * frame_rate)
    print(f"单个倾斜事件的持续时间范围: {min_tilt_duration_sec:.2f}s ({min_tilt_frames} 帧) 到 {max_tilt_duration_sec:.2f}s ({max_tilt_frames} 帧)")

    # --- 6. 检测单个倾斜区间 ---
    print(f"\n--- 使用列 '{analysis_col_final}' 检测单个倾斜区间 ---")
    head_tilt_intervals = detect_tilt_intervals(
        df_corrected,
        analysis_col_final,
        final_thresholds['upper_threshold'],
        final_thresholds['lower_threshold'],
        final_thresholds['mean'], # Pass the mean roll angle
        min_tilt_frames,
        max_tilt_frames,
        frame_rate
    )

    # --- 7. 生成报告 ---
    # Pass only tilt intervals to the report function
    generate_report(head_tilt_intervals)

    print("\n=" * 5)
    print(" 倾斜分析完成 ")
    print("=" * 5)

# ==============================================================
# 9. 主执行块 (Adapted from Yaw version)
# ==============================================================
if __name__ == "__main__":
    print("\n===== 头部倾斜 (Roll) 检测工具 (动态阈值) =====")

    file_path = input("输入 OpenFace CSV 文件路径 (或按 Enter 浏览): ").strip()

    if not file_path:
        try:
            print("正在打开文件对话框...")
            root = tk.Tk()
            root.withdraw()
            file_path = filedialog.askopenfilename(
                title="选择 OpenFace CSV 文件",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )
            if not file_path:
                print("未选择文件。退出。")
            else:
                 print(f"已选择文件: {file_path}")
            # 关闭隐藏的tk窗口
            root.destroy()
        except ImportError:
            print("Tkinter 未安装或无法用于文件对话框。请手动提供文件路径。")
            file_path = input("输入 OpenFace CSV 文件路径: ").strip()
        except Exception as e:
             print(f"文件选择过程中出错: {e}")
             file_path = "" # Ensure file_path is empty if dialog fails

    if file_path:
        try:
            analyze_head_tilt(file_path) # Call the new tilt analysis function
        except Exception as e:
            print("\n" + "="*2 + " 发生错误 " + "="*2)
            print(f"分析过程中发生意外错误: {e}")
            import traceback
            print("\n--- 错误追踪 ---")
            traceback.print_exc()
            print("="*5)
    else:
        if not file_path: # Check again in case the dialog was cancelled but didn't error
             print("未提供有效的文件路径。退出。")