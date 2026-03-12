'''
Author: Martinwang96 -git
Date: 2025-04-11 14:31:15
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
# No rcParams needed as there's no plotting.

# ==============================================================
# 1. 加载和准备数据 (基本不变)
# ==============================================================
def load_and_prepare_data(file_path):
    """Loads OpenFace CSV data, prepares yaw angle, and calculates frame rate."""
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

    if 'pose_Ry' not in df.columns:
        print("错误: CSV中未找到必需的列 'pose_Ry'。")
        print(f"可用列: {list(df.columns)}")
        return None, None

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

    if abs(df['pose_Ry']).max() < np.pi * 2:
        print("假设 'pose_Ry' 是弧度单位, 转换为角度。")
        df['yaw_deg'] = df['pose_Ry'] * 180 / np.pi
    else:
        print("假设 'pose_Ry' 已经是角度单位。")
        df['yaw_deg'] = df['pose_Ry']
    
    if 'confidence' in df.columns:
        before = len(df)
        df = df[df['confidence'] > 0].copy()
        removed = before - len(df)
        if removed > 0:
            print(f"过滤掉了 {removed} 帧（confidence == 0）。")
    else:
        print("警告：未检测到 'confidence' 列，跳过该过滤。")
    df = df.reset_index(drop=True)
    df['frame'] = df.index
    df['second'] = (df['frame'] / frame_rate).round(3)

    if 'success' in df.columns:
        initial_rows = len(df)
        df = df[df['success'] == 1].copy()
        removed_rows = initial_rows - len(df)
        if removed_rows > 0:
            print(f"根据 'success' 列过滤掉了 {removed_rows} 帧 (success != 1)。")

    df = df.dropna(subset=['yaw_deg'])
    if len(df) == 0:
        print("错误: 处理后无有效数据。")
        return None, None

    print(f"数据加载和准备完成。有效总帧数: {len(df)}")
    return df, frame_rate

# ==============================================================
# 2. 描述性统计 (不变)
# ==============================================================
def compute_statistics(df, column='yaw_deg'):
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
# 3. Savitzky-Golay 滤波器 (不变)
# ==============================================================
def apply_savgol_filter(df, column='yaw_deg', window_length=11, polyorder=2):
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
# 4. 时间一致性检查 (可选的异常值校正) (不变)
# ==============================================================
def apply_temporal_consistency(df, column='yaw_deg_filtered', window_size=5, std_multiplier=3.0):
    """Applies temporal consistency check to correct potential outliers."""
    if column not in df.columns:
        print(f"错误: 用于时间一致性检查的列 '{column}' 未找到。")
        return df

    corrected_df = df.copy()
    corrected_col_name = f'{column}_corrected'
    outlier_col_name = f'{column}_is_outlier'

    corrected_df[corrected_col_name] = df[column]
    corrected_df[outlier_col_name] = False

    if len(df) < window_size:
        print(f"警告: 数据长度 ({len(df)}) 小于窗口大小 ({window_size})。跳过时间一致性检查。")
        return corrected_df

    num_corrected = 0
    # Use rolling window for efficiency if possible, otherwise loop
    # Looping approach for clarity matching original script:
    for i in range(window_size, len(df)):
        window_indices = df.index[i-window_size : i]
        current_index = df.index[i]
        window_data = df.loc[window_indices, column].dropna()
        current_value = df.loc[current_index, column]

        if pd.isna(current_value) or window_data.empty:
            continue

        window_mean = np.mean(window_data.values)
        window_std = np.std(window_data.values)
        tolerance = 0.1 # degrees

        if window_std > 1e-6:
            lower_bound = window_mean - std_multiplier * window_std
            upper_bound = window_mean + std_multiplier * window_std
        else:
            lower_bound = window_mean - tolerance
            upper_bound = window_mean + tolerance

        if current_value < lower_bound or current_value > upper_bound:
            corrected_df.loc[current_index, outlier_col_name] = True
            corrected_df.loc[current_index, corrected_col_name] = window_mean
            num_corrected += 1

    if num_corrected > 0:
        print(f"已对 '{column}' 应用时间一致性检查。校正了 {num_corrected} 个潜在异常值。结果在 '{corrected_col_name}'。")
    else:
        print(f"已对 '{column}' 应用时间一致性检查。未检测到或校正异常值。结果在 '{corrected_col_name}'。")
    return corrected_df

# ==============================================================
# 5. 确定转头阈值 (基本不变)
# ==============================================================
def determine_head_turn_thresholds(df, column='yaw_deg_filtered_corrected', n_clusters=2):
    """Calculates head turn thresholds using statistics and optional clustering."""
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

    mean_val = data_series.mean()
    std_val = data_series.std()
    if std_val == 0: # Handle case with no variation
        print(f"警告: 列 '{column}' 的标准差为 0。将使用小的固定偏移量作为阈值。")
        std_val = 1.0 # Assign a small default std dev

    # Method 1: Statistical Threshold
    statistical_threshold_delta = 2 * std_val

    # Method 2: Clustering Threshold
    cluster_threshold_delta = statistical_threshold_delta # Default
    if len(data_series) >= n_clusters * 5:
        try:
            data_for_clustering = data_series.values.reshape(-1, 1)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto') # Use 'auto' for future compatibility
            clusters = kmeans.fit_predict(data_for_clustering)
            centers = kmeans.cluster_centers_.flatten()
            centers.sort()
            if len(centers) >= 2:
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
    # Handle IQR=0 case
    iqr_threshold_delta = 1.5 * IQR if IQR > 1e-6 else statistical_threshold_delta * 0.75 # Fallback if IQR is zero

    # Method 4: Weighted Average Threshold
    weights = {'statistical': 0.4, 'cluster': 0.4, 'iqr': 0.2}
    weighted_threshold_delta = (weights['statistical'] * statistical_threshold_delta +
                                weights['cluster'] * cluster_threshold_delta +
                                weights['iqr'] * iqr_threshold_delta)

    print("\n--- 转头 (Yaw) 阈值计算 ---")
    print(f"使用来自列的数据: '{column}'")
    print(f"基准平均角度: {mean_val:.2f}°")
    print("-" * 20)
    print(f"1. 统计 Delta (±2σ): {statistical_threshold_delta:.2f}°")
    # print(f"   => 阈值: < {mean_val - statistical_threshold_delta:.2f}° 或 > {mean_val + statistical_threshold_delta:.2f}°")
    print(f"2. 聚类 Delta (或备用值): {cluster_threshold_delta:.2f}°")
    # print(f"   => 阈值: < {mean_val - cluster_threshold_delta:.2f}° 或 > {mean_val + cluster_threshold_delta:.2f}°")
    print(f"3. IQR Delta (±1.5*IQR): {iqr_threshold_delta:.2f}°")
    # print(f"   => 阈值: < {mean_val - iqr_threshold_delta:.2f}° 或 > {mean_val + iqr_threshold_delta:.2f}°")
    print("-" * 20)
    print(f"【综合加权 Delta】: {weighted_threshold_delta:.2f}°")

    final_upper_threshold = mean_val + weighted_threshold_delta
    final_lower_threshold = mean_val - weighted_threshold_delta
    print(f"【最终建议阈值】: Yaw < {final_lower_threshold:.2f}° 或 Yaw > {final_upper_threshold:.2f}°")
    print("-" * 20)

    return {
        'mean': mean_val,
        'upper_threshold': final_upper_threshold,
        'lower_threshold': final_lower_threshold,
        'weighted_delta': weighted_threshold_delta,
        'statistical_delta': statistical_threshold_delta,
        'cluster_delta': cluster_threshold_delta,
        'iqr_delta': iqr_threshold_delta,
    }

# ==============================================================
# 6. 检测转头区间 (修改: 增加方向)
# ==============================================================
def detect_head_turn_intervals(df, column, upper_thresh, lower_thresh, mean_yaw, min_frames, max_frames, frame_rate):
    """
    检测头部偏航角超出阈值并满足持续时间标准的连续区间。
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
    temp_df = df.reset_index(drop=True) # Use reset index to ensure sequential access

    for idx in range(len(temp_df)):
        current_value = temp_df.loc[idx, column]
        current_frame = temp_df.loc[idx, 'frame']

        # 确定当前值是否超出阈值
        is_outside_upper = pd.notna(current_value) and current_value > upper_thresh
        is_outside_lower = pd.notna(current_value) and current_value < lower_thresh
        is_outside = is_outside_upper or is_outside_lower

        if is_outside and not in_interval:
            # 开始一个潜在的区间
            in_interval = True
            start_idx = idx
            # 确定初始方向 (哪个阈值被首先触发)
            current_direction = 'right' if is_outside_upper else 'left'

        elif in_interval:
            # 持续或结束区间
            current_duration_frames = idx - start_idx + 1

            # 结束条件:
            # 1. 值回到阈值内
            # 2. 达到最大持续时间
            # 3. 到达数据序列末尾
            # 4. **方向改变** (如果一个区间跨越了均值从右到左或反之，我们将其视为两个独立的转头事件，在这里通过`not is_outside`处理这种情况)
            end_loop = (idx == len(temp_df) - 1)
            # 如果当前值仍在区间内，检查是否仍在 *同一侧* (虽然严格的阈值检查已足够)
            # still_same_side = (current_direction == 'right' and is_outside_upper) or \
            #                   (current_direction == 'left' and is_outside_lower)

            # 结束区间的逻辑简化：只要当前值不再满足 *最初进入区间时* 的条件，或超时，或到末尾，就结束
            # if not is_outside or current_duration_frames > max_frames or end_loop:
            # 更精确的结束：如果当前点仍在 *某个* 阈值之外，但不是启动区间时的方向，或者回到了阈值内
            interval_should_end = False
            if not is_outside:
                 interval_should_end = True
            elif current_duration_frames > max_frames:
                 interval_should_end = True
            elif end_loop:
                 interval_should_end = True
            # Optional: Check if direction switched side (e.g., went from > upper to < lower directly)
            # This might split very fast shakes, but generally handled by threshold check
            # elif (current_direction == 'right' and is_outside_lower) or \
            #      (current_direction == 'left' and is_outside_upper):
            #      interval_should_end = True # Direction crossed over

            if interval_should_end:
                 # 确定区间的实际结束帧
                 # 如果是因为回到阈值内而结束，则结束帧是前一帧
                 # 如果是因为超时或到末尾而结束，且当前仍在阈值外，则结束帧是当前帧
                 end_idx = idx - 1 if not is_outside else idx
                 # 确保 end_idx 不在 start_idx 之前 (处理单帧区间)
                 if end_idx < start_idx:
                     end_idx = start_idx

                 interval_duration_frames = end_idx - start_idx + 1

                 # 检查持续时间是否有效
                 if min_frames <= interval_duration_frames <= max_frames:
                     interval_data = temp_df.iloc[start_idx:end_idx+1][column]
                     if not interval_data.empty: # 确保区间数据非空
                         start_time = temp_df.iloc[start_idx]['second']
                         end_time = temp_df.iloc[end_idx]['second']
                         start_frame = temp_df.iloc[start_idx]['frame']
                         end_frame = temp_df.iloc[end_idx]['frame']
                         duration_sec = (interval_duration_frames / frame_rate)
                         avg_angle = interval_data.mean()
                         # 重新确认最终方向基于平均值
                         interval_direction = 'right' if avg_angle > mean_yaw else 'left'

                         intervals.append({
                             'start_frame': start_frame,
                             'end_frame': end_frame,
                             'start_time': start_time,
                             'end_time': end_time,
                             'duration_frames': interval_duration_frames,
                             'duration_seconds': round(duration_sec, 3),
                             'max_angle': round(interval_data.max(), 2),
                             'min_angle': round(interval_data.min(), 2),
                             'avg_angle': round(avg_angle, 2),
                             'direction': interval_direction # **新增方向**
                         })

                 # 重置以寻找下一个潜在区间
                 in_interval = False
                 start_idx = None

                 # 特殊处理：如果当前点本身就 *开始* 了一个新的区间（紧随上一个结束之后）
                 # 这种情况发生在 `interval_should_end` 为 True 但 `is_outside` 也为 True 时
                 # (例如，因为超时或方向切换而结束，但当前点仍在阈值外)
                 if is_outside and interval_should_end and not end_loop:
                     # 检查这个新开始的点是否也满足最小帧数（虽然这里只是一帧）
                     # 实际上，我们应该让循环自然地在下一次迭代中处理这个新开始
                     in_interval = True
                     start_idx = idx
                     current_direction = 'right' if is_outside_upper else 'left'

    print(f"\n检测到 {len(intervals)} 个满足条件的 *单个* 转头区间。")
    return intervals

# ==============================================================
# 7. 新增: 检测摇头事件
# ==============================================================
def detect_head_shake_events(turn_intervals, min_shake_turns=4, max_gap_sec=1.0):
    """
    分析单个转头区间列表，识别连续的、方向交替的摇头事件。
    摇头定义为至少 `min_shake_turns` 次（例如 4 次表示 LRLR）方向交替的转头，
    且相邻反向转头之间的间隔不超过 `max_gap_sec` 秒。
    """
    if not turn_intervals or len(turn_intervals) < min_shake_turns:
        print("转头区间不足，无法检测摇头事件。")
        return []

    shake_events = []
    current_shake_sequence = [] # 存储构成当前潜在摇头的转头区间

    for i, current_turn in enumerate(turn_intervals):
        if not current_shake_sequence:
            # 开始一个新的潜在摇头序列
            current_shake_sequence.append(current_turn)
            continue

        # 获取序列中最后一个转头的信息
        last_turn = current_shake_sequence[-1]

        # 检查方向是否交替
        is_alternating = current_turn['direction'] != last_turn['direction']

        # 检查时间间隔是否足够小
        time_gap = current_turn['start_time'] - last_turn['end_time']
        is_close_enough = time_gap <= max_gap_sec

        if is_alternating and is_close_enough:
            # 方向交替且时间接近，将当前转头加入序列
            current_shake_sequence.append(current_turn)
        else:
            # 条件不满足（方向相同 或 间隔太长），当前序列中断
            # 检查已记录的序列是否满足最小摇头次数要求
            if len(current_shake_sequence) >= min_shake_turns:
                # 满足条件，记录为一个摇头事件
                first_turn = current_shake_sequence[0]
                final_turn = current_shake_sequence[-1]
                shake_events.append({
                    'start_time': first_turn['start_time'],
                    'end_time': final_turn['end_time'],
                    'duration_seconds': round(final_turn['end_time'] - first_turn['start_time'], 3),
                    'turn_count': len(current_shake_sequence),
                    'start_frame': first_turn['start_frame'],
                    'end_frame': final_turn['end_frame'],
                    'avg_abs_angle': round(np.mean([abs(t['avg_angle']) for t in current_shake_sequence]), 2), # 示例：平均绝对角度
                    'included_turns': current_shake_sequence # 可以选择性保留原始区间信息
                })

            # 无论是否记录成功，都要重置序列，并用当前转头开始新的序列
            current_shake_sequence = [current_turn]

    # 循环结束后，检查最后一个待处理的序列
    if len(current_shake_sequence) >= min_shake_turns:
        first_turn = current_shake_sequence[0]
        final_turn = current_shake_sequence[-1]
        shake_events.append({
            'start_time': first_turn['start_time'],
            'end_time': final_turn['end_time'],
            'duration_seconds': round(final_turn['end_time'] - first_turn['start_time'], 3),
            'turn_count': len(current_shake_sequence),
            'start_frame': first_turn['start_frame'],
            'end_frame': final_turn['end_frame'],
            'avg_abs_angle': round(np.mean([abs(t['avg_angle']) for t in current_shake_sequence]), 2),
            'included_turns': current_shake_sequence
        })

    print(f"检测到 {len(shake_events)} 个摇头事件 (至少 {min_shake_turns} 次交替转头，间隔 <= {max_gap_sec}s)。")
    return shake_events


# ==============================================================
# 8. 生成报告 (修改: 包含摇头事件)
# ==============================================================
def generate_report(head_turn_intervals, head_shake_events, output_dir="头部运动分析结果"):
    """Generates console output and saves CSV reports for head turns and shakes."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # --- 转头报告 ---
    turn_report_path = os.path.join(output_dir, f"转头事件报告_{timestamp}.csv")
    if not head_turn_intervals:
        print("\n未检测到满足条件的单个转头事件。")
        # Optionally create an empty file or just skip
    else:
        print("\n--- 检测到的单个转头事件 ---")
        turn_report_data = []
        for i, turn in enumerate(head_turn_intervals):
            direction_zh = "向右" if turn['direction'] == 'right' else "向左"
            print(f"转头事件 {i+1}:")
            print(f"  时间: {turn['start_time']:.2f}s - {turn['end_time']:.2f}s")
            print(f"  持续时间: {turn['duration_seconds']:.2f}s ({turn['duration_frames']} 帧)")
            print(f"  方向: {direction_zh} (平均角度: {turn['avg_angle']:.1f}°)")
            print(f"  角度范围: Min={turn['min_angle']:.1f}°, Max={turn['max_angle']:.1f}°")
            print("-" * 10)

            turn_report_data.append({
                '事件编号': i + 1,
                '类型': '转头',
                '开始时间(秒)': turn['start_time'],
                '结束时间(秒)': turn['end_time'],
                '持续时间(秒)': turn['duration_seconds'],
                '持续时间(帧)': turn['duration_frames'],
                '方向': direction_zh,
                '平均角度(度)': turn['avg_angle'],
                '最小角度(度)': turn['min_angle'],
                '最大角度(度)': turn['max_angle'],
                '开始帧': turn['start_frame'],
                '结束帧': turn['end_frame'],
            })

        try:
            turn_report_df = pd.DataFrame(turn_report_data)
            turn_report_df.to_csv(turn_report_path, index=False, encoding='utf-8-sig')
            print(f"\n转头事件报告已保存至: {turn_report_path}")
        except Exception as e:
            print(f"\n保存转头事件报告 CSV 时出错: {e}")

    # --- 摇头报告 ---
    shake_report_path = os.path.join(output_dir, f"摇头事件报告_{timestamp}.csv")
    if not head_shake_events:
        print("\n未检测到满足条件的摇头事件。")
        # Optionally create an empty file or just skip
    else:
        print("\n--- 检测到的摇头事件 ---")
        shake_report_data = []
        for i, shake in enumerate(head_shake_events):
            print(f"摇头事件 {i+1}:")
            print(f"  时间: {shake['start_time']:.2f}s - {shake['end_time']:.2f}s")
            print(f"  总持续时间: {shake['duration_seconds']:.2f}s")
            print(f"  包含转头次数: {shake['turn_count']}")
            print(f"  平均绝对角度: {shake['avg_abs_angle']:.1f}°") # Example metric
            print("-" * 10)

            shake_report_data.append({
                '事件编号': i + 1,
                '类型': '摇头',
                '开始时间(秒)': shake['start_time'],
                '结束时间(秒)': shake['end_time'],
                '总持续时间(秒)': shake['duration_seconds'],
                '包含转头次数': shake['turn_count'],
                '平均绝对角度(度)': shake['avg_abs_angle'], # Example
                '开始帧': shake['start_frame'],
                '结束帧': shake['end_frame'],
                # Optionally add more details like sequence of directions
            })

        try:
            shake_report_df = pd.DataFrame(shake_report_data)
            shake_report_df.to_csv(shake_report_path, index=False, encoding='utf-8-sig')
            print(f"\n摇头事件报告已保存至: {shake_report_path}")
        except Exception as e:
            print(f"\n保存摇头事件报告 CSV 时出错: {e}")

# ==============================================================
# 9. 主分析函数 (修改: 增加摇头检测步骤)
# ==============================================================
def analyze_head_movements(file_path): # Renamed for clarity
    """主函数，运行头部转动（转头+摇头）分析流程。"""
    print("=" * 50)
    print(" 开始头部运动分析 (转头 Yaw + 摇头 Shake) ")
    print("=" * 50)

    # --- 1. 加载数据 ---
    df, frame_rate = load_and_prepare_data(file_path)
    if df is None:
        return

    # --- 2. 原始数据统计 ---
    print("\n--- 原始 yaw_deg 基本统计 ---")
    stats_original = compute_statistics(df, 'yaw_deg')
    if stats_original:
        for key, value in stats_original.items():
            print(f"  {key}: {value:.3f}")
    else:
        print("无法计算原始数据的统计信息。")


    # --- 3. 数据平滑与校正 ---
    window_len_sec = 0.5
    window_length = int(window_len_sec * frame_rate)
    window_length = max(5, min(len(df) // 5 if len(df) > 25 else 5, window_length)) # 更安全的边界检查
    if window_length % 2 == 0: window_length += 1

    print(f"\n--- 应用数据平滑 (Savitzky-Golay 滤波器) ---")
    df_filtered = apply_savgol_filter(df, 'yaw_deg', window_length=window_length, polyorder=2)
    analysis_col_smooth = 'yaw_deg_filtered'

    print(f"\n--- 应用异常值校正 (时间一致性) ---")
    # 调整时间一致性窗口大小，例如 0.2 秒对应的帧数
    consistency_window = max(5, int(0.2 * frame_rate))
    df_corrected = apply_temporal_consistency(df_filtered, analysis_col_smooth, window_size=consistency_window, std_multiplier=3.0)
    analysis_col_final = 'yaw_deg_filtered_corrected' # 最终使用的列

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
    print("\n--- 确定转头/摇头检测阈值 ---")
    suggested_thresholds = determine_head_turn_thresholds(df_corrected, analysis_col_final)
    if suggested_thresholds is None:
        print("错误: 无法确定头部运动阈值。中止分析。")
        return # Changed from exit() to return

    mean_yaw_angle = suggested_thresholds['mean'] # 保存平均角度

    print("\n【建议阈值】")
    print(f"  建议上限: {suggested_thresholds['upper_threshold']:.2f}°")
    print(f"  建议下限: {suggested_thresholds['lower_threshold']:.2f}°")

    try:
        user_upper_input = input(f"请输入最终转头/摇头检测上限 (回车保留建议值 {suggested_thresholds['upper_threshold']:.2f}°): ").strip()
        user_lower_input = input(f"请输入最终转头/摇头检测下限 (回车保留建议值 {suggested_thresholds['lower_threshold']:.2f}°): ").strip()

        final_upper_threshold = float(user_upper_input) if user_upper_input else suggested_thresholds['upper_threshold']
        final_lower_threshold = float(user_lower_input) if user_lower_input else suggested_thresholds['lower_threshold']
    except ValueError:
        print("输入阈值无效，将使用建议值。")
        final_upper_threshold = suggested_thresholds['upper_threshold']
        final_lower_threshold = suggested_thresholds['lower_threshold']

    final_thresholds = {
        'mean': mean_yaw_angle, # 使用之前计算的均值
        'upper_threshold': final_upper_threshold,
        'lower_threshold': final_lower_threshold
    }

    print(f"\n【最终确定的阈值】: Yaw > {final_thresholds['upper_threshold']:.2f}° 或 Yaw < {final_thresholds['lower_threshold']:.2f}° 时判定为显著偏转")


    # --- 5. 获取用户输入: 转头持续时间 ---
    print("\n--- 设置 *单个转头* 的持续时间标准 ---")
    try:
        min_turn_duration_sec = float(input(f"输入单个转头事件的【最小】持续时间 (秒, 例如 0.3, 默认: 0.3): ") or 0.3)
        max_turn_duration_sec = float(input(f"输入单个转头事件的【最大】持续时间 (秒, 例如 3.0, 默认: 3.0): ") or 3.0)
    except ValueError:
        print("输入无效。使用默认值 (最小: 0.3s, 最大: 3.0s)。")
        min_turn_duration_sec = 0.3
        max_turn_duration_sec = 3.0

    min_turn_frames = max(1, int(min_turn_duration_sec * frame_rate))
    max_turn_frames = int(max_turn_duration_sec * frame_rate)
    print(f"单个转头事件的持续时间范围: {min_turn_duration_sec:.2f}s ({min_turn_frames} 帧) 到 {max_turn_duration_sec:.2f}s ({max_turn_frames} 帧)")

    # --- 6. 检测单个转头区间 ---
    print(f"\n--- 使用列 '{analysis_col_final}' 检测单个转头区间 ---")
    head_turn_intervals = detect_head_turn_intervals(
        df_corrected,
        analysis_col_final,
        final_thresholds['upper_threshold'],
        final_thresholds['lower_threshold'],
        final_thresholds['mean'], # 传入均值用于判断方向
        min_turn_frames,
        max_turn_frames,
        frame_rate
    )

    # --- 7. 获取用户输入: 摇头判断标准 ---
    print("\n--- 设置 *摇头* 事件的判断标准 ---")
    try:
        # 摇头定义为 LRLR (4次) 或 LRLRL (5次) 等
        min_shake_turns_input = int(input(f"定义一次摇头至少需要包含多少次方向交替的转头 (例如 4 表示 LRLR, 默认: 4): ") or 4)
        # 两次反向转头之间的最大间隔，超过这个间隔就不算连续摇头了
        max_gap_sec_input = float(input(f"摇头中相邻反向转头之间的最大时间间隔 (秒, 例如 0.8, 默认: 0.8): ") or 0.8)
    except ValueError:
        print("输入无效。使用默认值 (最小转头次数: 4, 最大间隔: 0.8s)。")
        min_shake_turns_input = 4
        max_gap_sec_input = 0.8

    print(f"摇头事件标准: 至少 {min_shake_turns_input} 次交替转头, 相邻反向转头间隔不超过 {max_gap_sec_input:.2f}s")

    # --- 8. 检测摇头事件 ---
    print(f"\n--- 基于检测到的转头区间，分析摇头事件 ---")
    head_shake_events = detect_head_shake_events(
        head_turn_intervals, # 输入是上一步检测到的转头列表
        min_shake_turns=min_shake_turns_input,
        max_gap_sec=max_gap_sec_input
    )

    # --- 9. 生成报告 ---
    generate_report(head_turn_intervals, head_shake_events) # 传递两种事件列表

    print("\n=" * 5)
    print(" 分析完成 ")
    print("=" * 5)

# ==============================================================
# 10. 主执行块 (基本不变)
# ==============================================================
if __name__ == "__main__":
    print("\n===== 头部运动检测工具 (转头+摇头, 动态阈值) =====")

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
             file_path = ""

    if file_path:
        try:
            analyze_head_movements(file_path) # 调用更新后的主函数
        except Exception as e:
            print("\n" + "="*20 + " 发生错误 " + "="*20)
            print(f"分析过程中发生意外错误: {e}")
            import traceback
            print("\n--- 错误追踪 ---")
            traceback.print_exc()
            print("="*5)
    else:
        print("未提供有效的文件路径。退出。")