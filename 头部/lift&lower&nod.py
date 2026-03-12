import pandas as pd
import numpy as np
from scipy import signal
from sklearn.cluster import KMeans
import os
from datetime import datetime
import tkinter as tk
from tkinter import filedialog
import warnings

# 抑制一些不必要的警告
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# 设置环境变量来避免KMeans的joblib警告
os.environ['LOKY_MAX_CPU_COUNT'] = '1'

# --- Tkinter Setup ---
try:
    from tkinter import Tk, filedialog
    USE_TKINTER = True
except ImportError:
    USE_TKINTER = False

# ==============================================================
# Utility Functions
# ==============================================================

def load_and_prepare_data(file_path):
    print(f"尝试从以下路径加载数据: {file_path}")
    try:
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip()
        print("CSV 文件加载成功。")
    except FileNotFoundError:
        print(f"错误: 文件未找到 {file_path}")
        return None, None, None
    except Exception as e:
        print(f"加载 CSV 文件时出错: {e}")
        return None, None, None

    core_required_columns = ['pose_Rx', 'timestamp']
    missing_core_cols = [col for col in core_required_columns if col not in df.columns]
    if missing_core_cols:
        print(f"错误: CSV中缺少核心必需的列: {', '.join(missing_core_cols)}。")
        return None, None, None

    optional_cols_to_check = ['confidence']
    for col_name in optional_cols_to_check:
        if col_name in df.columns:
            df[col_name] = pd.to_numeric(df[col_name], errors='coerce')
        else:
            df[col_name] = np.nan

    if 'confidence' in df.columns and df['confidence'].notna().any():
        confidence_threshold = 0.3
        low_confidence_mask = (df['confidence'] < confidence_threshold) | df['confidence'].isnull()
        if low_confidence_mask.sum() > 0:
            print(f"警告: {low_confidence_mask.sum()} 帧的 'confidence' 低于阈值或无效，其 'pose_Rx' 将被置为 NaN。")
            df.loc[low_confidence_mask, 'pose_Rx'] = np.nan

    df['pose_Rx'] = pd.to_numeric(df['pose_Rx'], errors='coerce')
    if df['pose_Rx'].isnull().all():
        print(f"错误: 'pose_Rx' 列在处理后全部为 NaN。无法继续。")
        return None, None, None

    if df['pose_Rx'].isnull().any():
        df['pose_Rx'] = df['pose_Rx'].interpolate(method='linear', limit_direction='both').bfill().ffill()
    if df['pose_Rx'].isnull().any():
        print("警告: 'pose_Rx' 插值后仍有 NaN。")
        return None, None, None

    df['pitch_deg'] = df['pose_Rx'] * (180.0 / np.pi)

    frame_rate = 30.0
    if 'timestamp' in df.columns and df['timestamp'].notna().sum() > 1:
        timestamps_numeric = pd.to_numeric(df['timestamp'], errors='coerce').dropna().sort_values()
        if len(timestamps_numeric) > 1:
            avg_time_diff = np.mean(np.diff(timestamps_numeric))
            if avg_time_diff > 1e-6:
                calculated_fps = 1.0 / avg_time_diff
                if 5.0 < calculated_fps < 200.0:
                    frame_rate = calculated_fps
    print(f"采用帧率: {frame_rate:.2f} FPS (基于timestamp计算或默认)。")

    df = df.reset_index(drop=True)
    df['frame'] = df.index
    if 'timestamp' in df.columns and pd.api.types.is_numeric_dtype(df['timestamp']) and df['timestamp'].notna().all():
        df['second'] = (df['timestamp'] - df['timestamp'].iloc[0]).round(3)
    else:
        df['second'] = (df['frame'] / frame_rate).round(3)

    df = df.dropna(subset=['pitch_deg', 'second', 'frame'])
    if df.empty:
        print("错误: 数据准备和清洗后，没有有效的数据行剩余。")
        return None, None, None

    print(f"数据加载和准备完成。最终有效帧数: {len(df)}")
    base_filename = os.path.splitext(os.path.basename(file_path))[0]
    return df, frame_rate, base_filename

def calculate_baseline_reference(df, column='pitch_deg', baseline_duration_sec=10.0, frame_rate=30.0):
    """
    计算基准参考角度 - 使用前几秒的数据作为中性位置参考
    """
    baseline_frames = int(baseline_duration_sec * frame_rate)
    baseline_frames = min(baseline_frames, len(df) // 4)  # 最多使用总数据的1/4
    baseline_frames = max(baseline_frames, 10)  # 至少10帧

    if len(df) < baseline_frames:
        baseline_data = df[column]
    else:
        baseline_data = df[column].head(baseline_frames)

    baseline_data = baseline_data.dropna()
    if len(baseline_data) == 0:
        return None

    # 使用中位数作为基准，比均值更稳定
    baseline_median = baseline_data.median()
    baseline_std = baseline_data.std()
    if pd.isna(baseline_std) or baseline_std < 1e-3:
        baseline_std = 3.0

    print(f"--- 基准参考计算 ---")
    print(f"  基准样本: {len(baseline_data)} 帧 ({len(baseline_data)/frame_rate:.2f}秒)")
    print(f"  基准中位数角度: {baseline_median:.2f}°")
    print(f"  基准标准差: {baseline_std:.2f}°")

    return {
        'baseline_median': baseline_median,
        'baseline_std': baseline_std,
        'baseline_frames_used': len(baseline_data)
    }

def apply_savgol_filter(df, column='pitch_deg', window_length=11, polyorder=2):
    if column not in df.columns:
        df[f'{column}_filtered'] = df.get(column, pd.Series(np.nan, index=df.index))
        return df
    filtered_df = df.copy()
    filtered_col_name = f'{column}_filtered'
    valid_data = df[column].dropna()
    if len(valid_data) <= window_length:
        filtered_df[filtered_col_name] = df[column]
        return filtered_df
    if window_length % 2 == 0:
        window_length += 1
    if polyorder >= window_length:
        polyorder = window_length - 1
    if polyorder < 1:
        filtered_df[filtered_col_name] = df[column]
        return filtered_df
    try:
        filtered_data = signal.savgol_filter(valid_data.values, window_length, polyorder)
        filtered_series = pd.Series(filtered_data, index=valid_data.index)
        filtered_df[filtered_col_name] = filtered_series
        filtered_df[filtered_col_name] = filtered_df[filtered_col_name].fillna(df[column])
    except Exception as e:
        print(f"Savitzky-Golay 滤波 '{column}' 时出错: {e}。将使用原始数据。")
        filtered_df[filtered_col_name] = df[column]
    return filtered_df

def apply_temporal_consistency(df, column='pitch_deg_filtered', window_size=5, std_multiplier=3.0):
    if column not in df.columns or df[column].isnull().all():
        df[f'{column}_corrected'] = df.get(column, pd.Series(np.nan, index=df.index))
        return df
    corrected_df = df.copy()
    corrected_col_name = f'{column}_corrected'
    corrected_df[corrected_col_name] = df[column]
    if len(df[column].dropna()) < window_size or window_size < 3:
        return corrected_df
    rolling_mean = df[column].rolling(window=window_size, center=True, min_periods=1).mean()
    rolling_std = df[column].rolling(window=window_size, center=True, min_periods=1).std().fillna(0)
    lower_bound = rolling_mean - std_multiplier * rolling_std
    upper_bound = rolling_mean + std_multiplier * rolling_std
    is_outlier = (df[column] < lower_bound) | (df[column] > upper_bound)
    num_outliers_found = is_outlier.sum()
    if num_outliers_found > 0:
        corrected_df.loc[is_outlier, corrected_col_name] = rolling_mean[is_outlier]
    corrected_df[corrected_col_name] = corrected_df[corrected_col_name].fillna(df[column])
    return corrected_df

def determine_pitch_thresholds_with_baseline(df, column='pitch_deg_filtered_corrected', baseline_info=None, n_clusters=2):
    """
    结合基准参考和动态分析来确定阈值
    """
    if column not in df.columns or df[column].isnull().all():
        return None

    data_series = df[column].dropna()
    if len(data_series) < 10:
        if len(data_series) > 0:
            mean_val = data_series.mean()
            std_val = data_series.std()
            if pd.isna(std_val) or std_val < 1e-3:
                std_val = 5.0
            return {
                'mean': mean_val,
                'upper_threshold': mean_val + std_val,
                'lower_threshold': mean_val - std_val,
                'weighted_delta': std_val,
                'baseline_adjusted': False
            }
        return None

    # 基础统计
    mean_val = data_series.mean()
    std_val = data_series.std()
    if pd.isna(std_val) or std_val < 1e-6:
        std_val = 1.0

    # 基准调整
    if baseline_info is not None:
        baseline_median = baseline_info['baseline_median']
        baseline_std = baseline_info['baseline_std']

        # 使用基准中位数作为参考点，而不是整体均值
        reference_angle = baseline_median
        print(f"  使用基准中位数 {baseline_median:.2f}° 作为参考点")

        # 结合基准标准差和整体标准差
        combined_std = (baseline_std + std_val) / 2.0
        stat_delta = 1.2 * combined_std  # 稍微保守一些

        baseline_adjusted = True
    else:
        reference_angle = mean_val
        stat_delta = 1.5 * std_val
        baseline_adjusted = False

    # KMeans聚类分析（修复版本）
    cluster_delta = stat_delta
    if len(data_series) >= n_clusters * 5:
        try:
            # 修复KMeans的参数设置
            kmeans = KMeans(
                n_clusters=n_clusters,
                random_state=42,
                n_init=10,  # 明确指定数值而不是'auto'
                max_iter=300,
                algorithm='lloyd'  # 明确指定算法
            )

            # 确保数据格式正确
            cluster_data = data_series.values.reshape(-1, 1)
            cluster_centers = kmeans.fit(cluster_data).cluster_centers_
            cluster_centers = np.sort(cluster_centers.flatten())

            if len(cluster_centers) >= 2:
                cluster_delta = abs(cluster_centers[0] - cluster_centers[-1]) / 2.0
                print(f"  KMeans聚类成功，聚类中心间距: {cluster_delta:.2f}°")
        except Exception as e:
            print(f"  KMeans聚类失败: {e}。使用统计方法。")

    # IQR方法
    Q1 = data_series.quantile(0.25)
    Q3 = data_series.quantile(0.75)
    IQR = Q3 - Q1
    iqr_delta = 1.0 * IQR if IQR > 1e-6 else stat_delta * 0.75

    # 加权组合
    weights = {'statistical': 0.4, 'cluster': 0.2, 'iqr': 0.4}
    stat_delta = abs(stat_delta)
    cluster_delta = abs(cluster_delta)
    iqr_delta = abs(iqr_delta)

    weighted_delta = (
        weights['statistical'] * stat_delta +
        weights['cluster'] * cluster_delta +
        weights['iqr'] * iqr_delta
    )

    # 基于参考角度（基准或均值）计算阈值
    suggested_upper_threshold = reference_angle + weighted_delta
    suggested_lower_threshold = reference_angle - weighted_delta

    print(f"--- 动态阈值计算完毕 ---")
    print(f"  参考角度: {reference_angle:.2f}°, 加权Delta: {weighted_delta:.2f}°")
    print(f"  建议抬头阈值: < {suggested_lower_threshold:.2f}°")
    print(f"  建议低头阈值: > {suggested_upper_threshold:.2f}°")
    if baseline_adjusted:
        print(f"  ✓ 已应用基准参考调整")

    return {
        'mean': reference_angle,
        'upper_threshold': suggested_upper_threshold,
        'lower_threshold': suggested_lower_threshold,
        'weighted_delta': weighted_delta,
        'baseline_adjusted': baseline_adjusted
    }

# ==============================================================
# Action Detection Functions
# ==============================================================

def detect_tilt_intervals(df, column, upper_thresh, lower_thresh, min_frames, frame_rate):
    """
    检测“抬头”和“低头”区间，只要持续帧数 >= min_frames 就算一次事件。
    """
    if column not in df.columns or df[column].isnull().all():
        return []
    if not all(c in df.columns for c in ['frame', 'second']):
        return []

    intervals = []
    in_interval = False
    start_idx = None
    current_tilt_type = None
    temp_df = df.reset_index(drop=True)

    for idx in range(len(temp_df)):
        current_value = temp_df.loc[idx, column]
        is_lookup = pd.notna(current_value) and current_value < lower_thresh
        is_lookdown = pd.notna(current_value) and current_value > upper_thresh
        is_outside = is_lookup or is_lookdown

        if is_outside and not in_interval:
            # 新区间开始
            in_interval = True
            start_idx = idx
            current_tilt_type = 'lookup' if is_lookup else 'lookdown'

        elif in_interval:
            end_loop = (idx == len(temp_df) - 1)
            # 判断是否切换到另一个类型或回到中立
            type_switched = (current_tilt_type == 'lookup' and is_lookdown) or \
                           (current_tilt_type == 'lookdown' and is_lookup)

            no_longer_meets_condition = False
            if current_tilt_type == 'lookup':
                no_longer_meets_condition = pd.isna(current_value) or current_value >= lower_thresh
            elif current_tilt_type == 'lookdown':
                no_longer_meets_condition = pd.isna(current_value) or current_value <= upper_thresh

            interval_should_end = False
            if no_longer_meets_condition or type_switched:
                interval_should_end = True
            elif end_loop:
                interval_should_end = True

            if interval_should_end:
                end_idx = idx - 1 if (no_longer_meets_condition or type_switched) and not end_loop else idx
                if end_idx < start_idx:
                    end_idx = start_idx

                interval_duration_frames = end_idx - start_idx + 1

                if interval_duration_frames >= min_frames:
                    interval_data = temp_df.iloc[start_idx : end_idx + 1]
                    valid_pitch_data = interval_data[column].dropna()
                    if not valid_pitch_data.empty:
                        start_time = interval_data['second'].iloc[0]
                        end_time = interval_data['second'].iloc[-1]
                        intervals.append({
                            'start_time': round(float(start_time), 3),
                            'end_time': round(float(end_time), 3),
                            'tilt_type': current_tilt_type
                        })

                # 重置状态
                in_interval = False
                if is_outside and interval_should_end and not end_loop:
                    # 如果当前帧依然满足抬头/低头条件，则开启下一个区间
                    in_interval = True
                    start_idx = idx
                    current_tilt_type = 'lookup' if is_lookup else 'lookdown'
                else:
                    start_idx = None
                    current_tilt_type = None

    return intervals

# ==============================================================
# Reporting Function for ELAN (CSV Output)
# ==============================================================

def generate_elan_report_csv(tilt_intervals, output_dir, base_filename):
    """
    只将“抬头”和“低头”事件导出为 ELAN CSV。
    """
    if not tilt_intervals:
        print("未检测到抬头/低头事件，无法生成 ELAN 报告。")
        return

    all_events_for_elan = []
    type_map_elan = {'lookup': '抬头', 'lookdown': '低头'}

    for interval in tilt_intervals:
        all_events_for_elan.append({
            'StartTime': interval['start_time'],
            'EndTime': interval['end_time'],
            'Annotation': type_map_elan[interval['tilt_type']]
        })

    df_elan_report = pd.DataFrame(all_events_for_elan)
    df_elan_report = df_elan_report.sort_values(by='StartTime').reset_index(drop=True)

    report_folder = os.path.join(output_dir, "ELAN导入报告_CSV")
    os.makedirs(report_folder, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(report_folder, f"{base_filename}_ELAN_{timestamp}.csv")

    try:
        df_elan_report.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\nELAN兼容报告已保存到 (CSV格式): {output_file}")
        print("列顺序: StartTime, EndTime, Annotation")

        # 显示检测结果摘要
        print(f"\n=== 检测结果摘要 ===")
        print(f"抬头事件: {len([x for x in tilt_intervals if x['tilt_type'] == 'lookup'])} 个")
        print(f"低头事件: {len([x for x in tilt_intervals if x['tilt_type'] == 'lookdown'])} 个")
        print(f"总计事件: {len(tilt_intervals)} 个")

    except Exception as e:
        print(f"\n保存 ELAN CSV 报告时出错: {e}")

# ==============================================================
# Main Analysis Function
# ==============================================================

def analyze_pitch_movements_for_elan(file_path):
    print("=" * 6)
    print(" 开始头部垂直运动分析 (为 ELAN 生成 CSV 报告) ")
    print("=" * 6)

    df, frame_rate, base_filename = load_and_prepare_data(file_path)
    if df is None:
        return

    output_dir = os.path.dirname(file_path) if os.path.dirname(file_path) else '.'

    # 不使用前几秒做基准
    baseline_info = None

    raw_data_points = len(df['pitch_deg'])
    analysis_col_final = 'pitch_deg'

    if raw_data_points > 20:
        window_len_sec = 0.3
        window_length_savgol = int(window_len_sec * frame_rate)
        window_length_savgol = max(5, min(raw_data_points - 2 if raw_data_points % 2 == 0 else raw_data_points - 1, window_length_savgol))
        if window_length_savgol % 2 == 0:
            window_length_savgol += 1
        if window_length_savgol < 3:
            window_length_savgol = 3

        consistency_window_frames = max(5, int(0.15 * frame_rate))
        consistency_window_frames = min(consistency_window_frames, raw_data_points - 1 if raw_data_points > 0 else 3)
        if consistency_window_frames < 3:
            consistency_window_frames = 3

        print(f"\n--- 应用数据平滑与校正 ---")
        df_filtered = apply_savgol_filter(df, 'pitch_deg', window_length=window_length_savgol, polyorder=2)
        analysis_col_smooth = 'pitch_deg_filtered' if 'pitch_deg_filtered' in df_filtered else 'pitch_deg'
        df_corrected = apply_temporal_consistency(df_filtered, analysis_col_smooth, window_size=consistency_window_frames, std_multiplier=2.5)
        analysis_col_final = 'pitch_deg_filtered_corrected' if 'pitch_deg_filtered_corrected' in df_corrected else analysis_col_smooth
    else:
        df_corrected = df.copy()
        if 'pitch_deg_filtered_corrected' not in df_corrected.columns and 'pitch_deg' in df_corrected.columns:
            df_corrected['pitch_deg_filtered_corrected'] = df_corrected['pitch_deg']
        analysis_col_final = 'pitch_deg_filtered_corrected' if 'pitch_deg_filtered_corrected' in df_corrected else 'pitch_deg'

    # 使用改进的阈值确定函数
    print(f"\n--- 确定抬头/低头动态阈值（结合基准参考）---")
    suggested_thresholds = determine_pitch_thresholds_with_baseline(df_corrected, analysis_col_final, baseline_info)

    if suggested_thresholds is None:
        print("错误: 无法确定俯仰角阈值。将使用固定回退阈值。")
        mean_pitch_fallback = df_corrected[analysis_col_final].mean() if analysis_col_final in df_corrected and not df_corrected[analysis_col_final].empty else 0
        std_pitch_fallback = df_corrected[analysis_col_final].std() if analysis_col_final in df_corrected and not df_corrected[analysis_col_final].empty and df_corrected[analysis_col_final].std() > 1e-3 else 10.0
        suggested_thresholds = {
            'mean': mean_pitch_fallback,
            'upper_threshold': mean_pitch_fallback + std_pitch_fallback,
            'lower_threshold': mean_pitch_fallback - std_pitch_fallback,
            'weighted_delta': std_pitch_fallback,
            'baseline_adjusted': False
        }

    # 用户输入阈值
    user_upper_input = input(f"输入最终低头阈值 (角度 >, 回车保留 {suggested_thresholds['upper_threshold']:.2f}°): ").strip()
    user_lower_input = input(f"输入最终抬头阈值 (角度 <, 回车保留 {suggested_thresholds['lower_threshold']:.2f}°): ").strip()

    try:
        final_upper_threshold = float(user_upper_input) if user_upper_input else suggested_thresholds['upper_threshold']
        final_lower_threshold = float(user_lower_input) if user_lower_input else suggested_thresholds['lower_threshold']
        if final_lower_threshold >= final_upper_threshold:
            print("警告：抬头阈值必须小于低头阈值。将使用建议值。")
            final_upper_threshold = suggested_thresholds['upper_threshold']
            final_lower_threshold = suggested_thresholds['lower_threshold']
    except ValueError:
        print("输入无效，将使用建议阈值。")
        final_upper_threshold = suggested_thresholds['upper_threshold']
        final_lower_threshold = suggested_thresholds['lower_threshold']

    final_thresholds = {
        'mean': suggested_thresholds['mean'],
        'upper_threshold': final_upper_threshold,
        'lower_threshold': final_lower_threshold,
        'weighted_delta': suggested_thresholds['weighted_delta']
    }

    print(f"\n【最终采用抬头/低头阈值】: 抬头 (Pitch < {final_thresholds['lower_threshold']:.2f}°)， 低头 (Pitch > {final_thresholds['upper_threshold']:.2f}°)")
    if suggested_thresholds.get('baseline_adjusted', False):
        print("✓ 已应用基准参考调整，应该能更准确地区分抬头和低头")

    # 抬头/低头检测参数
    min_tilt_duration_sec = 0.5
    min_tilt_frames = max(1, int(min_tilt_duration_sec * frame_rate))

    print(f"\n--- 开始检测抬头/低头动作 ---")
    print(f"最小抬头/低头持续时间: {min_tilt_duration_sec}秒 ({min_tilt_frames}帧)")

    # 检测抬头/低头间隔
    detected_tilt_intervals = detect_tilt_intervals(
        df_corrected,
        analysis_col_final,
        final_thresholds['upper_threshold'],
        final_thresholds['lower_threshold'],
        min_tilt_frames,
        frame_rate
    )

    print(f"\n=== 检测结果 ===")
    print(f"抬头事件: {len([x for x in detected_tilt_intervals if x['tilt_type'] == 'lookup'])} 个")
    print(f"低头事件: {len([x for x in detected_tilt_intervals if x['tilt_type'] == 'lookdown'])} 个")

    # 显示详细检测结果
    if detected_tilt_intervals:
        print(f"\n--- 抬头/低头详情 ---")
        for i, interval in enumerate(detected_tilt_intervals, 1):
            duration = interval['end_time'] - interval['start_time']
            action_name = "抬头" if interval['tilt_type'] == 'lookup' else "低头"
            print(f"{i:2d}. {action_name}: {interval['start_time']:.3f}s - {interval['end_time']:.3f}s (持续 {duration:.3f}s)")

    # 生成ELAN兼容的CSV报告
    generate_elan_report_csv(detected_tilt_intervals, output_dir, base_filename)

    print(f"\n{'='*6}")
    print(" 分析完成！")
    print(f"{'='*6}")

# ==============================================================
# Interactive File Selection and Main Execution
# ==============================================================

def select_csv_file():
    """交互式文件选择"""
    if USE_TKINTER:
        try:
            root = tk.Tk()
            root.withdraw()  # 隐藏主窗口
            root.attributes('-topmost', True)  # 置顶

            file_path = filedialog.askopenfilename(
                title="选择包含头部姿态数据的CSV文件",
                filetypes=[
                    ("CSV文件", "*.csv"),
                    ("所有文件", "*.*")
                ],
                initialdir=os.getcwd()
            )
            root.destroy()

            if file_path:
                return file_path
            else:
                print("未选择文件。")
                return None

        except Exception as e:
            print(f"GUI文件选择器错误: {e}")
            return manual_file_input()
    else:
        return manual_file_input()

def manual_file_input():
    """手动输入文件路径"""
    while True:
        file_path = input("\n请输入CSV文件的完整路径 (或输入 'q' 退出): ").strip()

        if file_path.lower() == 'q':
            return None

        if file_path.startswith('"') and file_path.endswith('"'):
            file_path = file_path[1:-1]

        if os.path.isfile(file_path) and file_path.lower().endswith('.csv'):
            return file_path
        else:
            print("文件不存在或不是CSV文件，请重新输入。")

def main():
    """主函数"""
    print("=" * 8)
    print("         头部垂直运动分析")
    print("=" * 8)
    print("支持检测: 抬头、低头动作")
    print("数据要求: CSV文件需包含 'pose_Rx' (弧度) 和 'timestamp' 列")
    print("-" * 8)

    csv_file_path = select_csv_file()

    if csv_file_path is None:
        print("程序退出。")
        return

    print(f"\n选择的文件: {csv_file_path}")

    try:
        analyze_pitch_movements_for_elan(csv_file_path)
    except KeyboardInterrupt:
        print("\n\n程序被用户中断。")
    except Exception as e:
        print(f"\n分析过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

    # 程序结束前暂停
    input("\n按回车键退出程序...")

if __name__ == "__main__":
    main()
