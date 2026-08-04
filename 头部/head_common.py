'''
头部运动分析共享模块
统一处理 pitch (抬头/低头)、roll (左倾/右倾)、yaw (转头/摇头) 三种运动
Author: Martinwang96
Copyright (c) 2025 by Martin Wang, SISU
'''

import os
import warnings
import numpy as np
import pandas as pd
from scipy import signal
from sklearn.cluster import KMeans
from datetime import datetime

try:
    import tkinter as tk
    from tkinter import filedialog
    USE_TKINTER = True
except ImportError:
    USE_TKINTER = False

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
os.environ.setdefault('LOKY_MAX_CPU_COUNT', '1')


# ==============================================================
# Axis configurations — 三个轴的差异化参数集中管理
# ==============================================================
AXIS_CONFIGS = {
    'pitch': {
        'pose_col': 'pose_Rx',
        'deg_col': 'pitch_deg',
        'title': '头部垂直运动',
        'subtitle': '抬头/低头',
        'upper_action': '低头',
        'lower_action': '抬头',
        'upper_key': 'lookdown',
        'lower_key': 'lookup',
        'filter_mode': 'confidence',   # confidence | success | none
        'output_format': 'elan',        # elan | custom
        'weights': {'statistical': 0.4, 'cluster': 0.2, 'iqr': 0.4},
        'stat_multiplier': 1.5,
        'min_duration_sec': 0.5,
        'max_duration_sec': 0,          # 0 = unlimited
        'has_shake': False,
        'data_requirement': "CSV文件需包含 'pose_Rx' (弧度) 和 'timestamp' 列",
    },
    'roll': {
        'pose_col': 'pose_Rz',
        'deg_col': 'roll_deg',
        'title': '头部倾斜',
        'subtitle': '左倾/右倾',
        'upper_action': '向右倾',
        'lower_action': '向左倾',
        'upper_key': 'right',
        'lower_key': 'left',
        'filter_mode': 'success',
        'output_format': 'custom',
        'weights': {'statistical': 0.4, 'cluster': 0.3, 'iqr': 0.3},
        'stat_multiplier': 1.5,
        'min_duration_sec': 0.25,
        'max_duration_sec': 2.5,
        'has_shake': False,
        'data_requirement': "CSV文件需包含 'pose_Rz' 列",
    },
    'yaw': {
        'pose_col': 'pose_Ry',
        'deg_col': 'yaw_deg',
        'title': '头部转动',
        'subtitle': '转头/摇头',
        'upper_action': '向右转',
        'lower_action': '向左转',
        'upper_key': 'right',
        'lower_key': 'left',
        'filter_mode': 'success',
        'output_format': 'custom',
        'weights': {'statistical': 0.4, 'cluster': 0.4, 'iqr': 0.2},
        'stat_multiplier': 2.0,
        'min_duration_sec': 0.3,
        'max_duration_sec': 3.0,
        'has_shake': True,
        'data_requirement': "CSV文件需包含 'pose_Ry' 列",
    },
}


# ==============================================================
# 1. 数据加载与预处理（统一版）
# ==============================================================
def load_and_prepare_data(file_path, config):
    """加载 OpenFace CSV，准备角度列，计算帧率，执行过滤与清洗。"""
    pose_col = config['pose_col']
    deg_col = config['deg_col']

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

    # 检查必需列
    if pose_col not in df.columns:
        print(f"错误: CSV中未找到必需的列 '{pose_col}'。")
        print(f"可用列: {list(df.columns)}")
        return None, None, None

    # 弧度/角度自动检测与转换
    if abs(df[pose_col]).max() < np.pi * 2:
        print(f"假设 '{pose_col}' 是弧度单位, 转换为角度。")
        df[deg_col] = df[pose_col] * 180.0 / np.pi
    else:
        print(f"'{pose_col}' 值域较大，假设已经是角度单位。")
        df[deg_col] = df[pose_col]

    # 低置信度帧处理
    if 'confidence' in df.columns:
        df['confidence'] = pd.to_numeric(df['confidence'], errors='coerce')
        if config['filter_mode'] == 'confidence':
            low_conf_mask = (df['confidence'] < 0.3) | df['confidence'].isnull()
            if low_conf_mask.sum() > 0:
                print(f"警告: {low_conf_mask.sum()} 帧的 'confidence' 低于阈值或无效，其角度将被置为 NaN。")
                df.loc[low_conf_mask, deg_col] = np.nan
        elif config['filter_mode'] == 'success':
            before = len(df)
            df = df[df['confidence'] > 0].copy()
            removed = before - len(df)
            if removed > 0:
                print(f"过滤掉了 {removed} 帧（confidence == 0）。")

    # success 列过滤
    if config['filter_mode'] == 'success' and 'success' in df.columns:
        initial_rows = len(df)
        df = df[df['success'] == 1].copy()
        removed_rows = initial_rows - len(df)
        if removed_rows > 0:
            print(f"根据 'success' 列过滤掉了 {removed_rows} 帧 (success != 1)。")

    # 角度列数值化与 NaN 插值
    df[deg_col] = pd.to_numeric(df[deg_col], errors='coerce')
    if df[deg_col].isnull().all():
        print(f"错误: '{deg_col}' 列在处理后全部为 NaN。无法继续。")
        return None, None, None

    if df[deg_col].isnull().any():
        df[deg_col] = df[deg_col].interpolate(method='linear', limit_direction='both').bfill().ffill()
    if df[deg_col].isnull().any():
        print(f"警告: '{deg_col}' 插值后仍有 NaN。")
        return None, None, None

    # 帧率计算
    frame_rate = 30.0
    if 'timestamp' in df.columns and df['timestamp'].notna().sum() > 1:
        ts = pd.to_numeric(df['timestamp'], errors='coerce').dropna().sort_values()
        if len(ts) > 1:
            avg_dt = np.mean(np.diff(ts))
            if avg_dt > 1e-6:
                calc_fps = 1.0 / avg_dt
                if 5.0 < calc_fps < 200.0:
                    frame_rate = calc_fps
                    print(f"计算得到的帧率: {frame_rate:.2f} FPS")
                else:
                    print(f"警告: 计算帧率 ({calc_fps:.2f}) 异常。使用默认 30 FPS。")
    print(f"采用帧率: {frame_rate:.2f} FPS")

    # frame / second 列
    df = df.reset_index(drop=True)
    df['frame'] = df.index
    if 'timestamp' in df.columns and pd.api.types.is_numeric_dtype(df['timestamp']) and df['timestamp'].notna().all():
        df['second'] = (df['timestamp'] - df['timestamp'].iloc[0]).round(3)
    else:
        df['second'] = (df['frame'] / frame_rate).round(3)

    df = df.dropna(subset=[deg_col, 'second', 'frame'])
    if df.empty:
        print("错误: 数据准备和清洗后，没有有效的数据行剩余。")
        return None, None, None

    print(f"数据加载和准备完成。最终有效帧数: {len(df)}")
    base_filename = os.path.splitext(os.path.basename(file_path))[0]
    return df, frame_rate, base_filename


# ==============================================================
# 2. 描述性统计
# ==============================================================
def compute_statistics(df, column):
    if column not in df.columns:
        print(f"错误: 用于统计的列 '{column}' 未找到。")
        return None
    if df[column].isnull().all():
        print(f"警告: 列 '{column}' 只包含空值。")
        return None
    return {
        '平均值': df[column].mean(),
        '标准差': df[column].std(),
        '最小值': df[column].min(),
        '25%分位数': df[column].quantile(0.25),
        '中位数': df[column].median(),
        '75%分位数': df[column].quantile(0.75),
        '最大值': df[column].max(),
        '范围': df[column].max() - df[column].min(),
        '偏度': df[column].skew(),
        '峰度': df[column].kurt(),
    }


# ==============================================================
# 3. Savitzky-Golay 滤波器
# ==============================================================
def apply_savgol_filter(df, column, window_length=11, polyorder=2):
    if column not in df.columns:
        print(f"错误: 用于 SavGol 滤波的列 '{column}' 未找到。")
        return df

    filtered_df = df.copy()
    filtered_col = f'{column}_filtered'
    valid_data = df[column].dropna()

    if len(valid_data) <= window_length:
        print(f"警告: 数据长度 ({len(valid_data)}) 过短，跳过 SavGol 滤波。")
        filtered_df[filtered_col] = df[column]
        return filtered_df

    if window_length % 2 == 0:
        window_length += 1
    if polyorder >= window_length:
        polyorder = window_length - 1
    if polyorder < 1:
        filtered_df[filtered_col] = df[column]
        return filtered_df

    try:
        filtered_data = signal.savgol_filter(valid_data.values, window_length, polyorder)
        filtered_df[filtered_col] = pd.Series(filtered_data, index=valid_data.index)
        filtered_df[filtered_col] = filtered_df[filtered_col].fillna(df[column])
        print(f"已应用 Savitzky-Golay 滤波 (窗口={window_length}, 阶数={polyorder})。")
    except Exception as e:
        print(f"SavGol 滤波出错: {e}。使用原始数据。")
        filtered_df[filtered_col] = df[column]
    return filtered_df


# ==============================================================
# 4. 时间一致性检查（rolling 版本，效率更高）
# ==============================================================
def apply_temporal_consistency(df, column, window_size=5, std_multiplier=3.0):
    if column not in df.columns or df[column].isnull().all():
        return df

    corrected_df = df.copy()
    corrected_col = f'{column}_corrected'
    corrected_df[corrected_col] = df[column]

    if len(df[column].dropna()) < window_size or window_size < 3:
        return corrected_df

    rolling_mean = df[column].rolling(window=window_size, center=True, min_periods=1).mean()
    rolling_std = df[column].rolling(window=window_size, center=True, min_periods=1).std().fillna(0)
    lower_bound = rolling_mean - std_multiplier * rolling_std
    upper_bound = rolling_mean + std_multiplier * rolling_std
    is_outlier = (df[column] < lower_bound) | (df[column] > upper_bound)

    num_corrected = is_outlier.sum()
    if num_corrected > 0:
        corrected_df.loc[is_outlier, corrected_col] = rolling_mean[is_outlier]
        print(f"时间一致性检查：校正了 {num_corrected} 个潜在异常值。")
    else:
        print("时间一致性检查：未检测到异常值。")

    corrected_df[corrected_col] = corrected_df[corrected_col].fillna(df[column])
    return corrected_df


# ==============================================================
# 5. 动态阈值计算（参数化版本）
# ==============================================================
def determine_thresholds(df, column, config, n_clusters=2):
    if column not in df.columns or df[column].isnull().all():
        return None

    data_series = df[column].dropna()
    if len(data_series) == 0:
        return None
    if len(data_series) < 10:
        print(f"警告: 数据点不足 ({len(data_series)})，阈值结果可能不可靠。")

    mean_val = data_series.mean()
    std_val = data_series.std()
    if pd.isna(std_val) or std_val < 1e-6:
        print(f"警告: 标准差接近 0，使用默认值。")
        std_val = 1.0

    # 方法1: 统计阈值
    stat_delta = config['stat_multiplier'] * std_val

    # 方法2: KMeans 聚类
    cluster_delta = stat_delta
    if len(data_series) >= n_clusters * 5:
        try:
            kmeans = KMeans(
                n_clusters=n_clusters, random_state=42,
                n_init=10, max_iter=300, algorithm='lloyd',
            )
            centers = kmeans.fit(data_series.values.reshape(-1, 1)).cluster_centers_
            centers = np.sort(centers.flatten())
            if len(centers) >= 2:
                cluster_delta = abs(centers[0] - centers[-1]) / 2.0
                print(f"  KMeans 聚类中心间距: {cluster_delta:.2f}°")
        except Exception as e:
            print(f"  KMeans 聚类失败: {e}。使用统计方法。")

    # 方法3: IQR
    Q1 = data_series.quantile(0.25)
    Q3 = data_series.quantile(0.75)
    IQR = Q3 - Q1
    iqr_delta = 1.0 * IQR if IQR > 1e-6 else stat_delta * 0.75

    # 加权组合
    w = config['weights']
    weighted_delta = (
        w['statistical'] * abs(stat_delta) +
        w['cluster'] * abs(cluster_delta) +
        w['iqr'] * abs(iqr_delta)
    )

    upper = mean_val + weighted_delta
    lower = mean_val - weighted_delta

    print(f"\n--- 动态阈值计算 ---")
    print(f"  参考角度: {mean_val:.2f}°, 加权Delta: {weighted_delta:.2f}°")
    print(f"  建议下限: {lower:.2f}° ({config['lower_action']})")
    print(f"  建议上限: {upper:.2f}° ({config['upper_action']})")

    return {
        'mean': mean_val,
        'upper_threshold': upper,
        'lower_threshold': lower,
        'weighted_delta': weighted_delta,
    }


# ==============================================================
# 6. 区间检测（统一版，含方向跟踪）
# ==============================================================
def detect_intervals(df, column, upper_thresh, lower_thresh, mean_val,
                     min_frames, max_frames, frame_rate, config):
    """
    检测角度超出阈值并满足持续时间标准的连续区间。
    返回区间列表，每个区间包含方向信息。
    """
    if column not in df.columns or df[column].isnull().all():
        return []
    if not all(c in df.columns for c in ['frame', 'second']):
        return []

    intervals = []
    in_interval = False
    start_idx = None
    current_direction = None
    temp_df = df.reset_index(drop=True)

    upper_key = config['upper_key']
    lower_key = config['lower_key']

    for idx in range(len(temp_df)):
        val = temp_df.loc[idx, column]
        is_upper = pd.notna(val) and val > upper_thresh
        is_lower = pd.notna(val) and val < lower_thresh
        is_outside = is_upper or is_lower

        if is_outside and not in_interval:
            in_interval = True
            start_idx = idx
            current_direction = upper_key if is_upper else lower_key

        elif in_interval:
            end_loop = (idx == len(temp_df) - 1)
            duration_frames = idx - start_idx + 1

            # 判断是否结束
            no_longer = False
            if current_direction == upper_key:
                no_longer = pd.isna(val) or val <= upper_thresh
            elif current_direction == lower_key:
                no_longer = pd.isna(val) or val >= lower_thresh

            type_switched = (current_direction == upper_key and is_lower) or \
                            (current_direction == lower_key and is_upper)

            max_exceeded = max_frames > 0 and duration_frames > max_frames
            interval_should_end = no_longer or type_switched or max_exceeded or end_loop

            if interval_should_end:
                end_idx = idx - 1 if (no_longer or type_switched) and not end_loop else idx
                if end_idx < start_idx:
                    end_idx = start_idx

                dur_frames = end_idx - start_idx + 1
                if dur_frames >= min_frames:
                    interval_data = temp_df.iloc[start_idx:end_idx + 1]
                    valid = interval_data[column].dropna()
                    if not valid.empty:
                        avg_angle = valid.mean()
                        # 最终方向基于平均值
                        final_dir = upper_key if avg_angle > mean_val else lower_key
                        intervals.append({
                            'start_frame': temp_df.loc[start_idx, 'frame'],
                            'end_frame': temp_df.loc[end_idx, 'frame'],
                            'start_time': round(float(temp_df.loc[start_idx, 'second']), 3),
                            'end_time': round(float(temp_df.loc[end_idx, 'second']), 3),
                            'duration_frames': dur_frames,
                            'duration_seconds': round(dur_frames / frame_rate, 3),
                            'avg_angle': round(avg_angle, 2),
                            'max_angle': round(valid.max(), 2),
                            'min_angle': round(valid.min(), 2),
                            'direction': final_dir,
                        })

                # 重置
                in_interval = False
                if is_outside and not end_loop:
                    in_interval = True
                    start_idx = idx
                    current_direction = upper_key if is_upper else lower_key
                else:
                    start_idx = None
                    current_direction = None

    print(f"\n检测到 {len(intervals)} 个区间。")
    return intervals


# ==============================================================
# 7. 摇头事件检测（仅 yaw 使用）
# ==============================================================
def detect_shake_events(intervals, min_turns=4, max_gap_sec=0.8):
    """分析转头区间列表，识别方向交替的摇头事件。"""
    if not intervals or len(intervals) < min_turns:
        print("转头区间不足，无法检测摇头事件。")
        return []

    shake_events = []
    sequence = []

    for turn in intervals:
        if not sequence:
            sequence.append(turn)
            continue

        last = sequence[-1]
        is_alternating = turn['direction'] != last['direction']
        time_gap = turn['start_time'] - last['end_time']
        is_close = time_gap <= max_gap_sec

        if is_alternating and is_close:
            sequence.append(turn)
        else:
            if len(sequence) >= min_turns:
                shake_events.append(_build_shake_event(sequence))
            sequence = [turn]

    if len(sequence) >= min_turns:
        shake_events.append(_build_shake_event(sequence))

    print(f"检测到 {len(shake_events)} 个摇头事件 (≥{min_turns}次交替, 间隔≤{max_gap_sec}s)。")
    return shake_events


def _build_shake_event(sequence):
    first, last = sequence[0], sequence[-1]
    return {
        'start_time': first['start_time'],
        'end_time': last['end_time'],
        'duration_seconds': round(last['end_time'] - first['start_time'], 3),
        'turn_count': len(sequence),
        'start_frame': first['start_frame'],
        'end_frame': last['end_frame'],
        'avg_abs_angle': round(np.mean([abs(t['avg_angle']) for t in sequence]), 2),
    }


# ==============================================================
# 8. 报告生成（ELAN CSV 和 自定义 CSV）
# ==============================================================
def generate_elan_report(intervals, config, output_dir, base_filename):
    """生成 ELAN 兼容的 CSV 报告 (StartTime, EndTime, Annotation)。"""
    if not intervals:
        print("未检测到事件，无法生成 ELAN 报告。")
        return

    type_map = {
        config['upper_key']: config['upper_action'],
        config['lower_key']: config['lower_action'],
    }

    rows = [{
        'StartTime': iv['start_time'],
        'EndTime': iv['end_time'],
        'Annotation': type_map[iv['direction']],
    } for iv in intervals]

    df_report = pd.DataFrame(rows).sort_values(by='StartTime').reset_index(drop=True)

    report_folder = os.path.join(output_dir, "ELAN导入报告_CSV")
    os.makedirs(report_folder, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(report_folder, f"{base_filename}_ELAN_{timestamp}.csv")

    try:
        df_report.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\nELAN 兼容报告已保存: {output_file}")
        print("列顺序: StartTime, EndTime, Annotation")

        upper_count = len([x for x in intervals if x['direction'] == config['upper_key']])
        lower_count = len([x for x in intervals if x['direction'] == config['lower_key']])
        print(f"\n=== 检测结果摘要 ===")
        print(f"{config['upper_action']}事件: {upper_count} 个")
        print(f"{config['lower_action']}事件: {lower_count} 个")
        print(f"总计: {len(intervals)} 个")
    except Exception as e:
        print(f"保存 ELAN CSV 报告时出错: {e}")


def generate_custom_report(intervals, config, output_dir, shake_events=None):
    """生成自定义 CSV 报告（含角度详情）。"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    event_type = config['title']

    # 主事件报告
    report_path = os.path.join(output_dir, f"{event_type}报告_{timestamp}.csv")
    if not intervals:
        print(f"\n未检测到{event_type}事件。")
    else:
        print(f"\n--- 检测到的{event_type}事件 ---")
        report_data = []
        for i, iv in enumerate(intervals):
            direction_zh = config['upper_action'] if iv['direction'] == config['upper_key'] else config['lower_action']
            print(f"事件 {i+1}: {iv['start_time']:.2f}s - {iv['end_time']:.2f}s "
                  f"({iv['duration_seconds']:.2f}s, {direction_zh}, 均值{iv['avg_angle']:.1f}°)")
            report_data.append({
                '事件编号': i + 1,
                '类型': event_type,
                '开始时间(秒)': iv['start_time'],
                '结束时间(秒)': iv['end_time'],
                '持续时间(秒)': iv['duration_seconds'],
                '持续时间(帧)': iv['duration_frames'],
                '方向': direction_zh,
                '平均角度(度)': iv['avg_angle'],
                '最小角度(度)': iv['min_angle'],
                '最大角度(度)': iv['max_angle'],
                '开始帧': iv['start_frame'],
                '结束帧': iv['end_frame'],
            })
        try:
            pd.DataFrame(report_data).to_csv(report_path, index=False, encoding='utf-8-sig')
            print(f"\n{event_type}报告已保存: {report_path}")
        except Exception as e:
            print(f"保存报告时出错: {e}")

    # 摇头报告（仅 yaw）
    if shake_events is not None:
        shake_path = os.path.join(output_dir, f"摇头事件报告_{timestamp}.csv")
        if not shake_events:
            print("\n未检测到摇头事件。")
        else:
            print(f"\n--- 检测到的摇头事件 ---")
            shake_data = []
            for i, sh in enumerate(shake_events):
                print(f"摇头 {i+1}: {sh['start_time']:.2f}s - {sh['end_time']:.2f}s "
                      f"({sh['duration_seconds']:.2f}s, {sh['turn_count']}次转头)")
                shake_data.append({
                    '事件编号': i + 1,
                    '类型': '摇头',
                    '开始时间(秒)': sh['start_time'],
                    '结束时间(秒)': sh['end_time'],
                    '总持续时间(秒)': sh['duration_seconds'],
                    '包含转头次数': sh['turn_count'],
                    '平均绝对角度(度)': sh['avg_abs_angle'],
                    '开始帧': sh['start_frame'],
                    '结束帧': sh['end_frame'],
                })
            try:
                pd.DataFrame(shake_data).to_csv(shake_path, index=False, encoding='utf-8-sig')
                print(f"\n摇头报告已保存: {shake_path}")
            except Exception as e:
                print(f"保存摇头报告时出错: {e}")


# ==============================================================
# 9. 交互式文件选择
# ==============================================================
def select_csv_file(title="选择包含头部姿态数据的CSV文件"):
    if USE_TKINTER:
        try:
            root = tk.Tk()
            root.withdraw()
            root.attributes('-topmost', True)
            file_path = filedialog.askopenfilename(
                title=title,
                filetypes=[("CSV文件", "*.csv"), ("所有文件", "*.*")],
                initialdir=os.getcwd(),
            )
            root.destroy()
            if file_path:
                return file_path
            print("未选择文件。")
            return None
        except Exception as e:
            print(f"GUI文件选择器错误: {e}")
            return _manual_file_input()
    return _manual_file_input()


def _manual_file_input():
    while True:
        file_path = input("\n请输入CSV文件的完整路径 (或输入 'q' 退出): ").strip()
        if file_path.lower() == 'q':
            return None
        if file_path.startswith('"') and file_path.endswith('"'):
            file_path = file_path[1:-1]
        if os.path.isfile(file_path) and file_path.lower().endswith('.csv'):
            return file_path
        print("文件不存在或不是CSV文件，请重新输入。")


# ==============================================================
# 10. 统一分析主流程
# ==============================================================
def analyze(file_path, axis):
    """运行指定轴的头部运动分析完整流程。"""
    config = AXIS_CONFIGS[axis]
    deg_col = config['deg_col']

    print("=" * 50)
    print(f" 开始{config['title']}分析 ({config['subtitle']}) ")
    print("=" * 50)

    # 1. 加载数据
    df, frame_rate, base_filename = load_and_prepare_data(file_path, config)
    if df is None:
        return

    output_dir = os.path.dirname(file_path) if os.path.dirname(file_path) else '.'

    # 2. 统计
    print(f"\n--- 原始 {deg_col} 统计 ---")
    stats = compute_statistics(df, deg_col)
    if stats:
        for k, v in stats.items():
            print(f"  {k}: {v:.3f}")

    # 3. 平滑与校正
    raw_points = len(df[deg_col])
    if raw_points > 20:
        window_len = int(0.3 * frame_rate)
        window_len = max(5, min(raw_points - 2 if raw_points % 2 == 0 else raw_points - 1, window_len))
        if window_len % 2 == 0:
            window_len += 1
        if window_len < 3:
            window_len = 3
    else:
        window_len = 5

    print(f"\n--- 应用数据平滑 ---")
    df_filtered = apply_savgol_filter(df, deg_col, window_length=window_len, polyorder=2)
    smooth_col = f'{deg_col}_filtered'

    print(f"\n--- 应用异常值校正 ---")
    consistency_window = max(5, int(0.15 * frame_rate))
    df_corrected = apply_temporal_consistency(df_filtered, smooth_col, window_size=consistency_window, std_multiplier=3.0)
    analysis_col = f'{deg_col}_filtered_corrected'

    # 4. 动态阈值
    print(f"\n--- 确定{config['title']}阈值 ---")
    suggested = determine_thresholds(df_corrected, analysis_col, config)
    if suggested is None:
        print("错误: 无法确定阈值。使用回退值。")
        mean_fb = df_corrected[analysis_col].mean() if analysis_col in df_corrected else 0
        std_fb = df_corrected[analysis_col].std() if analysis_col in df_corrected else 10.0
        if pd.isna(std_fb) or std_fb < 1e-3:
            std_fb = 10.0
        suggested = {
            'mean': mean_fb,
            'upper_threshold': mean_fb + std_fb,
            'lower_threshold': mean_fb - std_fb,
            'weighted_delta': std_fb,
        }

    # 5. 用户确认阈值
    try:
        if config['output_format'] == 'elan':
            # ELAN 模式：分别输入上下限
            user_upper = input(f"输入最终{config['upper_action']}阈值 (角度 >, 回车保留 {suggested['upper_threshold']:.2f}°): ").strip()
            user_lower = input(f"输入最终{config['lower_action']}阈值 (角度 <, 回车保留 {suggested['lower_threshold']:.2f}°): ").strip()
            final_upper = float(user_upper) if user_upper else suggested['upper_threshold']
            final_lower = float(user_lower) if user_lower else suggested['lower_threshold']
            if final_lower >= final_upper:
                print("警告：下限必须小于上限。使用建议值。")
                final_upper = suggested['upper_threshold']
                final_lower = suggested['lower_threshold']
        else:
            # 自定义模式：输入 Delta
            user_delta = input(f"输入最终检测Delta (度, 回车保留 {suggested['weighted_delta']:.2f}°): ").strip()
            if user_delta:
                final_delta = abs(float(user_delta))
                final_upper = suggested['mean'] + final_delta
                final_lower = suggested['mean'] - final_delta
            else:
                final_upper = suggested['upper_threshold']
                final_lower = suggested['lower_threshold']
    except ValueError:
        print("输入无效，使用建议值。")
        final_upper = suggested['upper_threshold']
        final_lower = suggested['lower_threshold']

    print(f"\n【最终阈值】: {config['lower_action']} < {final_lower:.2f}°, {config['upper_action']} > {final_upper:.2f}°")

    # 6. 持续时间参数
    min_dur = config['min_duration_sec']
    max_dur = config['max_duration_sec']
    try:
        user_min = input(f"输入最小持续时间 (秒, 默认 {min_dur}): ").strip()
        if user_min:
            min_dur = float(user_min)
        if max_dur > 0:
            user_max = input(f"输入最大持续时间 (秒, 默认 {max_dur}): ").strip()
            if user_max:
                max_dur = float(user_max)
    except ValueError:
        print("输入无效，使用默认值。")

    min_frames = max(1, int(min_dur * frame_rate))
    max_frames = int(max_dur * frame_rate) if max_dur > 0 else 0
    print(f"持续时间范围: {min_dur:.2f}s ({min_frames}帧)" + (f" ~ {max_dur:.2f}s ({max_frames}帧)" if max_dur > 0 else ""))

    # 7. 检测区间
    print(f"\n--- 检测{config['subtitle']}区间 ---")
    intervals = detect_intervals(
        df_corrected, analysis_col, final_upper, final_lower, suggested['mean'],
        min_frames, max_frames, frame_rate, config,
    )

    # 显示详情
    if intervals:
        for i, iv in enumerate(intervals, 1):
            dir_zh = config['upper_action'] if iv['direction'] == config['upper_key'] else config['lower_action']
            print(f"{i:2d}. {dir_zh}: {iv['start_time']:.3f}s - {iv['end_time']:.3f}s ({iv['duration_seconds']:.3f}s)")

    # 8. 摇头检测（仅 yaw）
    shake_events = None
    if config['has_shake']:
        try:
            min_shake = int(input(f"摇头最少交替转头次数 (默认 4): ").strip() or 4)
            max_gap = float(input(f"摇头最大间隔 (秒, 默认 0.8): ").strip() or 0.8)
        except ValueError:
            min_shake, max_gap = 4, 0.8
        shake_events = detect_shake_events(intervals, min_turns=min_shake, max_gap_sec=max_gap)

    # 9. 生成报告
    if config['output_format'] == 'elan':
        generate_elan_report(intervals, config, output_dir, base_filename)
    else:
        generate_custom_report(intervals, config, output_dir, shake_events)

    print(f"\n{'='*50}")
    print(f" {config['title']}分析完成！")
    print(f"{'='*50}")


def preview_thresholds(file_path, axis):
    """
    预览模式：加载数据并计算推荐阈值与统计信息，但不执行区间检测。
    供 Web 两阶段工作流第一阶段调用。
    返回 dict: {status, stats, recommended, frame_rate, frame_count, duration_sec}
    """
    config = AXIS_CONFIGS.get(axis)
    if not config:
        return {'status': 'error', 'message': f'未知轴: {axis}'}
    deg_col = config['deg_col']

    df, frame_rate, base_filename = load_and_prepare_data(file_path, config)
    if df is None:
        return {'status': 'error', 'message': '数据加载失败'}

    stats = compute_statistics(df, deg_col) or {}

    # 平滑与校正（与正式分析一致，保证阈值建议准确）
    raw_points = len(df[deg_col])
    window_len = int(0.3 * frame_rate) if raw_points > 20 else 5
    window_len = max(5, window_len)
    if window_len % 2 == 0:
        window_len += 1
    df_filtered = apply_savgol_filter(df, deg_col, window_length=window_len, polyorder=2)
    smooth_col = f'{deg_col}_filtered'
    consistency_window = max(5, int(0.15 * frame_rate))
    df_corrected = apply_temporal_consistency(df_filtered, smooth_col, window_size=consistency_window, std_multiplier=3.0)
    analysis_col = f'{deg_col}_filtered_corrected'

    suggested = determine_thresholds(df_corrected, analysis_col, config)
    if suggested is None:
        mean_fb = df_corrected[analysis_col].mean() if analysis_col in df_corrected else 0
        std_fb = df_corrected[analysis_col].std() if analysis_col in df_corrected else 10.0
        if pd.isna(std_fb) or std_fb < 1e-3:
            std_fb = 10.0
        suggested = {'mean': mean_fb, 'upper_threshold': mean_fb + std_fb,
                     'lower_threshold': mean_fb - std_fb, 'weighted_delta': std_fb}

    duration_sec = round(len(df) / frame_rate, 2) if frame_rate > 0 else 0

    return {
        'status': 'ok',
        'axis': axis,
        'axis_label': config['subtitle'],
        'stats': {k: round(float(v), 3) for k, v in stats.items()} if stats else {},
        'recommended': {
            'upper_threshold': round(float(suggested['upper_threshold']), 3),
            'lower_threshold': round(float(suggested['lower_threshold']), 3),
            'delta': round(float(suggested['weighted_delta']), 3),
            'mean': round(float(suggested['mean']), 3),
            'min_duration': config['min_duration_sec'],
            'max_duration': config['max_duration_sec'],
        },
        'frame_rate': round(float(frame_rate), 2),
        'frame_count': int(len(df)),
        'duration_sec': duration_sec,
        'upper_action': config['upper_action'],
        'lower_action': config['lower_action'],
    }


def analyze_programmatic(file_path, axis, params=None, output_dir=None):
    """
    非交互式分析接口（供 Web/批量调用）。
    params 可选键：upper_threshold, lower_threshold, delta, min_duration, max_duration,
                   min_shake_turns, max_shake_gap。缺省时使用建议值/默认值。
    返回结果 dict：{status, intervals, shake_events, report_files, frame_rate, stats}
    """
    config = AXIS_CONFIGS[axis]
    deg_col = config['deg_col']
    params = params or {}

    df, frame_rate, base_filename = load_and_prepare_data(file_path, config)
    if df is None:
        return {'status': 'error', 'message': '数据加载失败'}

    out_dir = output_dir or (os.path.dirname(file_path) or '.')

    stats = compute_statistics(df, deg_col) or {}

    # 平滑与校正
    raw_points = len(df[deg_col])
    window_len = int(0.3 * frame_rate) if raw_points > 20 else 5
    window_len = max(5, window_len)
    if window_len % 2 == 0:
        window_len += 1
    df_filtered = apply_savgol_filter(df, deg_col, window_length=window_len, polyorder=2)
    smooth_col = f'{deg_col}_filtered'
    consistency_window = max(5, int(0.15 * frame_rate))
    df_corrected = apply_temporal_consistency(df_filtered, smooth_col, window_size=consistency_window, std_multiplier=3.0)
    analysis_col = f'{deg_col}_filtered_corrected'

    suggested = determine_thresholds(df_corrected, analysis_col, config)
    if suggested is None:
        mean_fb = df_corrected[analysis_col].mean() if analysis_col in df_corrected else 0
        std_fb = df_corrected[analysis_col].std() if analysis_col in df_corrected else 10.0
        if pd.isna(std_fb) or std_fb < 1e-3:
            std_fb = 10.0
        suggested = {'mean': mean_fb, 'upper_threshold': mean_fb + std_fb,
                      'lower_threshold': mean_fb - std_fb, 'weighted_delta': std_fb}

    # 参数覆盖
    if 'upper_threshold' in params and 'lower_threshold' in params:
        final_upper = params['upper_threshold']
        final_lower = params['lower_threshold']
    elif 'delta' in params:
        final_upper = suggested['mean'] + params['delta']
        final_lower = suggested['mean'] - params['delta']
    else:
        final_upper = suggested['upper_threshold']
        final_lower = suggested['lower_threshold']

    min_dur = params.get('min_duration', config['min_duration_sec'])
    max_dur = params.get('max_duration', config['max_duration_sec'])
    min_frames = max(1, int(min_dur * frame_rate))
    max_frames = int(max_dur * frame_rate) if max_dur > 0 else 0

    intervals = detect_intervals(
        df_corrected, analysis_col, final_upper, final_lower, suggested['mean'],
        min_frames, max_frames, frame_rate, config,
    )

    shake_events = None
    if config['has_shake']:
        min_shake = params.get('min_shake_turns', 4)
        max_gap = params.get('max_shake_gap', 0.8)
        shake_events = detect_shake_events(intervals, min_turns=min_shake, max_gap_sec=max_gap)

    # 生成报告
    report_files = []
    if config['output_format'] == 'elan':
        report_folder = os.path.join(out_dir, "ELAN导入报告_CSV")
        os.makedirs(report_folder, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        type_map = {config['upper_key']: config['upper_action'], config['lower_key']: config['lower_action']}
        rows = [{'StartTime': iv['start_time'], 'EndTime': iv['end_time'],
                 'Annotation': type_map[iv['direction']]} for iv in intervals]
        if rows:
            rep = os.path.join(report_folder, f"{base_filename}_ELAN_{ts}.csv")
            pd.DataFrame(rows).sort_values('StartTime').to_csv(rep, index=False, encoding='utf-8-sig')
            report_files.append(rep)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        if intervals:
            rep = os.path.join(out_dir, f"{config['title']}报告_{ts}.csv")
            data_rows = [{
                '事件编号': i + 1, '类型': config['title'],
                '开始时间(秒)': iv['start_time'], '结束时间(秒)': iv['end_time'],
                '持续时间(秒)': iv['duration_seconds'], '方向': config['upper_action'] if iv['direction'] == config['upper_key'] else config['lower_action'],
                '平均角度(度)': iv['avg_angle'],
            } for i, iv in enumerate(intervals)]
            pd.DataFrame(data_rows).to_csv(rep, index=False, encoding='utf-8-sig')
            report_files.append(rep)
        if shake_events:
            rep = os.path.join(out_dir, f"摇头事件报告_{ts}.csv")
            shake_rows = [{
                '事件编号': i + 1, '类型': '摇头',
                '开始时间(秒)': sh['start_time'], '结束时间(秒)': sh['end_time'],
                '总持续时间(秒)': sh['duration_seconds'], '包含转头次数': sh['turn_count'],
            } for i, sh in enumerate(shake_events)]
            pd.DataFrame(shake_rows).to_csv(rep, index=False, encoding='utf-8-sig')
            report_files.append(rep)

    return {
        'status': 'ok',
        'intervals': intervals,
        'shake_events': shake_events or [],
        'report_files': report_files,
        'frame_rate': frame_rate,
        'stats': stats,
        'thresholds': {'upper': final_upper, 'lower': final_lower, 'mean': suggested['mean']},
    }


def run(axis):
    """入口点：选择文件并运行指定轴的分析。"""
    config = AXIS_CONFIGS[axis]
    print("=" * 50)
    print(f"         {config['title']}分析")
    print("=" * 50)
    print(f"支持检测: {config['subtitle']}")
    print(f"数据要求: {config['data_requirement']}")
    print("-" * 50)

    file_path = select_csv_file(f"选择包含头部姿态数据的CSV文件")
    if file_path is None:
        print("程序退出。")
        return

    print(f"\n选择的文件: {file_path}")
    try:
        analyze(file_path, axis)
    except KeyboardInterrupt:
        print("\n\n程序被用户中断。")
    except Exception as e:
        print(f"\n分析过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

    input("\n按回车键退出程序...")
