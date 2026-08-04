'''
面部分析共享工具模块
抽取 facialexpression / mi&zha&bi / fix&sca 三个文件的公共逻辑
Author: Martinwang96
Copyright (c) 2025 by Martin Wang, SISU
'''

import os
import numpy as np
import pandas as pd
from datetime import datetime

# matplotlib 中文字体设置（各文件按需调用）
def setup_matplotlib(backend=None):
    """配置 matplotlib 中文字体和后端。"""
    import matplotlib
    if backend:
        matplotlib.use(backend)
    from matplotlib import rcParams
    rcParams['font.sans-serif'] = ['SimHei']
    rcParams['axes.unicode_minus'] = False
    return rcParams


# ==============================================================
# CSV 加载与帧率计算
# ==============================================================
def load_openface_csv(file_path):
    """读取 OpenFace CSV，自动去除列名空格。"""
    print(f"正在读取文件: {file_path}")
    try:
        data = pd.read_csv(file_path)
        data.columns = data.columns.str.strip()
        print(f"成功读取数据，共 {len(data)} 行")
        return data
    except FileNotFoundError:
        print(f"错误: 文件未找到 {file_path}")
        return None
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return None


def calculate_frame_rate(data, default=30.0):
    """从 timestamp 列计算帧率，失败时返回默认值。"""
    if 'timestamp' not in data.columns:
        print(f"警告: 未找到 'timestamp' 列，使用默认帧率 {default} fps")
        return default

    timestamps = pd.to_numeric(data['timestamp'], errors='coerce').dropna().sort_values()
    if len(timestamps) <= 1:
        print(f"警告: 时间戳数据不足，使用默认帧率 {default} fps")
        return default

    avg_dt = np.mean(np.diff(timestamps))
    if avg_dt > 1e-6:
        fps = 1.0 / avg_dt
        if 5.0 < fps < 200.0:
            print(f"检测到帧率: {fps:.1f} fps")
            return fps
        print(f"警告: 计算帧率 {fps:.1f} 异常，使用默认 {default} fps")
    else:
        print(f"警告: 时间戳间隔无效，使用默认帧率 {default} fps")
    return default


def ensure_frame_second_cols(data, frame_rate):
    """确保 data 包含 frame 和 second 列。"""
    if 'frame' not in data.columns:
        data['frame'] = np.arange(len(data))
    if 'timestamp' in data.columns:
        ts = pd.to_numeric(data['timestamp'], errors='coerce')
        first_valid = ts.dropna().iloc[0] if ts.dropna().size > 0 else 0
        data['second'] = (ts.fillna(method='ffill').fillna(method='bfill') - first_valid)
    else:
        data['second'] = data['frame'] / frame_rate
    return data


# ==============================================================
# 交互式输入辅助
# ==============================================================
def ask(prompt, default=None):
    """交互式输入，支持默认值。"""
    suffix = f" [{default}]" if default is not None else ""
    v = input(f"{prompt}{suffix}: ").strip()
    if v:
        return v
    return default if default is not None else ""


def ask_float(prompt, default):
    """交互式浮点输入。"""
    raw = ask(prompt, str(default))
    try:
        return float(raw)
    except (ValueError, TypeError):
        print(f"输入无效，使用默认值 {default}")
        return float(default)


def ask_bool(prompt, default=True):
    """交互式布尔输入。"""
    hint = "Y/n" if default else "y/N"
    v = input(f"{prompt} ({hint}): ").strip().lower()
    if not v:
        return default
    return v in {"y", "yes", "1", "true"}


# ==============================================================
# 报告导出
# ==============================================================
def get_timestamp_str():
    """获取当前时间戳字符串。"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_csv_report(df, output_dir, filename_prefix, timestamp=None):
    """保存 CSV 报告到指定目录，文件名含时间戳。"""
    os.makedirs(output_dir, exist_ok=True)
    ts = timestamp or get_timestamp_str()
    filepath = os.path.join(output_dir, f"{filename_prefix}_{ts}.csv")
    try:
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        print(f"报告已保存: {filepath}")
        return filepath
    except Exception as e:
        print(f"保存报告时出错: {e}")
        return None


def save_elan_csv(events, output_dir, base_filename, timestamp=None):
    """保存 ELAN 兼容 CSV (StartTime, EndTime, Annotation)。"""
    os.makedirs(output_dir, exist_ok=True)
    ts = timestamp or get_timestamp_str()
    filepath = os.path.join(output_dir, f"{base_filename}_ELAN_{ts}.csv")
    try:
        events.to_csv(filepath, index=False, encoding='utf-8-sig')
        print(f"ELAN 兼容报告已保存: {filepath}")
        return filepath
    except Exception as e:
        print(f"保存 ELAN 报告时出错: {e}")
        return None


# ==============================================================
# 通用区间检测
# ==============================================================
def detect_flag_intervals(df, flag_col, min_frames, frame_rate):
    """
    检测布尔标记列的持续区间。
    返回区间列表，每个含 start_frame, end_frame, start_time, end_time, duration。
    """
    intervals = []
    in_interval = False
    start_idx = None

    for idx in df.index:
        if df.loc[idx, flag_col]:
            if not in_interval:
                in_interval = True
                start_idx = idx
        else:
            if in_interval:
                duration = idx - start_idx
                if duration >= min_frames:
                    end_idx = idx - 1
                    intervals.append({
                        'start_frame': df.loc[start_idx, 'frame'] if 'frame' in df.columns else start_idx,
                        'end_frame': df.loc[end_idx, 'frame'] if 'frame' in df.columns else end_idx,
                        'start_time': df.loc[start_idx, 'second'] if 'second' in df.columns else start_idx / frame_rate,
                        'end_time': df.loc[end_idx, 'second'] if 'second' in df.columns else end_idx / frame_rate,
                        'duration': round(duration / frame_rate, 2),
                    })
                in_interval = False
                start_idx = None

    # 处理末尾区间
    if in_interval:
        end_idx = df.index[-1]
        duration = end_idx - start_idx + 1
        if duration >= min_frames:
            intervals.append({
                'start_frame': df.loc[start_idx, 'frame'] if 'frame' in df.columns else start_idx,
                'end_frame': df.loc[end_idx, 'frame'] if 'frame' in df.columns else end_idx,
                'start_time': df.loc[start_idx, 'second'] if 'second' in df.columns else start_idx / frame_rate,
                'end_time': df.loc[end_idx, 'second'] if 'second' in df.columns else end_idx / frame_rate,
                'duration': round(duration / frame_rate, 2),
            })

    return intervals


# ==============================================================
# 交互式文件选择
# ==============================================================
def select_csv_file(title="选择OpenFace CSV文件"):
    """交互式选择 CSV 文件。"""
    try:
        from tkinter import Tk, filedialog
        root = Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        file_path = filedialog.askopenfilename(
            title=title,
            filetypes=[("CSV文件", "*.csv"), ("所有文件", "*.*")],
        )
        root.destroy()
        if file_path:
            return file_path
        print("未选择文件。")
        return None
    except ImportError:
        file_path = input("请输入CSV文件路径: ").strip()
        return file_path if file_path else None
    except Exception as e:
        print(f"文件选择器错误: {e}")
        return input("请输入CSV文件路径: ").strip() or None
