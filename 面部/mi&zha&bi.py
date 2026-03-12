# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import os
from datetime import datetime

# 设置字体为支持中文的字体
rcParams['font.sans-serif'] = ['SimHei']  # 设置为 SimHei 或其他支持中文的字体
rcParams['axes.unicode_minus'] = False  # 防止负号显示为方块

# --- detect_eye_states 函数 (保持不变) ---
def detect_eye_states(file_path=None, visualize=True):
    """
    从OpenFace输出CSV文件中检测眯眼、闭眼和眨眼状态

    参数:
    file_path (str): OpenFace输出CSV文件的路径
    visualize (bool): 是否可视化结果

    返回:
    tuple: (眯眼区间, 闭眼区间, 眨眼区间, 带有眼部状态标注的数据, 帧率)
    """
    # 1. 获取文件路径（如果未提供）
    if file_path is None:
        file_path = input("请输入OpenFace CSV文件的路径：")

    # 2. 读取并清理数据
    print(f"正在读取文件: {file_path}")
    try:
        data = pd.read_csv(file_path)
        # 清理列名中的空格
        data.columns = data.columns.str.strip()
        print(f"成功读取数据，共 {len(data)} 行")
        if data.empty:
            print("警告: 文件为空或无法正确解析。")
            return [], [], [], None, 30 # 返回默认帧率
    except FileNotFoundError:
        print(f"错误: 文件未找到 {file_path}")
        return [], [], [], None, 30
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return [], [], [], None, 30

    # 3. 定义眼睑关键点 - 增加检查确保列存在
    required_eye_lmk_cols = [
        'eye_lmk_x_37', 'eye_lmk_x_38', 'eye_lmk_y_37', 'eye_lmk_y_38',
        'eye_lmk_x_40', 'eye_lmk_x_41', 'eye_lmk_y_40', 'eye_lmk_y_41',
        'eye_lmk_x_43', 'eye_lmk_x_44', 'eye_lmk_y_43', 'eye_lmk_y_44',
        'eye_lmk_x_46', 'eye_lmk_x_47', 'eye_lmk_y_46', 'eye_lmk_y_47'
    ]
    missing_cols = [col for col in required_eye_lmk_cols if col not in data.columns]
    if missing_cols:
        print(f"警告: 缺少必要的眼睛关键点列: {', '.join(missing_cols)}")
        print("将无法计算眼睑距离和EAR。")
        return [], [], [], data, 30 # 返回原始数据和默认帧率

    try:
        # 右眼上眼睑点
        right_upper_x = data[['eye_lmk_x_37', 'eye_lmk_x_38']].mean(axis=1)
        right_upper_y = data[['eye_lmk_y_37', 'eye_lmk_y_38']].mean(axis=1)
        # 右眼下眼睑点
        right_lower_x = data[['eye_lmk_x_40', 'eye_lmk_x_41']].mean(axis=1)
        right_lower_y = data[['eye_lmk_y_40', 'eye_lmk_y_41']].mean(axis=1)

        # 左眼上眼睑点
        left_upper_x = data[['eye_lmk_x_43', 'eye_lmk_x_44']].mean(axis=1)
        left_upper_y = data[['eye_lmk_y_43', 'eye_lmk_y_44']].mean(axis=1)
        # 左眼下眼睑点
        left_lower_x = data[['eye_lmk_x_46', 'eye_lmk_x_47']].mean(axis=1)
        left_lower_y = data[['eye_lmk_y_46', 'eye_lmk_y_47']].mean(axis=1)
    except Exception as e:
         print(f"计算眼睑点时出错: {e}")
         return [], [], [], data, 30

    # 4. 计算眼睑距离
    right_eye_distance = np.sqrt((right_upper_x - right_lower_x)**2 +
                                (right_upper_y - right_lower_y)**2)
    left_eye_distance = np.sqrt((left_upper_x - left_lower_x)**2 +
                               (left_upper_y - left_lower_y)**2)

    # 5. 计算眼睛纵横比(EAR)
    required_ear_lmk_cols = [
        'eye_lmk_x_36', 'eye_lmk_x_39', 'eye_lmk_y_36', 'eye_lmk_y_39',
        'eye_lmk_x_42', 'eye_lmk_x_45', 'eye_lmk_y_42', 'eye_lmk_y_45'
    ]
    missing_ear_cols = [col for col in required_ear_lmk_cols if col not in data.columns]
    can_calculate_ear = not missing_ear_cols

    if can_calculate_ear:
        try:
            # 右眼宽度（眼角间距离）
            right_eye_width = np.sqrt((data['eye_lmk_x_36'] - data['eye_lmk_x_39'])**2 +
                                     (data['eye_lmk_y_36'] - data['eye_lmk_y_39'])**2)
            # 左眼宽度（眼角间距离）
            left_eye_width = np.sqrt((data['eye_lmk_x_42'] - data['eye_lmk_x_45'])**2 +
                                    (data['eye_lmk_y_42'] - data['eye_lmk_y_45'])**2)

            # 避免除以零
            right_ear = np.divide(right_eye_distance, right_eye_width, out=np.zeros_like(right_eye_distance), where=right_eye_width!=0)
            left_ear = np.divide(left_eye_distance, left_eye_width, out=np.zeros_like(left_eye_distance), where=left_eye_width!=0)
            print("成功计算眼睛纵横比(EAR)")
        except Exception as e:
             print(f"计算EAR时出错: {e}. 将使用眼睑距离代替。")
             right_ear = right_eye_distance
             left_ear = left_eye_distance
             can_calculate_ear = False # 标记无法计算EAR
    else:
        print(f"警告: 缺少计算EAR所需的列: {', '.join(missing_ear_cols)}. 将使用眼睑距离代替。")
        right_ear = right_eye_distance
        left_ear = left_eye_distance

    # 添加测量数据到数据框
    data['right_eye_distance'] = right_eye_distance
    data['left_eye_distance'] = left_eye_distance
    data['right_ear'] = right_ear
    data['left_ear'] = left_ear

    # 6. 检查是否存在AU45（眨眼）
    has_au45 = False
    au45_cols = [col for col in data.columns if 'AU45' in col]
    has_au45 = bool(au45_cols) # True if list is not empty

    if has_au45:
        print(f"找到AU45眨眼相关列: {', '.join(au45_cols)}")
        # 优先查找 AU45_c (分类)
        au45_c_col = [col for col in au45_cols if '_c' in col.lower()]

        if au45_c_col:
            au45_c = au45_c_col[0]
            # 使用分类列 > 0.5 作为眨眼判断依据 (OpenFace通常输出0或1)
            data['is_blinking'] = data[au45_c] > 0.5
            print(f"使用列 '{au45_c}' 进行眨眼检测。")
        else:
            # 如果没有 _c 列，尝试用 _r 列 (强度)，设置一个阈值
            au45_r_col = [col for col in au45_cols if '_r' in col.lower()]
            if au45_r_col:
                 au45_r = au45_r_col[0]
                 # 强度阈值需要根据数据调整，这里先设为1.0
                 blink_intensity_threshold = 1.0
                 data['is_blinking'] = data[au45_r] > blink_intensity_threshold
                 print(f"警告: 未找到AU45_c列，使用AU45_r > {blink_intensity_threshold} 进行眨眼检测。")
            else:
                 # 如果两者都没有，使用找到的第一个AU45列
                 first_au45_col = au45_cols[0]
                 data['is_blinking'] = data[first_au45_col] > 0.5 # 假设是分类或设置阈值
                 print(f"警告: 未找到AU45_c或AU45_r列，使用找到的第一个AU45列 '{first_au45_col}' > 0.5 进行眨眼检测。")
    else:
        print("未找到AU45眨眼相关列，将不进行眨眼检测")
        data['is_blinking'] = False # 添加一个全为False的列，以便后续代码统一处理

    # 7. 获取帧率信息
    frame_rate = 30.0 # 默认值
    if 'timestamp' in data.columns:
        timestamps = data['timestamp'].dropna().unique() # 去重和去NaN
        if len(timestamps) > 1:
            # 计算有效时间戳之间的平均间隔
            valid_diffs = np.diff(timestamps)
            valid_diffs = valid_diffs[valid_diffs > 0] # 只考虑正间隔
            if len(valid_diffs) > 0:
                avg_frame_time = np.mean(valid_diffs)
                if avg_frame_time > 1e-6: # 避免除以极小值
                    frame_rate = 1.0 / avg_frame_time
                    print(f"从时间戳计算的帧率: {frame_rate:.2f} fps")
                else:
                    print("时间戳间隔过小或无效，使用默认帧率 30 fps")
            else:
                 print("未找到有效的时间戳间隔，使用默认帧率 30 fps")
        else:
            print("时间戳数量不足，无法计算帧率，使用默认帧率 30 fps")
            # 尝试从文件名猜测帧率 (可选，比较复杂，暂不实现)
    else:
        print("警告: CSV文件中未找到 'timestamp' 列。")
        # 尝试用帧号估算，如果帧号存在且看似连续
        if 'frame' in data.columns and len(data) > 1:
            frame_diff = np.mean(np.diff(data['frame'].dropna()))
            if frame_diff == 1: # 如果帧号大致连续递增1
                 try:
                      user_fps = input(f"未找到时间戳，请确认视频帧率 (默认 {frame_rate}): ")
                      frame_rate = float(user_fps) if user_fps else frame_rate
                 except ValueError:
                      print("输入无效，使用默认帧率 30 fps")
            else:
                 print("帧号不连续，无法估算帧率，使用默认帧率 30 fps")
        else:
             try:
                 user_fps = input(f"未找到时间戳或帧号，请输入视频帧率 (默认 {frame_rate}): ")
                 frame_rate = float(user_fps) if user_fps else frame_rate
             except ValueError:
                 print("输入无效，使用默认帧率 30 fps")

    # 8. 获取用户参数
    ear_threshold = 0.18 # 默认值
    try:
        user_thresh = input(f"请输入眼睛纵横比(EAR)的阈值 (推荐0.1-0.25，默认 {ear_threshold})：")
        ear_threshold = float(user_thresh) if user_thresh else ear_threshold
    except ValueError:
        print(f"输入无效，使用默认阈值 {ear_threshold}")

    min_squint_duration = 0.5 # 秒，默认值
    try:
        user_squint_dur = input(f"请输入最小眯眼持续时间（秒，默认 {min_squint_duration})：")
        min_squint_duration = float(user_squint_dur) if user_squint_dur else min_squint_duration
    except ValueError:
        print(f"输入无效，使用默认持续时间 {min_squint_duration} 秒")

    long_closed_eyes_threshold = 1.0 # 秒，默认值
    try:
        user_closed_dur = input(f"请输入闭眼的阈值 (秒, 默认 {long_closed_eyes_threshold}): ")
        long_closed_eyes_threshold = float(user_closed_dur) if user_closed_dur else long_closed_eyes_threshold
    except ValueError:
        print(f"输入无效，使用默认阈值 {long_closed_eyes_threshold} 秒")

    min_blink_duration = 0.3 # 秒，默认值
    if has_au45:
        try:
            user_blink_dur = input(f"请输入最小眨眼持续时间（秒，基于AU45，默认 {min_blink_duration})：")
            min_blink_duration = float(user_blink_dur) if user_blink_dur else min_blink_duration
        except ValueError:
            print(f"输入无效，使用默认持续时间 {min_blink_duration} 秒")
    else:
        min_blink_duration = 0 # 如果没有AU45数据，则不检测持续眨眼

    # 计算帧数
    # 确保帧率大于0
    if frame_rate <= 0:
        print("错误：帧率无效，无法计算帧数阈值。将使用默认帧数。")
        frame_rate = 30.0 # 重置为默认值
        min_squint_frames = int(min_squint_duration * frame_rate)
        min_closed_frames = int(long_closed_eyes_threshold * frame_rate)
        min_blink_frames = int(min_blink_duration * frame_rate) if has_au45 else 0
    else:
        min_squint_frames = max(1, int(min_squint_duration * frame_rate)) # 至少1帧
        min_closed_frames = max(1, int(long_closed_eyes_threshold * frame_rate)) # 至少1帧
        min_blink_frames = max(1, int(min_blink_duration * frame_rate)) if has_au45 else 0 # 至少1帧

    # 9. 检测眼部状态
    # 使用 EAR 或 眼睑距离 进行判断
    metric_to_use = left_ear if can_calculate_ear else left_eye_distance # 优先用左眼或平均值？这里用左眼示例
    metric_threshold = ear_threshold if can_calculate_ear else ear_threshold # 如果用距离，阈值可能需要调整

    # 分别计算左右眼状态，但最终判断可能基于两者或平均值
    data['right_eye_low'] = data['right_ear'] < ear_threshold if can_calculate_ear else data['right_eye_distance'] < ear_threshold # 阈值可能需调整
    data['left_eye_low'] = data['left_ear'] < ear_threshold if can_calculate_ear else data['left_eye_distance'] < ear_threshold   # 阈值可能需调整
    data['any_eye_low'] = data['right_eye_low'] | data['left_eye_low']

    # 添加帧和秒信息
    if 'frame' not in data.columns:
        data['frame'] = np.arange(len(data))

    if 'timestamp' in data.columns:
        # 使用时间戳计算秒，处理非单调或NaN的情况
        first_valid_timestamp = data['timestamp'].dropna().iloc[0] if not data['timestamp'].dropna().empty else 0
        data['second'] = ((data['timestamp'].fillna(method='ffill').fillna(method='bfill') - first_valid_timestamp)).round().astype(int)
        # 如果时间戳不从0开始，需要调整
        min_second = data['second'].min()
        if min_second > 0:
             data['second'] = data['second'] - min_second
    elif frame_rate > 0:
        data['second'] = (data['frame'] // frame_rate).astype(int)
    else:
        data['second'] = 0 # 如果无法计算秒，则都设为0
        print("警告: 无法确定秒数信息。")


    # 10. 过滤持续性低开度状态 (潜在眯眼/闭眼)
    # 使用滚动窗口判断连续性
    # 窗口大小应该基于帧数阈值
    low_state_window = min(min_squint_frames, len(data)) # 窗口不超过数据长度
    if low_state_window > 0:
        # 使用 .rolling().sum() 计算窗口内True的个数是否达到阈值
        data['sustained_low_state'] = data['any_eye_low'].rolling(window=low_state_window, center=True, min_periods=1).sum() >= low_state_window
        # 或者用 mean > 阈值 (e.g., > 0.8 表示窗口内80%以上是低开度)
        # data['sustained_low_state'] = data['any_eye_low'].rolling(window=low_state_window, center=True, min_periods=1).mean() > 0.8
    else:
         data['sustained_low_state'] = False # 如果窗口为0，则无持续状态


    # 11. 处理持续性眨眼状态（如果有AU45数据）
    if has_au45 and min_blink_frames > 0:
        blink_window = min(min_blink_frames, len(data))
        if blink_window > 0:
             data['sustained_blink'] = data['is_blinking'].rolling(window=blink_window, center=True, min_periods=1).sum() >= blink_window
        else:
             data['sustained_blink'] = False
    else:
        data['sustained_blink'] = False # 如果没有AU45或眨眼帧阈值为0


    # 12. 按秒聚合状态 (用于检测区间)
    # 注意：这里聚合的是原始低开度状态，不是持续状态
    # 聚合 sustained_low_state 可能更准确反映区间
    if 'second' in data.columns and data['second'].nunique() > 1 : # 确保有秒列且不止一个值
         # 聚合持续低开度状态
         low_state_per_second = data.groupby('second')['sustained_low_state'].mean() > 0.5 # 该秒内大部分时间是持续低开度
         # 聚合持续眨眼状态
         blink_state_per_second = data.groupby('second')['sustained_blink'].mean() > 0.5 if has_au45 else pd.Series(False, index=data['second'].unique())
    else: # 如果秒信息不可靠，则无法按秒聚合
         print("警告：秒信息不足或无效，无法按秒检测区间。将尝试基于帧进行检测。")
         # 这里可以添加基于帧的区间检测逻辑，但会更复杂
         # 暂时返回空区间
         squint_intervals = []
         long_closed_eyes_intervals = []
         blink_intervals_frames = [] # 眨眼仍然基于帧检测
         # ... (此处需要实现基于帧的区间查找逻辑，类似下面的秒逻辑，但操作对象是帧) ...
         # 为了简化，当前版本在秒信息无效时，眯眼和长闭眼区间将为空

    # --- 基于秒的区间检测 ---
    if 'second' in data.columns and data['second'].nunique() > 1 :
        squint_intervals = []
        start_sec = None
        current_duration = 0

        for sec, is_low in low_state_per_second.items():
            if is_low:
                if start_sec is None:
                    start_sec = sec
                current_duration += 1 # 假设秒是连续的
            else:
                if start_sec is not None:
                    # 检查持续时间是否满足眯眼阈值
                    # 注意：这里的duration是秒数，不是帧数
                    if current_duration >= min_squint_duration:
                        squint_intervals.append((start_sec, sec - 1)) # 区间是 [start, end]
                    start_sec = None
                    current_duration = 0

        # 处理末尾的区间
        if start_sec is not None:
            if current_duration >= min_squint_duration:
                squint_intervals.append((start_sec, low_state_per_second.index[-1]))

        # 14. 过滤出闭眼区间
        long_closed_eyes_intervals = []
        final_squint_intervals = [] # 存储不是长闭眼的眯眼区间
        for start, end in squint_intervals:
            duration_sec = end - start + 1
            if duration_sec >= long_closed_eyes_threshold:
                long_closed_eyes_intervals.append((start, end))
            else:
                # 只有当它不是长闭眼时，才算作普通眯眼
                final_squint_intervals.append((start, end))

        # 更新 squint_intervals 为最终的纯眯眼区间
        squint_intervals = final_squint_intervals


    # 15. 检测眨眼区间（基于帧，如果有AU45数据）
    blink_intervals_frames = [] # 存储 (start_frame_idx, end_frame_idx, duration_sec)
    if has_au45 and min_blink_frames > 0:
        in_blink = False
        blink_start_idx = None
        blink_length_frames = 0

        # 遍历每一帧检测持续性眨眼
        for idx, is_sustained_blink in data['sustained_blink'].items(): # 使用 .items() 获取索引和值
            if is_sustained_blink:
                if not in_blink:
                    in_blink = True
                    blink_start_idx = idx
                    blink_length_frames = 1
                else:
                    blink_length_frames += 1
            else:
                if in_blink:
                    # 结束一个眨眼区间
                    in_blink = False
                    # 检查帧数是否达到阈值
                    if blink_length_frames >= min_blink_frames:
                        end_idx = idx - 1 # 结束帧是当前帧的前一帧
                        duration_sec = blink_length_frames / frame_rate if frame_rate > 0 else 0
                        blink_intervals_frames.append((blink_start_idx, end_idx, duration_sec))
                    blink_start_idx = None
                    blink_length_frames = 0

        # 处理视频结束时仍在眨眼的情况
        if in_blink and blink_length_frames >= min_blink_frames:
             end_idx = data.index[-1] # 最后一帧的索引
             duration_sec = blink_length_frames / frame_rate if frame_rate > 0 else 0
             blink_intervals_frames.append((blink_start_idx, end_idx, duration_sec))

    # 16. 输出结果
    print("\n================ 眼部状态检测结果 ================")

    if squint_intervals:
        print(f"\n检测到的眯眼时间区间 (持续 > {min_squint_duration}秒, 且 < {long_closed_eyes_threshold}秒)：")
        for i, interval in enumerate(squint_intervals, 1):
            duration = interval[1] - interval[0] + 1
            print(f"{i}. 第 {interval[0]} 秒 到 第 {interval[1]} 秒 (持续 {duration} 秒)")
    else:
        print(f"\n未检测到持续时间在 ({min_squint_duration}秒, {long_closed_eyes_threshold}秒) 范围内的眯眼状态")

    if long_closed_eyes_intervals:
        print(f"\n检测到的闭眼区间 (持续 >= {long_closed_eyes_threshold}秒)：")
        for i, interval in enumerate(long_closed_eyes_intervals, 1):
            duration = interval[1] - interval[0] + 1
            print(f"{i}. 第 {interval[0]} 秒 到 第 {interval[1]} 秒 (持续 {duration} 秒)")
    else:
        print(f"\n未检测到持续超过 {long_closed_eyes_threshold}秒 的闭眼状态")

    if has_au45 and blink_intervals_frames:
        print(f"\n检测到的持续眨眼区间 (持续 >= {min_blink_duration}秒)：")
        for i, (start_idx, end_idx, duration) in enumerate(blink_intervals_frames, 1):
             # 尝试获取对应的帧号和秒数
             start_frame = data.loc[start_idx, 'frame'] if start_idx in data.index and 'frame' in data.columns else f"索引{start_idx}"
             end_frame = data.loc[end_idx, 'frame'] if end_idx in data.index and 'frame' in data.columns else f"索引{end_idx}"
             start_sec = data.loc[start_idx, 'second'] if start_idx in data.index and 'second' in data.columns else "?"
             end_sec = data.loc[end_idx, 'second'] if end_idx in data.index and 'second' in data.columns else "?"
             print(f"{i}. 帧 {start_frame}-{end_frame} (约第{start_sec}-{end_sec}秒) 持续 {duration:.2f}秒")
    elif has_au45:
        print(f"\n未检测到持续超过 {min_blink_duration}秒 的眨眼状态")

    # 17. 可视化结果
    if visualize and len(data) > 0:
        # 注意 visualize_eye_states 可能需要更新以处理 blink_intervals_frames
        visualize_eye_states(data, squint_intervals, long_closed_eyes_intervals,
                            blink_intervals_frames, # 传递基于帧的眨眼区间
                            ear_threshold if can_calculate_ear else -1, # 传递阈值，如果不可用则为-1
                            frame_rate)

    # 返回秒区间和基于帧的眨眼区间
    return squint_intervals, long_closed_eyes_intervals, blink_intervals_frames, data, frame_rate

# --- visualize_eye_states 函数 (略作修改以处理blink_intervals_frames) ---
def visualize_eye_states(data, squint_intervals, long_closed_eyes_intervals, blink_intervals_frames, ear_threshold, frame_rate):
    """创建眼部状态可视化图表"""
    if data is None or data.empty:
        print("无数据可供可视化。")
        return

    fig = plt.figure(figsize=(14, 10))
    has_ear = 'right_ear' in data.columns and 'left_ear' in data.columns and ear_threshold != -1
    has_au45_data = 'sustained_blink' in data.columns

    # X轴数据：优先用 frame，如果不存在则用 index
    x_axis = data['frame'] if 'frame' in data.columns else data.index

    # 1. 绘制EAR/距离随时间变化
    ax1 = plt.subplot(4, 1, 1)
    if has_ear:
        ax1.plot(x_axis, data['right_ear'], 'b-', label='右眼EAR', alpha=0.7, linewidth=1)
        ax1.plot(x_axis, data['left_ear'], 'g-', label='左眼EAR', alpha=0.7, linewidth=1)
        ax1.axhline(y=ear_threshold, color='r', linestyle='--', label=f'EAR阈值 ({ear_threshold:.3f})')
        ax1.set_ylabel('眼睛纵横比(EAR)')
        ax1.set_title('眼睛开合度 (EAR) 随时间变化')
    elif 'right_eye_distance' in data.columns and 'left_eye_distance' in data.columns:
        ax1.plot(x_axis, data['right_eye_distance'], 'b-', label='右眼距离', alpha=0.7, linewidth=1)
        ax1.plot(x_axis, data['left_eye_distance'], 'g-', label='左眼距离', alpha=0.7, linewidth=1)
        # 对于距离，阈值可能不同，暂不绘制
        ax1.set_ylabel('眼睑距离')
        ax1.set_title('眼睑距离随时间变化 (未使用EAR)')
    else:
        ax1.text(0.5, 0.5, '缺少眼部度量数据', ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('眼部度量')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # 2. 绘制低开度状态
    ax2 = plt.subplot(4, 1, 2, sharex=ax1) # 共享X轴
    if 'right_eye_low' in data.columns and 'left_eye_low' in data.columns and 'sustained_low_state' in data.columns:
        ax2.plot(x_axis, data['right_eye_low'].astype(int), 'b-', label='右眼低开度', alpha=0.5)
        ax2.plot(x_axis, data['left_eye_low'].astype(int), 'g-', label='左眼低开度', alpha=0.5)
        ax2.plot(x_axis, data['sustained_low_state'].astype(int), 'r-', label='持续低开度', linewidth=1.5)

        # 标记眯眼区间 (基于秒区间转换回帧)
        has_second_col = 'second' in data.columns and 'frame' in data.columns
        if has_second_col:
             plotted_squint_label = False
             for i, (start_sec, end_sec) in enumerate(squint_intervals):
                 # 找到对应秒区间的帧范围
                 interval_frames = data[(data['second'] >= start_sec) & (data['second'] <= end_sec)]['frame']
                 if not interval_frames.empty:
                     start_frame = interval_frames.min()
                     end_frame = interval_frames.max()
                     ax2.axvspan(start_frame, end_frame, alpha=0.2, color='orange',
                                label='眯眼区间' if not plotted_squint_label else "")
                     plotted_squint_label = True
                 else:
                      print(f"警告: 无法为眯眼区间 {start_sec}-{end_sec} 找到对应的帧。")

        ax2.set_title('低开度状态检测 (潜在眯眼/闭眼)')
        ax2.set_ylabel('状态 (1=低)')
        ax2.set_ylim(-0.1, 1.1)
        ax2.legend(loc='upper right')
    else:
        ax2.text(0.5, 0.5, '缺少低开度状态数据', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('低开度状态')
    ax2.grid(True, alpha=0.3)


    # 3. 绘制闭眼状态
    ax3 = plt.subplot(4, 1, 3, sharex=ax1)
    if 'sustained_low_state' in data.columns:
        ax3.plot(x_axis, data['sustained_low_state'].astype(int), 'k-', alpha=0.7)

        # 标记闭眼区间 (基于秒区间转换回帧)
        if has_second_col:
            plotted_closed_label = False
            for i, (start_sec, end_sec) in enumerate(long_closed_eyes_intervals):
                 interval_frames = data[(data['second'] >= start_sec) & (data['second'] <= end_sec)]['frame']
                 if not interval_frames.empty:
                    start_frame = interval_frames.min()
                    end_frame = interval_frames.max()
                    ax3.axvspan(start_frame, end_frame, alpha=0.3, color='red',
                               label='闭眼' if not plotted_closed_label else "")
                    plotted_closed_label = True
                 else:
                      print(f"警告: 无法为长闭眼区间 {start_sec}-{end_sec} 找到对应的帧。")

        ax3.set_title('闭眼检测')
        ax3.set_ylabel('闭眼状态 (1=是)')
        ax3.set_ylim(-0.1, 1.1)
        if long_closed_eyes_intervals:
            ax3.legend(loc='upper right')
    else:
         ax3.text(0.5, 0.5, '缺少低开度状态数据', ha='center', va='center', transform=ax3.transAxes)
         ax3.set_title('闭眼')
    ax3.grid(True, alpha=0.3)


    # 4. 绘制眨眼状态 (如果有)
    ax4 = plt.subplot(4, 1, 4, sharex=ax1)
    if has_au45_data and 'is_blinking' in data.columns:
        ax4.plot(x_axis, data['is_blinking'].astype(int), 'c-', label='单帧眨眼(AU45)', alpha=0.5) # 青色
        ax4.plot(x_axis, data['sustained_blink'].astype(int), 'm-', label='持续眨眼', linewidth=1.5) # 洋红

        # 标记眨眼区间 (基于帧索引)
        plotted_blink_label = False
        for i, (start_idx, end_idx, _) in enumerate(blink_intervals_frames):
             # 获取对应的帧号 (如果可用)
             start_frame = data.loc[start_idx, 'frame'] if start_idx in data.index and 'frame' in data.columns else start_idx
             end_frame = data.loc[end_idx, 'frame'] if end_idx in data.index and 'frame' in data.columns else end_idx
             # 使用帧号或索引进行标记
             ax4.axvspan(start_frame, end_frame, alpha=0.3, color='purple',
                        label='眨眼区间' if not plotted_blink_label else "")
             plotted_blink_label = True

        ax4.set_title('眨眼检测 (基于AU45)')
        if blink_intervals_frames:
            ax4.legend(loc='upper right')
    else:
        ax4.text(0.5, 0.5, '未检测到或无AU45眨眼数据',
                horizontalalignment='center', verticalalignment='center',
                transform=ax4.transAxes, fontsize=12)
        ax4.set_title('眨眼检测')

    ax4.set_xlabel('帧号' if 'frame' in data.columns else '数据索引')
    ax4.set_ylabel('眨眼状态 (1=是)')
    ax4.set_ylim(-0.1, 1.1)
    ax4.grid(True, alpha=0.3)

    # 添加一个总标题
    fig.suptitle('眼部状态检测结果', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # 为了给总标题留出空间

    # 创建输出目录
    output_dir = "眼部状态检测结果" # 保持一致
    os.makedirs(output_dir, exist_ok=True)

    # 生成时间戳文件名
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"eye_detection_visualization_{timestamp_str}.png")

    # 保存图像
    try:
        plt.savefig(output_file, dpi=150)
        print(f"\n可视化结果已保存至: {output_file}")
        plt.show()
    except Exception as e:
        print(f"\n保存可视化图像时出错: {e}")
        plt.show() # 仍然尝试显示图像


# --- generate_report 函数 (不再需要，但保留以防万一或用于打印总结) ---
def generate_report(data, squint_intervals, long_closed_eyes_intervals, blink_intervals_frames, frame_rate):
    """生成检测结果的统计信息字典"""
    report = {
        "总帧数": 0,
        "总时长(秒)": 0.0,
        "帧率(fps)": frame_rate,
        "眯眼统计": {
            "眯眼区间数": len(squint_intervals),
            "眯眼总时长(秒)": sum([(end-start+1) for start, end in squint_intervals]),
            "眯眼占总时长百分比": 0.0,
            "眯眼区间详情(秒)": [{"开始(秒)": start, "结束(秒)": end, "持续时间(秒)": end-start+1} for start, end in squint_intervals]
        },
        "闭眼统计": {
            "闭眼区间数": len(long_closed_eyes_intervals),
            "闭眼总时长(秒)": sum([(end-start+1) for start, end in long_closed_eyes_intervals]),
            "闭眼占总时长百分比": 0.0,
            "闭眼区间详情(秒)": [{"开始(秒)": start, "结束(秒)": end, "持续时间(秒)": end-start+1} for start, end in long_closed_eyes_intervals]
        },
        "眨眼统计": {
            "持续眨眼次数": len(blink_intervals_frames),
            "平均眨眼持续时间(秒)": 0.0,
            "眨眼区间详情(帧)": [] # 稍后填充
        }
    }

    if data is not None and not data.empty:
        report["总帧数"] = len(data)
        total_duration = 0
        if 'timestamp' in data.columns and not data['timestamp'].dropna().empty:
            timestamps = data['timestamp'].dropna()
            total_duration = timestamps.max() - timestamps.min()
        elif 'frame' in data.columns and frame_rate > 0:
            total_duration = (data['frame'].max() - data['frame'].min() + 1) / frame_rate
        report["总时长(秒)"] = round(total_duration, 2)

        if total_duration > 0:
            report["眯眼统计"]["眯眼占总时长百分比"] = round(report["眯眼统计"]["眯眼总时长(秒)"] / total_duration * 100, 2)
            report["闭眼统计"]["闭眼占总时长百分比"] = round(report["闭眼统计"]["闭眼总时长(秒)"] / total_duration * 100, 2)

        if blink_intervals_frames:
            total_blink_duration = sum([duration for _, _, duration in blink_intervals_frames])
            report["眨眼统计"]["平均眨眼持续时间(秒)"] = round(total_blink_duration / len(blink_intervals_frames), 3)

            blink_details = []
            for start_idx, end_idx, duration in blink_intervals_frames:
                 start_frame = data.loc[start_idx, 'frame'] if start_idx in data.index and 'frame' in data.columns else f"索引{start_idx}"
                 end_frame = data.loc[end_idx, 'frame'] if end_idx in data.index and 'frame' in data.columns else f"索引{end_idx}"
                 blink_details.append({"开始帧": start_frame, "结束帧": end_frame, "持续时间(秒)": round(duration, 3)})
            report["眨眼统计"]["眨眼区间详情(帧)"] = blink_details

    return report


# --- 主程序块 ---
if __name__ == "__main__":
    # 运行眼部状态检测，获取帧率
    squint_intervals, long_closed_eyes_intervals, blink_intervals_frames, annotated_data, frame_rate = detect_eye_states()

    if annotated_data is not None and not annotated_data.empty:

        # 生成统计信息 (用于打印总结)
        report = generate_report(annotated_data, squint_intervals, long_closed_eyes_intervals, blink_intervals_frames, frame_rate)

        # 打印简要总结
        print("\n================ 检测结果总结 ================")
        print(f"总分析时长: {report['总时长(秒)']} 秒 ({report['总帧数']} 帧), 帧率: {report['帧率(fps)']:.2f} fps")
        print(f"检测到 {report['眯眼统计']['眯眼区间数']} 个眯眼区间，总计 {report['眯眼统计']['眯眼总时长(秒)']} 秒 ({report['眯眼统计']['眯眼占总时长百分比']}%)")
        print(f"检测到 {report['闭眼统计']['闭眼区间数']} 个闭眼区间，总计 {report['闭眼统计']['闭眼总时长(秒)']} 秒 ({report['闭眼统计']['闭眼占总时长百分比']}%)")
        has_blink_data = report['眨眼统计']['持续眨眼次数'] > 0
        if has_blink_data:
            print(f"检测到 {report['眨眼统计']['持续眨眼次数']} 次持续眨眼，平均持续时间 {report['眨眼统计']['平均眨眼持续时间(秒)']} 秒")
        else:
            print("未检测到（或未分析）持续眨眼事件。")

        # 询问是否保存结果 (现在只保存CSV)
        save_results = input("\n是否保存眼部事件摘要 CSV 文件? (y/n, 默认 n): ").lower() == 'y'

        if save_results:
            output_dir = "眼部状态检测结果"
            os.makedirs(output_dir, exist_ok=True)
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

            event_list = []
            has_timestamp = 'timestamp' in annotated_data.columns
            has_frame = 'frame' in annotated_data.columns

            # --- 精确时间计算函数 (同上) ---
            def get_precise_time(index):
                if has_timestamp and index in annotated_data.index:
                    # 尝试直接访问时间戳，如果失败则尝试用帧计算
                    ts = annotated_data.loc[index, 'timestamp']
                    # 检查 ts 是否为有效数值
                    if pd.notna(ts):
                         return round(ts, 2)
                    elif has_frame and frame_rate > 0: # 如果时间戳无效，尝试用帧
                         fr = annotated_data.loc[index, 'frame']
                         if pd.notna(fr):
                              # 计算相对时间，假设第一帧时间为0或第一个有效时间戳
                              first_valid_time = 0
                              if not annotated_data['timestamp'].dropna().empty:
                                   first_valid_time = annotated_data['timestamp'].dropna().iloc[0]
                              # 如果用帧号估算，需要知道视频起始时间，这里简化处理，认为相对时间即可
                              return round(fr / frame_rate, 2) # 可能需要调整基准时间
                         else: return None # 帧号也无效
                    else: return None # 时间戳无效且无帧信息或帧率
                elif has_frame and frame_rate > 0 and index in annotated_data.index:
                     fr = annotated_data.loc[index, 'frame']
                     if pd.notna(fr):
                        return round(fr / frame_rate, 2) # 同样，基准时间问题
                     else: return None
                else:
                    return None

            # --- 处理眯眼事件 ---
            if 'second' in annotated_data.columns: # 确保秒列存在
                for start_sec, end_sec in squint_intervals:
                    interval_indices = annotated_data[
                        (annotated_data['second'] >= start_sec) &
                        (annotated_data['second'] <= end_sec)
                    ].index
                    if interval_indices.empty: continue

                    first_frame_idx = interval_indices.min()
                    last_frame_idx = interval_indices.max()
                    precise_start_time = get_precise_time(first_frame_idx)
                    precise_end_time = get_precise_time(last_frame_idx)

                    if precise_start_time is not None and precise_end_time is not None:
                        duration = round(precise_end_time - precise_start_time, 2)
                        duration = max(0, duration) # 确保持续时间非负
                        event_list.append({
                            "开始时间(秒)": precise_start_time,
                            "结束时间(秒)": precise_end_time,
                            "事件类型": "眯眼",
                            "持续时间(秒)": duration
                        })
                    else:
                        print(f"警告: 无法计算眯眼区间 {start_sec}-{end_sec} 的精确时间。")

            # --- 处理闭眼事件 ---
            if 'second' in annotated_data.columns: # 确保秒列存在
                for start_sec, end_sec in long_closed_eyes_intervals:
                    interval_indices = annotated_data[
                        (annotated_data['second'] >= start_sec) &
                        (annotated_data['second'] <= end_sec)
                    ].index
                    if interval_indices.empty: continue

                    first_frame_idx = interval_indices.min()
                    last_frame_idx = interval_indices.max()
                    precise_start_time = get_precise_time(first_frame_idx)
                    precise_end_time = get_precise_time(last_frame_idx)

                    if precise_start_time is not None and precise_end_time is not None:
                        duration = round(precise_end_time - precise_start_time, 2)
                        duration = max(0, duration)
                        event_list.append({
                            "开始时间(秒)": precise_start_time,
                            "结束时间(秒)": precise_end_time,
                            "事件类型": "闭眼",
                            "持续时间(秒)": duration
                        })
                    else:
                         print(f"警告: 无法计算长闭眼区间 {start_sec}-{end_sec} 的精确时间。")


            # --- 处理眨眼事件 ---
            if blink_intervals_frames:
                 for start_idx, end_idx, _ in blink_intervals_frames:
                     precise_start_time = get_precise_time(start_idx)
                     precise_end_time = get_precise_time(end_idx)

                     if precise_start_time is not None and precise_end_time is not None:
                         duration = round(precise_end_time - precise_start_time, 2)
                         min_frame_duration = round(1/frame_rate, 2) if frame_rate > 0 else 0.01
                         if duration < 0: duration = min_frame_duration
                         elif duration == 0 and precise_start_time != precise_end_time : duration = min_frame_duration

                         event_list.append({
                            "开始时间(秒)": precise_start_time,
                            "结束时间(秒)": precise_end_time,
                            "事件类型": "眨眼",
                            "持续时间(秒)": duration
                        })
                     else:
                          print(f"警告: 无法计算眨眼区间索引 {start_idx}-{end_idx} 的精确时间。")


            # --- 创建并保存摘要CSV ---
            if event_list:
                summary_df = pd.DataFrame(event_list)
                summary_df = summary_df.sort_values(by="开始时间(秒)").reset_index(drop=True)
                summary_df = summary_df[["开始时间(秒)", "结束时间(秒)", "持续时间(秒)", "事件类型"]]

                summary_csv_file = os.path.join(output_dir, f"eye_events_summary_{timestamp_str}.csv")
                try:
                    summary_df.to_csv(summary_csv_file, index=False, encoding='utf-8-sig', float_format='%.2f')
                    print(f"\n眼部事件摘要 (精确到0.01秒) 已保存至: {summary_csv_file}")
                except Exception as e:
                    print(f"\n保存摘要CSV文件时出错: {e}")
            else:
                print("\n未检测到任何可记录的眼部事件，未生成摘要CSV文件。")

            # --- 文本报告保存部分已被移除 ---

    elif annotated_data is None:
        print("未能加载数据，无法生成报告或保存结果。")
    else: # annotated_data is empty
         print("数据为空，无法进行分析或保存结果。")