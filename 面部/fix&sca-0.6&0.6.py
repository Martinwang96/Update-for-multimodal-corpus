import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import datetime
from matplotlib import rcParams
import os

# 设置 matplotlib 使用非交互式后端
import matplotlib
matplotlib.use('Agg')
from matplotlib import rcParams
rcParams['font.sans-serif'] = ['SimHei']  # 设置为 SimHei 或其他支持中文的字体
rcParams['axes.unicode_minus'] = False  # 防止负号显示为方块

def load_openface_data(file_path: str) -> pd.DataFrame:
    """读取 OpenFace CSV 数据文件，并去除所有列名前后的空格"""
    try:
        data = pd.read_csv(file_path, encoding='utf-8-sig')
    except Exception as e:
        print(f"读取 CSV 文件失败: {e}")
        sys.exit(1)
    data.columns = [col.strip() for col in data.columns]
    print(f"数据加载完成，共 {len(data)} 行")
    print("列名列表：", list(data.columns)[:10], "...等" if len(data.columns) > 10 else "")
    return data

def compute_eye_velocity(data: pd.DataFrame, eye_prefix: str = 'gaze_0') -> pd.Series:
    """
    计算指定 eye（例如 gaze_0）的帧间速度（仅使用 X、Y 轴），返回每帧速度大小。
    使用当前帧与前一帧差值的欧氏距离。
    """
    if f"{eye_prefix}_x" not in data.columns or f"{eye_prefix}_y" not in data.columns:
        raise ValueError(f"数据中缺少 {eye_prefix}_x 或 {eye_prefix}_y 列")
    dx = data[f"{eye_prefix}_x"].diff()
    dy = data[f"{eye_prefix}_y"].diff()
    velocity = np.sqrt(dx**2 + dy**2)
    velocity.fillna(0, inplace=True)
    return velocity

def compute_combined_eyes_velocity(data: pd.DataFrame) -> pd.Series:
    """结合两只眼睛的数据计算更准确的眼动速度"""
    velocities = []
    
    # 尝试计算两只眼睛的速度
    for prefix in ['gaze_0', 'gaze_1']:
        try:
            vel = compute_eye_velocity(data, prefix)
            velocities.append(vel)
        except ValueError:
            print(f"警告: {prefix} 数据不可用")
    
    # 如果两只眼都有数据，取平均值
    if len(velocities) == 2:
        print("使用两只眼睛的平均速度")
        return (velocities[0] + velocities[1]) / 2
    elif len(velocities) == 1:
        print("仅使用单眼速度数据")
        return velocities[0]
    else:
        raise ValueError("无法计算眼动速度，两只眼睛的gaze数据都不可用")

def compute_eye_state(data: pd.DataFrame, fixation_factor: float = 0.5, use_combined: bool = True) -> pd.DataFrame:
    """
    根据眼动数据计算每帧的眼动状态：
      - fixation：当帧间眼动速度 < (平均速度 × fixation_factor)
      - saccade：否则
    
    参数:
    - data: 包含眼动数据的DataFrame
    - fixation_factor: 用于计算凝视阈值的因子
    - use_combined: 是否使用两只眼睛的综合数据
    """
    try:
        if use_combined:
            # 尝试使用两只眼睛的数据
            velocity = compute_combined_eyes_velocity(data)
        else:
            # 只使用左眼数据
            velocity = compute_eye_velocity(data, 'gaze_0')
    except ValueError as e:
        print(f"错误: {e}")
        print("尝试替代方案...")
        try:
            velocity = compute_eye_velocity(data, 'gaze_0')
        except ValueError:
            try:
                velocity = compute_eye_velocity(data, 'gaze_1')
            except ValueError:
                print("致命错误: 无法获取任何眼动速度数据")
                sys.exit(1)
    
    mean_velocity = velocity.mean()
    threshold = mean_velocity * fixation_factor
    print(f"平均眼动速度：{mean_velocity:.4f}，设定 fixation 阈值为：{threshold:.4f}")
    data['is_fixation'] = velocity < threshold
    data['is_saccade'] = ~data['is_fixation']
    data['eye_velocity'] = velocity  # 保存速度数据以便后续分析
    return data

def second_level_segmentation(data: pd.DataFrame, fps: float = None, time_unit: float = 1.0) -> pd.DataFrame:
    """
    将帧级眼动状态聚合到指定时间单位级别：
      每个时间单位内若 fixation 帧比例 ≥ 50%，则该单位为"凝视"，否则为"扫视"。
    
    参数:
    - data: 包含is_fixation列的DataFrame
    - fps: 视频帧率
    - time_unit: 时间分段单位，默认1.0秒，可以设为0.5表示半秒
    """
    if 'timestamp' in data.columns:
        data['time_in_sec'] = data['timestamp'].astype(float)
    elif 'frame' in data.columns and fps is not None:
        data['time_in_sec'] = data['frame'] / fps
    else:
        raise ValueError("必须存在 'timestamp' 列，或同时提供 'frame' 列和 fps 参数。")
    
    # 根据time_unit划分时间段
    data['time_unit'] = (data['time_in_sec'] / time_unit).apply(np.floor).astype(int)
    
    # 按时间单位分组统计
    groups = data.groupby('time_unit')
    results = []
    
    for unit, grp in groups:
        n_total = len(grp)
        n_fix = grp['is_fixation'].sum()
        fix_ratio = n_fix / n_total
        label = 'fixation' if fix_ratio >= 0.5 else 'saccade'
        
        results.append({
            'time_unit': unit,
            'start_time': unit * time_unit,
            'end_time': (unit + 1) * time_unit,
            'n_frames': n_total,
            'fix_ratio': fix_ratio,
            'sac_ratio': 1 - fix_ratio,
            'label': label
        })
    
    df_res = pd.DataFrame(results)
    return df_res

def merge_consecutive_segments(df: pd.DataFrame) -> pd.DataFrame:
    """合并连续相同状态的时间段记录"""
    if df.empty:
        return df
    
    df = df.sort_values(by='time_unit').reset_index(drop=True)
    merged = []
    current = df.iloc[0].to_dict()
    
    for i in range(1, len(df)):
        row = df.iloc[i].to_dict()
        if (row['label'] == current['label'] and 
            abs(row['start_time'] - current['end_time']) < 0.001):  # 允许浮点误差
            # 合并时间段
            total_frames = current['n_frames'] + row['n_frames']
            current['n_frames'] = total_frames
            current['end_time'] = row['end_time']
            # 更新比例
            current['fix_ratio'] = (current['fix_ratio'] * current['n_frames'] + 
                                  row['fix_ratio'] * row['n_frames']) / total_frames
            current['sac_ratio'] = 1 - current['fix_ratio']
        else:
            merged.append(current)
            current = row
    
    merged.append(current)
    return pd.DataFrame(merged)

def adjust_short_fixations(df: pd.DataFrame, min_duration: float = 1.0) -> pd.DataFrame:
    """
    如果某段状态为 fixation 且持续时间小于 min_duration 秒，
    则将该段状态调整为 saccade。
    
    参数:
    - df: 包含眼动段的DataFrame
    - min_duration: 最小凝视持续时间（秒）
    """
    if df.empty:
        return df
    
    df = df.sort_values(by='start_time').reset_index(drop=True)
    df_adjusted = df.copy()
    
    for i in range(len(df_adjusted)):
        duration = df_adjusted.loc[i, 'end_time'] - df_adjusted.loc[i, 'start_time']
        if df_adjusted.loc[i, 'label'] == 'fixation' and duration < min_duration:
            df_adjusted.loc[i, 'label'] = 'saccade'
            df_adjusted.loc[i, 'fix_ratio'] = 0.0
            df_adjusted.loc[i, 'sac_ratio'] = 1.0
    
    merged_df = merge_consecutive_segments(df_adjusted)
    return merged_df

def print_eye_data_info(data: pd.DataFrame):
    """打印数据中的眼动相关列信息"""
    eye_columns = [col for col in data.columns if any(key in col.lower() for key in ['gaze', 'eye', 'au45', 'pupil'])]
    print(f"\n找到 {len(eye_columns)} 个可能的眼动相关列:")
    
    categories = {
        'gaze_0': [],
        'gaze_1': [],
        'eye_lmk': [],
        'au45': [],
        'other_eye': []
    }
    
    for col in eye_columns:
        if 'gaze_0' in col:
            categories['gaze_0'].append(col)
        elif 'gaze_1' in col:
            categories['gaze_1'].append(col)
        elif 'eye_lmk' in col:
            categories['eye_lmk'].append(col)
        elif 'au45' in col:
            categories['au45'].append(col)
        else:
            categories['other_eye'].append(col)
    
    for category, cols in categories.items():
        if cols:
            print(f"\n{category} 相关列 ({len(cols)}):")
            for col in cols[:5]:
                print(f"  - {col}")
            if len(cols) > 5:
                print(f"  ... 以及 {len(cols) - 5} 个其他列")

def main():
    print("=== OpenFace眼动分析程序 ===")
    file_path = input("请输入 OpenFace CSV 文件路径: ").strip()
    data = load_openface_data(file_path)
    
    print_eye_data_info(data)
    
    use_combined = True
    try:
        if 'gaze_0_x' in data.columns and 'gaze_0_y' in data.columns:
            print("检测到左眼(gaze_0)数据")
        else:
            print("警告: 未检测到左眼(gaze_0)数据")
            use_combined = False
            
        if 'gaze_1_x' in data.columns and 'gaze_1_y' in data.columns:
            print("检测到右眼(gaze_1)数据")
        else:
            print("警告: 未检测到右眼(gaze_1)数据")
            use_combined = False
    except Exception as e:
        print(f"检查眼动数据时出错: {e}")
        use_combined = False
    
    if 'is_fixation' not in data.columns or 'is_saccade' not in data.columns:
        print("\n未发现眼动状态标记，开始依据 gaze 数据计算眼动状态...")
        data = compute_eye_state(data, fixation_factor=0.5, use_combined=use_combined)
    
    time_unit = 0.5
    try:
        time_unit_input = input(f"请输入时间分析单位(默认{time_unit}秒): ").strip()
        if time_unit_input:
            time_unit = float(time_unit_input)
    except:
        print(f"使用默认时间单位: {time_unit}秒")
    
    min_fixation = 1.0
    try:  
        min_fix_input = input(f"请输入最小凝视持续时间(默认{min_fixation}秒): ").strip()
        if min_fix_input:
            min_fixation = float(min_fix_input)
    except:
        print(f"使用默认最小凝视持续时间: {min_fixation}秒")
    
    fps = None
    if 'timestamp' not in data.columns:
        if 'frame' in data.columns:
            try:
                fps = float(input("请输入视频帧率 (fps): "))
            except Exception as e:
                print(f"无效的帧率输入: {e}")
                sys.exit(1)
        else:
            print("数据中既没有 'timestamp' 也没有 'frame' 列，无法计算时间。")
            sys.exit(1)
    
    print(f"\n开始按 {time_unit} 秒为单位进行眼动状态聚合...")
    time_df = second_level_segmentation(data, fps=fps, time_unit=time_unit)
    
    print("合并连续相同状态的时间段...")
    merged_df = merge_consecutive_segments(time_df)
    
    print(f"调整短于 {min_fixation} 秒的凝视段...")
    final_df = adjust_short_fixations(merged_df, min_duration=min_fixation)
    
    print("\n最终眼动状态时间段:")
    for _, row in final_df.iterrows():
        duration = row['end_time'] - row['start_time']
        if row['label'] == 'fixation':
            print(f"{row['start_time']:.2f}秒 ~ {row['end_time']:.2f}秒 (持续{duration:.2f}秒) 为凝视，累计 {row['n_frames']} 帧")
        else:
            print(f"{row['start_time']:.2f}秒 ~ {row['end_time']:.2f}秒 (持续{duration:.2f}秒) 为扫视，累计 {row['n_frames']} 帧")
    
    fix_segments = final_df[final_df['label'] == 'fixation']
    sac_segments = final_df[final_df['label'] == 'saccade']
    
    total_time = final_df['end_time'].max() - final_df['start_time'].min()
    fix_time = fix_segments['end_time'].sum() - fix_segments['start_time'].sum()
    sac_time = sac_segments['end_time'].sum() - sac_segments['start_time'].sum()
    
    print(f"\n总分析时长: {total_time:.2f}秒")
    print(f"总凝视时长: {fix_time:.2f}秒 ({fix_time/total_time*100:.1f}%)")
    print(f"总扫视时长: {sac_time:.2f}秒 ({sac_time/total_time*100:.1f}%)")
    print(f"凝视段数量: {len(fix_segments)}个")
    print(f"扫视段数量: {len(sac_segments)}个")
    
    if not fix_segments.empty:
        avg_fix_duration = (fix_segments['end_time'] - fix_segments['start_time']).mean()
        print(f"平均凝视持续时间: {avg_fix_duration:.2f}秒")
    
    if not sac_segments.empty:
        avg_sac_duration = (sac_segments['end_time'] - sac_segments['start_time']).mean()
        print(f"平均扫视持续时间: {avg_sac_duration:.2f}秒")
    
    # 保存结果到 "扫视凝视结果" 文件夹
    output_folder = "扫视凝视结果"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_file = os.path.join(output_folder, "eye_movement_analysis.csv")
    final_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n分析结果已保存到 '{output_file}'")
    
    try:
        plt.figure(figsize=(12, 6))
        for i, row in final_df.iterrows():
            start = row['start_time']
            end = row['end_time']
            y = 1 if row['label'] == 'fixation' else 0
            color = 'blue' if row['label'] == 'fixation' else 'red'
            plt.plot([start, end], [y, y], linewidth=8, color=color)
        
        plt.yticks([0, 1], ['扫视', '凝视'])
        plt.xlabel('时间(秒)')
        plt.title('眼动状态时间段')
        plt.grid(True, axis='x')
        
        plt_file = os.path.join(output_folder, "eye_movement_timeline.png")
        plt.savefig(plt_file)
        print(f"时间段图表已保存到 '{plt_file}'")
    except Exception as e:
        print(f"绘制图表时出错: {e}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n程序执行出错: {e}")
        import traceback
        traceback.print_exc()
