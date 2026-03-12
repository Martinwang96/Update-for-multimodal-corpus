'''
Author: Martinwang96 -git
Date: 2025-04-02 16:45:25
Contact: martingwang01@163.com
LONG LIVE McDonald's
Copyright (c) 2025 by Martin Wang in Language of Sciences, Shanghai International Studies University, All Rights Reserved. 
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import os
from datetime import datetime

rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False

def detect_expression(file_path=None, visualize=True):
    """检测指定的表情组合，结合存在性（c）和强度（r）"""
    
    if file_path is None:
        file_path = input("请输入OpenFace CSV文件路径：")
    
    try:
        data = pd.read_csv(file_path)
        data.columns = data.columns.str.strip()
    except Exception as e:
        print(f"文件读取失败：{e}")
        return {}, None
    
    # 必要表情列检查（新增AU9, AU15, AU20相关列）
    required_columns = [
        'AU01_c', 'AU02_c', 'AU04_c', 'AU05_c', 'AU06_c', 'AU07_c', 'AU09_c', 
        'AU10_c', 'AU12_c', 'AU14_c', 'AU15_c', 'AU17_c', 'AU20_c', 'AU23_c', 
        'AU25_c', 'AU26_c', 'AU28_c', 'AU45_c',
        'AU06_r', 'AU07_r', 'AU09_r', 'AU10_r', 'AU12_r', 'AU15_r', 'AU17_r',
        'AU20_r', 'AU25_r', 'AU26_r', 'AU05_r', 'AU14_r', 'AU45_r'
    ]
    missing = [col for col in required_columns if col not in data.columns]
    if missing:
        print(f"数据缺少必要列：{', '.join(missing)}")
        return {}, data
    
    # 计算帧率
    if 'timestamp' in data.columns and len(data) > 1:
        avg_time = np.mean(np.diff(data['timestamp']))
        frame_rate = 1 / avg_time if avg_time > 0 else 30
    else:
        frame_rate = 30
    print(f"检测到帧率：{frame_rate:.1f} fps")
    
    # 用户参数设置（新增厌烦相关参数）
    print("\n请设置检测参数（回车使用默认值）：")
    
    # 微笑参数（更新增强版的微笑参数）
    happy_au6_r = float(input("微笑（AU06）强度阈值，默认1.5：") or 1.5)
    happy_au7_r = float(input("微笑（AU07）强度阈值，默认1.5：") or 1.5)
    happy_au12_r = float(input("微笑（AU12）强度阈值，默认1.5：") or 1.5)
    happy_au25_r = float(input("微笑（AU25）强度阈值，默认1.5：") or 1.5)
    happy_au26_r = float(input("微笑（AU26）强度阈值，默认1.5：") or 1.5)
    
    # 张嘴参数
    surprise_au26_r = float(input("张嘴（AU26）强度阈值，默认1.5：") or 1.5)
    
    # 皱眉参数
    confused_au4_r = float(input("皱眉（AU04）强度阈值，默认1.5：") or 1.5)

    
    # 专注参数
    focused_au5_r = float(input("专注（AU05）强度阈值，默认1.5：") or 1.5)
    focused_au14_r = float(input("专注（AU14）强度阈值，默认1.5：") or 1.5)
    
    # 厌烦参数（新增）
    # bored_au9_r = float(input("厌烦（AU09 鼻子皱起）强度阈值，默认1.5：") or 1.5)
    # bored_au15_r = float(input("厌烦（AU15 嘴角下拉）强度阈值，默认1.5：") or 1.5)
    # bored_au10_r = float(input("厌烦（AU10 上唇提升）强度阈值，默认1.5：") or 1.5)
    # bored_au17_r = float(input("厌烦（AU17 下巴提升）强度阈值，默认1.5：") or 1.5)
    # bored_au20_r = float(input("厌烦（AU20 嘴角水平）强度阈值，默认1.5：") or 1.5)
    
    # 持续时间参数
    min_duration = float(input("表情最小持续时间（秒，默认1.0）：") or 1.0)
    min_frames = int(min_duration * frame_rate)
    
    # 初始化帧和秒信息
    if 'frame' not in data.columns:
        data['frame'] = np.arange(len(data))
    
    if 'timestamp' in data.columns:
        data['second'] = data['timestamp'].round(2)
    else:
        data['second'] = data['frame'] / frame_rate
    
    # 定义表情组合规则
    def is_happy(row):
        # 原有规则
        original_condition = (
            (row['AU06_c'] and row['AU12_c'] and 
             row['AU06_r'] > happy_au6_r and row['AU12_r'] > happy_au12_r) or
            (row['AU06_c'] and row['AU26_c'] and 
             row['AU06_r'] > happy_au6_r and row['AU26_r'] > happy_au26_r)
        )
        
        # 新增规则: AU7+AU12+AU25+AU26的同时出现和AU45的缺失
        enhanced_condition = (
            row['AU07_c'] and row['AU12_c'] and row['AU25_c'] and row['AU26_c'] and
            not row['AU45_c'] and
            row['AU07_r'] > happy_au7_r and 
            row['AU12_r'] > happy_au12_r and
            row['AU25_r'] > happy_au25_r and
            row['AU26_r'] > happy_au26_r
        )
        
        return original_condition or enhanced_condition

    def is_surprise(row):
        return (
            (row['AU26_c'] and row['AU26_r'] > surprise_au26_r) 
        )

    def is_confused(row):
        return (
            (row['AU04_c'] and row['AU04_r'] > confused_au4_r)
        )

    def is_focused(row):
        return (
            (row['AU05_c'] and row['AU14_c'] and 
             row['AU05_r'] > focused_au5_r and row['AU14_r'] > focused_au14_r)
        )
    
    # def is_bored(row):  # 新增厌烦判断函数
    #     return (
    #         # AU9 单独存在
    #         (row['AU09_c'] and row['AU09_r'] > bored_au9_r) or
    #         # AU15 单独存在
    #         (row['AU15_c'] and row['AU15_r'] > bored_au15_r) or
    #         # AU9 + AU15 组合
    #         (row['AU09_c'] and row['AU15_c'] and 
    #          row['AU09_r'] > bored_au9_r and row['AU15_r'] > bored_au15_r) or
    #         # AU9 + AU20 组合
    #         (row['AU09_c'] and row['AU20_c'] and 
    #          row['AU09_r'] > bored_au9_r and row['AU20_r'] > bored_au20_r) or
    #         # AU10 单独存在
    #         (row['AU10_c'] and row['AU10_r'] > bored_au10_r) or
    #         # AU10 + AU17 组合
    #         (row['AU10_c'] and row['AU17_c'] and 
    #          row['AU10_r'] > bored_au10_r and row['AU17_r'] > bored_au17_r)
    #    )
    
    # 初始化表情标记列
    data['is_happy'] = data.apply(is_happy, axis=1)
    data['is_surprise'] = data.apply(is_surprise, axis=1)
    data['is_confused'] = data.apply(is_confused, axis=1)
    data['is_focused'] = data.apply(is_focused, axis=1)
    # data['is_bored'] = data.apply(is_bored, axis=1)  # 新增厌烦标记
    
    # 检测持续区间函数
    def detect_expression_intervals(expr_col, min_f):
        intervals = []
        in_interval = False
        start = None
        
        for idx in data.index:
            if data.loc[idx, expr_col]:
                if not in_interval:
                    in_interval = True
                    start = idx
            else:
                if in_interval:
                    duration = idx - start
                    if duration >= min_f:
                        end = idx-1
                        intervals.append({
                            'start_frame': data.loc[start, 'frame'],
                            'end_frame': data.loc[end, 'frame'],
                            'start_time': data.loc[start, 'second'],
                            'end_time': data.loc[end, 'second'],
                            'duration': duration / frame_rate
                        })
                    in_interval = False
                    start = None
        
        if in_interval:
            end = data.index[-1]
            duration = end - start
            if duration >= min_f:
                intervals.append({
                    'start_frame': data.loc[start, 'frame'],
                    'end_frame': data.loc[end, 'frame'],
                    'start_time': data.loc[start, 'second'],
                    'end_time': data.loc[end, 'second'],
                    'duration': duration / frame_rate
                })
        
        return intervals
    
    # 检测各表情组合
    results = {}
    results['微笑'] = detect_expression_intervals('is_happy', min_frames)
    results['张嘴'] = detect_expression_intervals('is_surprise', min_frames)
    results['皱眉'] = detect_expression_intervals('is_confused', min_frames)
    results['专注'] = detect_expression_intervals('is_focused', min_frames)
    # results['厌烦'] = detect_expression_intervals('is_bored', min_frames)  # 新增厌烦检测结果
    
    # 可视化
    if visualize:
        fig, ax = plt.subplots(figsize=(14, 7))
        colors = {
            '微笑': 'limegreen',
            '张嘴': 'red',
            '皱眉': 'purple',
            '专注': 'blue',
            '厌烦': 'darkorange'  # 新增厌烦颜色
        }
        
        # 绘制关键AU强度曲线
        ax.plot(data['frame'], data['AU06_r'], label='AU06（脸颊提升）', alpha=0.3)
        ax.plot(data['frame'], data['AU07_r'], label='AU07（眼睑收紧）', alpha=0.3)
        ax.plot(data['frame'], data['AU12_r'], label='AU12（嘴角上扬）', alpha=0.3)
        ax.plot(data['frame'], data['AU25_r'], label='AU25（嘴唇分开）', alpha=0.3)
        ax.plot(data['frame'], data['AU26_r'], label='AU26（下巴下降）', alpha=0.3)
        ax.plot(data['frame'], data['AU05_r'], label='AU05（上眼睑抬起）', alpha=0.3)
        ax.plot(data['frame'], data['AU09_r'], label='AU09（鼻子皱起）', alpha=0.3)  # 新增
        ax.plot(data['frame'], data['AU10_r'], label='AU10（上唇提升）', alpha=0.3)
        ax.plot(data['frame'], data['AU14_r'], label='AU14（嘴角凹陷）', alpha=0.3)
        ax.plot(data['frame'], data['AU15_r'], label='AU15（嘴角下拉）', alpha=0.3)  # 新增
        ax.plot(data['frame'], data['AU17_r'], label='AU17（下巴提升）', alpha=0.3)
        ax.plot(data['frame'], data['AU20_r'], label='AU20（嘴角拉伸）', alpha=0.3)  # 新增
        
        # 标记表情区间
        for expr_type, intervals in results.items():
            for interval in intervals:
                ax.axvspan(
                    interval['start_frame'],
                    interval['end_frame'],
                    color=colors[expr_type],
                    alpha=0.3,
                    label=f'{expr_type}动作' if interval['start_frame'] == intervals[0]['start_frame'] else ""
                )
        
        ax.set_title('表情检测分析（强度与存在性结合）')
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        ax.set_xlabel('帧编号')
        ax.set_ylabel('表情强度（0-5）')
        plt.tight_layout()
        
        # 保存结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = "表情分析结果"
        os.makedirs(output_dir, exist_ok=True)
        
        plt.savefig(f"{output_dir}/表情分析_{timestamp}.png", dpi=150, bbox_inches='tight')
        
        # 生成CSV报告
        report = []
        for expr_type, intervals in results.items():
            for interval in intervals:
                report.append({
                    '表情类型': expr_type,
                    '开始时间': interval['start_time'],
                    '结束时间': interval['end_time'],
                    '持续时间(s)': round(interval['duration'], 2)
                })
        
        df = pd.DataFrame(report)
        df.to_csv(f"{output_dir}/表情报告_{timestamp}.csv", index=False)
        print(f"结果已保存到：{output_dir}")
        plt.show()
    
    return results

def main():
    print("====== 专业表情检测工具（强度+存在性） ======")
    
    file_path = input("输入CSV文件路径（回车使用文件选择器）：")
    if not file_path.strip():
        try:
            from tkinter import Tk, filedialog
            root = Tk()
            root.withdraw()
            file_path = filedialog.askopenfilename(
                title="选择OpenFace CSV文件",
                filetypes=[("CSV文件", "*.csv")]
            )
            if not file_path:
                print("未选择文件，退出程序。")
                return
        except ImportError:
            print("手动输入路径：")
            file_path = input()
    
    try:
        results = detect_expression(file_path, visualize=True)
        print("检测完成！")
    except Exception as e:
        print(f"运行出错：{str(e)}")

if __name__ == "__main__":
    main()