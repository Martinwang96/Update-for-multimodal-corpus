'''
头部运动分析封装器
封装 head_common 模块，提供 pitch/roll/yaw 三轴统一接口
'''

import os
import sys
from pathlib import Path

# 将 头部/ 目录加入 sys.path
_HEAD_DIR = Path(__file__).resolve().parent.parent / '头部'
if str(_HEAD_DIR) not in sys.path:
    sys.path.insert(0, str(_HEAD_DIR))

import head_common


class HeadAnalyzer:
    """头部运动分析统一封装（抬头/低头、左倾/右倾、转头/摇头）。"""

    AXES = {
        'pitch': '抬头/低头 (pose_Rx)',
        'roll': '左倾/右倾 (pose_Rz)',
        'yaw': '转头/摇头 (pose_Ry)',
    }

    def preview(self, csv_path, axis):
        """预览模式：返回推荐阈值与统计（不执行区间检测）。"""
        if axis not in head_common.AXIS_CONFIGS:
            return {'status': 'error', 'message': f'未知轴: {axis}'}
        try:
            return head_common.preview_thresholds(file_path=csv_path, axis=axis)
        except Exception as e:
            return {'status': 'error', 'message': f'预览出错: {e}'}

    def analyze(self, csv_path, axis, params=None, output_dir=None):
        """
        运行指定轴的头部运动分析。
        axis: 'pitch' | 'roll' | 'yaw'
        params: 可选参数 dict (upper_threshold, lower_threshold, delta,
                min_duration, max_duration, min_shake_turns, max_shake_gap)
        返回结果 dict。
        """
        if axis not in head_common.AXIS_CONFIGS:
            return {'status': 'error', 'message': f'未知轴: {axis}'}

        try:
            result = head_common.analyze_programmatic(
                file_path=csv_path,
                axis=axis,
                params=params,
                output_dir=output_dir,
            )
        except Exception as e:
            return {'status': 'error', 'message': f'分析出错: {e}'}

        return result

    def list_axes(self):
        """列出可用分析轴。"""
        return self.AXES
