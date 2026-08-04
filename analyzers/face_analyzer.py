'''
面部分析封装器
将 facialexpression / mi&zha&bi / fix&sca 三个模块统一封装为可编程调用接口
供 Flask Web 应用调用
'''

import os
import sys
import importlib.util
from pathlib import Path

# 将 面部/ 目录加入 sys.path 以便导入其模块
_FACE_DIR = Path(__file__).resolve().parent.parent / '面部'
if str(_FACE_DIR) not in sys.path:
    sys.path.insert(0, str(_FACE_DIR))


def _load_module(filename, alias):
    """从 面部/ 目录动态加载模块。"""
    p = _FACE_DIR / filename
    if not p.is_file():
        raise FileNotFoundError(f"模块文件未找到: {p}")
    spec = importlib.util.spec_from_file_location(alias, str(p))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class FaceAnalyzer:
    """面部表情/眼部状态/眼动 统一分析封装。"""

    MODULES = {
        'expression': {'file': 'facialexpression.py', 'label': '面部表情（微笑/张嘴/皱眉/专注）'},
        'eye': {'file': 'mi&zha&bi.py', 'label': '眼部状态（眯眼/闭眼/眨眼）'},
        'gaze': {'file': 'fix&sca-0.6&0.6.py', 'label': '眼动分析（注视/扫视）'},
    }

    def preview(self, csv_path, subtype='expression'):
        """预览模式：返回数据统计与推荐参数（不执行区间检测）。"""
        from face_common import load_openface_csv
        try:
            data = load_openface_csv(csv_path)
            if data is None or data.empty:
                return {'status': 'error', 'message': 'CSV 为空或无法解析'}
        except Exception as e:
            return {'status': 'error', 'message': f'读取 CSV 失败: {e}'}

        frame_count = len(data)
        # 帧率
        frame_rate = 30.0
        if 'timestamp' in data.columns:
            ts = data['timestamp'].dropna()
            if len(ts) > 1:
                dt = ts.diff().dropna().mean()
                if dt and 1/dt > 5 and 1/dt < 200:
                    frame_rate = round(1/dt, 2)
        duration_sec = round(frame_count / frame_rate, 2) if frame_rate else 0

        result = {
            'status': 'ok',
            'subtype': subtype,
            'frame_count': frame_count,
            'frame_rate': frame_rate,
            'duration_sec': duration_sec,
            'columns': list(data.columns)[:80],
        }

        if subtype == 'expression':
            # AU 强度统计（_r 列）
            au_r_cols = [c for c in data.columns if c.endswith('_r') and c.startswith('AU')]
            au_stats = {}
            for c in au_r_cols:
                col = data[c]
                au_stats[c] = {
                    'mean': round(float(col.mean()), 3),
                    'std': round(float(col.std()), 3),
                    'min': round(float(col.min()), 3),
                    'max': round(float(col.max()), 3),
                    'p90': round(float(col.quantile(0.9)), 3),
                }
            result['au_stats'] = au_stats
            result['recommended'] = {
                'happy_au6_r': 1.5, 'happy_au12_r': 1.5, 'confused_au4_r': 1.5,
                'min_duration': 1.0,
            }
            result['subtype_label'] = '面部表情（微笑/张嘴/皱眉/专注）'
        elif subtype == 'eye':
            # 眨眼分类列 AU45_c 统计
            if 'AU45_c' in data.columns:
                blink_frames = int((data['AU45_c'] > 0.5).sum())
                result['au45_c_rate'] = round(blink_frames / frame_count, 3) if frame_count else 0
            result['recommended'] = {
                'ear_threshold': 0.18,
                'min_squint_duration': 0.5,
                'long_closed_eyes_threshold': 1.0,
                'min_blink_duration': 0.3,
            }
            result['subtype_label'] = '眼部状态（眯眼/闭眼/眨眼）'
        elif subtype == 'gaze':
            if 'gaze_angle_x' in data.columns:
                result['gaze_stats'] = {
                    'gaze_angle_x_std': round(float(data['gaze_angle_x'].std()), 3),
                    'gaze_angle_y_std': round(float(data['gaze_angle_y'].std()), 3) if 'gaze_angle_y' in data.columns else 0,
                }
            result['recommended'] = {}
            result['subtype_label'] = '眼动分析（注视/扫视）'
        return result

    def analyze_expression(self, csv_path, params=None, output_dir=None, visualize=False):
        """运行表情检测，返回结果 dict。"""
        mod = _load_module('facialexpression.py', 'face_expr_mod')
        results = mod.detect_expression(file_path=csv_path, visualize=visualize, params=params)
        # results 是 (dict_of_intervals, data) 但原函数返回 {} 或 results dict
        # detect_expression 返回 results dict (包含 微笑/张嘴/皱眉/专注 区间列表)
        if isinstance(results, tuple):
            intervals_dict, data = results
        else:
            intervals_dict, data = results, None

        summary = {}
        total = 0
        for expr, ivs in (intervals_dict or {}).items():
            summary[expr] = len(ivs)
            total += len(ivs)

        return {
            'status': 'ok',
            'intervals': intervals_dict or {},
            'summary': summary,
            'total_events': total,
            'report_dir': output_dir or os.path.dirname(csv_path),
        }

    def analyze_eye(self, csv_path, params=None, output_dir=None):
        """运行眼部状态检测（眯眼/闭眼/眨眼）。"""
        mod = _load_module('mi&zha&bi.py', 'face_eye_mod')
        # detect_eye_states 返回 (眯眼区间, 闭眼区间, 眨眼区间, data, frame_rate)
        # 必须传params（哪怕是空 dict），否则会触发模块内 ask_float() 交互式
        # input() 调用，在 Web 场景下会一直阻塞等待终端输入，导致前端请求“假死”。
        try:
            squint, closed, blink, data, fps = mod.detect_eye_states(
                file_path=csv_path, visualize=False, params=params or {}
            )
        except Exception as e:
            return {'status': 'error', 'message': str(e)}

        return {
            'status': 'ok',
            'intervals': {'眯眼': squint, '闭眼': closed, '眨眼': blink},
            'summary': {'眯眼': len(squint), '闭眼': len(closed), '眨眼': len(blink)},
            'total_events': len(squint) + len(closed) + len(blink),
            'frame_rate': fps,
            'report_dir': output_dir or os.path.dirname(csv_path),
        }

    def analyze_gaze(self, csv_path, params=None, output_dir=None):
        """运行眼动分析（注视/扫视）。"""
        mod = _load_module('fix&sca-0.6&0.6.py', 'face_gaze_mod')
        try:
            # fix&sca 模块的主函数签名可能不同，尝试通用调用
            if hasattr(mod, 'detect_fixations_and_saccades'):
                result = mod.detect_fixations_and_saccades(csv_path)
            elif hasattr(mod, 'main'):
                # 退而求其次，使用 main 但需要非交互
                return {'status': 'error', 'message': '眼动模块需要非交互式接口，暂未适配'}
            else:
                return {'status': 'error', 'message': '眼动模块未找到可调用入口'}
            return {
                'status': 'ok',
                'result': result,
                'report_dir': output_dir or os.path.dirname(csv_path),
            }
        except Exception as e:
            return {'status': 'error', 'message': f'眼动分析出错: {e}'}

    def list_modules(self):
        """列出当前 Web 端开放的模块。"""
        return {
            'expression': self.MODULES['expression']['label'],
            'eye': self.MODULES['eye']['label'],
        }
