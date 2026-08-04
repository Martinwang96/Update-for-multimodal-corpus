'''
躯体动作分析封装器
完整流程：上传视频 → MMPose 识别 → 关键点 JSON → 4 模块分析 → 事件 CSV
- analyze_from_video(): 视频输入主流程（依赖 mmpose）
- analyze_from_json():  已有关键点 JSON，跳过识别阶段（不需 mmpose）
'''

import os
import sys
import importlib.util
from pathlib import Path

# 将 躯体/ 目录加入 sys.path
_BODY_DIR = Path(__file__).resolve().parent.parent / '躯体'
if str(_BODY_DIR) not in sys.path:
    sys.path.insert(0, str(_BODY_DIR))


def _load_pipeline_module():
    """动态加载 躯体/综合处理-2d.py 模块。"""
    p = _BODY_DIR / '综合处理-2d.py'
    if not p.is_file():
        raise FileNotFoundError(f"躯体分析模块未找到: {p}")
    spec = importlib.util.spec_from_file_location('body_pipeline', str(p))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class BodyAnalyzer:
    """躯体动作分析统一封装（倾斜/耸肩/位移/转动）。"""

    MODULES = {
        'tilt': '身体倾斜',
        'shrug': '耸肩',
        'displacement': '身体位移',
        'rotation': '身体转动',
    }

    def default_params(self):
        """返回躯体四个子模块的默认参数。"""
        mod = _load_pipeline_module()
        return mod.build_default_params()

    def _normalize_params(self, params):
        """兼容两种传参方式：顶层配置覆盖 / params 内嵌模块参数。"""
        if not params:
            return None
        if 'params' in params:
            return params
        if any(k in self.MODULES for k in params.keys()):
            return {'params': params}
        return params

    def analyze_from_json(self, json_path, output_dir, params=None, progress_callback=None):
        """从 MMPose 关键点 JSON 运行 4 模块分析（跳过识别阶段）。"""
        mod = _load_pipeline_module()
        config = {
            'entry_mode': 'json',
            'run_root': str(output_dir),
            'json_path': str(json_path),
        }
        normalized_params = self._normalize_params(params)
        if normalized_params:
            config.update(normalized_params)
        config['parallel'] = False
        try:
            summary = mod.run_pipeline(config, progress_callback=progress_callback)
            return {'status': 'ok', 'summary': summary, 'output_dir': str(output_dir)}
        except Exception as e:
            return {'status': 'error', 'message': f'躯体分析出错: {e}'}

    def analyze_from_video(self, video_path, output_dir, recognition_cfg=None,
                           params=None, progress_callback=None):
        """从视频运行完整流水线（识别 + 分析）。"""
        mod = _load_pipeline_module()
        config = {
            'entry_mode': 'video',
            'run_root': str(output_dir),
            'video_path': str(video_path),
            'recognition': recognition_cfg or {},
        }
        normalized_params = self._normalize_params(params)
        if normalized_params:
            config.update(normalized_params)
        config['parallel'] = False
        try:
            summary = mod.run_pipeline(config, progress_callback=progress_callback)
            return {'status': 'ok', 'summary': summary, 'output_dir': str(output_dir)}
        except Exception as e:
            return {'status': 'error', 'message': f'躯体分析出错: {e}'}

    def list_modules(self):
        """列出可用分析模块。"""
        return self.MODULES
