#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
非言语行为分析平台 - 统一 Flask 应用
整合面部 / 头部 / 躯体三大分析模块
Author: Martinwang96
'''

import json
import math
import os
import sys
import uuid
import threading
import traceback
from pathlib import Path

from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge


def json_safe(obj):
    """递归转换对象为 JSON 可序列化的原生类型。"""
    if obj is None:
        return None
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, int):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, str):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [json_safe(v) for v in obj]
    # numpy 标量/数组
    if hasattr(obj, 'item') and callable(getattr(obj, 'item')):
        try:
            return json_safe(obj.item())
        except Exception:
            pass
    if hasattr(obj, 'tolist') and callable(getattr(obj, 'tolist')):
        try:
            return json_safe(obj.tolist())
        except Exception:
            pass
    # 兜底：转字符串，避免整个响应崩溃
    try:
        import json as _json
        _json.dumps(obj)
        return obj
    except Exception:
        return str(obj)


def api_error(message, status=400, code=None, detail=None):
    payload = {'status': 'error', 'error': message}
    if code:
        payload['code'] = code
    if detail is not None:
        payload['detail'] = json_safe(detail)
    return jsonify(payload), status

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / 'uploads'
UPLOAD_DIR.mkdir(exist_ok=True)
RESULT_DIR = BASE_DIR / 'results'
RESULT_DIR.mkdir(exist_ok=True)

app = Flask(__name__, template_folder=str(BASE_DIR / 'templates'))
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB

# 任务存储（生产环境应换用 Redis/数据库）
TASKS = {}
TASK_LOCK = threading.Lock()

# 预览暂存：preview_id -> {csv_path, module, subtype/axis, ...}
PREVIEWS = {}
PREVIEW_LOCK = threading.Lock()

# 共享上传文件：file_id -> {path, filename, kind, openface_csv}
# 一份文件可被多个模块共享预览/分析；视频转 CSV 结果缓存复用
UPLOADS = {}
UPLOAD_LOCK = threading.Lock()
# 每个文件的 OpenFace 转换锁，避免并发重复转换
CONVERT_LOCKS = {}

VIDEO_EXTS = ('.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv')
# 文件类型 -> 可用模块
KIND_MODULES = {
    'video': ('face', 'head', 'body'),
    'csv': ('face', 'head'),
    'json': ('body',),
}


def _get_convert_lock(file_id):
    with UPLOAD_LOCK:
        if file_id not in CONVERT_LOCKS:
            CONVERT_LOCKS[file_id] = threading.Lock()
        return CONVERT_LOCKS[file_id]


# ==============================================================
# 分析器懒加载
# ==============================================================
def get_face_analyzer():
    from analyzers import FaceAnalyzer
    return FaceAnalyzer()

def get_head_analyzer():
    from analyzers import HeadAnalyzer
    return HeadAnalyzer()

def get_body_analyzer():
    from analyzers import BodyAnalyzer
    return BodyAnalyzer()


# ==============================================================
# 后台任务执行
# ==============================================================
def run_task(task_id, func, *args, **kwargs):
    """在后台线程执行分析任务并更新状态。"""
    with TASK_LOCK:
        TASKS[task_id]['status'] = 'running'
    try:
        result = json_safe(func(*args, **kwargs))
        if isinstance(result, dict) and result.get('status') == 'error':
            raise RuntimeError(result.get('message') or result.get('error') or '分析失败')
        with TASK_LOCK:
            TASKS[task_id]['status'] = 'done'
            TASKS[task_id]['result'] = result
            TASKS[task_id]['error'] = None
            TASKS[task_id]['traceback'] = None
    except Exception as e:
        with TASK_LOCK:
            TASKS[task_id]['status'] = 'error'
            TASKS[task_id]['result'] = None
            TASKS[task_id]['error'] = str(e)
            TASKS[task_id]['traceback'] = traceback.format_exc()


def start_task(func, *args, **kwargs):
    """启动后台分析任务，返回 task_id。"""
    task_id = str(uuid.uuid4())[:8]
    with TASK_LOCK:
        TASKS[task_id] = {'status': 'pending', 'result': None, 'error': None}
    t = threading.Thread(target=run_task, args=(task_id, func, *args), kwargs=kwargs, daemon=True)
    t.start()
    return task_id


# ==============================================================
# 路由
# ==============================================================
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/modules')
def api_modules():
    """返回可用模块列表。"""
    try:
        face = get_face_analyzer().list_modules()
    except Exception:
        face = {}
    try:
        head = get_head_analyzer().list_axes()
    except Exception:
        head = {}
    try:
        body = get_body_analyzer().list_modules()
    except Exception:
        body = {}
    return jsonify({'face': face, 'head': head, 'body': body})


@app.route('/api/openface/status')
def api_openface_status():
    """检查 OpenFace 是否可用。"""
    from openface_runner import find_openface_binary
    binary = find_openface_binary()
    return jsonify({
        'available': binary is not None,
        'binary': binary,
        'env_hint': '设置 OPENFACE_BIN 环境变量指定 FeatureExtraction 路径',
    })


# ==============================================================
# 共享上传：一份文件，多模块共用
# ==============================================================
@app.route('/api/upload', methods=['POST'])
def api_upload():
    """上传文件（CSV/JSON/视频），返回 file_id 与可用模块列表。"""
    if 'file' not in request.files:
        return api_error('未上传文件', 400)
    f = request.files['file']
    fname = f.filename or ''
    lower = fname.lower()

    if lower.endswith(VIDEO_EXTS):
        kind = 'video'
    elif lower.endswith('.csv'):
        kind = 'csv'
    elif lower.endswith('.json'):
        kind = 'json'
    else:
        return api_error('不支持的文件类型（需 .csv / .json / 视频）', 400)

    file_id = f"f_{uuid.uuid4().hex[:10]}"
    ext = Path(fname).suffix
    save_path = UPLOAD_DIR / f"{file_id}{ext}"
    f.save(str(save_path))

    with UPLOAD_LOCK:
        UPLOADS[file_id] = {
            'path': str(save_path),
            'filename': fname,
            'kind': kind,
            'openface_csv': None,
        }

    return jsonify({
        'file_id': file_id,
        'filename': fname,
        'kind': kind,
        'modules': list(KIND_MODULES[kind]),
    })


def _ensure_openface_csv(file_id):
    """
    确保该视频文件已有 OpenFace CSV（缓存复用，仅转换一次）。
    返回 CSV 路径；失败抛 RuntimeError。
    """
    lock = _get_convert_lock(file_id)
    with lock:
        with UPLOAD_LOCK:
            up = UPLOADS.get(file_id)
            cached = up.get('openface_csv') if up else None
        if cached and Path(cached).is_file():
            return cached

        if not up:
            raise RuntimeError('文件不存在或已过期')

        from openface_runner import find_openface_binary, run_openface
        if not find_openface_binary():
            raise RuntimeError('视频分析需要 OpenFace，未安装（请配置 OPENFACE_BIN）')

        of_out = UPLOAD_DIR / file_id / 'openface_csv'
        result = run_openface(up['path'], of_out)
        csv_path = result['csv_path']
        with UPLOAD_LOCK:
            UPLOADS[file_id]['openface_csv'] = csv_path
        return csv_path


# ==============================================================
# 预览接口：两阶段工作流第一阶段（推荐阈值）
# ==============================================================
@app.route('/api/preview', methods=['POST'])
def api_preview():
    """
    预览阶段：基于共享上传文件（file_id）返回推荐参数与数据统计。
    - 视频 + face/head: OpenFace 转 CSV（仅首次，缓存复用）
    - 视频 + body: 返回视频基本信息
    - CSV + face/head: 直接计算
    - JSON + body: 直接返回
    """
    file_id = request.form.get('file_id', '')
    module = request.form.get('module', 'face')

    with UPLOAD_LOCK:
        up = UPLOADS.get(file_id)
    if not up:
        return api_error('文件不存在或已过期，请重新上传', 404)

    kind = up['kind']
    if module not in KIND_MODULES[kind]:
        return api_error(f'当前文件类型（{kind}）不支持该模块', 400)

    preview_id = f"pv_{uuid.uuid4().hex[:10]}"

    try:
        if module == 'head':
            axis = request.form.get('axis', 'pitch')
            csv_path = _ensure_openface_csv(file_id) if kind == 'video' else up['path']
            result = get_head_analyzer().preview(csv_path, axis)
            csv_for_analysis = csv_path
        elif module == 'face':
            sub = request.form.get('subtype', 'expression')
            if sub == 'gaze':
                return api_error('面部注视方向暂未开放，请先使用表情识别或眼部状态', 400)
            csv_path = _ensure_openface_csv(file_id) if kind == 'video' else up['path']
            result = get_face_analyzer().preview(csv_path, sub)
            csv_for_analysis = csv_path
        else:  # body
            if kind == 'video':
                import cv2
                cap = cv2.VideoCapture(up['path'])
                fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS) or 30
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()
                result = {
                    'status': 'ok', 'module': 'body',
                    'frame_count': fc, 'frame_rate': round(fps, 2),
                    'duration_sec': round(fc / fps, 2) if fps else 0,
                    'resolution': f'{w}x{h}',
                    'recognition': {'device': 'auto', 'pose2d': 'rtmo'},
                    'recommended': get_body_analyzer().default_params(),
                    'note': '可调整躯体四个子模块的默认参数；视频输入会先走 MMPose 识别。',
                }
            else:
                result = {
                    'status': 'ok', 'module': 'body',
                    'note': '可直接调整躯体四个子模块参数后开始分析。',
                    'recommended': get_body_analyzer().default_params(),
                }
            csv_for_analysis = up['path']
    except RuntimeError as e:
        return api_error(str(e), 500)
    except Exception as e:
        return api_error(f'预览失败: {e}', 500)

    if isinstance(result, dict) and result.get('status') == 'error':
        return api_error(result.get('message') or result.get('error') or '预览失败', 400)

    with PREVIEW_LOCK:
        PREVIEWS[preview_id] = {
            'csv_path': csv_for_analysis,
            'video_path': up['path'] if kind == 'video' else None,
            'kind': kind,
            'module': module,
            'subtype': request.form.get('subtype', ''),
            'axis': request.form.get('axis', ''),
            'result': json_safe(result),
        }

    return jsonify({'preview_id': preview_id, 'preview': result})


@app.route('/api/preview/<preview_id>')
def api_get_preview(preview_id):
    """获取已暂存的预览结果。"""
    with PREVIEW_LOCK:
        p = PREVIEWS.get(preview_id)
        snapshot = json_safe(dict(p)) if p else None
    if not snapshot:
        return api_error('预览不存在或已过期', 404)
    return jsonify({'preview_id': preview_id, 'preview': snapshot['result'],
                    'module': snapshot['module'], 'subtype': snapshot.get('subtype'), 'axis': snapshot.get('axis')})


@app.route('/api/analyze/final', methods=['POST'])
def api_analyze_final():
    """
    两阶段工作流第二阶段：用预览暂存的文件 + 用户确认的参数执行最终分析。
    请求体: preview_id + 各项参数。
    """
    preview_id = request.form.get('preview_id', '')
    with PREVIEW_LOCK:
        p = PREVIEWS.get(preview_id)
        snapshot = dict(p) if p else None
    if not snapshot:
        return api_error('预览不存在或已过期，请重新上传', 404)

    module = snapshot['module']
    csv_path = snapshot['csv_path']
    task_id_name = f"final_{module}_{uuid.uuid4().hex[:8]}"
    out_dir = RESULT_DIR / task_id_name
    out_dir.mkdir(parents=True, exist_ok=True)

    if module == 'head':
        axis = snapshot.get('axis') or 'pitch'
        params = {}
        for key in ['delta', 'min_duration', 'max_duration', 'upper_threshold', 'lower_threshold']:
            v = request.form.get(key)
            if v:
                try:
                    params[key] = float(v)
                except ValueError:
                    pass
        task_id = start_task(get_head_analyzer().analyze, csv_path, axis, params, str(out_dir))
    elif module == 'face':
        sub = snapshot.get('subtype') or 'expression'
        params = {}
        if sub == 'expression':
            keys = ['min_duration', 'happy_au6_r', 'happy_au7_r', 'happy_au12_r',
                     'happy_au25_r', 'happy_au26_r', 'surprise_au26_r',
                     'confused_au4_r', 'focused_au5_r', 'focused_au14_r']
        elif sub == 'eye':
            keys = ['ear_threshold', 'min_squint_duration',
                     'long_closed_eyes_threshold', 'min_blink_duration']
        else:
            keys = []
        for key in keys:
            v = request.form.get(key)
            if v:
                try:
                    params[key] = float(v)
                except ValueError:
                    pass
        analyzer = get_face_analyzer()
        if sub == 'expression':
            func = analyzer.analyze_expression
        elif sub == 'eye':
            func = analyzer.analyze_eye
        elif sub == 'gaze':
            return api_error('面部注视方向暂未开放，请先使用表情识别或眼部状态', 400)
        else:
            return api_error(f'未知子模块: {sub}', 400)
        # expression/eye 模块必须传params（哪怕是空 dict）以避免触发内部交互式input()
        task_id = start_task(func, csv_path, params if sub in ('expression', 'eye') else None, str(out_dir))
    elif module == 'body':
        body_params = {}
        raw_body_params = request.form.get('body_params_json', '')
        if raw_body_params:
            try:
                parsed = json.loads(raw_body_params)
            except json.JSONDecodeError:
                return api_error('躯体参数格式错误', 400)
            if not isinstance(parsed, dict):
                return api_error('躯体参数必须是对象', 400)
            body_params = parsed
        # body 直接分析（视频走 MMPose）
        if snapshot.get('video_path'):
            rec_cfg = {
                'python': sys.executable,
                'device': request.form.get('device', 'auto'),
                'pose2d': request.form.get('pose2d', 'rtmo'),
            }
            task_id = start_task(get_body_analyzer().analyze_from_video,
                                 snapshot['video_path'], str(out_dir), rec_cfg, body_params)
        else:
            task_id = start_task(get_body_analyzer().analyze_from_json, csv_path, str(out_dir), body_params)
    else:
        return api_error(f'未知模块: {module}', 400)

    return jsonify({'task_id': task_id, 'module': module})


@app.route('/api/task/<task_id>')
def api_task_status(task_id):
    """查询任务状态。"""
    with TASK_LOCK:
        task = TASKS.get(task_id)
        snapshot = json_safe(dict(task)) if task else None
    if not snapshot:
        return api_error('任务不存在', 404)
    return jsonify(snapshot)


@app.route('/api/results/<path:filename>')
def api_download_result(filename):
    """下载结果文件。"""
    return send_from_directory(str(RESULT_DIR), filename, as_attachment=True)


@app.route('/api/results')
def api_list_results():
    """列出结果目录。"""
    items = []
    for root, dirs, files in os.walk(RESULT_DIR):
        for fn in files:
            full = Path(root) / fn
            rel = full.relative_to(RESULT_DIR)
            items.append({
                'path': str(rel),
                'name': fn,
                'size': full.stat().st_size,
            })
    return jsonify({'results': items})


@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(error):
    return api_error('上传文件过大，请压缩后重试', 413, code='file_too_large')


@app.errorhandler(HTTPException)
def handle_http_exception(error):
    if request.path.startswith('/api/'):
        return api_error(error.description or '请求失败', error.code or 500)
    return error


@app.errorhandler(Exception)
def handle_unexpected_exception(error):
    if request.path.startswith('/api/'):
        return api_error('服务器内部错误', 500, detail=str(error))
    raise error


# ==============================================================
# 入口
# ==============================================================
if __name__ == '__main__':
    print('=' * 50)
    print('  非言语行为分析平台')
    print('  http://127.0.0.1:5050')
    print('=' * 50)
    app.run(host='0.0.0.0', port=5050, debug=True)
