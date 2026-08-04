#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
冒烟测试 — 验证统一应用的核心组件可正常导入和实例化
运行: python test_smoke.py
'''

import os
import sys
import traceback

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

PASS = 0
FAIL = 0


def check(name, func):
    global PASS, FAIL
    try:
        func()
        print(f'  [PASS] {name}')
        PASS += 1
    except Exception as e:
        print(f'  [FAIL] {name}: {e}')
        traceback.print_exc()
        FAIL += 1


def test_imports():
    import flask, pandas, numpy, scipy, sklearn, matplotlib
    from importlib.metadata import version
    print(f'    Flask {version("flask")}, pandas {version("pandas")}, numpy {version("numpy")}')


def test_analyzers_package():
    from analyzers import FaceAnalyzer, HeadAnalyzer, BodyAnalyzer
    print('    analyzers 导出: FaceAnalyzer, HeadAnalyzer, BodyAnalyzer')


def test_face_analyzer():
    from analyzers import FaceAnalyzer
    fa = FaceAnalyzer()
    mods = fa.list_modules()
    assert 'expression' in mods and 'eye' in mods
    print(f'    面部模块: {mods}')


def test_head_analyzer():
    from analyzers import HeadAnalyzer
    ha = HeadAnalyzer()
    axes = ha.list_axes()
    assert 'pitch' in axes and 'roll' in axes and 'yaw' in axes
    print(f'    头部轴: {axes}')


def test_body_analyzer():
    from analyzers import BodyAnalyzer
    ba = BodyAnalyzer()
    mods = ba.list_modules()
    assert 'tilt' in mods and 'shrug' in mods
    # 真正尝试加载 pipeline 模块（综合处理-2d.py 本身不依赖 mmpose）
    from analyzers.body_analyzer import _load_pipeline_module
    mod = _load_pipeline_module()
    assert hasattr(mod, 'run_pipeline'), '综合处理-2d.py 缺少 run_pipeline'
    print(f'    躯体模块: {mods}, pipeline 可加载: True')


def test_body_video_deps():
    """检测躯体视频识别所需的 mmpose 是否可用（视频输入主流程依赖）。"""
    try:
        from mmpose.apis import MMPoseInferencer  # noqa: F401
        print('    mmpose 可用：视频识别路径就绪')
        return 'ok'
    except ImportError:
        print('    mmpose 未安装：视频识别主路径不可用')
        raise  # 视为主流程依赖，FAIL 而非 SKIP


def test_head_common_config():
    sys.path.insert(0, os.path.join(BASE_DIR, '头部'))
    import head_common
    for axis in ['pitch', 'roll', 'yaw']:
        cfg = head_common.AXIS_CONFIGS[axis]
        for key in ['pose_col', 'deg_col', 'title', 'subtitle', 'upper_action', 'lower_action']:
            assert key in cfg, f'{axis} 缺少 {key}'
    print('    三轴配置完整: pitch/roll/yaw')


def test_flask_app_import():
    import app as flask_app
    assert flask_app.app is not None
    print('    Flask app 实例化成功')


def test_flask_routes():
    import app as flask_app
    client = flask_app.app.test_client()
    resp = client.get('/')
    assert resp.status_code == 200, f'首页返回 {resp.status_code}'
    resp = client.get('/api/modules')
    assert resp.status_code == 200, f'/api/modules 返回 {resp.status_code}'
    import json
    data = json.loads(resp.data)
    assert 'face' in data and 'head' in data and 'body' in data
    print('    路由: GET / -> 200, GET /api/modules -> OK')


def test_template_exists():
    tpl = os.path.join(BASE_DIR, 'templates', 'index.html')
    assert os.path.isfile(tpl), f'模板不存在: {tpl}'
    with open(tpl, encoding='utf-8') as f:
        content = f.read()
    assert '非言语行为分析' in content
    print('    templates/index.html 存在且内容正常')


def test_requirements():
    req = os.path.join(BASE_DIR, 'requirements.txt')
    assert os.path.isfile(req), 'requirements.txt 不存在'
    with open(req) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                assert '>' in line or '=' in line, f'依赖行无版本约束: {line}'
    print('    requirements.txt 格式正确')


if __name__ == '__main__':
    print('=' * 50)
    print('  冒烟测试 — 非言语行为分析平台')
    print('=' * 50)

    print('\n[1/10] 核心依赖导入')
    check('imports', test_imports)

    print('\n[2/10] analyzers 包')
    check('analyzers package', test_analyzers_package)

    print('\n[3/10] FaceAnalyzer')
    check('FaceAnalyzer', test_face_analyzer)

    print('\n[4/10] HeadAnalyzer')
    check('HeadAnalyzer', test_head_analyzer)

    print('\n[5/10] BodyAnalyzer（含 pipeline 加载）')
    check('BodyAnalyzer', test_body_analyzer)

    print('\n[6/10] 躯体视频路径依赖 (mmpose)')
    check('body video deps (mmpose)', test_body_video_deps)

    print('\n[7/10] head_common 配置')
    check('head_common config', test_head_common_config)

    print('\n[8/10] Flask app 导入')
    check('flask app import', test_flask_app_import)

    print('\n[9/10] Flask 路由')
    check('flask routes', test_flask_routes)

    print('\n[10/10] 模板与依赖文件')
    check('template exists', test_template_exists)
    check('requirements', test_requirements)

    print('\n' + '=' * 50)
    print(f'  结果: {PASS} 通过, {FAIL} 失败')
    print('=' * 50)
    sys.exit(0 if FAIL == 0 else 1)
