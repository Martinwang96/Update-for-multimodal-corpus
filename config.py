#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
统一配置文件
集中管理 OpenFace / MMPose 等外部工具路径和参数
'''

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# ==============================================================
# OpenFace 配置（面部/头部视频→CSV）
# ==============================================================
# OpenFace 的 FeatureExtraction 可执行文件路径
# 优先读环境变量 OPENFACE_BIN，其次用常见路径自动探测
OPENFACE_BIN = os.environ.get('OPENFACE_BIN', '')

# 常见安装路径（按顺序探测）
_OPENFACE_CANDIDATES = [
    '/usr/local/bin/FeatureExtraction',
    '/opt/homebrew/bin/FeatureExtraction',
    str(Path.home() / 'OpenFace' / 'build' / 'bin' / 'FeatureExtraction'),
    str(Path.home() / 'openface' / 'build' / 'bin' / 'FeatureExtraction'),
    '/Applications/OpenFace/build/bin/FeatureExtraction',
]

def find_openface():
    """自动探测 OpenFace 可执行文件。"""
    if OPENFACE_BIN and os.path.isfile(OPENFACE_BIN):
        return OPENFACE_BIN
    for p in _OPENFACE_CANDIDATES:
        if os.path.isfile(p):
            return p
    return None

OPENFACE_PATH = find_openface()

# ==============================================================
# MMPose 配置（躯体视频→关键点JSON）
# ==============================================================
# 设备：'cpu' 或 'cuda'
MMPONSE_DEVICE = os.environ.get('MMPONSE_DEVICE', 'cpu')
# 2D 姿态识别模型
MMPONSE_POSE2D = os.environ.get('MMPONSE_POSE2D', 'rtmo')

# ==============================================================
# 目录配置
# ==============================================================
UPLOAD_DIR = BASE_DIR / 'uploads'
RESULT_DIR = BASE_DIR / 'results'
UPLOAD_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)

# ==============================================================
# 分析默认参数
# ==============================================================
# 面部表情
FACE_DEFAULTS = {
    'min_duration': 1.0,
    'happy_au6_r': 1.5, 'happy_au7_r': 1.5, 'happy_au12_r': 1.5,
    'happy_au25_r': 1.5, 'happy_au26_r': 1.5,
    'surprise_au26_r': 1.5,
    'confused_au4_r': 1.5,
    'focused_au5_r': 1.5, 'focused_au14_r': 1.5,
}

# 头部运动
HEAD_DEFAULTS = {
    'pitch': {'min_duration': 0.5, 'max_duration': 0},
    'roll': {'min_duration': 0.25, 'max_duration': 2.5},
    'yaw': {'min_duration': 0.3, 'max_duration': 3.0},
}
