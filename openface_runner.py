#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
OpenFace FeatureExtraction 调用封装
职责：视频 -> OpenFace CSV 的转换
安全：列表传参（无 shell）、路径校验、超时控制、绝对路径调用
'''

import os
import shutil
import subprocess
from pathlib import Path

# FeatureExtraction 候选路径（按优先级）：
#  1. 环境变量 OPENFACE_BIN（文件或目录）
#  2. 常见编译产物路径
#  3. PATH 中的 FeatureExtraction
_DEFAULT_CANDIDATES = [
    '/usr/local/bin/FeatureExtraction',
    '/opt/homebrew/bin/FeatureExtraction',
    os.path.expanduser('~/OpenFace/build/bin/FeatureExtraction'),
    os.path.expanduser('~/openface/build/bin/FeatureExtraction'),
    os.path.expanduser('~/Downloads/OpenFace/build/bin/FeatureExtraction'),
]

# 允许的视频扩展名
ALLOWED_VIDEO_EXT = ('.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv')


def find_openface_binary():
    """返回 FeatureExtraction 绝对路径，找不到返回 None。"""
    env_path = os.environ.get('OPENFACE_BIN', '').strip()
    if env_path:
        p = Path(env_path)
        if p.is_file():
            return str(p)
        # 可能传入的是 OpenFace 根目录或 build 目录
        for sub in ('bin/FeatureExtraction', 'FeatureExtraction'):
            cand = p / sub
            if cand.is_file():
                return str(cand)
    for c in _DEFAULT_CANDIDATES:
        if Path(c).is_file():
            return c
    found = shutil.which('FeatureExtraction')
    return found  # 可能为 None


def is_available():
    """OpenFace 是否可用。"""
    return find_openface_binary() is not None


def _resolve(p):
    """解析为绝对路径，避免符号链接造成的相对路径问题。"""
    return str(Path(p).resolve())


def run_openface(video_path, out_dir, timeout=3600):
    """
    调用 FeatureExtraction 将视频转为 CSV。

    参数：
        video_path: 视频文件路径（绝对路径最佳）
        out_dir:    CSV 输出目录
        timeout:    超时秒数（默认 1 小时，兜底防卡死）
    返回：
        dict: {
            'csv_path': 生成的 CSV 绝对路径,
            'stdout':   标准输出,
            'stderr':   标准错误,
            'binary':   使用的可执行文件路径,
        }
    异常：
        RuntimeError: 未安装 / 处理失败 / 超时
    """
    binary = find_openface_binary()
    if not binary:
        raise RuntimeError(
            '未找到 OpenFace 的 FeatureExtraction 可执行文件。\n'
            '请通过环境变量 OPENFACE_BIN 指定其绝对路径，例如：\n'
            '  export OPENFACE_BIN=/path/to/OpenFace/build/bin/FeatureExtraction\n'
            '编译说明: https://github.com/TadasBaltrusaitis/OpenFace/wiki'
        )

    video_path = _resolve(video_path)
    if not Path(video_path).is_file():
        raise RuntimeError(f'视频文件不存在: {video_path}')

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_dir_abs = _resolve(out_dir)

    # 列表传参，绝不使用 shell=True，防止命令注入
    # -2Dfp/-3Dfp/-pose/-aus/-gaze 全开，保证面部/头部/眼动分析所需列齐全
    cmd = [
        binary,
        '-f', video_path,
        '-out_dir', out_dir_abs,
        '-q',       # 静默，不弹 GUI 窗口
        '-2Dfp',    # 2D 面部关键点
        '-3Dfp',    # 3D 面部关键点
        '-pose',    # 头部姿态 (pose_Rx/Ry/Rz)
        '-aus',     # Action Units（表情/眼部检测依赖）
        '-gaze',    # 眼动数据
    ]

    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(f'OpenFace 处理超时（超过 {timeout} 秒），视频可能过长')

    stdout = proc.stdout.decode('utf-8', errors='replace')
    stderr = proc.stderr.decode('utf-8', errors='replace')

    if proc.returncode != 0:
        raise RuntimeError(
            f'OpenFace 处理失败（返回码 {proc.returncode}）:\n'
            f'{stderr[-2000:]}'
        )

    # FeatureExtraction 对 video.mp4 默认生成 video.csv
    base = Path(video_path).stem
    csv_path = out_dir / f'{base}.csv'
    if not csv_path.is_file():
        # 兜底：取目录下最新生成的 CSV
        csvs = sorted(
            out_dir.glob('*.csv'),
            key=lambda f: f.stat().st_mtime,
            reverse=True,
        )
        if not csvs:
            raise RuntimeError(
                'OpenFace 执行完成但未找到输出 CSV。\n'
                f'stdout 末尾: {stdout[-1000:]}\n'
                f'stderr 末尾: {stderr[-1000:]}'
            )
        csv_path = csvs[0]

    return {
        'csv_path': str(csv_path),
        'stdout': stdout,
        'stderr': stderr,
        'binary': binary,
    }
