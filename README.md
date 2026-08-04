# 非言语行为分析平台

基于 OpenFace 和 MMPose 的多模态非言语行为（体态语言）分析系统，整合面部表情、头部运动、躯体动作三大模块为统一的 Web 应用。

## 目录结构

```
项目代码优化/
├── app.py                      # 统一 Flask Web 应用入口
├── requirements.txt            # Python 依赖（版本范围兼容）
├── test_smoke.py               # 冒烟测试
├── analyzers/                  # 分析器统一封装包
│   ├── __init__.py
│   ├── face_analyzer.py        # 面部分析封装
│   ├── head_analyzer.py        # 头部分析封装
│   └── body_analyzer.py        # 躯体分析封装
├── 面部/                        # 面部分析模块
│   ├── face_common.py          # 面部共享工具
│   ├── facialexpression.py     # 表情检测
│   ├── mi&zha&bi.py            # 眼部状态检测
│   └── fix&sca-0.6&0.6.py      # 眼动分析
├── 头部/                        # 头部运动模块
│   ├── head_common.py          # 头部共享分析模块
│   ├── 抬头低头.py              # pitch 入口
│   ├── 左倾右倾.py              # roll 入口
│   └── 转头摇头.py              # yaw 入口
├── 躯体/                        # 躯体动作模块
│   ├── 综合处理-2d.py           # 统一流水线
│   ├── 综合处理-web.py          # 原独立 Web 版（参考）
│   ├── body_recognition_runner.py
│   ├── 倾斜-2d.py / 耸肩-2d.py / 位移-2d.py / 转动-2d.py
│   ├── templates/ static/ uploads/ combined_2d_runs/
└── templates/                  # 统一 Web 模板
    └── index.html
```

## 安装

```bash
cd 项目代码优化
pip install -r requirements.txt
```

> **pip 兼容性**：所有依赖使用 `>=min,<next_major` 范围，兼容 Python 3.8+。

### 安装 MMPose（CPU 版，躯体视频识别必需）

躯体分析流程是「上传视频 → MMPose 识别 → 关键点 JSON → 4 模块分析 → 事件 CSV」，MMPose 是核心依赖。CPU 版无需 CUDA，支持 macOS / Linux / Windows：

```bash
# 1. 安装 PyTorch CPU 版（推荐 conda，也可用 pip）
conda install pytorch torchvision cpuonly -c pytorch
# 或 pip: pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 2. 安装 OpenMMLab 工具
pip install -U openmim

# 3. 安装 MMEngine 和 MMCV（注意：mmpose 1.x 需要 mmcv 2.x）
mim install mmengine
mim install "mmcv>=2.0.1"

# 4. 安装 MMPose
mim install "mmpose>=1.1.0"
```

验证安装：
```python
from mmpose.apis import MMPoseInferencer
inferencer = MMPoseInferencer('rtmo', device='cpu')
```

> 若已有现成的关键点 JSON，可跳过识别阶段，不装 mmpose 也能用 `analyze_from_json()`。

### 安装 OpenFace（面部/头部 AU 与 pose 检测，C++ 源码编译）

面部和头部分析的流程是「上传视频 → OpenFace `FeatureExtraction` → CSV（AU + pose_Rx/Ry/Rz）→ 分析模块」。OpenFace 不是 pip 包，需从源码编译：

```bash
# macOS
brew install openblas opencv boost cmake
git clone https://github.com/TadasBaltrusaitis/OpenFace.git
cd OpenFace
mkdir build && cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE ..
make
./download_models.sh
# 产物：bin/FeatureExtraction
```

详见 [OpenFace Wiki](https://github.com/TadasBaltrusaitis/OpenFace/wiki)。

## 运行

```bash
python app.py
```

浏览器打开 http://127.0.0.1:5000

## 功能

| 模块 | 输入 | 输出 |
|------|------|------|
| 面部表情 | OpenFace CSV | 微笑/张嘴/皱眉/专注 区间 + CSV 报告 |
| 眼部状态 | OpenFace CSV | 眯眼/闭眼/眨眼 区间 + CSV 报告 |
| 头部运动 | OpenFace CSV | 抬头/低头/左倾/右倾/转头/摇头 区间 + ELAN CSV |
| 躯体动作 | 关键点 JSON 或视频 | 倾斜/耸肩/位移/转动 事件 + 汇总 JSON |

## 测试

```bash
python test_smoke.py
```

## 技术栈

Flask · pandas · numpy · scipy · scikit-learn · matplotlib · MMPose(可选)

---
Author: Martinwang96 · Copyright (c) 2025 by Martin Wang, SISU
